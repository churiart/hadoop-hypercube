"""
Construct a cluster, run the simulator for two strategies
(Default vs Hypercube) and produce a compact summary + basic plots.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from .nodes import (
    DataNode, generate_nodes, kmeans_cluster, compute_centroid_bit_mapping,
    assign_centroids_and_addresses, build_registry
)
from .placement import compute_decile_cutpoints
from .simulator_v2 import execute_single_strategy_simulation
from .config import default_runtime_params, default_spec_params
from .placement import CONTROLLED_TARGETS, CONTROLLED_WEIGHTS
from .runtime import metrics_to_percentiles
from . import runtime as rt
import sys
import subprocess

# Shared, strategy-agnostic params
rt_params = default_runtime_params()
# Enable Hamming-based compute scaling - penalty on centroid distance
rt_params.enable_hamming_penalty = True
rt_params.hamming_gamma = 3.0
sp_params = default_spec_params()

# --- Global Simulation Parameters ---
NUM_NODES = 400
NUM_RACKS = 10
K = 81
NUM_BLOCKS = 1000
SEED = 42

plt.style.use('ieee')
plt.rcParams.update({'font.serif': ['Times New Roman'], 'font.family': 'serif'})

def _build_cluster(num_nodes: int = NUM_NODES, num_racks: int = NUM_RACKS, k: int = K, seed: int = SEED):
    print("Generating canonical cluster layout...")
    nodes = generate_nodes(num_nodes, num_racks, seed=seed, variance_factor=1.0)

    labels, centroids = kmeans_cluster(nodes, k, seed=seed)

    b_r = int(math.ceil(math.log2(num_racks)))
    b_c = int(math.ceil(math.log2(k)))

    mapping = compute_centroid_bit_mapping(centroids, num_bits=b_c)
    assign_centroids_and_addresses(nodes, labels, b_r, b_c, mapping)

    registry = build_registry(nodes)
    cutpoints = compute_decile_cutpoints(nodes)

    # --- Minimal wiring for Hamming-based compute penalty ---
    ideal_bits_by_tag = {}
    dims = ['cpu', 'ram', 'disk', 'net']
    def _cost_on_centroid(c_vec: np.ndarray, targets: dict, weights: dict) -> float:
        p = metrics_to_percentiles(c_vec, cutpoints)
        cost = 0.0
        for i, dim in enumerate(dims):
            t = targets[dim]
            w = weights[dim]
            gap = max(0.0, (t - p[i]) / max(t, 1e-9))
            cost += w * gap
        return cost

    addr_len = len(nodes[0].addr_bits)
    b_c = int(math.ceil(math.log2(k)))
    b_r = addr_len - b_c
    centroid_bits_by_id = {}
    for n in nodes:
        if n.centroid_id not in centroid_bits_by_id:
            centroid_bits_by_id[n.centroid_id] = n.addr_bits[-b_c:]

    # Determine ideal centroid bits for the default profile)
    for tag, targets in CONTROLLED_TARGETS.items():
        weights = CONTROLLED_WEIGHTS[tag]
        scores = {cid: _cost_on_centroid(centroids[cid], targets, weights) for cid in range(len(centroids))}
        ideal_cid = min(scores, key=scores.get)
        ideal_bits_by_tag[tag] = centroid_bits_by_id.get(ideal_cid, format(ideal_cid, f"0{b_c}b"))

    rt.set_hamming_context(b_r=b_r, b_c=b_c, ideal_bits_by_tag=ideal_bits_by_tag)

    total_capacity = sum(n.max_replicas for n in nodes)
    print(f"Total cluster capacity: {total_capacity} replicas.")
    return nodes, registry, centroids, cutpoints

def _run_one_strategy(label: str, nodes, registry, centroids, cutpoints, speculative: bool, num_blocks: int = None) -> Dict:
    print(f"  Executing {label} strategy...")
    return execute_single_strategy_simulation(
        num_blocks=NUM_BLOCKS if num_blocks is None else num_blocks,
        tag_distribution={
            "balanced": 0.4,
            "cpu-bound": 0.2,
            "disk-bound": 0.2,
            "ram-bound": 0.1,
            "net-bound": 0.1,
        },
        nodes=nodes,
        registry=registry,
        centroids=centroids,
        cutpoints=cutpoints,
        placement_strategy="default" if label.lower().startswith("default") else "hyper",
        speculative=speculative,
        num_replicas=3,
        progress_every=20,
        idle_stall_limit=2000,
        collect_details=True,
        runtime_params=rt_params,
        spec_params=sp_params,
    )


def _apply_stragglers(nodes, fraction: float = 0.1, seed: int = 42, degrade_factor: float = 0.5) -> None:
    """
    Degrade a small fraction of nodes' s_vectors to emulate stragglers.
    """
    if not nodes or fraction <= 0:
        return
    random.seed(seed)
    num = max(1, int(len(nodes) * fraction))
    chosen_ids = set(random.sample([n.node_id for n in nodes], num))

    # Degrade in the nodes list copy (CPU, RAM, DISK only; keep NET to avoid network penalties on JCT)
    nid_to_node = {n.node_id: n for n in nodes}
    for nid in chosen_ids:
        node = nid_to_node.get(nid)
        if node is not None:
            sv = node.s_vector.copy()
            sv[:3] = sv[:3] * degrade_factor 
            node.s_vector = sv

def plot_summary(results: Dict[str, Dict], output_dir: str = "dist") -> None:
    labels = list(results.keys())
    net = [results[k]["network_cost_MB"] for k in labels]
    req = [results[k]["spec"]["requests"] for k in labels]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    bars1 = axs[0].bar(labels, net)
    #axs[0].set_title("Network MB (speculative non-local reads)")
    axs[0].set_ylim(bottom=0)

    bars2 = axs[1].bar(labels, req)
    #axs[1].set_title("Speculation requests")
    axs[1].set_ylim(bottom=0)
    
    for bar, val in zip(bars1, net):
        axs[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.1f}', ha='center', va='bottom')
    for bar, val in zip(bars2, req):
        axs[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val}', ha='center', va='bottom')

    for k in labels:
        arr = np.sort(np.array(results[k].get("completion_times", []), dtype=float))
        if arr.size == 0:
            continue
        y = np.arange(1, arr.size + 1, dtype=float) / float(arr.size)
        x_clip = np.percentile(arr, 99) if arr.size > 1 else arr[-1]
        axs[2].step(np.clip(arr, arr[0], x_clip), y, where="post", label=k)
    #axs[2].set_title("CDF of completion times")
    axs[2].legend()
    plt.tight_layout()

    try:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "summary.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    except Exception as e:
        print(f"Warning: failed to save plot: {e}")

    #plt.show()


def plot_replica_and_task_distributions(results_by_condition: Dict[str, Dict[str, Dict]], output_dir: str = "dist") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for cond, data in results_by_condition.items():
        # Replica counts
        rep_default = data.get("default", {}).get("replica_counts_per_node", {})
        rep_hyper = data.get("hyper", {}).get("replica_counts_per_node", {})
        if rep_default and rep_hyper:
            node_ids = sorted(set(rep_default.keys()) | set(rep_hyper.keys()))
            vals_def = [rep_default.get(n, 0) for n in node_ids]
            vals_hyp = [rep_hyper.get(n, 0) for n in node_ids]
            import numpy as np
            order = np.argsort(vals_def)
            x = np.arange(len(node_ids))
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(x, np.array(vals_def)[order], label="Default", lw=1.6)
            ax.plot(x, np.array(vals_hyp)[order], label="Hypercube", lw=1.6)
            #ax.set_title(f"Replica (block) count per DataNode ({cond})")
            ax.set_xlabel("Node index (sorted by Default)")
            ax.set_ylabel("Replica count")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"replicas_per_node_{cond}.png"), dpi=150)
            plt.close(fig)

        # Tasks per node
        tasks_default = data.get("default", {}).get("tasks_per_node", {})
        tasks_hyper = data.get("hyper", {}).get("tasks_per_node", {})
        if tasks_default and tasks_hyper:
            node_ids = sorted(set(tasks_default.keys()) | set(tasks_hyper.keys()))
            vals_def = [tasks_default.get(n, 0) for n in node_ids]
            vals_hyp = [tasks_hyper.get(n, 0) for n in node_ids]
            import numpy as np
            order = np.argsort(vals_def)
            x = np.arange(len(node_ids))
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(x, np.array(vals_def)[order], label="Default", lw=1.6)
            ax.plot(x, np.array(vals_hyp)[order], label="Hypercube", lw=1.6)
            #ax.set_title(f"Tasks per DataNode ({cond})")
            ax.set_xlabel("Node index (sorted by Default)")
            ax.set_ylabel("Tasks")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"tasks_per_node_{cond}.png"), dpi=150)
            plt.close(fig)


def plot_per_strategy_sorted(results_by_condition: Dict[str, Dict[str, Dict]], output_dir: str = "dist") -> None:
    os.makedirs(output_dir, exist_ok=True)
    import numpy as np
    import matplotlib.pyplot as plt
    for cond, data in results_by_condition.items():
        t_def = data.get("default", {}).get("tasks_per_node", {})
        t_hyp = data.get("hyper", {}).get("tasks_per_node", {})
        if t_def and t_hyp:
            v_def = np.array(sorted(t_def.values()))
            v_hyp = np.array(sorted(t_hyp.values()))
            n_def, n_hyp = v_def.size, v_hyp.size
            xmax = max(n_def, n_hyp)
            ymax = float(max(v_def.max(initial=0), v_hyp.max(initial=0)))
            fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
            axs[0].plot(np.arange(n_def), v_def, color="C0")
            #axs[0].set_title(f"Default (sorted by Default)")
            axs[0].set_xlabel("Node index")
            axs[0].set_ylabel("Tasks")
            axs[1].plot(np.arange(n_hyp), v_hyp, color="C1")
            #axs[1].set_title(f"Hypercube (sorted by Hypercube)")
            axs[1].set_xlabel("Node index")
            for ax in axs:
                ax.set_ylim(0, max(1.0, ymax))
            total_def = int(v_def.sum())
            total_hyp = int(v_hyp.sum())
            axs[0].text(0.02, 0.94, f"Total={total_def}", transform=axs[0].transAxes, ha="left", va="top")
            axs[1].text(0.02, 0.94, f"Total={total_hyp}", transform=axs[1].transAxes, ha="left", va="top")
            #fig.suptitle(f"Tasks per DataNode (per-strategy sort) — {cond}")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"tasks_per_node_per_strategy_sorted_{cond}.png"), dpi=150)
            plt.close(fig)

        # Replicas: per-strategy sort
        r_def = data.get("default", {}).get("replica_counts_per_node", {})
        r_hyp = data.get("hyper", {}).get("replica_counts_per_node", {})
        if r_def and r_hyp:
            v_def = np.array(sorted(r_def.values()))
            v_hyp = np.array(sorted(r_hyp.values()))
            n_def, n_hyp = v_def.size, v_hyp.size
            ymax = float(max(v_def.max(initial=0), v_hyp.max(initial=0)))
            fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
            axs[0].plot(np.arange(n_def), v_def, color="C0")
            #axs[0].set_title(f"Default (sorted by Default)")
            axs[0].set_xlabel("Node index")
            axs[0].set_ylabel("Replica count")
            axs[1].plot(np.arange(n_hyp), v_hyp, color="C1")
            #axs[1].set_title(f"Hypercube (sorted by Hypercube)")
            axs[1].set_xlabel("Node index")
            for ax in axs:
                ax.set_ylim(0, max(1.0, ymax))
            total_def = int(v_def.sum())
            total_hyp = int(v_hyp.sum())
            axs[0].text(0.02, 0.94, f"Total={total_def}", transform=axs[0].transAxes, ha="left", va="top")
            axs[1].text(0.02, 0.94, f"Total={total_hyp}", transform=axs[1].transAxes, ha="left", va="top")
            #fig.suptitle(f"Replica count per DataNode (per-strategy sort) — {cond}")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"replicas_per_node_per_strategy_sorted_{cond}.png"), dpi=150)
            plt.close(fig)

def plot_spec_histograms(results_by_condition: Dict[str, Dict[str, Dict]], output_dir: str = "dist/spec") -> None:
    os.makedirs(output_dir, exist_ok=True)
    import numpy as np
    import matplotlib.pyplot as plt
    conditions = [k for k in ["no_stragglers", "with_stragglers"] if k in results_by_condition]

    # 1) Speculation Requests (straggler jobs)
    req_default = []
    req_hyper = []
    for cond in conditions:
        req_default.append(results_by_condition[cond]["default"].get("spec", {}).get("requests", 0))
        req_hyper.append(results_by_condition[cond]["hyper"].get("spec", {}).get("requests", 0))
    x = np.arange(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width/2, req_default, width, label="Default")
    bars2 = ax.bar(x + width/2, req_hyper, width, label="Hypercube")
    
    # Add value labels on top of each bar
    for bar, val in zip(bars1, req_default):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{val}', ha='center', va='bottom')
    for bar, val in zip(bars2, req_hyper):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{val}', ha='center', va='bottom')
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ') for c in conditions])
    ax.set_ylabel("Speculation requests")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "spec_requests.png"), dpi=150)
    plt.close(fig)

    # 2) Speculative Tasks (with network transfer)
    net_default = []
    net_hyper = []
    for cond in conditions:
        sdef = results_by_condition[cond]["default"].get("spec", {})
        shyp = results_by_condition[cond]["hyper"].get("spec", {})
        net_default.append(int(sdef.get("launched_rack", 0)) + int(sdef.get("launched_remote", 0)))
        net_hyper.append(int(shyp.get("launched_rack", 0)) + int(shyp.get("launched_remote", 0)))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width/2, net_default, width, label="Default")
    bars2 = ax.bar(x + width/2, net_hyper, width, label="Hypercube")

    for bar, val in zip(bars1, net_default):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{val}', ha='center', va='bottom')
    for bar, val in zip(bars2, net_hyper):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{val}', ha='center', va='bottom')
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ') for c in conditions])
    ax.set_ylabel("Speculative requests")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "spec_tasks_remote.png"), dpi=150)
    plt.close(fig)


def plot_spec_remote_reasons(results_by_condition: Dict[str, Dict[str, Dict]], output_dir: str = "dist/spec") -> None:
    os.makedirs(output_dir, exist_ok=True)
    import numpy as np
    import matplotlib.pyplot as plt
    conditions = [k for k in ["no_stragglers", "with_stragglers"] if k in results_by_condition]

    def get_counts(method: str, cond: str, key: str) -> int:
        return int(results_by_condition[cond][method].get("spec", {}).get(key, 0))

    data = {
        cond: {
            "default": (
                get_counts("default", cond, "remote_due_to_no_local"),
                get_counts("default", cond, "remote_despite_local"),
            ),
            "hyper": (
                get_counts("hyper", cond, "remote_due_to_no_local"),
                get_counts("hyper", cond, "remote_despite_local"),
            ),
        }
        for cond in conditions
    }

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    keys = ("remote_due_to_no_local", "remote_despite_local")
    labels = ("No local slots", "Despite local slots")

    for ax, method, color_base in zip(
        axs, ["default", "hyper"], ["C0", "C1"],
    ):
        no_local = [data[c][method][0] for c in conditions]
        despite = [data[c][method][1] for c in conditions]
        x = np.arange(len(conditions))
        width = 0.35
        ax.bar(x - width/2, no_local, width, label=labels[0], color=color_base)
        ax.bar(x + width/2, despite, width, label=labels[1], color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', ' ') for c in conditions])
        #ax.set_title(method.capitalize())
        ax.set_ylabel("Remote spec starts")
        ax.legend()

    #fig.suptitle("Remote speculative starts: reasons")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "spec_remote_reasons.png"), dpi=150)
    plt.close(fig)

def _print_completion_percentiles(condition: str, res_default: Dict, res_hyper: Dict) -> None:
    def _pcts(arr):
        a = np.array(arr, dtype=float)
        if a.size == 0:
            return (0.0, 0.0)
        return (float(np.percentile(a, 95)), float(np.percentile(a, 99)))
    p95_d, p99_d = _pcts(res_default.get("completion_times", []))
    p95_h, p99_h = _pcts(res_hyper.get("completion_times", []))
    print(f"\n[Completion percentiles] {condition}")
    print(f"  Default : p95={p95_d:.3f}, p99={p99_d:.3f}")
    print(f"  Hypercube: p95={p95_h:.3f}, p99={p99_h:.3f}")

def _print_cdf_completion_times(condition: str, res_default: Dict, res_hyper: Dict) -> None:
    def _cdf_stats(arr):
        a = np.array(arr, dtype=float)
        if a.size == 0:
            return []
        sorted_arr = np.sort(a)
        # Calculate percentiles at key points
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        values = [float(np.percentile(sorted_arr, p)) for p in percentiles]
        return values
    
    cdf_default = _cdf_stats(res_default.get("completion_times", []))
    cdf_hyper = _cdf_stats(res_hyper.get("completion_times", []))
    
    print(f"\n[CDF completion times] {condition}")
    print(f"  Percentile:  10%    25%    50%    75%    90%    95%    99%")
    
    if cdf_default:
        print(f"  Default : {cdf_default[0]:6.3f} {cdf_default[1]:6.3f} {cdf_default[2]:6.3f} {cdf_default[3]:6.3f} {cdf_default[4]:6.3f} {cdf_default[5]:6.3f} {cdf_default[6]:6.3f}")
    else:
        print(f"  Default : No completion times available")
    
    if cdf_hyper:
        print(f"  Hypercube: {cdf_hyper[0]:6.3f} {cdf_hyper[1]:6.3f} {cdf_hyper[2]:6.3f} {cdf_hyper[3]:6.3f} {cdf_hyper[4]:6.3f} {cdf_hyper[5]:6.3f} {cdf_hyper[6]:6.3f}")
    else:
        print(f"  Hypercube: No completion times available")

def plot_completion_percentiles(results_by_condition: Dict[str, Dict[str, Dict]], output_dir: str = "dist/percentiles") -> None:

    #Plot p95 and p99 completion times per strategy, per condition.
    os.makedirs(output_dir, exist_ok=True)

    def _pcts(arr):
        a = np.array(arr, dtype=float)
        if a.size == 0:
            return (0.0, 0.0)
        return (float(np.percentile(a, 95)), float(np.percentile(a, 99)))

    for cond, pair in results_by_condition.items():
        res_def = pair.get("default", {})
        res_hyp = pair.get("hyper", {})
        p95_d, p99_d = _pcts(res_def.get("completion_times", []))
        p95_h, p99_h = _pcts(res_hyp.get("completion_times", []))

        fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        # p95
        axs[0].bar(["Default", "Hypercube"], [p95_d, p95_h], color=["C0", "C1"])
        #axs[0].set_title("p95")
        axs[0].set_ylabel("Completion time (a.u.)")
        for x, v in zip([0, 1], [p95_d, p95_h]):
            axs[0].text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        # p99
        axs[1].bar(["Default", "Hypercube"], [p99_d, p99_h], color=["C0", "C1"])
        #axs[1].set_title("p99")
        for x, v in zip([0, 1], [p99_d, p99_h]):
            axs[1].text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        #fig.suptitle(f"Completion time percentiles — {cond.replace('_', ' ')}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{cond}_p95_p99.png"), dpi=150)
        plt.close(fig) 

def run_systematic_degradation_analysis_alongside() -> None:
    try:
        print("\n--- Running Systematic Degradation Analysis ---")
        subprocess.run([sys.executable, "systematic_degradation_analysis.py"], check=False)
    except Exception as e:
        print(f"Warning: failed to run systematic degradation analysis: {e}")

def main() -> Dict[str, Dict]:
    nodes, registry, centroids, cutpoints = _build_cluster()

    import copy
    results = {}
    jct_results = {}

    print("\n\n--- Running Simulation Condition: no_stragglers ---")
    res_default = _run_one_strategy("Default HDFS", copy.deepcopy(nodes), registry, centroids, cutpoints, speculative=True)

    nodes_hyper = copy.deepcopy(nodes)
    registry_hyper = build_registry(nodes_hyper)
    res_hyper   = _run_one_strategy("Hypercube",   nodes_hyper, registry_hyper, centroids, cutpoints, speculative=True)
    results["no_stragglers"] = {"default": res_default, "hyper": res_hyper}

    jct_results["no_stragglers"] = {
        "default": res_default.get("individual_job_completion_times", []),
        "hyper": res_hyper.get("individual_job_completion_times", [])
    }

    # Quick plots
    flat_no = {"Default": res_default, "Hypercube": res_hyper}
    plot_summary(flat_no, output_dir=os.path.join("dist", "no_stragglers"))
    _print_completion_percentiles("no_stragglers", res_default, res_hyper)
    _print_cdf_completion_times("no_stragglers", res_default, res_hyper)

    # Degrade a larger fraction of nodes to trigger more speculation
    print("\n\n--- Running Simulation Condition: with_stragglers ---")
    nodes_ws = copy.deepcopy(nodes)
    _apply_stragglers(nodes_ws, fraction=0.3, seed=SEED)
    registry_ws = build_registry(nodes_ws)
    res_default_ws = _run_one_strategy("Default HDFS", nodes_ws, registry_ws, centroids, cutpoints, speculative=True)
    # Recreate nodes copy for hypercube to ensure equal starting point
    nodes_ws_h = copy.deepcopy(nodes)
    _apply_stragglers(nodes_ws_h, fraction=0.3, seed=SEED)
    registry_ws_h = build_registry(nodes_ws_h)
    res_hyper_ws   = _run_one_strategy("Hypercube",   nodes_ws_h, registry_ws_h, centroids, cutpoints, speculative=True)
    results["with_stragglers"] = {"default": res_default_ws, "hyper": res_hyper_ws}

    jct_results["with_stragglers"] = {
        "default": res_default_ws.get("individual_job_completion_times", []),
        "hyper": res_hyper_ws.get("individual_job_completion_times", [])
    }

    flat_ws = {"Default": res_default_ws, "Hypercube": res_hyper_ws}
    plot_summary(flat_ws, output_dir=os.path.join("dist", "with_stragglers"))
    _print_completion_percentiles("with_stragglers", res_default_ws, res_hyper_ws)
    _print_cdf_completion_times("with_stragglers", res_default_ws, res_hyper_ws)

    plot_replica_and_task_distributions(
        {"no_stragglers": {"default": res_default, "hyper": res_hyper},
         "with_stragglers": {"default": res_default_ws, "hyper": res_hyper_ws}},
        output_dir=os.path.join("dist", "distributions")
    )
    plot_per_strategy_sorted(
        {"no_stragglers": {"default": res_default, "hyper": res_hyper},
         "with_stragglers": {"default": res_default_ws, "hyper": res_hyper_ws}},
        output_dir=os.path.join("dist", "per_strategy")
    )
    plot_spec_histograms(
        {"no_stragglers": {"default": res_default, "hyper": res_hyper},
         "with_stragglers": {"default": res_default_ws, "hyper": res_hyper_ws}},
        output_dir=os.path.join("dist", "spec")
    )
    plot_spec_remote_reasons(
        {"no_stragglers": {"default": res_default, "hyper": res_hyper},
         "with_stragglers": {"default": res_default_ws, "hyper": res_hyper_ws}},
        output_dir=os.path.join("dist", "spec")
    )

    plot_completion_percentiles(
        {"no_stragglers": {"default": res_default, "hyper": res_hyper},
         "with_stragglers": {"default": res_default_ws, "hyper": res_hyper_ws}},
        output_dir=os.path.join("dist", "percentiles")
    )

    plot_jct_percentiles(jct_results, output_dir=os.path.join("dist", "jct_percentiles"))
    plot_jct_cdf(jct_results, output_dir=os.path.join("dist", "jct_cdf"))

    # run_sensitivity_analysis()

    return results

def plot_jct_percentiles(jct_results: Dict[str, Dict[str, List[float]]], output_dir: str = "dist/jct_percentiles") -> None:
    """p95 and p99 JCT per strategy, per condition."""
    os.makedirs(output_dir, exist_ok=True)

    def _pcts(arr_list):
        if not arr_list:
            return (0.0, 0.0)
        a = np.array(arr_list, dtype=float)
        if a.size == 0:
            return (0.0, 0.0)
        q = [95, 99]
        if a.size == 1:
             p_vals = [float(a[0]), float(a[0])]
        else:
             p_vals = [float(np.percentile(a, p)) for p in q]
        return tuple(p_vals)

    conditions = ["no_stragglers", "with_stragglers"]
    percentile_values = {}

    print("\n--- JCT percentiles ---")
    for cond in conditions:
        print(f"\nCondition: {cond.replace('_', ' ')}")
        percentile_values[cond] = {}
        res_def_jcts = jct_results.get(cond, {}).get("default", [])
        res_hyp_jcts = jct_results.get(cond, {}).get("hyper", [])

        p95_d, p99_d = _pcts(res_def_jcts)
        p95_h, p99_h = _pcts(res_hyp_jcts)
        percentile_values[cond]["default"] = (p95_d, p99_d)
        percentile_values[cond]["hyper"] = (p95_h, p99_h)
        print(f"  Default : p95={p95_d:.3f}, p99={p99_d:.3f} (from {len(res_def_jcts)} jobs)")
        print(f"  Hypercube: p95={p95_h:.3f}, p99={p99_h:.3f} (from {len(res_hyp_jcts)} jobs)")

        fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
        strategies = ["Default", "Hypercube"]

        p95_vals = [p95_d, p95_h]
        p99_vals = [p99_d, p99_h]

        max_val_p95 = max(p95_vals) if p95_vals else 0
        max_val_p99 = max(p99_vals) if p99_vals else 0
        upper_limit = max(max_val_p95, max_val_p99) * 1.15

        bars_95 = axs[0].bar(strategies, p95_vals, color=["C0", "C1"])
        #axs[0].set_title("P95 JCT")
        axs[0].set_ylabel("Job Completion Time")
        for bar, val in zip(bars_95, p95_vals):
            axs[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}",
                        ha="center", va="bottom", fontsize=9)

        bars_99 = axs[1].bar(strategies, p99_vals, color=["C0", "C1"])
        #axs[1].set_title("P99 JCT")
        for bar, val in zip(bars_99, p99_vals):
            axs[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}",
                        ha="center", va="bottom", fontsize=9)

        axs[0].set_ylim(0, upper_limit if upper_limit > 0 else 1)

        #fig.suptitle(f"JCT percentiles — {cond.replace('_', ' ')}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(output_dir, f"{cond}_jct_p95_p99.png")
        try:
            fig.savefig(plot_path, dpi=150)
            print(f"Saved JCT percentile plot to {plot_path}")
        except Exception as e:
            print(f"Warning: failed to save JCT percentile plot for {cond}: {e}")
        plt.close(fig)


def plot_jct_cdf(jct_results: Dict[str, Dict[str, List[float]]], output_dir: str = "dist/jct_cdf") -> None:
    """CDF of JCT"""
    os.makedirs(output_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = ["no_stragglers", "with_stragglers"]
    fig, axs = plt.subplots(1, len(conditions), figsize=(12, 5), sharey=True)

    if len(conditions) == 1:
        axs = [axs]

    print("\n--- Generating JCT CDF Plots ---")
    for i, cond in enumerate(conditions):
        ax = axs[i]
        condition_label = cond.replace('_', ' ')
        print(f"  Processing condition: {condition_label}")

        has_data = False
        max_x_val = 0

        jcts_def = jct_results.get(cond, {}).get("default", [])
        if jcts_def:
            sorted_jcts_def = np.sort(np.array(jcts_def, dtype=float))
            y_def = np.arange(1, len(sorted_jcts_def) + 1) / len(sorted_jcts_def)
            ax.step(sorted_jcts_def, y_def, where="post", label="Default", color="C0", linewidth=1.8)
            max_jct_def = sorted_jcts_def[-1]
            max_x_val = max(max_x_val, max_jct_def)
            ax.text(max_jct_def, 1.0, f'{max_jct_def:.2f}', color='C0', ha='right', va='bottom', fontsize=8)
            print(f"    Default: {len(jcts_def)} JCTs, Max={max_jct_def:.2f}")
            has_data = True

        jcts_hyp = jct_results.get(cond, {}).get("hyper", [])
        if jcts_hyp:
            sorted_jcts_hyp = np.sort(np.array(jcts_hyp, dtype=float))
            y_hyp = np.arange(1, len(sorted_jcts_hyp) + 1) / len(sorted_jcts_hyp)
            ax.step(sorted_jcts_hyp, y_hyp, where="post", label="Hypercube", color="C1", linewidth=1.8)
            max_jct_hyp = sorted_jcts_hyp[-1]
            max_x_val = max(max_x_val, max_jct_hyp)
            ax.text(max_jct_hyp, 1.0, f'{max_jct_hyp:.2f}', color='C1', ha='right', va='bottom', fontsize=8)
            print(f"    Hypercube: {len(jcts_hyp)} JCTs, Max={max_jct_hyp:.2f}")
            has_data = True

        #ax.set_title(f"JCT CDF ({condition_label})")
        ax.set_xlabel("Job Completion Time")
        if i == 0:
            ax.set_ylabel("Cumulative probability")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(left=0, right=max_x_val * 1.1 if max_x_val > 0 else 1)
        if has_data:
            ax.legend()
        else:
             ax.text(0.5, 0.5, "No JCT data available", ha='center', va='center', transform=ax.transAxes)


    fig.tight_layout()
    plot_path = os.path.join(output_dir, "jct_cdf_comparison.png")
    try:
        fig.savefig(plot_path, dpi=150)
        print(f"Saved JCT CDF plot to {plot_path}")
    except Exception as e:
        print(f"Warning: failed to save JCT CDF plot: {e}")
    plt.close(fig)

def run_sensitivity_analysis() -> None:
    """
    Sensitivity of Hypercube gain to hardware heterogeneity (variance factor).
    """
    print("\n--- Running Sensitivity Analysis for Hardware Variance ---")
    variance_levels = [0.20, 0.15, 0.10, 0.05, 0.01]
    num_blocks_sensitivity = NUM_BLOCKS
    repeats = 3

    def _build_with_variance(v: float):
        nodes = generate_nodes(NUM_NODES, NUM_RACKS, seed=SEED, variance_factor=v)
        labels, centroids = kmeans_cluster(nodes, K, seed=SEED)
        b_r = int(math.ceil(math.log2(NUM_RACKS)))
        b_c = int(math.ceil(math.log2(K)))
        mapping = compute_centroid_bit_mapping(centroids, num_bits=b_c)
        assign_centroids_and_addresses(nodes, labels, b_r, b_c, mapping)
        registry = build_registry(nodes)
        cutpoints = compute_decile_cutpoints(nodes)
        return nodes, registry, centroids, cutpoints

    gains = []
    xs = []
    for v in variance_levels:
        print(f"  variance_factor={v} ...")
        nodes, registry, centroids, cutpoints = _build_with_variance(v)
        # Run quick experiments (balanced tag)
        import copy
        means_def, means_hyp = [], []
        for r in range(repeats):
            # Reseed so both strategies see comparable randomness per repeat
            np.random.seed(SEED + r); random.seed(SEED + r)
            nodes_def = copy.deepcopy(nodes)
            registry_def = build_registry(nodes_def)
            res_def = _run_one_strategy(
                "Default HDFS", nodes_def, registry_def, centroids, cutpoints,
                speculative=True, num_blocks=num_blocks_sensitivity,
            )
            np.random.seed(SEED + r); random.seed(SEED + r)
            nodes_hyp = copy.deepcopy(nodes)
            registry_hyp = build_registry(nodes_hyp)
            res_hyp = _run_one_strategy(
                "Hypercube", nodes_hyp, registry_hyp, centroids, cutpoints,
                speculative=True, num_blocks=num_blocks_sensitivity,
            )
            ct_def = res_def.get("completion_times", [])
            ct_hyp = res_hyp.get("completion_times", [])
            if len(ct_def) > 0:
                means_def.append(float(np.mean(ct_def)))
            if len(ct_hyp) > 0:
                means_hyp.append(float(np.mean(ct_hyp)))
        m_def = float(np.mean(means_def)) if means_def else 0.0
        m_hyp = float(np.mean(means_hyp)) if means_hyp else 0.0
        gain = (m_def - m_hyp) / m_def * 100.0 if m_def > 0 else 0.0
        def _loc_mix(res):
            loc = res.get("launch_locality", {})
            total = sum(loc.values()) or 1
            return {
                "local%": 100.0 * loc.get("starts_local", 0) / total,
                "rack%": 100.0 * loc.get("starts_rack", 0) / total,
                "remote%": 100.0 * loc.get("starts_remote", 0) / total,
            }
        def _netp(res):
            s = res.get("exec_net_p_summary", {})
            return f"net_p[p10={s.get('p10',0):.2f}, p50={s.get('p50',0):.2f}, p90={s.get('p90',0):.2f}]"
        def _levels(res):
            lv = res.get("starts_by_level", {})
            tot = sum(lv.values()) or 1
            return f"levels[L0={lv.get('level0',0)}, L1={lv.get('level1',0)}, L2={lv.get('level2',0)}]"

        print(
            f"    mean_default={m_def:.3f} mean_hyper={m_hyp:.3f}  gain={gain:.2f}%\n"
            f"      default: {_loc_mix(res_def)} {_netp(res_def)} {_levels(res_def)}\n"
            f"      hyper  : {_loc_mix(res_hyp)} {_netp(res_hyp)} {_levels(res_hyp)}"
        )

        # Per-variance summary stats requested
        def _replica_stats(res):
            rc = res.get("replica_counts_per_node", {})
            vals = list(rc.values())
            if not vals:
                return (0.0, 0.0)
            arr = np.array(vals, dtype=float)
            return float(arr.mean()), float(arr.std(ddof=0))

        def _spec_requests(res):
            return int(res.get("spec", {}).get("requests", 0))

        def _spec_with_network(res):
            sp = res.get("spec", {})
            return int(sp.get("launched_rack", 0)) + int(sp.get("launched_remote", 0))

        mean_blocks_def, std_blocks_def = _replica_stats(res_def)
        mean_blocks_hyp, std_blocks_hyp = _replica_stats(res_hyp)
        req_def, req_hyp = _spec_requests(res_def), _spec_requests(res_hyp)
        spec_net_def, spec_net_hyp = _spec_with_network(res_def), _spec_with_network(res_hyp)

        print(
            "      blocks_per_node: "
            f"Default mean={mean_blocks_def:.2f} std={std_blocks_def:.2f} | "
            f"Hypercube mean={mean_blocks_hyp:.2f} std={std_blocks_hyp:.2f}"
        )
        print(
            "      speculation: "
            f"requests Default={req_def} Hyper={req_hyp}; "
            f"with_network Default={spec_net_def} Hyper={spec_net_hyp}"
        )
        gains.append(gain); xs.append(v)

    # Plot
    try:
        out_dir = os.path.join("dist", "sensitivity")
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, gains, marker='o')
        ax.set_xlabel("Hardware variance factor")
        ax.set_ylabel("Hypercube gain (%)")
        #ax.set_title("Sensitivity to hardware heterogeneity")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.invert_xaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "hardware_variance.png"), dpi=150)
        plt.close(fig)
        print(f"Saved sensitivity plot to {os.path.join(out_dir, 'hardware_variance.png')}")
    except Exception as e:
        print(f"Warning: failed to plot sensitivity analysis: {e}")


if __name__ == "__main__":
    main()


def run_remote_demo() -> None:
    """
    Construct a small, highly fragmented cluster to force off-rack (level-2) starts.
    """
    import copy
    print("\n--- Remote-starts demonstration (stress) ---")
    num_nodes = 200
    num_racks = 100  # ~2 nodes per rack → tiny rack capacity
    k = 64
    blocks = 5000
    rng_seed = SEED

    # Build cluster
    nodes = generate_nodes(num_nodes, num_racks, seed=rng_seed, variance_factor=1.0)
    # Cap compute slots to 1 to saturate quickly
    for n in nodes:
        n.max_compute_slots = 1
    labels, centroids = kmeans_cluster(nodes, k, seed=rng_seed)
    b_r = int(math.ceil(math.log2(num_racks)))
    b_c = int(math.ceil(math.log2(k)))
    mapping = compute_centroid_bit_mapping(centroids, num_bits=b_c)
    assign_centroids_and_addresses(nodes, labels, b_r, b_c, mapping)
    registry = build_registry(nodes)
    cutpoints = compute_decile_cutpoints(nodes)

    def _mix(res):
        loc = res.get("launch_locality", {})
        tot = sum(loc.values()) or 1
        return {k: round(100.0 * v / tot, 2) for k, v in loc.items()}
    def _levels(res):
        lv = res.get("starts_by_level", {})
        return lv

    print("Running Default...")
    res_def = execute_single_strategy_simulation(
        num_blocks=blocks,
        tag_distribution={"balanced": 1.0},
        nodes=copy.deepcopy(nodes),
        registry=build_registry(copy.deepcopy(nodes)),
        centroids=centroids,
        cutpoints=cutpoints,
        placement_strategy="default",
        collect_details=True,
        progress_every=100,
        runtime_params=rt_params,
        spec_params=sp_params,
    )
    print("Running Hypercube...")
    res_hyp = execute_single_strategy_simulation(
        num_blocks=blocks,
        tag_distribution={"balanced": 1.0},
        nodes=copy.deepcopy(nodes),
        registry=build_registry(copy.deepcopy(nodes)),
        centroids=centroids,
        cutpoints=cutpoints,
        placement_strategy="hyper",
        collect_details=True,
        progress_every=100,
        runtime_params=rt_params,
        spec_params=sp_params,
    )

    print("Default locality:", _mix(res_def), "levels:", _levels(res_def))
    print("Hypercube locality:", _mix(res_hyp), "levels:", _levels(res_hyp))
