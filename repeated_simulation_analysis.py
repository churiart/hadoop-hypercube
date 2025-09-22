from __future__ import annotations

import copy
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from hypercube.nodes import (
    DataNode, generate_nodes, kmeans_cluster, compute_centroid_bit_mapping,
    assign_centroids_and_addresses, build_registry
)
from hypercube.placement import compute_decile_cutpoints, CONTROLLED_TARGETS, CONTROLLED_WEIGHTS
from hypercube.simulator_v2 import execute_single_strategy_simulation
from hypercube.config import default_runtime_params, default_spec_params
from hypercube.runtime import metrics_to_percentiles
import hypercube.runtime as rt

# Simulation parameters
NUM_ITERATIONS = 10
NUM_NODES = 400
NUM_RACKS = 10
K = 81
NUM_BLOCKS = 1000
STRAGGLER_FRACTION = 0.1
STRAGGLER_DEGRADE_FACTOR = 0.3

# Runtime and speculation parameters (same as main client)
rt_params = default_runtime_params()
rt_params.enable_hamming_penalty = True
rt_params.hamming_gamma = 3.0
sp_params = default_spec_params()


def generate_random_tag_distribution() -> Dict[str, float]:
    """Generate a random tag distribution that sums to 1.0."""
    tags = ["balanced", "cpu-bound", "disk-bound", "ram-bound", "net-bound"]
    weights = np.random.rand(len(tags))
    weights = weights / weights.sum()
    return {tag: float(w) for tag, w in zip(tags, weights)}


def build_cluster_with_hamming_setup(num_nodes: int, num_racks: int, k: int, seed: int) -> Tuple:
    """Build a cluster and set up Hamming context for runtime penalties."""
    nodes = generate_nodes(num_nodes, num_racks, seed=seed, variance_factor=1.0)
    labels, centroids = kmeans_cluster(nodes, k, seed=seed)
    
    b_r = int(math.ceil(math.log2(num_racks)))
    b_c = int(math.ceil(math.log2(k)))
    
    mapping = compute_centroid_bit_mapping(centroids, num_bits=b_c)
    assign_centroids_and_addresses(nodes, labels, b_r, b_c, mapping)
    
    registry = build_registry(nodes)
    cutpoints = compute_decile_cutpoints(nodes)
    
    # Set up Hamming context
    addr_len = len(nodes[0].addr_bits)
    b_c_actual = addr_len - b_r
    centroid_bits_by_id = {}
    for n in nodes:
        if n.centroid_id not in centroid_bits_by_id:
            centroid_bits_by_id[n.centroid_id] = n.addr_bits[-b_c_actual:]
    
    # Determine ideal centroid bits for controlled targets
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
    
    ideal_bits_by_tag = {}
    for tag, targets in CONTROLLED_TARGETS.items():
        weights = CONTROLLED_WEIGHTS[tag]
        scores = {cid: _cost_on_centroid(centroids[cid], targets, weights) for cid in range(len(centroids))}
        ideal_cid = min(scores, key=scores.get)
        ideal_bits_by_tag[tag] = centroid_bits_by_id.get(ideal_cid, format(ideal_cid, f"0{b_c_actual}b"))
    
    rt.set_hamming_context(b_r=b_r, b_c=b_c_actual, ideal_bits_by_tag=ideal_bits_by_tag)
    
    return nodes, registry, centroids, cutpoints


def apply_stragglers(nodes: List[DataNode], fraction: float, seed: int, degrade_factor: float) -> None:
    """Apply straggler degradation to a fraction of nodes."""
    if not nodes or fraction <= 0:
        return
    
    random.seed(seed)
    num = max(1, int(len(nodes) * fraction))
    chosen_ids = set(random.sample([n.node_id for n in nodes], num))
    
    nid_to_node = {n.node_id: n for n in nodes}
    for nid in chosen_ids:
        node = nid_to_node.get(nid)
        if node is not None:
            sv = node.s_vector.copy()
            sv[:3] = sv[:3] * degrade_factor  # degrade CPU, RAM, DISK
            node.s_vector = sv


def run_single_iteration(iteration: int) -> Dict[str, Dict[str, List[float]]]:
    """Run one iteration with fresh cluster and tag generation."""
    print(f"Running iteration {iteration + 1}/{NUM_ITERATIONS}...")
    
    # Generate unique seed for this iteration
    base_seed = 42 + iteration * 1000
    
    # Generate random (tag) distribution for this iteration
    np.random.seed(base_seed)
    random.seed(base_seed)
    tag_distribution = generate_random_tag_distribution()
    
    # Build fresh cluster
    nodes, registry, centroids, cutpoints = build_cluster_with_hamming_setup(
        NUM_NODES, NUM_RACKS, K, base_seed
    )
    
    results = {}
    
    # No stragglers condition
    nodes_no_strag = copy.deepcopy(nodes)
    registry_no_strag = build_registry(nodes_no_strag)
    
    # Default strategy (no stragglers)
    np.random.seed(base_seed + 1)
    random.seed(base_seed + 1)
    res_default_no_strag = execute_single_strategy_simulation(
        num_blocks=NUM_BLOCKS,
        tag_distribution=tag_distribution,
        nodes=copy.deepcopy(nodes_no_strag),
        registry=build_registry(copy.deepcopy(nodes_no_strag)),
        centroids=centroids,
        cutpoints=cutpoints,
        placement_strategy="default",
        speculative=True,
        num_replicas=3,
        collect_details=True,
        runtime_params=rt_params,
        spec_params=sp_params,
    )
    
    # Hypercube strategy (no stragglers)
    np.random.seed(base_seed + 1)
    random.seed(base_seed + 1)
    res_hyper_no_strag = execute_single_strategy_simulation(
        num_blocks=NUM_BLOCKS,
        tag_distribution=tag_distribution,
        nodes=copy.deepcopy(nodes_no_strag),
        registry=build_registry(copy.deepcopy(nodes_no_strag)),
        centroids=centroids,
        cutpoints=cutpoints,
        placement_strategy="hyper",
        speculative=True,
        num_replicas=3,
        collect_details=True,
        runtime_params=rt_params,
        spec_params=sp_params,
    )
    
    # With stragglers condition
    nodes_with_strag = copy.deepcopy(nodes)
    apply_stragglers(nodes_with_strag, STRAGGLER_FRACTION, base_seed + 2, STRAGGLER_DEGRADE_FACTOR)
    
    # Default strategy (with stragglers)
    np.random.seed(base_seed + 3)
    random.seed(base_seed + 3)
    res_default_with_strag = execute_single_strategy_simulation(
        num_blocks=NUM_BLOCKS,
        tag_distribution=tag_distribution,
        nodes=copy.deepcopy(nodes_with_strag),
        registry=build_registry(copy.deepcopy(nodes_with_strag)),
        centroids=centroids,
        cutpoints=cutpoints,
        placement_strategy="default",
        speculative=True,
        num_replicas=3,
        collect_details=True,
        runtime_params=rt_params,
        spec_params=sp_params,
    )
    
    # Hypercube strategy (with stragglers)
    np.random.seed(base_seed + 3)
    random.seed(base_seed + 3)
    res_hyper_with_strag = execute_single_strategy_simulation(
        num_blocks=NUM_BLOCKS,
        tag_distribution=tag_distribution,
        nodes=copy.deepcopy(nodes_with_strag),
        registry=build_registry(copy.deepcopy(nodes_with_strag)),
        centroids=centroids,
        cutpoints=cutpoints,
        placement_strategy="hyper",
        speculative=True,
        num_replicas=3,
        collect_details=True,
        runtime_params=rt_params,
        spec_params=sp_params,
    )

    results = {
        "no_stragglers": {
            "default": res_default_no_strag.get("completion_times", []),
            "hyper": res_hyper_no_strag.get("completion_times", []),
        },
        "with_stragglers": {
            "default": res_default_with_strag.get("completion_times", []),
            "hyper": res_hyper_with_strag.get("completion_times", []),
        }
    }
    
    return results


def compute_percentiles(completion_times: List[float]) -> Tuple[float, float]:
    """Compute P95 and P99 from completion times."""
    if completion_times is None or len(completion_times) == 0:
        return 0.0, 0.0
    
    arr = np.array(completion_times, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    return p95, p99


def run_repeated_simulation() -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Run the simulation NUM_ITERATIONS times and collect percentile data."""
    print(f"Starting repeated simulation with {NUM_ITERATIONS} iterations...")
    print(f"Parameters: {NUM_NODES} nodes, {NUM_RACKS} racks, {K} centroids, {NUM_BLOCKS} blocks")
    percentile_data = {
        "no_stragglers": {
            "default": {"p95": [], "p99": []},
            "hyper": {"p95": [], "p99": []},
        },
        "with_stragglers": {
            "default": {"p95": [], "p99": []},
            "hyper": {"p95": [], "p99": []},
        }
    }

    all_completion_times = {
        "no_stragglers": {
            "default": [],
            "hyper": []
        },
        "with_stragglers": {
            "default": [],
            "hyper": []
        }
    }
    
    for iteration in range(NUM_ITERATIONS):
        try:
            results = run_single_iteration(iteration)
            
            for condition in ["no_stragglers", "with_stragglers"]:
                for strategy in ["default", "hyper"]:
                    completion_times = results[condition][strategy]
                    p95, p99 = compute_percentiles(completion_times)
                    percentile_data[condition][strategy]["p95"].append(p95)
                    percentile_data[condition][strategy]["p99"].append(p99)
                    all_completion_times[condition][strategy].extend(completion_times)
            
            # Progress update every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"Completed {iteration + 1}/{NUM_ITERATIONS} iterations")
                
        except Exception as e:
            import traceback
            print(f"Error in iteration {iteration + 1}: {e}")
            traceback.print_exc()
            continue
    
    actual_percentiles = {}
    for condition in ["no_stragglers", "with_stragglers"]:
        actual_percentiles[condition] = {}
        for strategy in ["default", "hyper"]:
            all_times = all_completion_times[condition][strategy]
            if all_times:
                p95, p99 = compute_percentiles(all_times)
                actual_percentiles[condition][strategy] = {"p95": p95, "p99": p99}
            else:
                actual_percentiles[condition][strategy] = {"p95": 0.0, "p99": 0.0}
    
    percentile_data["actual_percentiles"] = actual_percentiles
    percentile_data["all_completion_times"] = all_completion_times
    
    return percentile_data


def plot_percentile_distributions(percentile_data: Dict, output_dir: str = "dist/repeated_analysis") -> None:
    """Plot distributions of P95 and P99 percentiles across iterations."""
    os.makedirs(output_dir, exist_ok=True)
    
    conditions = ["no_stragglers", "with_stragglers"]
    strategies = ["default", "hyper"]
    percentiles = ["p95", "p99"]
    
    # Create summary statistics
    print("\n--- Summary Statistics (across all iterations) ---")
    for condition in conditions:
        print(f"\n{condition.replace('_', ' ').title()}:")
        for strategy in strategies:
            for percentile in percentiles:
                values = percentile_data[condition][strategy][percentile]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    print(f"  {strategy} {percentile.upper()}: "
                          f"mean={mean_val:.3f} ±{std_val:.3f} "
                          f"[{min_val:.3f}, {max_val:.3f}]")
    
    print("\n--- Computing actual 95th percentile across all iterations ---")
    
    # Get actual percentiles computed across all iterations
    if "actual_percentiles" in percentile_data:
        actual_p95 = percentile_data["actual_percentiles"]
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        
        # Prepare data for merged plot
        plot_labels = ["no stragglers", "with stragglers"]
        x_pos = np.arange(len(plot_labels))
        width = 0.35
        
        # Get P95 values for both strategies across conditions
        default_p95_values = [
            actual_p95["no_stragglers"]["default"]["p95"],
            actual_p95["with_stragglers"]["default"]["p95"]
        ]
        hyper_p95_values = [
            actual_p95["no_stragglers"]["hyper"]["p95"],
            actual_p95["with_stragglers"]["hyper"]["p95"]
        ]
        
        bars1 = ax.bar(x_pos - width/2, default_p95_values, width, label="Default")
        bars2 = ax.bar(x_pos + width/2, hyper_p95_values, width, label="Hypercube")
        
        for bar, val in zip(bars1, default_p95_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom')
        for bar, val in zip(bars2, hyper_p95_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_labels)
        ax.set_ylabel("95th Percentile Completion Time")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "actual_p95_merged.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\n--- Actual P95 across all iterations ---")
        for condition in ["no_stragglers", "with_stragglers"]:
            print(f"\n{condition.replace('_', ' ').title()}:")
            for strategy in ["default", "hyper"]:
                actual_p95_val = actual_p95[condition][strategy]["p95"]
                actual_p99_val = actual_p95[condition][strategy]["p99"]
                print(f"  {strategy}: P95={actual_p95_val:.3f}, P99={actual_p99_val:.3f}")
    else:
        print("Note: Actual percentiles not available in data")
    
    # Plot P95 distributions
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, condition in enumerate(["no_stragglers", "with_stragglers"]):
        ax = axs[i]
        
        # Get P95 data
        default_p95 = percentile_data[condition]["default"]["p95"]
        hyper_p95 = percentile_data[condition]["hyper"]["p95"]
        
        # Box plots
        box_data = [default_p95, hyper_p95]
        labels = ["Default", "Hypercube"]
        
        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(f"P95 Completion Times\n({condition.replace('_', ' ').title()})")
        ax.set_ylabel("P95 Completion Time (s)")
        ax.grid(True, alpha=0.3)
        
        # Add mean markers
        means = [np.mean(data) for data in box_data]
        ax.scatter([1, 2], means, color='red', marker='D', s=50, zorder=10, label='Mean')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "p95_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot P99 distributions
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, condition in enumerate(conditions):
        ax = axs[i]
        
        # Get P99 data
        default_p99 = percentile_data[condition]["default"]["p99"]
        hyper_p99 = percentile_data[condition]["hyper"]["p99"]
        
        # Box plots
        box_data = [default_p99, hyper_p99]
        labels = ["Default", "Hypercube"]
        
        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(f"P99 Completion Times\n({condition.replace('_', ' ').title()})")
        ax.set_ylabel("P99 Completion Time (s)")
        ax.grid(True, alpha=0.3)
        
        # Add mean markers
        means = [np.mean(data) for data in box_data]
        ax.scatter([1, 2], means, color='red', marker='D', s=50, zorder=10, label='Mean')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "p99_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Combined plot showing both P95 and P99
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, condition in enumerate(conditions):
        for j, percentile in enumerate(percentiles):
            ax = axs[j, i]
            
            # Get data
            default_data = percentile_data[condition]["default"][percentile]
            hyper_data = percentile_data[condition]["hyper"][percentile]
            
            # Histograms
            ax.hist(default_data, bins=20, alpha=0.6, label="Default", color='lightblue', density=True)
            ax.hist(hyper_data, bins=20, alpha=0.6, label="Hypercube", color='lightcoral', density=True)
            
            # Add mean lines
            ax.axvline(np.mean(default_data), color='blue', linestyle='--', alpha=0.8, label=f'Default Mean')
            ax.axvline(np.mean(hyper_data), color='red', linestyle='--', alpha=0.8, label=f'Hypercube Mean')
            
            ax.set_title(f"{percentile.upper()} - {condition.replace('_', ' ').title()}")
            ax.set_xlabel(f"{percentile.upper()} Completion Time (s)")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "percentile_histograms.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


def main():
    print("=" * 60)
    print("REPEATED SIMULATION ANALYSIS")
    print("=" * 60)
    
    # Run the repeated simulation
    percentile_data = run_repeated_simulation()
    
    # Create plots
    plot_percentile_distributions(percentile_data)
    
    if "actual_percentiles" in percentile_data:
        print("\n" + "=" * 60)
        print("ACTUAL P95 SUMMARY (Across All Iterations Combined)")
        print("=" * 60)
        print(f"{'Condition':<20} {'Strategy':<10} {'P95':<10} {'P99':<10}")
        print("-" * 60)
        
        for condition in ["no_stragglers", "with_stragglers"]:
            condition_name = condition.replace("_", " ").title()
            for strategy in ["default", "hyper"]:
                strategy_name = "Hypercube" if strategy == "hyper" else "Default"
                p95_val = percentile_data["actual_percentiles"][condition][strategy]["p95"]
                p99_val = percentile_data["actual_percentiles"][condition][strategy]["p99"]
                print(f"{condition_name:<20} {strategy_name:<10} {p95_val:<10.3f} {p99_val:<10.3f}")
        
        print("-" * 60)
        print("Note: These are the actual 95th and 99th percentiles computed")
        print("across ALL completion times from ALL iterations combined.")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        try:
            NUM_ITERATIONS = int(sys.argv[1])
            print(f"Using {NUM_ITERATIONS} iterations from command line")
        except Exception:
            print(f"Warning: bad NUM_ITERATIONS arg '{sys.argv[1]}', using default {NUM_ITERATIONS}")
    main()
