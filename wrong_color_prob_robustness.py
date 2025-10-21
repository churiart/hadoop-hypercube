
#!/usr/bin/env python3

import argparse, os, sys, math, importlib.util, types
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

ROOT = Path(__file__).resolve().parent

plt.style.use('ieee')
plt.rcParams.update({'font.serif': ['Times New Roman'], 'font.family': 'serif'})

def load_as(subname: str, file: str):
    modname = f"hypercube.{subname}"
    spec = importlib.util.spec_from_file_location(modname, str(ROOT / file))
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "hypercube"
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module

# Load core modules in dependency order
nodes = load_as("nodes", "nodes.py")
placement = load_as("placement", "placement.py")
runtime = load_as("runtime", "runtime.py")
speculation = load_as("speculation", "speculation.py")
sim = load_as("simulator_v2", "simulator_v2.py")
config = load_as("config", "config.py")

def build_cluster(num_nodes=400, num_racks=10, k=81, seed=42):
    nlist = nodes.generate_nodes(num_nodes=num_nodes, num_racks=num_racks, seed=seed, variance_factor=1.0)
    labels, cents = nodes.kmeans_cluster(nlist, k, seed=seed)
    # assign addresses
    max_rack = max(n.rack_id for n in nlist)
    b_r = max(1, int(math.ceil(math.log2(max_rack+1))))
    b_c = int(math.ceil(math.log2(k)))
    mapping = nodes.compute_centroid_bit_mapping(cents, num_bits=b_c)
    nodes.assign_centroids_and_addresses(nlist, labels, b_r, b_c, mapping)
    reg = nodes.build_registry(nlist)
    cuts = placement.compute_decile_cutpoints(nlist)
    return nlist, reg, cents, cuts, b_r, b_c

# ---------- Workloads ----------
CONTROLLED_TARGETS = placement.CONTROLLED_TARGETS
CONTROLLED_WEIGHTS = placement.CONTROLLED_WEIGHTS
ALL_TAGS = list(CONTROLLED_TARGETS.keys())

def sample_true_tags(num_blocks: int, seed: int = 42) -> List[str]:
    rng = np.random.default_rng(seed)
    # Uniform over the available controlled tags
    return list(rng.choice(ALL_TAGS, size=num_blocks))

def dominant_dim_of_tag(tag: str) -> int:
    w = CONTROLLED_WEIGHTS.get(tag, CONTROLLED_WEIGHTS["balanced"])
    dims = ["cpu","ram","disk","net"]
    mx = -1.0; md = 0
    for i, d in enumerate(dims):
        if w[d] > mx:
            mx = w[d]; md = i
    return md

def wrong_tag_for(true_tag: str, rng: np.random.Generator) -> str:
    d0 = dominant_dim_of_tag(true_tag)
    candidates = [t for t in ALL_TAGS if dominant_dim_of_tag(t) != d0]
    # bias toward the most "peaky" among candidates
    def peak(tag):
        w = CONTROLLED_WEIGHTS[tag]
        return max(w.values())
    # small randomness
    if rng.random() < 0.3:
        return str(rng.choice(candidates))
    return max(candidates, key=peak)

def make_declared_tags(true_tags: List[str], p: float, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    decl = []
    for t in true_tags:
        if rng.random() < p:
            decl.append(wrong_tag_for(t, rng))
        else:
            decl.append(t)
    return decl

class PreparedTagStream:
    def __init__(self, head_then_tail: List[str]):
        self._data = list(head_then_tail)
        self._i = 0
    def __call__(self, dist: Dict[str, float]) -> str:
        if self._i >= len(self._data):
            # fallback: balanced
            return "balanced"
        out = self._data[self._i]
        self._i += 1
        return out

from contextlib import contextmanager
@contextmanager
def patch_sample_tag(prepared_stream):
    """Temporarily replace sim._sample_tag with a prepared stream callable."""
    old = getattr(sim, "_sample_tag")
    setattr(sim, "_sample_tag", prepared_stream)
    try:
        yield
    finally:
        setattr(sim, "_sample_tag", old)

# ---------- Run one p ----------
def run_once(num_blocks: int,
             nodes_list,
             registry,
             centroids,
             cutpoints,
             true_tags: List[str],
             declared_tags: List[str],
             rt_params=None,
             sp_params=None,
             seed: int = 42):
    # disable storage capacity constraints by making max_replicas enormous
    for n in nodes_list:
        n.max_replicas = 10**9
        n.replica_count = 0
        n.compute_load = 0

    # The simulator samples tags twice in one call: first for placement, then for task runtime.
    # Prepare a tag stream that yields [declared_tags ... , true_tags ...] in that order.
    stream = PreparedTagStream(head_then_tail = list(declared_tags) + list(true_tags))

    # Ensure determinism across strategies (writer rack sampling etc.)
    import random as pyrand
    pyrand.seed(seed)
    np.random.seed(seed)

    with patch_sample_tag(stream):
        res_default = sim.execute_single_strategy_simulation(
            num_blocks=num_blocks,
            tag_distribution={},
            nodes=nodes_list,
            registry=registry,
            centroids=centroids,
            cutpoints=cutpoints,
            placement_strategy="default",
            speculative=True,
            num_replicas=3,
            collect_details=True,
            runtime_params=rt_params,
            spec_params=sp_params,
        )

    # Reset counters and the prepared stream again for Hypercube run
    for n in nodes_list:
        n.max_replicas = 10**9
        n.replica_count = 0
        n.compute_load = 0
    stream = PreparedTagStream(head_then_tail = list(declared_tags) + list(true_tags))
    pyrand.seed(seed)
    np.random.seed(seed)

    with patch_sample_tag(stream):
        res_hyper = sim.execute_single_strategy_simulation(
            num_blocks=num_blocks,
            tag_distribution={},
            nodes=nodes_list,
            registry=registry,
            centroids=centroids,
            cutpoints=cutpoints,
            placement_strategy="hypercube",
            speculative=True,
            num_replicas=3,
            collect_details=True,
            runtime_params=rt_params,
            spec_params=sp_params,
        )

    import numpy as _np
    d_times = _np.array(res_default["completion_times"], dtype=float)
    h_times = _np.array(res_hyper["completion_times"], dtype=float)
    mean_d = float(d_times.mean())
    mean_h = float(h_times.mean())
    p95_d = float(_np.percentile(d_times, 95))
    p95_h = float(_np.percentile(h_times, 95))
    gain_pct = ( (mean_d - mean_h) / mean_d ) * 100.0

    return {
        "mean_default": mean_d,
        "mean_hypercube": mean_h,
        "p95_default": p95_d,
        "p95_hypercube": p95_h,
        "gain_pct": gain_pct,
        "default_details": res_default,
        "hypercube_details": res_hyper,
    }

def run_sweep(num_blocks=1000, seed=42, p_values=None, num_nodes=400, num_racks=10, k=81):
    if p_values is None:
        p_values = [round(x,2) for x in np.linspace(0.0, 1.0, 11)]
    nodes_list, registry, cents, cuts, b_r, b_c = build_cluster(num_nodes=num_nodes, num_racks=num_racks, k=k, seed=seed)
    true_tags = sample_true_tags(num_blocks=num_blocks, seed=seed)

    # params
    rt_params = config.default_runtime_params()  # runtime.RuntimeParams()
    sp_params = config.default_spec_params()     # speculation.SpecParams()

    rows = []
    for p in p_values:
        declared = make_declared_tags(true_tags, p=float(p), seed=int(seed + 1000*p))
        result = run_once(num_blocks, nodes_list, registry, cents, cuts, true_tags, declared, rt_params, sp_params, seed=seed)
        row = {"p": float(p), "mean_default": result["mean_default"], "mean_hypercube": result["mean_hypercube"],
               "p95_default": result["p95_default"], "p95_hypercube": result["p95_hypercube"],
               "gain_pct": result["gain_pct"]}
        rows.append(row)
    return rows

def plot_gain(rows, outdir="dist/mis_tag_prob"):
    os.makedirs(outdir, exist_ok=True)
    p = [r["p"] for r in rows]
    g = [r["gain_pct"] for r in rows]
    plt.figure(figsize=(7,4))
    plt.plot(p, g, marker="o")
    plt.xlabel("Wrong color with probability $p$")
    plt.ylabel("Gain over default (%)")
    plt.grid(True, alpha=0.3)
    path = os.path.join(outdir, "gain_vs_p.png")
    plt.tight_layout(); plt.savefig(path, dpi=160)
    print("Saved plot:", path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--racks", type=int, default=10)
    ap.add_argument("--k", type=int, default=81)
    args = ap.parse_args()
    rows = run_sweep(num_blocks=args.blocks, seed=args.seed, num_nodes=args.nodes, num_racks=args.racks, k=args.k)
    print("\n p    mean_default   mean_hypercube   p95_def   p95_hyp   gain(%)")
    for r in rows:
        print(f"{r['p']:>3.1f}  {r['mean_default']:>13.4f}  {r['mean_hypercube']:>15.4f}  {r['p95_default']:>8.3f}  {r['p95_hypercube']:>8.3f}  {r['gain_pct']:>7.3f}")
    plot_gain(rows)

if __name__ == "__main__":
    main()
