#!/usr/bin/env python3
"""
Systematic Degradation Analysis

Instead of probabilistic noise (p=X), this analysis uses controlled degradation:
- I=1: Best placement (1st best centroid for each block)
- I=2: 2nd best centroid for each block  
- I=3: 3rd best centroid for each block
- I=4: 4th best centroid for each block
- I=5: 5th best centroid for each block
- ...
- I=10: 10th best centroid for each block

This gives us a clear, systematic degradation curve from optimal to suboptimal.
"""

from hypercube import nodes, placement, runtime, config
from hypercube.client import _build_cluster
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple

def create_heterogeneous_cluster(num_nodes: int = 200, num_racks: int = 20, k: int = 50, seed: int = 42):
    """Create a heterogeneous cluster with wide range of hardware capabilities."""
    
    print(f"Creating heterogeneous cluster: {num_nodes} nodes, {num_racks} racks, {k} centroids")
    
    nodes_list, cluster_registry, centroids, cutpoints = _build_cluster(
        num_nodes=num_nodes, num_racks=num_racks, k=k, seed=seed
    )
    
    return nodes_list, cluster_registry, centroids, cutpoints

def generate_diverse_workloads(num_blocks: int, rng: np.random.Generator):
    """Generate diverse workload definitions for systematic testing."""
    
    # More diverse workload types for better heterogeneity testing
    workload_types = {
        "cpu-intensive": {"targets": [0.9, 0.2, 0.2, 0.2], "weights": [0.7, 0.1, 0.1, 0.1]},
        "cpu-moderate": {"targets": [0.7, 0.4, 0.3, 0.3], "weights": [0.5, 0.2, 0.15, 0.15]},
        "ram-intensive": {"targets": [0.2, 0.9, 0.2, 0.2], "weights": [0.1, 0.7, 0.1, 0.1]},
        "ram-moderate": {"targets": [0.3, 0.7, 0.4, 0.3], "weights": [0.15, 0.5, 0.2, 0.15]},
        "disk-intensive": {"targets": [0.2, 0.2, 0.9, 0.2], "weights": [0.1, 0.1, 0.7, 0.1]},
        "disk-moderate": {"targets": [0.3, 0.3, 0.7, 0.4], "weights": [0.15, 0.15, 0.5, 0.2]},
        "net-intensive": {"targets": [0.2, 0.2, 0.2, 0.9], "weights": [0.1, 0.1, 0.1, 0.7]},
        "net-moderate": {"targets": [0.3, 0.3, 0.4, 0.7], "weights": [0.15, 0.15, 0.2, 0.5]},
        "balanced": {"targets": [0.5, 0.5, 0.5, 0.5], "weights": [0.25, 0.25, 0.25, 0.25]},
        "mixed-compute": {"targets": [0.8, 0.6, 0.3, 0.4], "weights": [0.4, 0.3, 0.15, 0.15]},
        "mixed-io": {"targets": [0.3, 0.4, 0.8, 0.6], "weights": [0.15, 0.15, 0.4, 0.3]}
    }
    
    type_names = list(workload_types.keys())
    # More varied distribution to test different scenarios
    type_probs = [0.12, 0.08, 0.12, 0.08, 0.12, 0.08, 0.12, 0.08, 0.1, 0.05, 0.05]
    
    baseline_targets = []
    baseline_weights = []
    block_tags = []
    
    for block_id in range(num_blocks):
        workload_type = rng.choice(type_names, p=type_probs)
        baseline_targets.append(workload_types[workload_type]["targets"])
        baseline_weights.append(workload_types[workload_type]["weights"])
        block_tags.append(workload_type)
    
    return baseline_targets, baseline_weights, block_tags

def rank_centroids_by_cost(
    centroids,
    targets,
    weights,
    cutpoints,
    use_dominant_dim_cost: bool = False,
    dominant_cost_mode: str = "one_sided",
):
    """Rank all centroids by cost for given workload (best to worst)."""
    
    targets = np.array(targets)
    weights = np.array(weights)
    
    centroid_costs = []
    
    for centroid_id, centroid in enumerate(centroids):
        # Convert centroid hardware vector to percentiles using cutpoints (0..1)
        c_percentiles = runtime.metrics_to_percentiles(centroid, cutpoints)
        
        if use_dominant_dim_cost:
            # Use only the dominant (highest weight) dimension for cost
            idx = int(np.argmax(weights))
            t = float(targets[idx])
            p = float(c_percentiles[idx])
            eps = 1e-6
            if dominant_cost_mode == "abs":
                cost = abs(p - t)
            else:  # one_sided under-provision penalty
                cost = max(0.0, (t - p) / max(t, eps))
        else:
            # Multi-dimensional raw gap cost (negative means over-provisioned)
            gaps = (targets - c_percentiles) / np.maximum(targets, 1e-6)
            cost = np.sum(weights * gaps)
        
        centroid_costs.append((centroid_id, cost))
    
    # Sort by cost (ascending = best to worst)
    centroid_costs.sort(key=lambda x: x[1])
    
    return centroid_costs

def force_placement_to_centroid_rank(
    nodes_list,
    centroids,
    baseline_targets,
    baseline_weights,
    cutpoints,
    degradation_level: int,
    rng: np.random.Generator,
    use_dominant_dim_cost: bool = False,
    dominant_cost_mode: str = "one_sided",
):
    """Force placement to use centroid at specific rank (1=best, 2=2nd best, 3=3rd best, etc.)."""
    
    block_replicas = []
    ideal_centroid_ids = []
    actual_centroid_ids = []
    
    for block_id, (targets, weights) in enumerate(zip(baseline_targets, baseline_weights)):
        # Rank all centroids for this workload
        ranked_centroids = rank_centroids_by_cost(
            centroids,
            targets,
            weights,
            cutpoints,
            use_dominant_dim_cost=use_dominant_dim_cost,
            dominant_cost_mode=dominant_cost_mode,
        )
        
        # Get ideal (best) centroid
        ideal_centroid_id = ranked_centroids[0][0]  # Best centroid
        ideal_centroid_ids.append(ideal_centroid_id)
        
        # Choose centroid based on degradation level
        # I=1: 1st best (index 0), I=2: 2nd best (index 1), etc.
        target_index = degradation_level - 1  # Convert to 0-based index
        
        if target_index < 0:
            target_index = 0  # Default to best
        elif target_index >= len(ranked_centroids):
            target_index = len(ranked_centroids) - 1  # Use worst available
        
        target_centroid_id = ranked_centroids[target_index][0]
        actual_centroid_ids.append(target_centroid_id)
        
        # Find nodes in the target centroid
        target_nodes = [n for n in nodes_list if n.centroid_id == target_centroid_id]
        
        # Select 3 replicas from target centroid (or fewer if not available)
        num_replicas = min(3, len(target_nodes))
        if num_replicas == 0:
            # Fallback: if no nodes in target centroid, use best centroid
            target_nodes = [n for n in nodes_list if n.centroid_id == ideal_centroid_id]
            num_replicas = min(3, len(target_nodes))
        
        selected_replicas = rng.choice(target_nodes, size=num_replicas, replace=False)
        block_replicas.append(selected_replicas.tolist())
    
    return block_replicas, ideal_centroid_ids, actual_centroid_ids

def run_default_placement(nodes_list, num_blocks: int, rng: np.random.Generator):
    """Run default placement for comparison."""
    
    block_replicas = []
    
    for block_id in range(num_blocks):
        # Choose random writer rack
        writer_rack = rng.integers(0, max(node.rack_id for node in nodes_list) + 1)
        
        # Default placement
        replicas = placement.default_hdfs_placement(nodes_list, writer_rack, num_replicas=3)
        block_replicas.append(replicas)
    
    return block_replicas

def compute_runtimes_with_systematic_penalty(
    block_replicas, ideal_centroid_ids, actual_centroid_ids,
    centroid_bits_by_id, cutpoints, b_r: int, b_c: int, hamming_gamma: float,
    centroids=None, baseline_targets=None, baseline_weights=None
):
    """Compute runtimes with Hamming penalty based on systematic degradation."""
    
    rt_params = config.RuntimeParams(
        enable_hamming_penalty=True,
        hamming_gamma=hamming_gamma,
        service_time_cap=1000.0
    )
    
    all_completion_times = []
    hamming_distances = []
    cost_ratios = []
    per_block_costs = []
    
    for block_id, (replicas, ideal_centroid_id, actual_centroid_id) in enumerate(
        zip(block_replicas, ideal_centroid_ids, actual_centroid_ids)
    ):
        # Get ideal centroid bits
        ideal_centroid_bits = centroid_bits_by_id[ideal_centroid_id]

        # Calculate cost ratio if data is available
        if centroids is not None and baseline_targets is not None and baseline_weights is not None and block_id < len(baseline_targets):
            targets = np.array(baseline_targets[block_id])
            weights = np.array(baseline_weights[block_id])
            
            # Calculate ideal cost (using raw gaps, can be negative)
            ideal_centroid = centroids[ideal_centroid_id]
            ideal_percentiles = runtime.metrics_to_percentiles(ideal_centroid, cutpoints)
            ideal_gaps = (targets - ideal_percentiles) / np.maximum(targets, 1e-6)
            ideal_cost = np.sum(weights * ideal_gaps)
            
            # Calculate actual cost (using raw gaps, can be negative)
            actual_centroid = centroids[actual_centroid_id]
            actual_percentiles = runtime.metrics_to_percentiles(actual_centroid, cutpoints)
            actual_gaps = (targets - actual_percentiles) / np.maximum(targets, 1e-6)
            actual_cost = np.sum(weights * actual_gaps)
            
            # Store raw costs (can be negative)
            cost_ratio = actual_cost  # Using actual cost directly
            per_block_costs.append(actual_cost)
        else:
            cost_ratio = 1.0
            per_block_costs.append(cost_ratio)
        
        # Set up runtime context for this block
        runtime.set_hamming_context(
            b_r=b_r,
            b_c=b_c,
            ideal_bits_by_tag={"task": ideal_centroid_bits}
        )
        
        # Compute runtime for each replica and track Hamming distances
        for replica_node in replicas:
            service_time = runtime.compute_service_time(
                node=replica_node,
                tag="task",
                cutpoints=cutpoints,
                params=rt_params
            )
            all_completion_times.append(service_time)
            
            # Calculate Hamming distance for this replica
            actual_bits = replica_node.addr_bits[-b_c:]
            hamming_dist = sum(1 for a, b in zip(actual_bits, ideal_centroid_bits) if a != b)
            hamming_distances.append(hamming_dist)
            cost_ratios.append(cost_ratio)
    
    return all_completion_times, hamming_distances, cost_ratios, per_block_costs

def run_systematic_analysis(
    num_blocks: int = 2000,
    degradation_levels: List[int] = range(1, 82),
    hamming_gamma: float = 2.0,
    repeats: int = 3,
    seed: int = 42,
    use_dominant_dim_cost: bool = False,
    dominant_cost_mode: str = "one_sided",
):
    """Run systematic degradation analysis."""
    
    print("=== Systematic Degradation Analysis ===")
    print(f"Blocks: {num_blocks}, Degradation levels: {degradation_levels}, Gamma: {hamming_gamma}")
    if use_dominant_dim_cost:
        print(f"Ranking cost: dominant-dimension ({dominant_cost_mode})")
    else:
        print(f"Ranking cost: multi-dimensional raw gaps (signed)")
    
    # Create output directory
    os.makedirs("dist/systematic", exist_ok=True)
    
    # Build heterogeneous cluster
    print("\nBuilding heterogeneous cluster...")
    nodes_list, cluster_registry, centroids, cutpoints = create_heterogeneous_cluster(
        num_nodes=400, num_racks=10, k=81, seed=42
    )
    
    # Get cluster info
    addr_len = len(nodes_list[0].addr_bits)
    k = len(centroids)
    b_c = int(np.ceil(np.log2(k)))
    b_r = addr_len - b_c
    
    print(f"Cluster: {len(nodes_list)} nodes, {k} centroids, {b_r} rack bits, {b_c} centroid bits")
    
    # Build centroid_bits_by_id mapping
    centroid_bits_by_id = {}
    for node in nodes_list:
        if node.centroid_id not in centroid_bits_by_id:
            centroid_bits_by_id[node.centroid_id] = node.addr_bits[-b_c:]
    
    # Generate workload definitions once
    print(f"Generating {num_blocks} diverse workload definitions...")
    base_rng = np.random.default_rng(seed)
    baseline_targets, baseline_weights, block_tags = generate_diverse_workloads(num_blocks, base_rng)
    
    # Run analysis for each degradation level
    results_by_level = {}
    
    for degradation_level in degradation_levels:
        print(f"\n=== Degradation Level I={degradation_level} ===")
        if degradation_level == 1:
            print("  (Optimal placement - 1st best centroid for each block)")
        elif degradation_level == 2:
            print("  (2nd best centroid for each block)")
        elif degradation_level == 3:
            print("  (3rd best centroid for each block)")
        else:
            print(f"  ({degradation_level}th best centroid for each block)")
        
        level_results = []
        
        for repeat in range(repeats):
            print(f"    Repeat {repeat + 1}/{repeats}")
            
            repeat_rng = np.random.default_rng(base_rng.integers(0, 2**32))
            
            # Run Default strategy (always random, unaffected by degradation level)
            default_replicas = run_default_placement(nodes_list, num_blocks, repeat_rng)
            
            # For Default, we still need ideal centroids for Hamming penalty calculation
            ideal_centroid_ids = []
            for targets, weights in zip(baseline_targets, baseline_weights):
                ranked_centroids = rank_centroids_by_cost(
                    centroids,
                    targets,
                    weights,
                    cutpoints,
                    use_dominant_dim_cost=use_dominant_dim_cost,
                    dominant_cost_mode=dominant_cost_mode,
                )
                ideal_centroid_ids.append(ranked_centroids[0][0])
            
            # Default uses random placement, so actual centroids are whatever nodes it picked
            default_actual_centroid_ids = [replicas[0].centroid_id for replicas in default_replicas]
            
            default_times, default_hamming, default_cost_ratios, default_per_block_costs = compute_runtimes_with_systematic_penalty(
                default_replicas, ideal_centroid_ids, default_actual_centroid_ids,
                centroid_bits_by_id, cutpoints, b_r, b_c, hamming_gamma,
                centroids, baseline_targets, baseline_weights
            )
            
            # Run Hypercube strategy with forced degradation
            hypercube_replicas, ideal_centroid_ids, actual_centroid_ids = force_placement_to_centroid_rank(
                nodes_list,
                centroids,
                baseline_targets,
                baseline_weights,
                cutpoints,
                degradation_level,
                repeat_rng,
                use_dominant_dim_cost=use_dominant_dim_cost,
                dominant_cost_mode=dominant_cost_mode,
            )
            
            hypercube_times, hypercube_hamming, hypercube_cost_ratios, hypercube_per_block_costs = compute_runtimes_with_systematic_penalty(
                hypercube_replicas, ideal_centroid_ids, actual_centroid_ids,
                centroid_bits_by_id, cutpoints, b_r, b_c, hamming_gamma,
                centroids, baseline_targets, baseline_weights
            )
            
            # Calculate statistics
            mean_default = np.mean(default_times)
            mean_hypercube = np.mean(hypercube_times)
            gain_pct = ((mean_default - mean_hypercube) / mean_default) * 100
            
            mean_default_hamming = np.mean(default_hamming)
            mean_hypercube_hamming = np.mean(hypercube_hamming)
            mean_default_cost = np.mean(default_cost_ratios)
            mean_hypercube_cost = np.mean(hypercube_cost_ratios)
            
            level_results.append({
                'repeat': repeat,
                'mean_default': mean_default,
                'mean_hypercube': mean_hypercube,
                'gain_pct': gain_pct,
                'mean_default_hamming': mean_default_hamming,
                'mean_hypercube_hamming': mean_hypercube_hamming,
                'mean_default_cost': mean_default_cost,
                'mean_hypercube_cost': mean_hypercube_cost,
                'per_block_costs_hypercube': hypercube_per_block_costs
            })
        
        # Calculate summary statistics
        gains = [r['gain_pct'] for r in level_results]
        hamming_hyper = [r['mean_hypercube_hamming'] for r in level_results]
        hamming_default = [r['mean_default_hamming'] for r in level_results]
        cost_hyper = [r['mean_hypercube_cost'] for r in level_results]
        cost_default = [r['mean_default_cost'] for r in level_results]
        
        results_by_level[degradation_level] = {
            'mean_gain': np.mean(gains),
            'std_gain': np.std(gains),
            'min_gain': np.min(gains),
            'max_gain': np.max(gains),
            'mean_hamming_hypercube': np.mean(hamming_hyper),
            'mean_hamming_default': np.mean(hamming_default),
            'mean_cost_hypercube': np.mean(cost_hyper),
            'mean_cost_default': np.mean(cost_default),
            'results': level_results
        }
        
        print(f"    Average Gain: {np.mean(gains):.2f}%")
        print(f"    Gain Std: {np.std(gains):.2f}%")
        print(f"    Mean Hamming Distance - Hypercube: {np.mean(hamming_hyper):.2f}, Default: {np.mean(hamming_default):.2f}")
        print(f"    Mean Cost - Hypercube: {np.mean(cost_hyper):.3f}, Default: {np.mean(cost_default):.3f}")
    
    # Plot results
    print(f"\nGenerating plots...")
    
    plt.figure(figsize=(18, 10))
    
    # Plot 1: Gain vs Degradation Level
    plt.subplot(2, 3, 1)
    levels = list(results_by_level.keys())
    mean_gains = [results_by_level[l]['mean_gain'] for l in levels]
    std_gains = [results_by_level[l]['std_gain'] for l in levels]
    
    plt.errorbar(levels, mean_gains, yerr=std_gains, marker='o', capsize=3, capthick=1, linewidth=1, markersize=4)
    plt.xlabel('Degradation Level (I)')
    plt.ylabel('Hypercube Gain (%)')
    plt.title('Systematic Degradation: Gain vs Level')
    plt.grid(True, alpha=0.3)
    if len(levels) > 15:
        tick_positions = levels[::5]
        plt.xticks(tick_positions)
    
    # Plot 2: Raw Cost Comparison
    plt.subplot(2, 3, 2)
    cost_hyper = [results_by_level[l]['mean_cost_hypercube'] for l in levels]
    cost_default = [results_by_level[l]['mean_cost_default'] for l in levels]
    
    plt.plot(levels, cost_hyper, 'o-', label='Hypercube', linewidth=1, markersize=3, color='red')
    plt.plot(levels, cost_default, 's-', label='Default', linewidth=1, markersize=3, color='blue')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlabel('Degradation Level (I)')
    plt.ylabel('Mean Cost (raw, can be negative)')
    plt.title('Raw Cost by Strategy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if len(levels) > 15:
        tick_positions = levels[::5]
        plt.xticks(tick_positions)
    
    # Plot 3: points ordered by I
    plt.subplot(2, 3, 3)
    gains_by_level = [results_by_level[l]['mean_gain'] for l in levels]
    # Use per-level mean of hypercube runtimes as total runtime proxy
    runtimes_by_level = [np.mean([r['mean_hypercube'] for r in results_by_level[l]['results']]) for l in levels]

    # Order is already by I (levels). Plot scatter and a connecting line to show trend along I.
    plt.plot(runtimes_by_level, gains_by_level, '--', alpha=0.4, color='gray')
    # Use reversed colormap so smaller I (least degradation) are lightest
    sc = plt.scatter(runtimes_by_level, gains_by_level, c=levels, cmap='viridis_r', s=45, alpha=0.85)
    cbar = plt.colorbar(sc)
    cbar.set_label('I (Degradation Level)')

    # Annotate a few evenly spaced I levels for readability
    annotate_idxs = np.linspace(0, len(levels) - 1, num=min(8, len(levels)), dtype=int)
    for i in annotate_idxs:
        plt.annotate(f'I={levels[i]}', (runtimes_by_level[i], gains_by_level[i]),
                     xytext=(4, 4), textcoords='offset points', fontsize=8)

    plt.xlabel('Mean Hypercube Runtime (s)')
    plt.ylabel('Hypercube Gain (%)')
    plt.title('Gain vs Total Runtime (ordered by I)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Degradation curve
    plt.subplot(2, 3, 4)
    baseline_gain = results_by_level[1]['mean_gain']  # I=1 is now the baseline (best)
    degradation = [(baseline_gain - results_by_level[l]['mean_gain']) for l in levels]
    
    plt.plot(levels, degradation, 'ro-', linewidth=2, markersize=4)
    plt.xlabel('Degradation Level (I)')
    plt.ylabel('Performance Loss (%)')
    plt.title('Performance Loss vs Degradation')
    plt.grid(True, alpha=0.3)
    if len(levels) > 15:
        tick_positions = levels[::5]
        plt.xticks(tick_positions)
    
    # Plot 5: Summary table (show key levels only)
    plt.subplot(2, 3, 5)
    plt.axis('off')
    
    # Show only key levels for readability (evenly spaced, always include last)
    if len(levels) <= 8:
        key_levels = levels
    else:
        idxs = np.linspace(0, len(levels) - 1, num=8, dtype=int)
        key_levels = [levels[i] for i in idxs]
    
    table_data = []
    for level in key_levels:
        summary = results_by_level[level]
        degradation_pct = baseline_gain - summary['mean_gain']
        table_data.append([
            f"I={level}",
            f"{summary['mean_gain']:.1f}%",
            f"{degradation_pct:.1f}%",
            f"{summary['mean_hamming_hypercube']:.1f}",
            f"{summary['mean_cost_hypercube']:.3f}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Level', 'Gain', 'Loss', 'Hamming', 'Cost'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)
    plt.title('Key Levels Summary')
    
    # Plot 6: Cost vs Hamming Distance scatter
    plt.subplot(2, 3, 6)
    hamming_distances = [results_by_level[l]['mean_hamming_hypercube'] for l in levels]
    costs = [results_by_level[l]['mean_cost_hypercube'] for l in levels]
    gains = [results_by_level[l]['mean_gain'] for l in levels]
    
    # Create scatter plot colored by degradation level I (lighter = low I)
    scatter = plt.scatter(hamming_distances, costs, c=levels, s=50, alpha=0.8, cmap='viridis_r')
    cbar2 = plt.colorbar(scatter)
    cbar2.set_label('I (Degradation Level)')
    
    # Add zero line to show over vs under provisioning
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Perfect match')
    
    annotate_idxs2 = np.linspace(0, len(levels) - 1, num=min(8, len(levels)), dtype=int)
    for i in annotate_idxs2:
        plt.annotate(f'I={levels[i]}', (hamming_distances[i], costs[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Mean Hamming Distance')
    plt.ylabel('Mean Cost (raw)')
    plt.title('Cost vs Hamming Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dist/systematic/systematic_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # High-resolution per-block cost plots for last 10 levels (I=20..39)
    print("\nGenerating per-block cost plots (I=20..39)...")
    os.makedirs('dist/systematic/per_block', exist_ok=True)
    last_levels = [l for l in levels if l >= 20][-10:] if len(levels) >= 20 else levels[-10:]
    for l in last_levels:
        # collect per-block costs from the first repeat entry for that level
        per_block = None
        for r in results_by_level[l]['results']:
            if 'per_block_costs_hypercube' in r:
                per_block = r['per_block_costs_hypercube']
                break
        if per_block is None:
            continue
        # Limit to first 100 blocks for clarity
        per_block = per_block[:100]
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(per_block)+1), per_block, marker='.', linewidth=1)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.xlabel('Block Index (first 100)')
        plt.ylabel('Raw Cost (can be negative)')
        plt.title(f'Per-block Costs for I={l} (Hypercube, first repeat)')
        plt.grid(True, alpha=0.3)
        out_path = f'dist/systematic/per_block/per_block_costs_I{l}.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Print final summary (show key levels only)
    print(f"\n=== Final Summary ===")
    # Print a dynamic, evenly spaced subset (same logic as table)
    if len(levels) <= 8:
        key_levels_summary = levels
    else:
        idxs = np.linspace(0, len(levels) - 1, num=8, dtype=int)
        key_levels_summary = [levels[i] for i in idxs]
    
    for level in key_levels_summary:
        summary = results_by_level[level]
        degradation_pct = baseline_gain - summary['mean_gain']
        print(f"I={level}: Gain = {summary['mean_gain']:.2f}% (loss: {degradation_pct:.2f}%, Hamming: {summary['mean_hamming_hypercube']:.1f}, Cost: {summary['mean_cost_hypercube']:.3f})")
    
    # Show total centroids info
    print(f"\nTotal centroids available: {len(centroids)}")
    print(f"Analyzed levels: I=1 to I={max(levels)} (top {max(levels)/len(centroids)*100:.1f}% of centroids)")
    if max(levels) < len(centroids):
        remaining = len(centroids) - max(levels)
        print(f"Remaining centroids not analyzed: {remaining} (from I={max(levels)+1} to I={len(centroids)})")
    
    print(f"\nResults saved to dist/systematic/systematic_analysis.png")
    
    return results_by_level

if __name__ == "__main__":
    run_systematic_analysis(
        num_blocks=400,
        degradation_levels=range(1, 82),
        hamming_gamma=2.0,
        repeats=1,
        seed=42,
        use_dominant_dim_cost=False,
        dominant_cost_mode="one_sided"
    )
