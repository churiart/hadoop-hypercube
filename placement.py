"""
placement.py
============

Block placement algorithms for the simulation.  This module
implements two strategies:

  1. default_hdfs_placement: replicates are placed uniformly at random
     across all nodes with capacity, subject to the rack constraint.

  2. hypercube_placement: Our approach which places two intra-rack
     and one inter-rack replica, with a robust, weighted global
     fallback mechanism.
"""

from __future__ import annotations

import random
import math
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np

from .nodes import DataNode

# Set this flag to False to use dynamically generated random workloads.
USE_CONTROLLED_WORKLOAD = False


# --- Controlled Workload Definitions ---
# These are used when USE_CONTROLLED_WORKLOAD is True.
CONTROLLED_TARGETS: Dict[str, Dict[str, float]] = {
    "balanced": {"cpu": 0.60, "ram": 0.60, "disk": 0.60, "net": 0.60},
    "cpu-bound": {"cpu": 0.80, "ram": 0.60, "disk": 0.40, "net": 0.50},
    "cpu-bound-hi": {"cpu": 0.90, "ram": 0.70, "disk": 0.50, "net": 0.60},
    "cpu-bound-lo": {"cpu": 0.70, "ram": 0.50, "disk": 0.30, "net": 0.40},
    "disk-bound": {"cpu": 0.40, "ram": 0.50, "disk": 0.80, "net": 0.50},
    "disk-bound-hi": {"cpu": 0.50, "ram": 0.60, "disk": 0.90, "net": 0.50},
    "disk-bound-lo": {"cpu": 0.30, "ram": 0.40, "disk": 0.70, "net": 0.40},
    "ram-bound": {"cpu": 0.50, "ram": 0.80, "disk": 0.40, "net": 0.60},
    "ram-bound-hi": {"cpu": 0.60, "ram": 0.90, "disk": 0.50, "net": 0.70},
    "ram-bound-lo": {"cpu": 0.40, "ram": 0.70, "disk": 0.30, "net": 0.50},
    "net-bound": {"cpu": 0.50, "ram": 0.60, "disk": 0.40, "net": 0.80},
    "net-bound-hi": {"cpu": 0.60, "ram": 0.70, "disk": 0.50, "net": 0.90},
    "net-bound-lo": {"cpu": 0.40, "ram": 0.50, "disk": 0.30, "net": 0.70},
}

CONTROLLED_WEIGHTS: Dict[str, Dict[str, float]] = {
    "balanced": {"cpu": 0.25, "ram": 0.25, "disk": 0.25, "net": 0.25},
    "cpu-bound": {"cpu": 0.55, "ram": 0.20, "disk": 0.15, "net": 0.10},
    "cpu-bound-hi": {"cpu": 0.55, "ram": 0.20, "disk": 0.15, "net": 0.10},
    "cpu-bound-lo": {"cpu": 0.55, "ram": 0.20, "disk": 0.15, "net": 0.10},
    "disk-bound": {"cpu": 0.15, "ram": 0.15, "disk": 0.55, "net": 0.15},
    "disk-bound-hi": {"cpu": 0.15, "ram": 0.15, "disk": 0.55, "net": 0.15},
    "disk-bound-lo": {"cpu": 0.15, "ram": 0.15, "disk": 0.55, "net": 0.15},
    "ram-bound": {"cpu": 0.15, "ram": 0.55, "disk": 0.15, "net": 0.15},
    "ram-bound-hi": {"cpu": 0.15, "ram": 0.55, "disk": 0.15, "net": 0.15},
    "ram-bound-lo": {"cpu": 0.15, "ram": 0.55, "disk": 0.15, "net": 0.15},
    "net-bound": {"cpu": 0.15, "ram": 0.15, "disk": 0.15, "net": 0.55},
    "net-bound-hi": {"cpu": 0.15, "ram": 0.15, "disk": 0.15, "net": 0.55},
    "net-bound-lo": {"cpu": 0.15, "ram": 0.15, "disk": 0.15, "net": 0.55},
}

# --- Dynamic Workload Generation ---
def _generate_random_workload_definitions() -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Generates a single, correlated random workload definition (targets and weights).
    A high weight for a resource will correlate with a high target for that resource.
    """
    dims = ['cpu', 'ram', 'disk', 'net']
    
    # 1. Generate random weights that sum to 1.0
    weights_raw = np.random.rand(len(dims))
    weights_normalized = weights_raw / weights_raw.sum()
    weights = {dim: w for dim, w in zip(dims, weights_normalized)}
    
    # 2. Generate correlated targets based on weights
    targets = {}
    for dim, w in weights.items():
        # Base target is proportional to the weight, ensuring correlation
        # Add some noise to decorrelate slightly
        base_target = 0.4 + w * 0.6 
        noise = np.random.uniform(-0.1, 0.1)
        final_target = np.clip(base_target + noise, 0.1, 1.0)
        targets[dim] = final_target
        
    return targets, weights


# --- Shared Constants ---
SOFTMAX_TEMPS: Dict[str, float] = {
    "balanced": 0.07, "cpu-bound": 0.05, "cpu-bound-hi": 0.05,
    "cpu-bound-lo": 0.05, "disk-bound": 0.08, "disk-bound-hi": 0.08,
    "disk-bound-lo": 0.08, "ram-bound": 0.06, "ram-bound-hi": 0.06,
    "ram-bound-lo": 0.06, "net-bound": 0.07, "net-bound-hi": 0.07,
    "net-bound-lo": 0.07,
}
LOCALITY_BONUS_FACTOR = 0.5


def compute_decile_cutpoints(nodes: List[DataNode]) -> np.ndarray:
    """Compute decile cut points for each dimension of the S vector."""
    s_matrix = np.vstack([node.s_vector for node in nodes])
    d = s_matrix.shape[1]
    cutpoints = np.zeros((d, 9))
    for i in range(d):
        col = s_matrix[:, i]
        sorted_vals = np.sort(col)
        for j in range(1, 10):
            pct = j / 10.0
            cutpoints[i, j - 1] = np.percentile(sorted_vals, pct * 100)
    return cutpoints


def s_to_percentiles(s_vector: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    """Map a single S vector to decile percentiles using precomputed cutpoints."""
    d = s_vector.shape[0]
    p = np.zeros(d)
    for i in range(d):
        bucket = np.searchsorted(cutpoints[i], s_vector[i], side="right")
        bucket = min(bucket, 9)
        p[i] = (bucket + 0.5) / 10.0
    return p


def default_hdfs_placement(
    nodes: List[DataNode],
    writer_rack: int,
    num_replicas: int = 3,
) -> List[DataNode]:
    """Default HDFS block placement: uniform random subject to rack constraint."""
    candidates = [n for n in nodes if n.has_capacity()]
    if len(candidates) < num_replicas:
        return candidates
    
    writer_rack_candidates = [n for n in candidates if n.rack_id == writer_rack]
    other_racks_candidates = [n for n in candidates if n.rack_id != writer_rack]
    
    chosen: List[DataNode] = []
    
    if writer_rack_candidates:
        chosen.append(random.choice(writer_rack_candidates))
    
    if other_racks_candidates:
        chosen.append(random.choice(other_racks_candidates))
    
    while len(chosen) < num_replicas:
        remaining_candidates = [n for n in candidates if n not in chosen]
        if not remaining_candidates: break
        chosen.append(random.choice(remaining_candidates))
        
    return chosen


def _select_nodes_from_pool(
    candidate_pool: List[DataNode],
    num_to_select: int,
    tag: str,
    cutpoints: np.ndarray,
    targets: Dict[str, float],
    weights: Dict[str, float],
    writer_rack: Optional[int] = None,
) -> List[DataNode]:
    """Helper function to score and select nodes from a candidate pool."""
    if not candidate_pool:
        return []

    dims = ['cpu', 'ram', 'disk', 'net']
    
    def cost_from_vector(vector: np.ndarray) -> float:
        cost = 0.0
        from .runtime import metrics_to_percentiles
        p = metrics_to_percentiles(vector, cutpoints)
        for i, dim in enumerate(dims):
            target = targets[dim]
            weight = weights[dim]
            gap = max(0.0, (target - p[i]) / target)
            cost += weight * gap
        return cost

    if len(candidate_pool) <= num_to_select:
        sorted_pool = sorted(candidate_pool, key=lambda n: cost_from_vector(n.s_vector))
        return sorted_pool[:num_to_select]

    scores: List[Tuple[DataNode, float]] = []
    T = SOFTMAX_TEMPS.get(tag, 0.07) # Default temp if tag is random
    for n in candidate_pool:
        cost = cost_from_vector(n.s_vector)
        if writer_rack is not None and n.rack_id == writer_rack:
            cost *= LOCALITY_BONUS_FACTOR
        
        score = math.exp(-cost / max(T, 1e-9))
        scores.append((n, score))
        
    total_score = sum(s for (_, s) in scores)
    if total_score <= 1e-9:
        return random.sample(candidate_pool, min(num_to_select, len(candidate_pool)))
        
    chosen: List[DataNode] = []
    
    for _ in range(num_to_select):
        if not scores: break
        
        current_total_score = sum(s for _, s in scores)
        if current_total_score <= 1e-9:
            nodes_only_degen = [n for n, _ in scores]
            if not nodes_only_degen: break
            chosen.extend(random.sample(nodes_only_degen, min(num_to_select - len(chosen), len(nodes_only_degen))))
            break

        probs = [s / current_total_score for _, s in scores]
        nodes_only = [n for n, _ in scores]
        
        if not nodes_only: break
        
        selected_node = np.random.choice(nodes_only, p=probs)
        chosen.append(selected_node)
        
        scores = [(n, s) for n, s in scores if n.node_id != selected_node.node_id]
            
    return chosen


def hypercube_softmax_placement(
    registry: Dict[str, List[DataNode]],
    nodes: List[DataNode],
    centroids: np.ndarray,
    writer_rack: int,
    tag: str,
    cutpoints: np.ndarray,
    num_replicas: int = 3,
) -> List[DataNode]:
    """
    Place replicas using a sequential, constrained hypercube search.
    """
    # --- Step 1: Get Workload Definition (Controlled or Random) ---
    if USE_CONTROLLED_WORKLOAD:
        targets = CONTROLLED_TARGETS[tag]
        weights = CONTROLLED_WEIGHTS[tag]
    else:
        targets, weights = _generate_random_workload_definitions()

    # --- Step 2: Determine Ideal Centroid based on workload ---
    dims = ['cpu', 'ram', 'disk', 'net']
    def cost_from_static_vector(vector: np.ndarray) -> float:
        cost = 0.0
        p = s_to_percentiles(vector, cutpoints)
        for i, dim in enumerate(dims):
            target = targets[dim]
            weight = weights[dim]
            gap = max(0.0, (target - p[i]) / target)
            cost += weight * gap
        return cost
        
    centroid_costs = {cid: cost_from_static_vector(c_vec) for cid, c_vec in enumerate(centroids)}
    ideal_centroid = min(centroid_costs, key=centroid_costs.get)
    
    max_rack_id = max(n.rack_id for n in nodes)
    b_r = max(1, int(math.ceil(math.log2(max_rack_id + 1))))
    addr_bits_len = len(nodes[0].addr_bits)
    b_c = addr_bits_len - b_r
    rack_bits_str = format(writer_rack, f"0{b_r}b")
    
    chosen_nodes: List[DataNode] = []

    # --- Stage 1: Intra-Rack Search ---
    intra_rack_candidates = []
    for radius in range(b_c + 1): 
        if len(intra_rack_candidates) >= 20: break 
        ideal_centroid_bits = format(ideal_centroid, f"0{b_c}b")
        indices = list(range(b_c))
        positions_list = combinations(indices, radius) if radius > 0 else [[]]

        for positions in positions_list:
            bits = list(ideal_centroid_bits)
            for pos in positions:
                bits[pos] = '1' if bits[pos] == '0' else '0'
            addr = rack_bits_str + ''.join(bits)
            nodes_at_addr = [n for n in registry.get(addr, []) if n.has_capacity()]
            intra_rack_candidates.extend(nodes_at_addr)
    
    intra_rack_candidates = list(dict.fromkeys(intra_rack_candidates))
    
    num_to_select = min(2, len(intra_rack_candidates))
    if num_to_select > 0:
        selected = _select_nodes_from_pool(intra_rack_candidates, num_to_select, tag, cutpoints, targets, weights)
        chosen_nodes.extend(selected)

    if len(chosen_nodes) >= num_replicas:
        return chosen_nodes

    # --- Stage 2: Inter-Rack Search ---
    num_needed = num_replicas - len(chosen_nodes)
    inter_rack_candidates = []
    all_other_racks = [r for r in range(max_rack_id + 1) if r != writer_rack]
    for other_rack in all_other_racks:
        if len(inter_rack_candidates) >= 20: break
        other_rack_str = format(other_rack, f"0{b_r}b")
        ideal_centroid_bits_str = format(ideal_centroid, f"0{b_c}b")
        addr = other_rack_str + ideal_centroid_bits_str
        nodes_at_addr = [n for n in registry.get(addr, []) if n.has_capacity() and n not in chosen_nodes]
        inter_rack_candidates.extend(nodes_at_addr)

    inter_rack_candidates = list(dict.fromkeys(inter_rack_candidates))
    
    if len(inter_rack_candidates) > 0:
        selected = _select_nodes_from_pool(inter_rack_candidates, num_needed, tag, cutpoints, targets, weights)
        chosen_nodes.extend(selected)

    if len(chosen_nodes) >= num_replicas:
        return chosen_nodes

    # --- Stage 3: Weighted Global Fallback Search ---
    num_needed = num_replicas - len(chosen_nodes)
    if num_needed > 0:
        global_candidates = [n for n in nodes if n.has_capacity() and n not in chosen_nodes]
        if global_candidates:
            selected = _select_nodes_from_pool(global_candidates, num_needed, tag, cutpoints, targets, weights, writer_rack=writer_rack)
            chosen_nodes.extend(selected)
            
    return chosen_nodes
