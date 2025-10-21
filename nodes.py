"""
nodes.py
============

This module defines the data structures and helper functions for
generating a synthetic fleet of DataNode objects and clustering them
into centroids using k-means. This includes heterogeneous
storage capacities and finite compute slots per hardware SKU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from itertools import product

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import MDS

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set this flag to False to use dynamically generated random hardware SKUs.
USE_CONTROLLED_SKUS = True


# --- Controlled SKU Definitions ---
# These are used when USE_CONTROLLED_SKUS is True.
CONTROLLED_SKUS = [
    {
        "name": "SKU 1 (General Purpose)",
        "means": np.array([0.8, 0.8, 0.8, 0.8]),
        "cov": np.array([[0.01, 0.005, 0.003, 0.003], [0.005, 0.01, 0.003, 0.003], [0.003, 0.003, 0.01, 0.002], [0.003, 0.003, 0.002, 0.01]]),
        "capacity": 25, "compute_slots": 8, "proportion": 0.3,
    },
    {
        "name": "SKU 2 (Balanced)",
        "means": np.array([1.2, 1.2, 1.2, 1.2]),
        "cov": np.array([[0.01, 0.006, 0.004, 0.004], [0.006, 0.01, 0.004, 0.004], [0.004, 0.004, 0.01, 0.003], [0.004, 0.004, 0.003, 0.01]]),
        "capacity": 35, "compute_slots": 12, "proportion": 0.3,
    },
    {
        "name": "SKU 3 (Compute-Optimized)",
        "means": np.array([1.8, 1.8, 1.0, 1.0]),
        "cov": np.array([[0.02, 0.018, 0.005, 0.005], [0.018, 0.02, 0.005, 0.005], [0.005, 0.005, 0.04, 0.008], [0.005, 0.005, 0.008, 0.04]]),
        "capacity": 20, "compute_slots": 24, "proportion": 0.2,
    },
    {
        "name": "SKU 4 (Storage-Archive)",
        "means": np.array([0.7, 0.7, 2.0, 0.8]),
        "cov": np.array([[0.01, 0.005, 0.002, 0.002], [0.005, 0.01, 0.002, 0.002], [0.002, 0.002, 0.05, 0.003], [0.002, 0.002, 0.003, 0.01]]),
        "capacity": 75, "compute_slots": 4, "proportion": 0.2,
    },
]

# --- Dynamic SKU Generation ---
def _generate_random_skus(num_skus: int = 4) -> List[Dict]:
    """
    Generates a list of random, but logically consistent, hardware SKUs.
    """
    logger.info(f"Generating {num_skus} random hardware SKUs...")
    skus = []
    
    # Generate random proportions that sum to 1.0
    proportions = np.random.rand(num_skus)
    proportions /= proportions.sum()
    
    for i in range(num_skus):
        # Create a base mean vector, then specialize it
        base_mean = np.random.uniform(0.7, 1.3, size=4)
        specialization = np.random.choice(['cpu', 'ram', 'disk', 'net', 'balanced'])
        
        capacity = np.random.randint(20, 80)

        if specialization == 'cpu':
            base_mean[0] *= np.random.uniform(1.5, 2.0) # Boost CPU
            capacity = np.random.randint(15, 30) # Lower capacity for compute
        elif specialization == 'ram':
            base_mean[1] *= np.random.uniform(1.5, 2.0) # Boost RAM
        elif specialization == 'disk':
            base_mean[2] *= np.random.uniform(1.8, 2.5) # Boost Disk
            capacity = np.random.randint(60, 100) # Higher capacity for storage
        elif specialization == 'net':
            base_mean[3] *= np.random.uniform(1.5, 2.0) # Boost Net

        # Create a plausible covariance matrix
        base_cov = np.random.uniform(0.002, 0.008, size=(4, 4))
        cov = (base_cov + base_cov.T) / 2 # Make it symmetric
        np.fill_diagonal(cov, np.random.uniform(0.01, 0.05, size=4)) # Ensure diagonal is dominant

        skus.append({
            "name": f"Random SKU {i+1} ({specialization})",
            "means": base_mean,
            "cov": cov,
            "capacity": int(capacity),
            "proportion": proportions[i],
            "compute_slots": max(4, int(np.random.normal(loc=12, scale=8))) # Add random compute slots
        })
    return skus


@dataclass
class DataNode:
    """Representation of a storage node in the synthetic cluster."""
    node_id: int
    rack_id: int
    c_vector: np.ndarray
    s_vector: np.ndarray
    max_replicas: int
    max_compute_slots: int
    centroid_id: int = -1
    addr_bits: str = ""
    replica_count: int = 0
    compute_load: int = 0

    def __hash__(self): return hash(self.node_id)
    def __eq__(self, other): return isinstance(other, DataNode) and self.node_id == other.node_id
    
    def has_capacity(self) -> bool: return self.replica_count < self.max_replicas
    def has_compute_slot(self) -> bool: return self.compute_load < self.max_compute_slots
    
    def start_task(self): self.compute_load += 1
    def finish_task(self): self.compute_load = max(0, self.compute_load - 1)
    def increment_replica_count(self): self.replica_count += 1


def generate_nodes(
    num_nodes: int,
    num_racks: int,
    seed: int | None = None,
    variance_factor: float = 1.0,
) -> List[DataNode]:
    """
    Generate a list of synthetic DataNodes from multiple hardware classes.
    """
    if seed is not None: np.random.seed(seed)

    sku_definitions = CONTROLLED_SKUS if USE_CONTROLLED_SKUS else _generate_random_skus()

    all_c_vectors, all_capacities, all_slots = [], [], []
    
    nodes_left = num_nodes
    for i, sku in enumerate(sku_definitions):
        n_sku = int(num_nodes * sku['proportion']) if i < len(sku_definitions) - 1 else nodes_left
        nodes_left -= n_sku
        
        c_vecs = np.random.multivariate_normal(mean=sku['means'], cov=sku['cov'] * variance_factor, size=n_sku)
        all_c_vectors.append(c_vecs)
        all_capacities.extend([sku['capacity']] * n_sku)
        all_slots.extend([sku['compute_slots']] * n_sku)

    c_vectors = np.vstack(all_c_vectors)
    capacities = np.array(all_capacities)
    slots = np.array(all_slots)
    
    indices = np.random.permutation(len(c_vectors))
    c_vectors, capacities, slots = c_vectors[indices], capacities[indices], slots[indices]

    nodes: List[DataNode] = []
    for i in range(num_nodes):
        nodes.append(DataNode(
            node_id=i, rack_id=i % num_racks,
            c_vector=np.clip(c_vectors[i], 0.1, None),
            s_vector=np.clip(c_vectors[i], 0.1, None).copy(),
            max_replicas=int(capacities[i]),
            max_compute_slots=int(slots[i])
        ))
    return nodes


def kmeans_cluster(
    nodes: List[DataNode],
    k: int,
    max_iter: int = 20,
    tol: float = 1e-3,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster nodes by their C vectors using k-means."""
    c_matrix = np.vstack([node.c_vector for node in nodes])
    if seed is not None:
        np.random.seed(seed)
    if SKLEARN_AVAILABLE:
        km = KMeans(n_clusters=k, max_iter=max_iter, tol=tol, random_state=seed, n_init=10)
        labels = km.fit_predict(c_matrix)
        centroids = km.cluster_centers_
    else:
        # sklearn is not available?
        num_nodes, dim = c_matrix.shape
        idx = np.random.choice(num_nodes, k, replace=False)
        centroids = c_matrix[idx].copy()
        for _ in range(max_iter):
            dists = np.linalg.norm(c_matrix[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([c_matrix[labels == j].mean(axis=0) for j in range(k)])
            if np.all(np.abs(new_centroids - centroids) < tol):
                break
            centroids = new_centroids
    return labels, centroids

def compute_centroid_bit_mapping(
    centroids: np.ndarray, num_bits: int, return_mds_coords: bool = False
) -> Dict[int, int] | Tuple[Dict[int, int], np.ndarray]:
    """
    Computes an optimal mapping from centroid ID to a binary code using MDS
    and the Hungarian algorithm to preserve semantic locality.
    """
    logger.info("Computing optimal centroid -> bit mapping using MDS + Hungarian...")
    num_centroids = len(centroids)

    hypercube_vertices = np.array(list(product([0, 1], repeat=num_bits)))

    dissimilarity_matrix = squareform(pdist(centroids, "euclidean"))
    mds = MDS(
        n_components=num_bits,
        dissimilarity="precomputed",
        random_state=42,
        n_init=10,
        max_iter=1000,
    )
    mds_coords = mds.fit_transform(dissimilarity_matrix)

    cost_matrix = np.linalg.norm(
        mds_coords[:, np.newaxis, :] - hypercube_vertices[np.newaxis, :, :], axis=2
    )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {
        int(centroid_idx): int(vertex_idx)
        for centroid_idx, vertex_idx in zip(row_ind, col_ind)
    }
    logger.info("Finished computing centroid bit mapping.")
    if return_mds_coords:
        return mapping, mds_coords
    return mapping


def assign_centroids_and_addresses(
    nodes: List[DataNode],
    labels: np.ndarray,
    b_r: int,
    b_c: int,
    centroid_mapping: Dict[int, int],
) -> None:
    """
    Assign centroid IDs and compute addr_bits for each node using the
    pre-computed optimal mapping.
    """
    for node, cluster_id in zip(nodes, labels):
        node.centroid_id = int(cluster_id)
        rack_bits = format(node.rack_id, f"0{b_r}b")
        
        semantic_code = centroid_mapping.get(node.centroid_id)
        if semantic_code is None:
            logger.warning(f"Centroid ID {node.centroid_id} not in mapping. Assigning default.")
            semantic_code = 0
        
        centroid_bits = format(semantic_code, f"0{b_c}b")
        node.addr_bits = rack_bits + centroid_bits


def build_registry(nodes: List[DataNode]) -> Dict[str, List[DataNode]]:
    """Build a registery-like mapping from addr_bits to node lists."""
    registry: Dict[str, List[DataNode]] = {}
    for node in nodes:
        registry.setdefault(node.addr_bits, []).append(node)
    return registry

# Default parameters
DEFAULT_NUM_NODES: int = 200
DEFAULT_NUM_RACKS: int = 5
DEFAULT_K: int = 64
