"""
runtime.py
===========

Strategy-agnostic runtime model for map attempts. This module provides
helpers to compute (1) node hardware percentiles, (2) base compute time,
and (3) total map time including read (locality) penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Protocol, Iterable
import numpy as np

@dataclass
class RuntimeParams:
    """
    Tunables for the runtime model.

    - net_remote_coef: scales the base read penalty.
    - rack_remote_factor: extra multiplier for rack-local reads vs node-local.
    - off_rack_factor: extra multiplier for off-rack reads vs node-local.
    - service_time_cap: a soft cap (abstract units) to avoid pathological tails.
    - epsilon: tiny positive to prevent division by zero.
    """
    net_remote_coef: float = 0.8
    rack_remote_factor: float = 1.4
    off_rack_factor: float = 2.2
    service_time_cap: float = 100.0
    epsilon: float = 1e-9
    # Optional compute-time multiplier based on Hamming distance in centroid code
    enable_hamming_penalty: bool = False
    hamming_gamma: float = 4.0


class NodeLike(Protocol):
    node_id: int
    rack_id: int
    s_vector: np.ndarray  # shape (4,) → [CPU, RAM, DISK, NET] performance scores


# ----------------------------
# Percentiles & base compute time
# ----------------------------

def metrics_to_percentiles(s_vector: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    """
    Convert a node's raw s_vector (CPU, RAM, DISK, NET scores) to per-dimension
    percentiles p in [0, 1] using global decile cutpoints.

    Parameters
    ----------
    s_vector : np.ndarray, shape (4,)
        Raw performance scores for [CPU, RAM, DISK, NET].
    cutpoints : np.ndarray, shape (4, 9)
        Decile thresholds for each dimension.
    """
    p = np.zeros(4, dtype=float)
    for i in range(4):
        # count how many cutpoints the value exceeds
        p[i] = float(np.sum(s_vector[i] > cutpoints[i])) / float(cutpoints[i].size)
    return p


def compute_service_time(node: NodeLike, tag: str, cutpoints: np.ndarray, params: RuntimeParams) -> float:
    """
    Base compute time T_compute for running a map locally on node.
    Higher percentiles imply faster hardware which means shorter times.

    The selection of bottleneck dimension is controlled by the task 'tag':
      - "cpu-bound": use CPU percentile p[0]
      - "mem-bound": use RAM percentile p[1]
      - "io-bound" : use DISK percentile p[2]
      - "net-bound": use NET percentile p[3]
      - otherwise  : use the mean of all four percentiles

    T_compute = min(service_time_cap, 1 / max(p_bottleneck, ε))
    where epsilon avoids division by zero.
    """
    p = metrics_to_percentiles(node.s_vector, cutpoints)
    eps = params.epsilon

    base = None
    if "cpu-bound" in tag:
        base = 1.0 / (p[0] + eps)
    elif "mem-bound" in tag:
        base = 1.0 / (p[1] + eps)
    elif "io-bound" in tag:
        base = 1.0 / (p[2] + eps)
    elif "net-bound" in tag:
        base = 1.0 / (p[3] + eps)
    else:
        # balanced (use average capacity)
        base = 4.0 / (np.sum(p) + eps)

    # Optional Hamming-distance penalty on compute time (strategy-agnostic)
    penalty_multiplier = _compute_hamming_multiplier_if_available(node, tag, params)
    return min(params.service_time_cap, penalty_multiplier * base)


# ----------------------------
# Locality (read) penalty and total time
# ----------------------------

def read_penalty(exec_node: NodeLike, replica_nodes: Iterable[NodeLike], cutpoints: np.ndarray, params: RuntimeParams) -> float:
    """
    - Node-local (level 0): 0.
    - Rack-local (level 1): net_remote_coef * rack_remote_factor / (p_net + ε).
    - Off-rack  (level 2): net_remote_coef * off_rack_factor  / (p_net + ε).
    """
    # If exec_node holds one of the replicas → no read penalty
    if any(exec_node.node_id == n.node_id for n in replica_nodes):
        return 0.0

    racks = {n.rack_id for n in replica_nodes}
    rack_local = (exec_node.rack_id in racks)

    net_p = metrics_to_percentiles(exec_node.s_vector, cutpoints)[3]
    eps = params.epsilon
    penalty = params.net_remote_coef / (net_p + eps)

    if rack_local:
        penalty *= params.rack_remote_factor
    else:
        penalty *= params.off_rack_factor

    return penalty


def predict_total_time(exec_node: NodeLike, replica_nodes: Iterable[NodeLike], tag: str, cutpoints: np.ndarray, params: RuntimeParams) -> float:
    """
    Total predicted time for a map attempt on exec_node that reads the
    block from 'replica_nodes' (the three replica holders).

    T = T_read + T_compute
      = read_penalty(exec_node, replicas, ...) + compute_service_time(exec_node, tag, ...)

    Parameters
    ----------
    exec_node : NodeLike
        The node that will execute the map attempt.
    replica_nodes : Iterable[NodeLike]
        The nodes that hold the data block replicas (usually 3 nodes).
    tag : str
        Workload tag (controls bottleneck dimension for compute).
    cutpoints : np.ndarray
        Global decile cutpoints (shape (4, 9)).
    params : RuntimeParams
        Runtime tuning parameters.
    """
    base = compute_service_time(exec_node, tag, cutpoints, params)
    rp = read_penalty(exec_node, replica_nodes, cutpoints, params)
    return base + rp


# ----------------------------
# Optional Hamming penalty context
# ----------------------------

_HAM_B_R: int | None = None
_HAM_B_C: int | None = None
_HAM_IDEAL_BITS_BY_TAG: dict | None = None


def set_hamming_context(b_r: int, b_c: int, ideal_bits_by_tag: dict | None) -> None:
    global _HAM_B_R, _HAM_B_C, _HAM_IDEAL_BITS_BY_TAG
    _HAM_B_R = int(b_r)
    _HAM_B_C = int(b_c)
    _HAM_IDEAL_BITS_BY_TAG = dict(ideal_bits_by_tag or {})


def _compute_hamming_multiplier_if_available(node: NodeLike, tag: str, params: RuntimeParams) -> float:
    if _HAM_B_R is None or _HAM_B_C is None or _HAM_IDEAL_BITS_BY_TAG is None:
        return 1.0
    if not getattr(params, 'enable_hamming_penalty', False):
        return 1.0
    try:
        addr: str = getattr(node, 'addr_bits', '')
        if not isinstance(addr, str) or len(addr) < (_HAM_B_R + _HAM_B_C):
            return 1.0
        node_centroid_bits = addr[-_HAM_B_C:]
        ideal_bits = _HAM_IDEAL_BITS_BY_TAG.get(tag)
        if not isinstance(ideal_bits, str) or len(ideal_bits) != _HAM_B_C:
            return 1.0
        d = sum(1 for a, b in zip(node_centroid_bits, ideal_bits) if a != b)
        frac = d / max(1, _HAM_B_C)
        gamma = float(getattr(params, 'hamming_gamma', 3.0))
        return max(1.0, 1.0 + gamma * frac)
    except Exception:
        return 1.0

def estimate_speed(start_time: float, now: float, progress: float, epsilon: float = 1e-9) -> float:
    """
    Estimate attempt speed v = progress / elapsed.

    - start_time: when the attempt started.
    - now: current time.
    - progress: fraction in [0, 1].
    - epsilon ε: small value to avoid division by zero.
    """
    elapsed = max(now - start_time, 0.0)
    denom = max(epsilon, elapsed)
    return max(0.0, min(1.0, progress / denom))


def estimate_remaining_time(start_time: float, now: float, progress: float, epsilon: float = 1e-9) -> float:
    """
    Estimate remaining time R = (1 - progress) / max(v, ε).

    - start_time: when the attempt began.
    - now: current time.
    - progress: fraction in [0, 1].
    - epsilon ε: tiny positive constant.
    """
    v = estimate_speed(start_time, now, progress, epsilon)
    return (1.0 - max(0.0, min(1.0, progress))) / max(v, epsilon)
