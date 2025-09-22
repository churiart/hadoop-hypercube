"""
Speculation logic that mirrors Hadoop MRv2 at a high level.
This module decides when to speculate and where to place the duplicate
attempt. It does not depend on the placement strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple

# We rely on the shared runtime model for time predictions.
from .runtime import (
    RuntimeParams,
    predict_total_time,
    estimate_remaining_time,
)


class NodeLike(Protocol):
    node_id: int
    rack_id: int
    s_vector: object  # only passed through to runtime; no direct use here
    def has_compute_slot(self) -> bool: ...


class AttemptLike(Protocol):
    task_id: int
    job_id: int
    start_time: float
    progress: float  # in [0, 1]
    is_speculative: bool


class TaskLike(Protocol):
    tag: str
    # The three replica holders for this block (data-local nodes)
    nodes: Sequence[NodeLike]

@dataclass
class SpecParams:
    """
    Tunables for speculation.

    - theta θ: speed trigger; consider speculation if v < θ * v_med.
    - delta δ: benefit margin; require T_new < (1 - δ) * R.
    - job_cap_beta β: at most β fraction of a job's running attempts may be speculative.
    - cluster_free_min f_min: speculate only if free_slots/total_slots ≥ f_min.
    - min_run_ticks: attempt must have run at least this long before speculation.
    """
    theta: float = 0.5
    delta: float = 0.20
    job_cap_beta: float = 0.10
    cluster_free_min: float = 0.20
    min_run_ticks: float = 1.0


@dataclass
class JobStats:
    running_attempts: Sequence[AttemptLike]
    running_count: int
    spec_running_count: int


@dataclass
class ClusterStats:
    free_slots: int
    total_slots: int


def locality_level(exec_node: NodeLike, replica_nodes: Sequence[NodeLike]) -> int:
    """
    Return locality level L of exec_node relative to replicas.
    0 if node-local, 1 if rack-local, 2 otherwise.
    """
    if any(exec_node.node_id == r.node_id for r in replica_nodes):
        return 0
    replica_racks = {r.rack_id for r in replica_nodes}
    return 1 if exec_node.rack_id in replica_racks else 2


def best_node_by_locality_and_time(
    task: TaskLike,
    candidates: Sequence[NodeLike],
    cutpoints,
    runtime_params: RuntimeParams
) -> Optional[Tuple[NodeLike, float]]:
    """
    Pick the node that minimizes total predicted time T while
    respecting Hadoop-like locality preference: consider L0 first; if none,
    consider L1; else L2.

    Returns (node, T) or None if no candidates have slots.
    """
    # Filter to nodes that actually have a free slot
    avail = [n for n in candidates if n.has_compute_slot()]
    if not avail:
        return None

    # Partition by locality level
    groups = {0: [], 1: [], 2: []}
    for n in avail:
        L = locality_level(n, task.nodes)
        groups[L].append(n)

    # Examine levels in order 0 -> 1 -> 2
    for L in (0, 1, 2):
        if groups[L]:
            best = None
            best_T = float("inf")
            for n in groups[L]:
                T = predict_total_time(n, task.nodes, task.tag, cutpoints, runtime_params)
                if T < best_T:
                    best_T = T
                    best = n
            return (best, best_T)
    return None


def _progress_value(obj, now: float) -> float:
    """Return numeric progress in [0,1] from either a .progress attribute or .progress(now) method."""
    p = getattr(obj, "progress", 0.0)
    try:
        if callable(p):
            try:
                val = p(now)
            except TypeError:
                val = p()  # in case it's a 0-arg method
        else:
            val = p
        val = float(val)
    except Exception:
        val = 0.0
    if val < 0.0: return 0.0
    if val > 1.0: return 1.0
    return val


# ----------------------------
# Main decisions
# ----------------------------

def should_speculate(
    attempt: AttemptLike,
    now: float,
    job_stats: JobStats,
    cluster_stats: ClusterStats,
    has_competing_attempt: bool,
    spec_params: SpecParams,
) -> bool:
    """
    Decide if we should request a speculative attempt for 'attempt'.

    Conditions (all must pass):
    1) Age gate: now - start_time ≥ min_run_ticks.
    2) Cluster gate: free_slots / total_slots ≥ f_min.
    3) Job cap: spec_running_count / running_count < β.
    4) Per-task: there is no competing attempt already (has_competing_attempt == False).
    5) Speed trigger: v < θ * v_med (computed over *other* running attempts when possible).

    Returns True if eligible, else False.
    """
    # 1) Age gate
    if now - attempt.start_time < spec_params.min_run_ticks:
        return False

    # 2) Cluster gate
    if cluster_stats.total_slots <= 0:
        return False
    f = cluster_stats.free_slots / float(cluster_stats.total_slots)  # "eff"
    if f < spec_params.cluster_free_min:
        return False

    # 3) Job cap
    running = max(1, job_stats.running_count)
    frac = job_stats.spec_running_count / float(running)  # speculative fraction (β check)
    if frac >= spec_params.job_cap_beta:
        return False

    # 4) Per-task (only one speculative copy at a time)
    if has_competing_attempt:
        return False

    # 5) Speed trigger (compare vs job median of others when available)
    # Compute v for the candidate attempt
    elapsed = max(0.0, now - attempt.start_time)
    if elapsed <= 0.0:
        return False
    v = max(0.0, min(1.0, attempt.progress / max(elapsed, 1e-9)))  # "vee"

    # Median over peers (exclude this attempt if present)
    peer_speeds: List[float] = []
    for a in job_stats.running_attempts:
        if a is attempt:
            continue
        e = max(0.0, now - a.start_time)
        if e > 0.0:
            peer_speeds.append(max(0.0, min(1.0, _progress_value(a, now) / e)))
    if not peer_speeds:
        # With no peers to compare to, be conservative; don't speculate.
        return False

    peer_speeds.sort()
    mid = len(peer_speeds) // 2
    if len(peer_speeds) % 2 == 1:
        v_med = peer_speeds[mid]
    else:
        v_med = 0.5 * (peer_speeds[mid - 1] + peer_speeds[mid])

    # Trigger only if v is clearly slower than peers
    return v < (spec_params.theta * v_med)


def pick_speculative_node(
    task: TaskLike,
    candidates: Sequence[NodeLike],
    cutpoints,
    runtime_params: RuntimeParams,
    original_remaining_time: float,
    spec_params: SpecParams,
) -> Optional[Tuple[NodeLike, float]]:
    """
    Pick a target node for a speculative copy (if any).

    - We prefer L0 nodes; if none, L1; else L2.
    - Among nodes at the chosen locality, we minimize total time T from runtime model.
    - Only return a node if min_T < (1 - δ) * R, where
      δ is the safety margin and R is remaining time of the original.

    Returns (node, predicted_T) or None if no beneficial candidate exists.
    """
    best = best_node_by_locality_and_time(task, candidates, cutpoints, runtime_params)
    if best is None:
        return None
    node, T_new = best

    # Benefit test with δ margin
    if T_new < (1.0 - spec_params.delta) * float(original_remaining_time):
        return (node, T_new)
    return None


# ----------------------------
# Bytes-based speculation (MRv2-style progress-bytes rate)
# ----------------------------

def should_speculate_bytes(
    attempt,
    now: float,
    job_attempts: Sequence,
    cluster_stats: ClusterStats,
    has_competing_attempt: bool,
    spec_params: SpecParams,
) -> bool:
    """
    Decide if we should request a speculative attempt using byte-rate progress.

    Conditions (all must pass):
    1) Age gate: now - start_time ≥ min_run_ticks.
    2) Cluster gate: free_slots / total_slots ≥ f_min.
    3) Job cap: spec_running_count / running_count < β.
    4) Per-task: there is no competing attempt already.
    5) Speed trigger: byte_rate < θ × median(peer byte_rate).
    """
    # 1) Age gate
    if now - getattr(attempt, "start_time", 0.0) < spec_params.min_run_ticks:
        return False

    # 2) Cluster gate
    if cluster_stats.total_slots <= 0:
        return False
    f = cluster_stats.free_slots / float(cluster_stats.total_slots)
    if f < spec_params.cluster_free_min:
        return False

    # 3) Job cap
    running_attempts = [a for a in job_attempts if not getattr(a, "is_speculative", False)]
    spec_attempts = [a for a in job_attempts if getattr(a, "is_speculative", False)]
    running_count = len(running_attempts)
    if running_count <= 0:
        return False
    if len(spec_attempts) / float(running_count) >= spec_params.job_cap_beta:
        return False

    # 4) Per-task competitor
    if has_competing_attempt:
        return False

    # 5) Speed trigger using byte rates
    if not hasattr(attempt, "byte_progress_rate"):
        return False
    attempt_rate = float(attempt.byte_progress_rate(now))

    peer_rates: list[float] = []
    for a in job_attempts:
        if a is attempt or getattr(a, "is_speculative", False):
            continue
        if now > getattr(a, "start_time", 0.0) and hasattr(a, "byte_progress_rate"):
            try:
                peer_rates.append(float(a.byte_progress_rate(now)))
            except Exception:
                pass
    if len(peer_rates) < 1:
        return False
    peer_rates.sort()
    median_rate = peer_rates[len(peer_rates) // 2]
    return attempt_rate < (spec_params.theta * median_rate)