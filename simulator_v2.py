
"""
Event-driven simulator for HDFS block placement + map-task execution.It delegates:

- Map attempt time model to `runtime.py`
- Speculation trigger and node selection to `speculation.py`
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np

from .nodes import DataNode
from .placement import default_hdfs_placement, hypercube_softmax_placement
from .runtime import RuntimeParams, predict_total_time, compute_service_time as rt_compute_service_time, metrics_to_percentiles
from .speculation import (
    SpecParams, should_speculate, should_speculate_bytes, pick_speculative_node, JobStats, ClusterStats
)

# ----------------------------
# Task / attempt data
# ----------------------------

@dataclass
class Task:
    task_id: int
    job_id: int
    tag: str
    # replica holders (data-local nodes)
    nodes: List[DataNode]
    # predicted total time per exec-node we actually used
    service_times: Dict[str, float] = field(default_factory=dict)
    creation_time: float = 0.0
    assigned_node: Optional[DataNode] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    is_speculative: bool = False
    # delay scheduling state: 0=node-local,1=rack-local,2=off-rack
    locality_level: int = 0
    locality_wait: int = 0
    # Progress tracking for realistic speculation (bytes-based)
    total_input_bytes: float = field(default_factory=lambda: BLOCK_MB * 1024 * 1024)  # Default 128MB in bytes
    bytes_processed: float = 0.0

    def progress(self, now: float) -> float:
        if self.assigned_node is None or self.start_time is None:
            return 0.0
        # Update bytes processed based on elapsed time and predicted total time
        tot = self.service_times.get(self.assigned_node.node_id, None)
        if tot is None or tot <= 0:
            return 0.0
        elapsed = now - self.start_time
        time_progress = max(0.0, min(1.0, elapsed / tot))
        # Update bytes processed (assume linear processing rate)
        self.bytes_processed = time_progress * self.total_input_bytes
        return time_progress

    def byte_progress_rate(self, now: float) -> float:
        """Return bytes processed per second estimated from predicted total time.

        We estimate a linear processing rate based on the predicted total time
        for the currently assigned node. This provides a stable, comparable
        metric across attempts without relying on external updates to
        `bytes_processed`.
        """
        if self.assigned_node is None or self.start_time is None:
            return 0.0
        tot = self.service_times.get(self.assigned_node.node_id, None)
        if tot is None or tot <= 0:
            return 0.0
        rate = self.total_input_bytes / max(tot, 1e-9)
        # keep bytes_processed approximately in sync for diagnostics
        elapsed = max(0.0, now - self.start_time)
        self.bytes_processed = min(self.total_input_bytes, rate * elapsed)
        return rate

# ----------------------------
# Scheduling constants
# ----------------------------

JOB_SIZE = 10                # tasks per synthetic job (controls peer stats)
NODE_LOCALITY_DELAY = 40     # ticks to wait before widening to rack
RACK_LOCALITY_DELAY = 80     # ticks to wait before widening to off-rack
BLOCK_MB = 128.0             # used for network cost accounting


def execute_single_strategy_simulation(
    num_blocks: int,
    tag_distribution: Dict[str, float],
    nodes: List[DataNode],
    registry: Dict[str, List[DataNode]],
    centroids: np.ndarray,
    cutpoints: np.ndarray,
    placement_strategy: str = "default",
    speculative: bool = True,
    num_replicas: int = 3,
    progress_every: Optional[int] = None,
    idle_stall_limit: int = 2000,
    collect_details: bool = True,
    runtime_params: Optional[RuntimeParams] = None,
    spec_params: Optional[SpecParams] = None,
) -> Dict:
    """
    Run a single experiment with one of the placement strategies.
    """
    rt = runtime_params or RuntimeParams()
    sp = spec_params or SpecParams()

    # --- Placement pre-pass: produce replica holders for each block ---
    # Choose a writer rack for each block uniformly
    max_rack = max(n.rack_id for n in nodes)
    blocks_replicas: List[List[DataNode]] = []
    placement_failures: List[int] = []

    # reset replica counts
    for n in nodes:
        n.replica_count = 0

    for b in range(num_blocks):
        writer_rack = random.randint(0, max_rack)
        tag = _sample_tag(tag_distribution)
        if placement_strategy == "default":
            repl = default_hdfs_placement(nodes, writer_rack, num_replicas=num_replicas)
        else:
            repl = hypercube_softmax_placement(
                registry=registry,
                nodes=nodes,
                centroids=centroids,
                writer_rack=writer_rack,
                tag=tag,
                cutpoints=cutpoints,
                num_replicas=num_replicas,
            )
        if len(repl) < num_replicas:
            placement_failures.append(b)
        for dn in repl:
            dn.increment_replica_count()
        blocks_replicas.append(repl)

    # --- Build tasks (grouped into jobs for speculation peer stats) ---
    tasks: List[Task] = []
    job_id = 0
    t_id = 0
    # Job Tracking
    job_task_counts: Dict[int, int] = {}
    job_finished_counts: Dict[int, int] = {}
    individual_jcts: List[float] = []

    for b in range(num_blocks):
        current_job_id = b // JOB_SIZE # job_id based on JOB_SIZE
        if current_job_id not in job_task_counts:
             job_task_counts[current_job_id] = 0
             job_finished_counts[current_job_id] = 0
        job_task_counts[current_job_id] += 1

        tag = _sample_tag(tag_distribution)
        tasks.append(Task(task_id=t_id, job_id=current_job_id, tag=tag, nodes=blocks_replicas[b], creation_time=float(b)))
        t_id += 1

    # Queues/state
    pending: List[Task] = tasks[:]  # FIFO
    running: List[Task] = []
    finished_task_ids: set[int] = set()
    duplicates: Dict[int, List[Task]] = {}  # task_id -> [running attempts]

    # Metrics
    network_cost_MB = 0.0
    spec_requests = 0
    spec_launched_local = spec_launched_rack = spec_launched_remote = 0
    spec_denied_no_slot = 0
    spec_won = spec_lost = spec_canceled = 0
    tasks_started_per_node: Dict[str, int] = {n.node_id: 0 for n in nodes}
    completion_times: List[float] = []
    # Diagnostics: reason for remote speculation
    spec_remote_due_to_no_local = 0
    spec_remote_despite_local = 0
    # Diagnostics: launch locality mix and exec-node network percentile summary
    starts_local = starts_rack = starts_remote = 0
    exec_net_p_values: List[float] = []
    # Diagnostics: starts by scheduler locality level (0=node,1=rack,2=off-rack allowed)
    starts_level0 = starts_level1 = starts_level2 = 0

    def _eligible_nodes_for(task: Task, level: int) -> List[DataNode]:
        if level == 0:
            return [n for n in task.nodes if n.has_compute_slot()]
        elif level == 1:
            racks = {n.rack_id for n in task.nodes}
            return [n for n in nodes if n.rack_id in racks and n.has_compute_slot()]
        else:
            return [n for n in nodes if n.has_compute_slot()]

    # Event loop: schedule -> next finish -> account -> maybe speculate -> repeat
    now = 0.0
    # min-heap of (finish_time, seq, Task)
    seq = 0
    event_q: List[Tuple[float, int, Task]] = []

    def _start(task: Task, exec_node: DataNode, is_spec: bool) -> None:
        nonlocal seq, network_cost_MB, spec_launched_local, spec_launched_rack, spec_launched_remote, starts_local, starts_rack, starts_remote, starts_level0, starts_level1, starts_level2
        # compute predicted time with shared model
        T = predict_total_time(exec_node, task.nodes, task.tag, cutpoints, rt)
        task.service_times[exec_node.node_id] = T
        task.assigned_node = exec_node
        task.is_speculative = is_spec
        task.start_time = now
        exec_node.start_task()
        seq += 1
        heapq.heappush(event_q, (now + T, seq, task))
        running.append(task)
        tasks_started_per_node[exec_node.node_id] = tasks_started_per_node.get(exec_node.node_id, 0) + 1
        # Record scheduler locality level at the moment of start
        if task.locality_level == 0:
            starts_level0 += 1
        elif task.locality_level == 1:
            starts_level1 += 1
        else:
            starts_level2 += 1
        # network accounting for non-local start
        if exec_node not in task.nodes:
            # rack-local vs remote
            if is_spec:
                if any(n.rack_id == exec_node.rack_id for n in task.nodes):
                    spec_launched_rack += 1
                else:
                    spec_launched_remote += 1
                # Only count network cost for speculative non-local starts
                network_cost_MB += BLOCK_MB
        else:
            if is_spec:
                spec_launched_local += 1

        # Launch locality classification (for all starts)
        if any(n.node_id == exec_node.node_id for n in task.nodes):
            starts_local += 1
        elif any(n.rack_id == exec_node.rack_id for n in task.nodes):
            starts_rack += 1
        else:
            starts_remote += 1

        # Record exec node network percentile for diagnostics
        try:
            net_p = float(metrics_to_percentiles(exec_node.s_vector, cutpoints)[3])
            exec_net_p_values.append(net_p)
        except Exception:
            pass

        duplicates.setdefault(task.task_id, []).append(task)

    # initial scheduling pass
    _fill_cluster(pending, _eligible_nodes_for, _start, nodes, progress_every, now)

    # Main loop
    while len(completion_times) < num_blocks:
        if not event_q:
            # nothing running—try to schedule again; if still nothing, progressively widen locality
            # to avoid exiting early with pending work that hasn't widened enough yet
            made_progress = False
            for _ in range(3):
                _fill_cluster(pending, _eligible_nodes_for, _start, nodes, progress_every, now)
                if event_q:
                    made_progress = True
                    break
                # Force-widen all pending tasks by one level (up to off-rack)
                for t in pending:
                    if t.locality_level < 2:
                        t.locality_level += 1
                        t.locality_wait = 0
            if not made_progress and not event_q:
                # still nothing runnable; break to avoid infinite loop
                break

        # Pop next completion(s)
        next_t, _, _ = event_q[0]
        now = next_t
        finished_this_tick: List[Task] = []
        while event_q and event_q[0][0] <= now + 1e-12:
            _, _, t = heapq.heappop(event_q)
            if t.task_id in finished_task_ids:
                # duplicate finishing after winner -> cancel if still running
                if t.assigned_node is not None:
                    t.assigned_node.finish_task()
                if t in running:
                    running.remove(t)
                spec_canceled += 1 if t.is_speculative else 0
                continue
            # mark winner
            t.finish_time = now
            # Record attempt duration (finish - start) to represent task runtime
            if t.start_time is not None:
                completion_times.append(t.finish_time - t.start_time)
            else:
                completion_times.append(0.0)
            finished_task_ids.add(t.task_id)
            t.assigned_node.finish_task()
            running.remove(t)
            finished_this_tick.append(t)

            current_job_id = t.job_id
            job_finished_counts[current_job_id] += 1
            if job_finished_counts[current_job_id] == job_task_counts[current_job_id]:
                individual_jcts.append(now)

            # cancel any duplicates still running
            for other in list(duplicates.get(t.task_id, [])):
                if other is t:
                    continue
                if other in running:
                    other.assigned_node.finish_task()
                    running.remove(other)
                    spec_lost += 1 if other.is_speculative else 0
            # winner may be spec
            if t.is_speculative:
                spec_won += 1

        # After some finish events, try to launch speculation for slow originals
        if speculative:
            free_slots = sum(max(0, n.max_compute_slots - n.compute_load) for n in nodes)
            total_slots = sum(n.max_compute_slots for n in nodes)
            cluster_stats = ClusterStats(free_slots=free_slots, total_slots=total_slots)
            # group running by job
            running_by_job: Dict[int, List[Task]] = {}
            for a in running:
                running_by_job.setdefault(a.job_id, []).append(a)
            for job, attempts in running_by_job.items():
                job_stats = JobStats(
                    running_attempts=attempts,
                    running_count=len(attempts),
                    spec_running_count=sum(1 for a in attempts if a.is_speculative),
                )
                for a in attempts:
                    if a.is_speculative:
                        continue
                    has_comp = any(x.is_speculative and x.task_id == a.task_id for x in attempts)
                    # Use speculation trigger driven by byte-rate progress
                    if should_speculate_bytes(a, now, attempts, cluster_stats, has_comp, sp):
                        # Check availability at data-local first for diagnostic classification (for this attempt 'a')
                        local_candidates = [n for n in a.nodes if n.has_compute_slot()]
                        local_available = len(local_candidates) > 0
                        choices = [n for n in nodes if n.has_compute_slot()]
                        if not choices:
                            spec_denied_no_slot += 1
                            continue
                        R = max(0.0, a.service_times[a.assigned_node.node_id] - (now - a.start_time))
                        pick = pick_speculative_node(a, choices, cutpoints, rt, R, sp)
                        if pick is not None:
                            spec_requests += 1
                            chosen, T = pick
                            # create a speculative attempt (shares task_id)
                            spec_attempt = Task(
                                task_id=a.task_id, job_id=a.job_id, tag=a.tag,
                                nodes=a.nodes, creation_time=a.creation_time, is_speculative=True
                            )
                            _start(spec_attempt, chosen, True)
                            # Classify remote reason if applicable
                            if chosen not in a.nodes:
                                if local_available:
                                    spec_remote_despite_local += 1
                                else:
                                    spec_remote_due_to_no_local += 1

        # Fill cluster with new regular launches after speculation
        _fill_cluster(pending, _eligible_nodes_for, _start, nodes, progress_every, now)

        # idle safety guard
        if len(running) == 0 and len(completion_times) < num_blocks and not pending:
            # this should not happen. break to avoid infinite loop
            break

    # Build result
    details = dict(
        completion_times=np.array(completion_times, dtype=float),
        network_cost_MB=float(network_cost_MB),
        spec=dict(
            requests=spec_requests,
            launched_local=spec_launched_local,
            launched_rack=spec_launched_rack,
            launched_remote=spec_launched_remote,
            denied_no_slot=spec_denied_no_slot,
            won=spec_won, lost=spec_lost, canceled=spec_canceled,
            remote_due_to_no_local=int(spec_remote_due_to_no_local),
            remote_despite_local=int(spec_remote_despite_local),
        ),
        tasks_per_node={k: int(v) for k, v in tasks_started_per_node.items()},
        replica_counts_per_node={n.node_id: int(n.replica_count) for n in nodes},
        placement_failures=placement_failures,
        launch_locality=dict(
            starts_local=int(starts_local),
            starts_rack=int(starts_rack),
            starts_remote=int(starts_remote),
        ),
        exec_net_p_summary=dict(
            mean=float(np.mean(exec_net_p_values)) if exec_net_p_values else 0.0,
            p50=float(np.percentile(exec_net_p_values, 50)) if exec_net_p_values else 0.0,
            p10=float(np.percentile(exec_net_p_values, 10)) if exec_net_p_values else 0.0,
            p90=float(np.percentile(exec_net_p_values, 90)) if exec_net_p_values else 0.0,
        ),
        starts_by_level=dict(level0=int(starts_level0), level1=int(starts_level1), level2=int(starts_level2)),
        individual_job_completion_times=individual_jcts
    ) if collect_details else {}

    return details

def _sample_tag(dist: Dict[str, float]) -> str:
    if not dist: return "balanced"
    keys = list(dist.keys())
    probs = np.array([max(0.0, dist[k]) for k in keys], dtype=float)
    probs = probs / probs.sum()
    return random.choices(keys, weights=probs, k=1)[0]

def _fill_cluster(pending: List[Task], elig_fn, start_fn, nodes: List[DataNode], progress_every: Optional[int], now: float) -> None:
    """Try to start as many regular tasks as possible respecting delay scheduling."""
    started = 0
    i = 0
    while i < len(pending):
        t = pending[i]
        choices = elig_fn(t, t.locality_level)
        if choices:
            exec_node = random.choice(choices)  # tie-break: same for all strategies
            start_fn(t, exec_node, False)
            # remove from pending
            pending.pop(i)
            started += 1
        else:
            # increase wait and maybe widen locality
            t.locality_wait += 1
            if t.locality_level == 0 and t.locality_wait >= NODE_LOCALITY_DELAY:
                t.locality_level = 1; t.locality_wait = 0
            elif t.locality_level == 1 and t.locality_wait >= RACK_LOCALITY_DELAY:
                t.locality_level = 2; t.locality_wait = 0
            i += 1

    if progress_every is not None and started and (int(now) % progress_every == 0):
        print(f"      progress: started={started} pending={len(pending)} running_slots={sum(n.compute_load for n in nodes)}", flush=True)
