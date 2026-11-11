"""Benchmark Finch on G-CARE subgraph-matching (counting) queries.

Each query graph is lowered to one einsum over the data graph's label
matrices/vectors (see :mod:`finchtensorbench.gcare`) and evaluated with
``finch.compute`` under one or more scheduler configurations
(:mod:`finchtensorbench.benchmarks.autoschedule`).  Every result is checked
against the G-CARE ground-truth count.

With ``--plan-only``, each query is only planned: once cold and then
``--runs`` times with fresh schedulers, stopping where the finished plan would
be compiled, so no kernel is compiled or run and no count is checked.

Each configuration runs in its own worker process.  Per query the worker:

1. ``first``: runs the query once with a fresh scheduler (cold: includes any
   Julia compilation of new kernels).
2. ``plan``: runs it again with another fresh scheduler, so stats, planning
   and lowering are redone with a warm kernel cache; stage times are recorded.
3. ``exec``: runs it ``--runs`` more times on that scheduler, which now hits
   its plan cache, so only execution is timed.

Step 1 may take ``--first-timeout`` seconds and steps 2-3 together
``--timeout`` seconds, for every query alike (no allowance depends on where a
query falls in a worker's queue).  A query exceeding either kills the worker; it
is recorded as a timeout and a new worker continues with the remaining queries.

Each query's plan, as step 1 produces it, is saved under ``<csv stem>_plans``
unless ``--no-plans`` is given.  Workers run with a fixed
``PYTHONHASHSEED`` (``--hash-seed``), and every row records the installed
finch-tensor revision.

Run with::

    pixi run python -m finchtensorbench.benchmarks.subgraph.subgraph_matching \
        --dataset human --configs default galley
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sps

from finchtensorbench import gcare
from finchtensorbench.benchmarks.autoschedule import (
    STAGES,
    config_label,
    parse_config,
)
from finchtensorbench.benchmarks.subgraph.results import print_summary

RESULTS_DIR = Path(__file__).resolve().parents[4] / "graph" / "subgraph_matching"
RUNNER_MODULE = "finchtensorbench.benchmarks.subgraph.subgraph_matching"

FIELDS = (
    "dataset",
    "config",
    "query",
    "family",
    "status",
    "ground_truth",
    "result",
    "correct",
    "first_s",
    "plan_total_s",
    *(f"{stage}_s" for stage in STAGES),
    "exec_median_s",
    "exec_min_s",
    "exec_mean_s",
    "exec_samples",
    "logic_order",
    "loop_order",
    "error",
    "plan_file",
    "cpu",
    "worker_query",
    "hash_seed",
    "finch_revision",
)


# ---------------------------------------------------------------------------
# Finch inputs
# ---------------------------------------------------------------------------


class GraphTensors:
    """Lazily converted, shared Finch tensors for one data graph."""

    def __init__(self, graph: gcare.GCareGraph):
        self.graph = graph
        self._tensors: dict[str, object] = {}

    def __getitem__(self, name: str):
        """Graph tensor *name* as 0/1 ``SparseList(Element)`` (a vector) or CSR
        ``Dense(SparseList(Element))`` (a matrix), converted on first use."""
        import finch as fl

        if name not in self._tensors:
            ndim, coords = gcare.query_tensor_coo(self.graph, name)
            n = self.graph.num_vertices
            # A vector is built as the single row of a 1 x n CSR matrix.
            cols = coords[-1]
            rows = coords[0] if ndim == 2 else np.zeros_like(cols)
            csr = sps.csr_matrix(
                (np.ones(len(cols)), (rows, cols)), shape=(n if ndim == 2 else 1, n)
            )
            csr.sum_duplicates()
            csr.data[:] = 1.0
            csr.sort_indices()
            f64 = np.float64
            level = fl.SparseListLevel(
                fl.ElementLevel(
                    fl.element(f64(0), fl.ftype(f64), fl.ftype(np.intp), fl.NumpyBufferFType),
                    fl.NumpyBuffer(csr.data.astype(f64)),
                ),
                np.intp(n),
                fl.NumpyBuffer(csr.indptr.astype(np.intp)),
                fl.NumpyBuffer(csr.indices.astype(np.intp)),
            )
            self._tensors[name] = fl.FiberTensor(
                level if ndim == 1 else fl.DenseLevel(level, np.intp(n))
            )
        return self._tensors[name]


def query_program(query: gcare.GCareQuery, tensors: GraphTensors):
    """The lazy Finch program counting *query*'s matches."""
    import finch as fl

    operands = [fl.defer(tensors[op.tensor]) for op in query.operands]
    return fl.einsum(query.subscripts(), *operands)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _run_once(query, tensors, context):
    """Compute *query*; return its count (None for a plan-only scheduler, which
    stops before compiling) and the seconds it took."""
    import finch as fl

    from finchtensorbench.benchmarks.autoschedule import PlanOnly

    context.timer.reset()
    started = time.perf_counter()
    try:
        result = fl.compute(query_program(query, tensors), ctx=context)
    except PlanOnly:
        return None, time.perf_counter() - started
    elapsed = time.perf_counter() - started
    return float(np.asarray(fl.to_numpy(result)).ravel()[0]), elapsed


def plan_path(plan_dir: Path, config_name: str, query_name: str) -> Path:
    """Where the plan of *query_name* under *config_name* is saved."""
    return plan_dir / config_name.replace("/", "+") / f"{query_name}.txt"


def _finch_revision() -> str:
    """The installed finch-tensor's version and, for a git install, the
    requested branch and commit."""
    from importlib import metadata

    try:
        dist = metadata.distribution("finch-tensor")
        info = json.loads(dist.read_text("direct_url.json") or "{}")
        vcs = info.get("vcs_info", {})
        ref = vcs.get("requested_revision", "")
        return f"{dist.version} {ref} @ {vcs.get('commit_id', '?')[:10]}".strip()
    except Exception:  # pragma: no cover - informational only
        return "unknown"


class PlanRecorder:
    """Write a query's plan to a text file as the scheduler produces it.

    Each stage is flushed as soon as it is seen, so the plan survives a kernel
    that times out, runs out of memory or crashes the worker.  It is given to
    the first, cold compute only.
    """

    def __init__(self, path: Path, query, config, tensors: "GraphTensors"):
        from finchtensorbench.benchmarks.autoschedule import config_label

        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        graph = tensors.graph
        lines = [
            f"# {query.dataset} {query.name}",
            f"# config: {config_label(config)} ({config.name})",
            f"# finch-tensor {_finch_revision()}",
            "",
            f"query:        {query.saps_expr()}",
            f"einsum:       {query.subscripts()}",
            f"ground truth: {query.ground_truth}",
            f"vertices:     {graph.num_vertices}",
            "operands:",
        ]
        for name in query.tensor_names:
            ndim, coords = gcare.query_tensor_coo(graph, name)
            lines.append(f"  {name:6} {ndim}-d  nnz {len(coords[0])}")
        self._write("\n".join(lines) + "\n", mode="w")

    def _write(self, text: str, mode: str = "a"):
        with self.path.open(mode, encoding="utf-8") as file:
            file.write(text)

    def on_program(self, stage, program, bindings):
        """Write the program entering *stage* (and, at the compiler, its formats)."""
        from finchtensorbench.benchmarks.autoschedule import PLAN_STAGES

        text = f"\n==== {PLAN_STAGES.get(stage, stage)} ====\n{program}\n"
        if stage == "compiler":
            text += "\nformats:\n" + "".join(
                f"  {str(alias):16} {ftype}\n" for alias, ftype in bindings.items()
            )
        self._write(text)


def _cold_run(query, tensors, config, recorder, on_cold_done, plan_only=False):
    """The first compute of *query*, with a fresh scheduler whose plan
    *recorder* writes; returns its count and seconds."""
    from finchtensorbench.benchmarks.autoschedule import build_scheduler

    on_program = recorder.on_program if recorder else None
    result = _run_once(
        query, tensors,
        build_scheduler(config, on_program=on_program, plan_only=plan_only),
    )  # fmt: skip
    on_cold_done()
    return result


def measure_plan_only(query, tensors, config, runs, recorder, on_cold_done):
    """Plan *query* without compiling or running it: once cold, then *runs*
    times (at least once) with fresh schedulers, reporting the medians."""
    from finchtensorbench.benchmarks.autoschedule import (
        build_scheduler,
        describe_orders,
    )

    _, first_s = _cold_run(query, tensors, config, recorder, on_cold_done, True)
    totals, stages = [], {stage: [] for stage in STAGES}
    for _ in range(max(runs, 1)):
        context = build_scheduler(config, plan_only=True)
        _, elapsed = _run_once(query, tensors, context)
        totals.append(elapsed)
        for stage in STAGES:
            stages[stage].append(context.timer.times[stage])
    return {
        "first_s": first_s,
        "plan_total_s": statistics.median(totals),
        **{f"{stage}_s": statistics.median(stages[stage]) for stage in STAGES},
        "logic_order": describe_orders(context.captures.get("logic"), loop_order=False),
        "loop_order": describe_orders(context.captures.get("loop"), loop_order=True),
    }


def measure_query(query, tensors, config, runs, recorder, on_cold_done):
    """Steps 1-3 of the module docstring: cold, fresh plan, cached runs."""
    from finchtensorbench.benchmarks.autoschedule import (
        build_scheduler,
        describe_orders,
    )

    _, first_s = _cold_run(query, tensors, config, recorder, on_cold_done)
    context = build_scheduler(config)
    result, plan_total_s = _run_once(query, tensors, context)
    stage_times = dict(context.timer.times)
    logic_order = describe_orders(context.captures.get("logic"), loop_order=False)
    loop_order = describe_orders(context.captures.get("loop"), loop_order=True)

    samples = []
    for _ in range(runs):
        value, elapsed = _run_once(query, tensors, context)
        if value != result:
            raise RuntimeError(f"result changed between runs: {result} -> {value}")
        samples.append(elapsed)

    return {
        "result": result,
        # Compared as float64 (finch's counts are float64): counts above 2**53
        # are not exact, so the ground truth is rounded the same way.
        "correct": query.ground_truth is not None
        and result == float(query.ground_truth),
        "first_s": first_s,
        "plan_total_s": plan_total_s,
        **{f"{stage}_s": stage_times[stage] for stage in STAGES},
        "exec_median_s": statistics.median(samples),
        "exec_min_s": min(samples),
        "exec_mean_s": statistics.fmean(samples),
        "exec_samples": ";".join(f"{s:.6g}" for s in samples),
        "logic_order": logic_order,
        "loop_order": loop_order,
    }


def _emit(file, record):
    file.write(json.dumps(record) + "\n")
    file.flush()


def _stdin_lines():
    """Lines of stdin until EOF, read with blocking reads.

    Julia's runtime switches fd 0 to non-blocking mode, after which Python's
    buffered ``readline`` returns "" (looks like EOF) whenever no line is
    waiting, so read the descriptor directly and force it back to blocking.
    """
    fd = sys.stdin.fileno()
    buffer = b""
    while True:
        while b"\n" not in buffer:
            os.set_blocking(fd, True)
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:  # flipped back to non-blocking meanwhile
                time.sleep(0.05)
                continue
            if not chunk:
                if buffer.strip():
                    yield buffer.decode().strip()
                return
            buffer += chunk
        line, buffer = buffer.split(b"\n", 1)
        yield line.decode().strip()


def run_worker(args):
    from finchtensorbench.benchmarks.autoschedule import build_scheduler

    revision = _finch_revision()
    config = parse_config(args.config)
    graph = gcare.load_graph(args.dataset, args.data_dir)
    tensors = GraphTensors(graph)
    # Serve query names from stdin, one per line, until it closes.
    by_name = {q.name: q for q in gcare.load_queries(graph, None, args.data_dir)}
    queries = (by_name[name] for name in _stdin_lines() if name)
    # Run (or plan) a tiny labelled path query first, so Julia start-up and
    # Finch's generic compilation are not charged to a timed query.
    warmup = gcare.GCareGraph(
        name="warmup", max_vid=3, continuous_ids=True,
        vectors={"V0": np.array([0, 2])},
        matrices={"E0": (np.array([0, 1, 2]), np.array([1, 2, 3]))},
    )  # fmt: skip
    path_query = gcare.GCareQuery(
        dataset="warmup", name="warmup/path", vertices=(0, 1, 2),
        operands=(gcare.Operand("V0", (0,)), gcare.Operand("E0", (0, 1)),
                  gcare.Operand("E0", (1, 2))),
    )  # fmt: skip
    _run_once(path_query, GraphTensors(warmup),
              build_scheduler(config, plan_only=args.plan_only))  # fmt: skip
    measure = measure_plan_only if args.plan_only else measure_query
    with args.worker_output.open("a", encoding="utf-8") as out:
        for query in queries:
            _emit(out, {"event": "start", "query": query.name, "time": time.time()})
            recorder = None
            if args.plan_dir:
                path = plan_path(args.plan_dir, args.config, query.name)
                recorder = PlanRecorder(path, query, config, tensors)
            try:
                record = measure(
                    query, tensors, config, args.runs, recorder,
                    on_cold_done=lambda name=query.name: _emit(
                        out, {"event": "cold", "query": name, "time": time.time()}
                    ),
                )  # fmt: skip
                record["status"] = (
                    "planned" if args.plan_only
                    else "ok" if record["correct"]
                    else "wrong" if query.ground_truth is not None
                    else "unchecked"
                )
            except Exception as error:  # record and keep going
                message = f"{type(error).__name__}: {error}"
                oom = isinstance(error, MemoryError) or "OutOfMemoryError" in message
                record = {"status": "oom" if oom else "error", "error": message[:2000]}
            if recorder:
                keys = ("status", "result", "first_s", "plan_total_s",
                        *(f"{s}_s" for s in STAGES), "exec_median_s",
                        "exec_samples", "error")  # fmt: skip
                recorder._write("\n==== measured ====\n" + "".join(
                    f"  {k:14} {record[k]}\n" for k in keys if k in record
                ))  # fmt: skip
            record["finch_revision"] = revision
            record["hash_seed"] = os.environ.get("PYTHONHASHSEED", "")
            _emit(out, {"event": "done", "query": query.name, "record": record})


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------


def _cgroup_dir() -> Path | None:
    """This process's cgroup v2 directory, if it has one."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    except OSError:
        pass
    return None


def _cgroup_oom_kills() -> int:
    folder = _cgroup_dir()
    try:
        for line in (folder / "memory.events").read_text().splitlines():
            key, value = line.split()
            if key == "oom_kill":
                return int(value)
    except (OSError, TypeError, ValueError):
        pass
    return 0


class WorkerSlot:
    """One long-lived worker process pinned to one CPU, fed queries on stdin.

    ``run(query)`` returns the worker's record, or a timeout/oom/crash record
    if the worker had to be killed or died; the next query then starts a
    fresh worker.
    """

    def __init__(self, args, config_name, cpu, folder: Path):
        self.args, self.config_name, self.cpu = args, config_name, cpu
        self.folder = folder
        self.process = None
        self.starts = 0

    def _start(self):
        self.starts += 1
        tag = f"cpu{self.cpu}-{self.starts}"
        self.output = self.folder / f"{tag}.jsonl"
        self.output.touch()
        self.log = self.folder / f"{tag}.log"
        self.offset = 0
        args = self.args
        command = [
            sys.executable, "-m", RUNNER_MODULE,
            "--worker-output", str(self.output),
            "--config", self.config_name,
            "--dataset", args.dataset,
            "--runs", str(args.runs),
            *(["--plan-only"] if args.plan_only else []),
        ]  # fmt: skip
        if args.data_dir:
            command += ["--data-dir", str(args.data_dir)]
        if args.plan_dir:
            command += ["--plan-dir", str(args.plan_dir)]
        env = os.environ.copy()
        src = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)
        for var in ("JULIA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):  # fmt: skip
            env[var] = "1"
        # Set iteration order (and so plan tie-breaking) is fixed across workers.
        env["PYTHONHASHSEED"] = str(args.hash_seed)
        cpu, limit_gb = self.cpu, args.memory_limit_gb

        def setup():
            # Pin the worker to its CPU and cap its address space; allocations
            # past the cap raise (Julia OutOfMemoryError, numpy MemoryError),
            # which the worker records as status oom.
            os.sched_setaffinity(0, {cpu})
            if limit_gb:
                limit = int(limit_gb * 1024**3)
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        self.log_file = self.log.open("w")
        self.process = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.PIPE,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=setup,
        )
        self.served = 0

    def _stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.log_file.close()
        self.process = None

    def run(self, query) -> dict:
        """Serve *query*; the record says which query of its worker it was
        (``worker_query``, 0 = the worker's first, cold, query)."""
        recycle = self.args.max_queries_per_worker
        if self.process is not None and recycle and self.served >= recycle:
            # Long-lived workers accumulate compiled kernels and caches until
            # they run out of memory, so start afresh now and then.
            self.close()
        if self.process is None or self.process.poll() is not None:
            self._stop()
            self._start()
        index = self.served
        record = self._serve(query)
        record["worker_query"] = index
        return record

    def _serve(self, query) -> dict:
        # Two phases, limited alike for every query: the cold first compute
        # (--first-timeout, which absorbs compilation wherever the query falls
        # in a worker's queue) and the timed fresh plan plus cached runs
        # (--timeout).
        self.process.stdin.write(query.name + "\n")
        self.process.stdin.flush()
        started = None
        phase, limit = "cold", self.args.first_timeout
        while True:
            # The worker's new complete event lines.
            with self.output.open(encoding="utf-8") as file:
                file.seek(self.offset)
                chunk = file.read()
            complete = chunk[: chunk.rfind("\n") + 1]
            self.offset += len(complete.encode("utf-8"))
            for event in map(json.loads, complete.splitlines()):
                if event["query"] != query.name:
                    continue
                if event["event"] == "start":
                    started = time.monotonic()
                elif event["event"] == "cold":
                    started = time.monotonic()
                    phase, limit = "timed", self.args.timeout
                else:
                    self.served += 1
                    return event["record"]
            if (self.process.poll() is not None
                    and self.output.stat().st_size <= self.offset):  # fmt: skip
                # The worker died with no record left to read.
                code = self.process.returncode
                self._stop()
                text = self.log.read_text()
                if started is None and self.served == 0:
                    # Died before serving anything (e.g. an import error): give up.
                    raise RuntimeError(
                        f"worker for {self.config_name} on cpu {self.cpu} failed:\n"
                        f"{text[-4000:]}"
                    )
                tail = text[-500:].replace("\n", " | ")
                kills = _cgroup_oom_kills()
                if kills > self.args.cgroup_oom_kills:
                    self.args.cgroup_oom_kills = kills
                    tail = f"[cgroup OOM kill: the whole run's memory cap was hit] {tail}"
                # We did not kill it, so SIGKILL means the kernel OOM killer.
                oom = code == -9 or any(
                    s in text for s in ("OutOfMemory", "MemoryError", "out of memory")
                )
                return {"status": "oom" if oom else "crash",
                        "error": f"exit {code}: {tail}"}  # fmt: skip
            if started and time.monotonic() - started > limit:
                self._stop()
                return {"status": "timeout", "error": f"{phase} phase exceeded {limit:g}s"}
            time.sleep(0.2)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        self._stop()


def run_config(args, config_name, queries, writer, csv_file):
    """Run every query under one config on ``args.jobs`` pinned workers."""
    pending: queue.Queue = queue.Queue()
    for query in queries:
        pending.put(query)
    rows: list[dict] = []
    lock = threading.Lock()

    def drain(cpu, folder):
        slot = WorkerSlot(args, config_name, cpu, folder)
        try:
            while True:
                try:
                    query = pending.get_nowait()
                except queue.Empty:
                    return
                record = slot.run(query)
                row = {"dataset": args.dataset, "config": config_name,
                       "query": query.name, "family": query.family,
                       "ground_truth": query.ground_truth, **record, "cpu": cpu}  # fmt: skip
                status = record["status"]
                if args.plan_dir:
                    path = plan_path(args.plan_dir, config_name, query.name)
                    row["plan_file"] = str(path.relative_to(args.plan_dir.parent))
                    # A killed or crashed worker could not note it in the plan.
                    killed = status in ("timeout", "crash") or (
                        status == "oom" and "exit" in record.get("error", "")
                    )
                    if killed and path.exists():
                        with path.open("a", encoding="utf-8") as file:
                            file.write(f"\n==== measured ====\n  status         {status}\n"
                                       f"  error          {record['error']}\n")  # fmt: skip
                if status in ("ok", "wrong", "unchecked"):
                    detail = (
                        f"exec {row['exec_median_s']:.4g}s  plan {row['plan_total_s']:.4g}s"
                        f"  first {row['first_s']:.4g}s  count {row['result']:.0f}"
                        f" (gt {row['ground_truth']})"
                    )
                elif status == "planned":
                    detail = (f"plan {row['plan_total_s']:.4g}s  first {row['first_s']:.4g}s"
                              f"  loops {row['loop_order']}")  # fmt: skip
                else:
                    detail = row.get("error", "")
                with lock:
                    rows.append(row)
                    writer.writerow(row)
                    csv_file.flush()
                    print(f"  {status:7} {query.name:22} {detail}", flush=True)
        finally:
            slot.close()

    cpus = args.cpus[: args.jobs]
    with tempfile.TemporaryDirectory(prefix="subgraph-workers-") as folder:
        with ThreadPoolExecutor(len(cpus)) as pool:
            futures = [pool.submit(drain, cpu, Path(folder)) for cpu in cpus]
            for future in futures:
                future.result()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="human")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["default", "galley"],
        help="presets (default, galley) or logic/loop/formatter/stats",
    )
    parser.add_argument("--queries", nargs="+", help="query names (default: all)")
    parser.add_argument("--families", nargs="+", help="e.g. Star_3 Graph_3")
    parser.add_argument(
        "--sample", type=int, help="run a random sample of this many of the queries"
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5, help="timed cached executions")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="only plan each query (stats through formats and protocols), "
        "--runs times with fresh schedulers; nothing is compiled or run, and "
        "rows get status 'planned'",
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per query")
    parser.add_argument(
        "--max-queries-per-worker",
        type=int,
        default=100,
        help="restart each worker after this many queries (0 never)",
    )
    parser.add_argument(
        "--first-timeout",
        type=float,
        help="seconds for each query's cold first compute, which compiles its "
        "kernels (default: --timeout); --timeout then limits the timed fresh "
        "plan and cached runs",
    )
    parser.add_argument(
        "--hash-seed", type=int, default=0, help="PYTHONHASHSEED for every worker"
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=16.0,
        help="address-space limit per worker process (0 disables)",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.75,
        help="cap all workers together at this fraction of RAM, lowering the "
        "per-worker limit if needed",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--no-plans", action="store_true", help="don't save each query's plan"
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="worker processes per config, one per CPU"
    )
    parser.add_argument(
        "--cpus",
        type=int,
        nargs="+",
        help="CPUs to pin workers to (default: one per physical core, fastest first)",
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--plan-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_output:
        return run_worker(args)
    if args.first_timeout is None:
        args.first_timeout = args.timeout
    args.cgroup_oom_kills = _cgroup_oom_kills()
    if args.cpus is None:
        # One logical CPU per physical core (so two workers never share a
        # core's hardware threads), or every allowed CPU if the topology is
        # unreadable.
        allowed = sorted(os.sched_getaffinity(0))
        cores: dict[tuple[str, str], int] = {}
        try:
            for cpu in allowed:
                base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
                key = ((base / "physical_package_id").read_text().strip(),
                       (base / "core_id").read_text().strip())  # fmt: skip
                cores.setdefault(key, cpu)
            args.cpus = sorted(cores.values())
        except OSError:
            args.cpus = allowed
    if args.jobs < 1 or args.jobs > len(args.cpus):
        parser.error(f"--jobs must be between 1 and the {len(args.cpus)} CPUs given")
    # Keep all workers together within a share of RAM, so that a query which
    # runs out of memory fails alone instead of the kernel's OOM killer
    # taking down other processes.
    total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    budget_gb = args.memory_fraction * total_gb
    # The memory cap of the cgroup this run is in (e.g. systemd-run -p MemoryMax).
    try:
        cap = (_cgroup_dir() / "memory.max").read_text().strip()
    except (OSError, TypeError):
        cap = "max"
    cap_gb = None if cap == "max" else int(cap) / 1024**3
    if cap_gb is not None and cap_gb < budget_gb:
        # The run's own cap (systemd-run MemoryMax) is tighter: keep the
        # workers' limits within it, so that a query that needs too much fails
        # alone (status oom) instead of the cgroup OOM killer picking a victim.
        budget_gb = 0.95 * cap_gb
    if args.memory_limit_gb and args.jobs * args.memory_limit_gb > budget_gb:
        limit = math.floor(10 * budget_gb / args.jobs) / 10
        print(
            f"note: {args.jobs} workers x {args.memory_limit_gb:g} GB would exceed "
            f"the {budget_gb:.0f} GB budget (min of {args.memory_fraction:.0%} of "
            f"{total_gb:.0f} GB RAM and this run's memory cap); limiting each "
            f"worker to {limit:g} GB"
        )
        args.memory_limit_gb = limit

    config_names = [config_label(parse_config(c)) for c in args.configs]
    graph = gcare.load_graph(args.dataset, args.data_dir)
    queries = gcare.load_queries(graph, args.queries, args.data_dir)
    if args.families:
        queries = [q for q in queries if q.family in args.families]
    if args.sample is not None and args.sample < len(queries):
        # Same sample for every config; reproducible from --sample-seed.
        picked = random.Random(args.sample_seed).sample(range(len(queries)), args.sample)
        queries = [queries[i] for i in sorted(picked)]
    print(
        f"{args.dataset}: {graph.num_vertices} vertices, "
        f"{sum(len(r) for r, _ in graph.matrices.values())} edges, "
        f"{len(queries)} queries"
        + (f" (sample, seed {args.sample_seed})" if args.sample is not None else "")
        + f", configs {config_names}, "
        f"{args.jobs} worker(s) on cpus {args.cpus[: args.jobs]}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.output_dir / f"{args.dataset}_{stamp}.csv"
    args.plan_dir = (
        None if args.no_plans else args.output_dir / f"{args.dataset}_{stamp}_plans"
    )
    print(f"finch-tensor {_finch_revision()}")
    rows = []
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for name in config_names:
            print(f"\n[{name}]", flush=True)
            rows += run_config(args, name, queries, writer, csv_file)
    print_summary(rows)
    print(f"\nwrote {path}")
    if args.plan_dir:
        print(f"plans in {args.plan_dir}")
    return path


if __name__ == "__main__":
    main()
