"""Benchmark Finch on G-CARE subgraph-matching (counting) queries.

Each query graph is lowered to one einsum over the data graph's label
matrices/vectors (see :mod:`finchtensorbench.gcare`) and evaluated with
``finch.compute`` under one or more scheduler configurations
(:mod:`finchtensorbench.benchmarks.autoschedule`).  Every result is checked
against the G-CARE ground-truth count.

Each configuration runs in its own worker process.  Per query the worker:

1. ``first``: runs the query once with a fresh scheduler (cold: includes any
   Julia compilation of new kernels).
2. ``plan``: runs it again with another fresh scheduler, so stats, planning
   and lowering are redone with a warm kernel cache; stage times are recorded.
3. ``exec``: runs it ``--runs`` more times on that scheduler, which now hits
   its plan cache, so only execution is timed.

A query exceeding ``--timeout`` seconds kills the worker; it is recorded as a
timeout and a new worker continues with the remaining queries.

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
)


# ---------------------------------------------------------------------------
# Finch inputs
# ---------------------------------------------------------------------------


def _element_level(values):
    import finch as fl

    dtype = np.float64
    return fl.ElementLevel(
        fl.element(dtype(0), fl.ftype(dtype), fl.ftype(np.intp), fl.NumpyBufferFType),
        fl.NumpyBuffer(np.asarray(values, dtype=dtype)),
    )


def sparse_vector(indices: np.ndarray, n: int):
    """0/1 vector of length *n* as ``SparseList(Element)``."""
    import finch as fl

    idx = np.unique(indices).astype(np.intp)
    return fl.FiberTensor(
        fl.SparseListLevel(
            _element_level(np.ones(len(idx))),
            np.intp(n),
            fl.NumpyBuffer(np.asarray([0, len(idx)], dtype=np.intp)),
            fl.NumpyBuffer(idx),
        )
    )


def sparse_matrix(rows: np.ndarray, cols: np.ndarray, n: int):
    """0/1 ``n x n`` matrix in CSR, i.e. ``Dense(SparseList(Element))``."""
    import finch as fl

    csr = sps.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    csr.sum_duplicates()
    csr.data[:] = 1.0
    csr.sort_indices()
    return fl.FiberTensor(
        fl.DenseLevel(
            fl.SparseListLevel(
                _element_level(csr.data),
                np.intp(n),
                fl.NumpyBuffer(csr.indptr.astype(np.intp)),
                fl.NumpyBuffer(csr.indices.astype(np.intp)),
            ),
            np.intp(n),
        )
    )


class GraphTensors:
    """Lazily converted, shared Finch tensors for one data graph."""

    def __init__(self, graph: gcare.GCareGraph):
        self.graph = graph
        self._tensors: dict[str, object] = {}

    def __getitem__(self, name: str):
        if name not in self._tensors:
            ndim, coords = gcare.query_tensor_coo(self.graph, name)
            n = self.graph.num_vertices
            self._tensors[name] = (
                sparse_vector(coords[0], n) if ndim == 1 else sparse_matrix(*coords, n)
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
    import finch as fl

    context.timer.reset()
    started = time.perf_counter()
    result = fl.compute(query_program(query, tensors), ctx=context)
    elapsed = time.perf_counter() - started
    return float(np.asarray(fl.to_numpy(result)).ravel()[0]), elapsed


def plan_path(plan_dir: Path, config_name: str, query_name: str) -> Path:
    """Where the plan of *query_name* under *config_name* is saved."""
    return plan_dir / config_name.replace("/", "+") / f"{query_name}.txt"


def _finch_revision() -> str:
    """The finch actually imported: a git checkout's branch and commit (with
    ``+dirty`` for uncommitted changes) when it is run from one, e.g. through
    PYTHONPATH, else the installed distribution's recorded revision."""
    from importlib import metadata

    import finch

    source = Path(finch.__file__ or ".").resolve().parent
    try:
        def git(where, *cmd):
            return subprocess.run(
                ["git", "-C", str(where), *cmd], capture_output=True, text=True,
                check=True,
            ).stdout.strip()  # fmt: skip

        top = Path(git(source, "rev-parse", "--show-toplevel"))
        if ".pixi" not in source.parts and source.is_relative_to(top):
            dirty = "+dirty" if git(top, "status", "--porcelain", "--", "src") else ""
            return (
                f"{git(top, 'rev-parse', '--abbrev-ref', 'HEAD')} @ "
                f"{git(top, 'rev-parse', '--short=10', 'HEAD')}{dirty} ({top})"
            )
    except (subprocess.CalledProcessError, OSError, ValueError):
        pass
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
    that times out, runs out of memory or crashes the worker.  Only the first
    compute is recorded (later fresh plans of the same query are identical
    up to names).
    """

    def __init__(self, path: Path, query, config, tensors: "GraphTensors"):
        from finchtensorbench.benchmarks.autoschedule import config_label

        self.path = path
        self.recording = True
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
        from finchtensorbench.benchmarks.autoschedule import PLAN_STAGES

        if not self.recording:
            return
        text = f"\n==== {PLAN_STAGES.get(stage, stage)} ====\n{program}\n"
        if stage == "compiler":
            text += "\nformats:\n" + "".join(
                f"  {str(alias):16} {ftype}\n" for alias, ftype in bindings.items()
            )
        self._write(text)

    def stop(self):
        self.recording = False

    def append_record(self, record: dict):
        keys = ("status", "result", "first_s", "plan_total_s",
                *(f"{s}_s" for s in STAGES), "exec_median_s", "exec_samples",
                "error")  # fmt: skip
        body = "".join(f"  {k:14} {record[k]}\n" for k in keys if k in record)
        self._write(f"\n==== measured ====\n{body}")


def append_plan_status(path: Path, status: str, error: str):
    """Record a failure the worker could not write itself (timeout, crash)."""
    if path.exists():
        with path.open("a", encoding="utf-8") as file:
            file.write(f"\n==== measured ====\n  status         {status}\n"
                       f"  error          {error}\n")  # fmt: skip


def measure_query(query, tensors, config, runs, recorder: PlanRecorder | None = None):
    from finchtensorbench.benchmarks.autoschedule import (
        build_scheduler,
        describe_orders,
    )

    on_program = recorder.on_program if recorder else None
    result, first_s = _run_once(
        query, tensors, build_scheduler(config, on_program=on_program)
    )
    if recorder:
        recorder.stop()

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
        "correct": query.ground_truth is not None and result == query.ground_truth,
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


def warm_up(config):
    """Run a tiny labelled path query so Julia start-up and Finch's generic
    compilation happen before any timed (and time-limited) query."""
    from finchtensorbench.benchmarks.autoschedule import build_scheduler

    graph = gcare.GCareGraph(
        name="warmup",
        max_vid=3,
        continuous_ids=True,
        vectors={"V0": np.array([0, 2])},
        matrices={"E0": (np.array([0, 1, 2]), np.array([1, 2, 3]))},
    )
    query = gcare.GCareQuery(
        dataset="warmup",
        name="warmup/path",
        vertices=(0, 1, 2),
        operands=(
            gcare.Operand("V0", (0,)),
            gcare.Operand("E0", (0, 1)),
            gcare.Operand("E0", (1, 2)),
        ),
    )
    _run_once(query, GraphTensors(graph), build_scheduler(config))


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
    config = parse_config(args.config)
    graph = gcare.load_graph(args.dataset, args.data_dir)
    tensors = GraphTensors(graph)
    if args.queries == ["-"]:
        # Serve query names from stdin, one per line, until it closes.
        by_name = {q.name: q for q in gcare.load_queries(graph, None, args.data_dir)}
        queries = (by_name[name] for name in _stdin_lines() if name)
    else:
        queries = gcare.load_queries(graph, args.queries, args.data_dir)
    warm_up(config)
    with args.worker_output.open("a", encoding="utf-8") as out:
        for query in queries:
            _emit(out, {"event": "start", "query": query.name, "time": time.time()})
            recorder = None
            if args.plan_dir:
                path = plan_path(args.plan_dir, args.config, query.name)
                recorder = PlanRecorder(path, query, config, tensors)
            try:
                record = measure_query(query, tensors, config, args.runs, recorder)
                record["status"] = (
                    "ok" if record["correct"]
                    else "wrong" if query.ground_truth is not None
                    else "unchecked"
                )
            except Exception as error:  # record and keep going
                message = f"{type(error).__name__}: {error}"
                oom = isinstance(error, MemoryError) or "OutOfMemoryError" in message
                record = {"status": "oom" if oom else "error", "error": message[:2000]}
            if recorder:
                recorder.append_record(record)
            _emit(out, {"event": "done", "query": query.name, "record": record})


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------


def _base_row(dataset, config_name, query, plan_dir=None):
    row = {
        "dataset": dataset,
        "config": config_name,
        "query": query.name,
        "family": query.family,
        "ground_truth": query.ground_truth,
    }
    if plan_dir is not None:
        path = plan_path(plan_dir, config_name, query.name)
        row["plan_file"] = str(path.relative_to(plan_dir.parent))
    return row


def _memory_limiter(limit_gb):
    """``preexec_fn`` capping the worker's address space (RLIMIT_AS) at *limit_gb*.

    Allocations past the cap raise (Julia ``OutOfMemoryError``, numpy
    ``MemoryError``), which the worker records as status ``oom``.
    """
    if not limit_gb:
        return None
    limit = int(limit_gb * 1024**3)

    def apply():
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return apply


def _worker_setup(limit_gb, cpu):
    """``preexec_fn`` for a worker: pin it to *cpu* and cap its address space."""
    limit_memory = _memory_limiter(limit_gb)

    def setup():
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})
        if limit_memory is not None:
            limit_memory()

    return setup


def default_cpus() -> list[int]:
    """The CPUs this process may use, one logical CPU per physical core (so two
    workers never share a core's hardware threads), in CPU order.  Falls back
    to every allowed CPU when the topology is not readable."""
    allowed = sorted(os.sched_getaffinity(0))
    topology = Path("/sys/devices/system/cpu")
    cores: dict[tuple[str, str], int] = {}
    try:
        for cpu in allowed:
            base = topology / f"cpu{cpu}" / "topology"
            key = (
                (base / "physical_package_id").read_text().strip(),
                (base / "core_id").read_text().strip(),
            )
            cores.setdefault(key, cpu)
    except OSError:
        return allowed
    return sorted(cores.values())


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
            "--queries", "-",
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
        self.log_file = self.log.open("w")
        self.process = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.PIPE,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=_worker_setup(args.memory_limit_gb, self.cpu),
        )
        self.served = 0

    def _events(self):
        with self.output.open(encoding="utf-8") as file:
            file.seek(self.offset)
            chunk = file.read()
        complete = chunk[: chunk.rfind("\n") + 1]
        self.offset += len(complete.encode("utf-8"))
        return [json.loads(line) for line in complete.splitlines()]

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
        record = self._serve(query, index)
        record["worker_query"] = index
        return record

    def _serve(self, query, index) -> dict:
        # A fresh worker (the first one or a restart) still has to compile
        # every kernel and planning path its first query touches, so that
        # query gets extra time; every worker gets the same allowance.
        limit = self.args.timeout + (self.args.first_query_grace if index == 0 else 0)
        self.process.stdin.write(query.name + "\n")
        self.process.stdin.flush()
        started = None
        while True:
            for event in self._events():
                if event["query"] != query.name:
                    continue
                if event["event"] == "start":
                    started = time.monotonic()
                else:
                    self.served += 1
                    return event["record"]
            if self.process.poll() is not None and not self._events_pending():
                return self._died(query, started)
            if started and time.monotonic() - started > limit:
                self._stop()
                return {"status": "timeout", "error": f"exceeded {limit:g}s"}
            time.sleep(0.2)

    def _events_pending(self) -> bool:
        return self.output.stat().st_size > self.offset

    def _died(self, query, started) -> dict:
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
        # We did not kill it, so SIGKILL means the kernel OOM killer.
        oom = code == -9 or any(
            s in text for s in ("OutOfMemory", "MemoryError", "out of memory")
        )
        return {"status": "oom" if oom else "crash", "error": f"exit {code}: {tail}"}

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
                row = _base_row(args.dataset, config_name, query, args.plan_dir)
                row.update(record, cpu=cpu)
                if args.plan_dir and "result" not in record:
                    path = plan_path(args.plan_dir, config_name, query.name)
                    if record["status"] in ("timeout", "crash") or (
                        record["status"] == "oom" and "exit" in record.get("error", "")
                    ):
                        append_plan_status(path, record["status"], record["error"])
                with lock:
                    rows.append(row)
                    writer.writerow(row)
                    csv_file.flush()
                    _print_row(row)
        finally:
            slot.close()

    cpus = args.cpus[: args.jobs]
    with tempfile.TemporaryDirectory(prefix="subgraph-workers-") as folder:
        with ThreadPoolExecutor(len(cpus)) as pool:
            futures = [pool.submit(drain, cpu, Path(folder)) for cpu in cpus]
            for future in futures:
                future.result()
    return rows


def _print_row(row):
    if row["status"] in ("ok", "wrong", "unchecked"):
        detail = (
            f"exec {row['exec_median_s']:.4g}s  plan {row['plan_total_s']:.4g}s  "
            f"first {row['first_s']:.4g}s  count {row['result']:.0f} "
            f"(gt {row['ground_truth']})"
        )
    else:
        detail = row.get("error", "")
    print(f"  {row['status']:7} {row['query']:22} {detail}", flush=True)


def _geomean(values):
    values = [v for v in values if v and v > 0]
    return math.exp(statistics.fmean(map(math.log, values))) if values else float("nan")


def print_summary(rows, config_names):
    statuses = ("ok", "unchecked", "wrong", "timeout", "oom", "error", "crash")
    header = "".join(f"{s:>10}" for s in statuses)
    print(f"\n{'config':38}{header}  geomean exec  geomean plan  (ok + unchecked)")
    for name in config_names:
        mine = [r for r in rows if r["config"] == name]
        ok = [r for r in mine if r["status"] in ("ok", "unchecked")]
        counts = "".join(
            f"{sum(r['status'] == s for r in mine):10}" for s in statuses
        )
        print(
            f"{name:38}{counts}  "
            f"{_geomean([r['exec_median_s'] for r in ok]):11.4g}s  "
            f"{_geomean([r['plan_total_s'] for r in ok]):11.4g}s"
        )


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
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per query")
    parser.add_argument(
        "--max-queries-per-worker",
        type=int,
        default=100,
        help="restart each worker after this many queries (0 never)",
    )
    parser.add_argument(
        "--first-query-grace",
        type=float,
        help="extra seconds for the first query of each worker process, which "
        "compiles its kernels from scratch (default: --timeout, i.e. doubled)",
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
    if args.first_query_grace is None:
        args.first_query_grace = args.timeout
    if args.cpus is None:
        args.cpus = default_cpus()
    if args.jobs < 1 or args.jobs > len(args.cpus):
        parser.error(f"--jobs must be between 1 and the {len(args.cpus)} CPUs given")
    # Keep all workers together within a share of RAM, so that a query which
    # runs out of memory fails alone instead of the kernel's OOM killer
    # taking down other processes.
    total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    budget_gb = args.memory_fraction * total_gb
    if args.memory_limit_gb and args.jobs * args.memory_limit_gb > budget_gb:
        limit = math.floor(10 * budget_gb / args.jobs) / 10
        print(
            f"note: {args.jobs} workers x {args.memory_limit_gb:g} GB would exceed "
            f"{args.memory_fraction:.0%} of {total_gb:.0f} GB RAM; limiting each "
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
    rows = []
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for name in config_names:
            print(f"\n[{name}]", flush=True)
            rows += run_config(args, name, queries, writer, csv_file)
    print_summary(rows, config_names)
    print(f"\nwrote {path}")
    if args.plan_dir:
        print(f"plans in {args.plan_dir}")
    return path


if __name__ == "__main__":
    main()
