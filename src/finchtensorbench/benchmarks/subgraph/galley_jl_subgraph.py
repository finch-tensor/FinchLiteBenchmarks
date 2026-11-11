"""Benchmark Galley.jl (Finch.jl backend) on the subgraph-matching workloads.

Galley.jl is installed by Julia's package manager into the project in
``galley_jl/`` (pinned there to a registered release), and the worker loads the
graphs and queries as Galley.jl's experiments do.  The data, queries and true
counts are the ones :mod:`finchtensorbench.gcare` downloads for the finch
benchmark (the same files as Galley.jl's paper used).  The timing follows
:mod:`finchtensorbench.benchmarks.subgraph.subgraph_matching`, so the two CSVs line up:

* ``first``: one ``galley()`` call, compiling the query's kernels;
* ``exec``/``plan``: ``--runs`` more calls; ``exec`` is Galley.jl's execution
  time and ``plan_total`` its optimization plus execution time.  Galley.jl has
  no plan cache, so each call re-plans; its data statistics are built once at
  load and are not part of either time.

``--jobs`` workers, each pinned to one of ``--cpus``, take the queries in
blocks of ``--max-queries-per-worker``; each worker runs one untimed query
first, like the finch workers' warm-up.  As in the finch benchmark, the cold
first call may take ``--first-timeout`` seconds and the timed calls together
``--timeout`` seconds, for every query alike.  A query exceeding either, or a
worker above ``--memory-limit-gb`` of resident memory, is killed and recorded;
a fresh worker continues with the next query.  Julia is told the limit
(``--heap-size-hint``, by default 80% of it), so its garbage collector works
within it instead of letting memory grow until the worker is killed.

Julia comes from ``--julia`` (default ``$JULIA``, else ``julia`` on the PATH);
the first run installs and precompiles Galley.jl.  Run with::

    pixi run python -m finchtensorbench.benchmarks.subgraph.galley_jl_subgraph --dataset aids
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import re
import signal
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from finchtensorbench import gcare

# G-CARE datasets (directed, labelled edges) and Sun & Luo subgraph-matching
# ones (gcare.SGM_DATASETS: one vertex label, undirected unlabelled edges)
GCARE_DATASETS = ("aids", "human", "lubm80", "yago")
DATASETS = (*GCARE_DATASETS, *gcare.SGM_DATASETS)
CONFIG = "galley.jl"
FIELDS = [
    "dataset", "config", "query", "family", "status", "ground_truth", "result",
    "correct", "first_s", "plan_total_s", "opt_s", "exec_median_s", "exec_min_s",
    "exec_mean_s", "exec_samples", "error", "cpu", "worker_query",
]
WORKER = Path(__file__).with_name("galley_jl_subgraph.jl")
PROJECT = Path(__file__).with_name("galley_jl")


def workload(args):
    """Download *args.dataset* if needed; return its graph kind ("gcare" or
    "sgm"), graph file, and queries as (name, family, file, true count)."""
    if args.dataset in gcare.SGM_DATASETS:
        root = gcare.ensure_sgm_downloaded(args.dataset, args.data_dir)
        kind, graph = "sgm", root / "graph.txt"
        query_root, truth_root = root / "queries", root / "truth"
    else:
        root = gcare.ensure_downloaded(args.data_dir)
        kind, graph = "gcare", root / "dataset" / args.dataset / f"{args.dataset}.txt"
        query_root = root / "queryset" / args.dataset
        truth_root = root / "ground_truth" / args.dataset
    queries = []
    for name in gcare.list_queries(args.dataset, args.data_dir):
        truth = truth_root / f"{name}.txt"
        count = truth.read_text().split()[0] if truth.is_file() else ""
        queries.append((name, name.split("/")[0], query_root / f"{name}.txt", count))
    return kind, graph, queries


def julia_cmd(args):
    return [*args.julia.split(), f"--project={PROJECT}", "--startup-file=no"]


def setup_julia(args):
    """Install and precompile the pinned Galley.jl once, before the workers
    start, so they neither race to do it nor spend their load time on it."""
    print(f"setting up Galley.jl in {PROJECT} ...", flush=True)
    subprocess.run(
        [*julia_cmd(args), "-e", "using Pkg; Pkg.instantiate(); using Galley"],
        check=True, timeout=args.load_timeout,
    )


def rss_gb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


class Worker:
    """One Julia process serving queries ``first..last``."""

    def __init__(self, args, cpu: int, first: int, last: int):
        hint = args.heap_size_hint_gb
        if hint is None:
            hint = 0.8 * args.memory_limit_gb
        cmd = [
            "taskset", "-c", str(cpu), *julia_cmd(args),
            *([f"--heap-size-hint={max(1, int(hint * 1024))}M"] if hint else []),
            str(WORKER), str(args.manifest), str(first), str(last), str(args.runs),
        ]
        self.log = open(args.log_dir / f"worker_{first}.log", "w")
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            self.log.write(line)
            self.log.flush()
            if line.startswith("@@"):
                self.lines.put(line.rstrip("\n").split("\t"))
        self.lines.put(None)

    def stop(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
        self.process.wait()
        self.log.close()


def record(args, cpu, index, status, worker_query, *, count="", first="",
           opt=(), exec_=(), error=""):
    name, family, _, expected = args.queries[index - 1]
    row = dict.fromkeys(FIELDS, "")
    row.update(dataset=args.dataset, config=CONFIG, query=name, family=family,
               status=status, ground_truth=expected, result=count, error=error,
               cpu=cpu, worker_query=worker_query)
    if status == "ok":
        # Compared as float64, as the finch benchmark does (finch's counts are
        # float64; counts above 2**53 are not exact in either).
        same = expected != "" and float(count) == float(expected)
        row["correct"] = "" if expected == "" else str(same)
        if expected == "":
            row["status"] = "unchecked"
        elif not same:
            row["status"] = "wrong"
        row["first_s"] = first
        row["opt_s"] = statistics.median(opt)
        row["plan_total_s"] = statistics.median(o + e for o, e in zip(opt, exec_))
        row["exec_median_s"] = statistics.median(exec_)
        row["exec_min_s"] = min(exec_)
        row["exec_mean_s"] = statistics.mean(exec_)
        row["exec_samples"] = ";".join(f"{e:.6g}" for e in exec_)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--data-dir", type=Path, help="G-CARE data root (see gcare.data_dir)")
    parser.add_argument("--julia", default=os.environ.get("JULIA", "julia"),
                        help="Julia command, e.g. 'julia +1.11' (default: $JULIA or julia)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="seconds for a query's timed calls")
    parser.add_argument("--first-timeout", type=float,
                        help="seconds for a query's cold first call (default: --timeout)")
    parser.add_argument("--heap-size-hint-gb", type=float,
                        help="Julia --heap-size-hint (default: 80%% of --memory-limit-gb;"
                        " 0 to omit)")
    parser.add_argument("--load-timeout", type=float, default=1800.0,
                        help="seconds a worker may take to load the workload")
    parser.add_argument("--max-queries-per-worker", type=int, default=50)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument("--jobs", type=int, default=1, help="Julia workers in parallel")
    parser.add_argument("--cpus", type=int, nargs="+",
                        help="CPUs to pin workers to, one per job (default: 0..jobs-1)")
    parser.add_argument("--limit", type=int, help="only the first N queries")
    parser.add_argument("--out-dir", type=Path, default=Path("graph/subgraph_matching"))
    args = parser.parse_args(argv)
    if args.first_timeout is None:
        args.first_timeout = args.timeout

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_dir / f"{args.dataset}_galleyjl_{stamp}.csv"
    args.log_dir = args.out_dir / f"{args.dataset}_galleyjl_{stamp}_logs"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    kind, graph, args.queries = workload(args)
    args.manifest = args.log_dir / "manifest.tsv"
    args.manifest.write_text(f"graph\t{kind}\t{graph.resolve()}\n" + "".join(
        f"query\t{path.resolve()}\n" for _, _, path, _ in args.queries))
    print(f"{args.dataset}: {len(args.queries)} queries", flush=True)
    setup_julia(args)
    cpus = args.cpus or list(range(args.jobs))
    if len(cpus) < args.jobs:
        parser.error("--cpus needs one CPU per job")
    cpus = cpus[:args.jobs]
    rows = []
    lock = threading.Lock()
    state = {"total": None, "next": 1}
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        def secs(row, key):
            return f"{row[key]:.4g}s" if row[key] != "" else "-"

        def emit(row):
            with lock:
                rows.append(row)
                writer.writerow(row)
                file.flush()
                print(f"  {row['status']:<9} {row['query']} exec {secs(row, 'exec_median_s')}"
                      f"  plan {secs(row, 'plan_total_s')}  first {secs(row, 'first_s')}"
                      f"  count {row['result'] or '-'} (gt {row['ground_truth'] or None})"
                      + (f"  {row['error'][:200]}" if row["error"] else ""), flush=True)

        def serve(cpu, next_q, last):
            """Serve queries next_q..last on fresh workers pinned to ``cpu``."""
            while next_q <= min(last, state["total"] or last):
                worker = Worker(args, cpu, next_q, last)
                deadline = time.monotonic() + args.load_timeout
                current = None  # (index, path, limit)
                served = 0
                while True:
                    try:
                        msg = worker.lines.get(timeout=0.5)
                    except queue.Empty:
                        msg = ()
                    now = time.monotonic()
                    if msg is None or (msg == () and worker.process.poll() is not None
                                       and worker.lines.empty()):
                        if current is None:
                            if next_q <= min(state["total"] or last, last):
                                worker.stop()
                                raise RuntimeError(
                                    f"worker exited before a query; see {worker.log.name}")
                            break
                        code = worker.process.wait()
                        tail = Path(worker.log.name).read_text()[-500:].replace("\n", " ")
                        status = "oom" if code == -9 or "OutOfMemory" in tail else "crash"
                        emit(record(args, cpu, current[1], status, served,
                                    error=f"exit {code}: {tail}"))
                        next_q = current[0] + 1
                        break
                    if msg and msg[0] == "@@READY":
                        with lock:
                            n = int(msg[1])
                            state["total"] = min(n, args.limit) if args.limit else n
                    elif msg and msg[0] == "@@BEGIN":
                        limit = args.first_timeout
                        current = (int(msg[1]), int(msg[1]), limit, "cold")
                        deadline = now + limit
                    elif msg and msg[0] == "@@COLD" and current is not None:
                        current = (current[0], current[1], args.timeout, "timed")
                        deadline = now + args.timeout
                    elif msg and msg[0] == "@@RESULT":
                        _, i, status, count, first, opt, exec_, err = msg
                        floats = lambda s: [float(x) for x in s.split(";") if x]
                        emit(record(args, cpu, int(i), status, served, count=count, first=float(first), opt=floats(opt),
                                    exec_=floats(exec_), error=err))
                        served += 1
                        next_q = int(i) + 1
                        current = None
                        deadline = float("inf")
                    elif msg and msg[0] == "@@DONE":
                        worker.stop()
                        next_q = max(next_q, last + 1)
                        break
                    if now > deadline or rss_gb(worker.process.pid) > args.memory_limit_gb:
                        over = now > deadline
                        worker.stop()
                        if current is None:
                            raise RuntimeError("worker failed to load the workload in time")
                        emit(record(args, cpu, current[1], "timeout" if over else "oom",
                                    served, error=(f"{current[3]} phase exceeded "
                                                   f"{current[2]:g}s" if over
                                                   else f"over {args.memory_limit_gb:g} GB")))
                        next_q = current[0] + 1
                        break

        def run_jobs(cpu):
            while True:
                with lock:
                    first = state["next"]
                    if state["total"] is not None and first > state["total"]:
                        return
                    state["next"] = first + args.max_queries_per_worker
                last = first + args.max_queries_per_worker - 1
                if args.limit:
                    last = min(last, args.limit)
                serve(cpu, first, last)

        errors = []

        def run_thread(cpu):
            try:
                run_jobs(cpu)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=run_thread, args=(cpu,)) for cpu in cpus]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]

    summary(rows)
    print(f"wrote {out}")


def summary(rows):
    statuses = ("ok", "unchecked", "wrong", "timeout", "oom", "error", "crash")
    done = [r for r in rows if r["status"] in ("ok", "unchecked")]

    def geo(key):
        xs = [float(r[key]) for r in done]
        return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")

    print("\n" + "".join(f"{s:>10}" for s in statuses) + "  geomean exec  geomean plan")
    print("".join(f"{sum(r['status'] == s for r in rows):>10}" for s in statuses)
          + f"  {geo('exec_median_s'):>11.4g}s  {geo('plan_total_s'):>11.4g}s")


if __name__ == "__main__":
    main()
