"""Benchmark Galley.jl (Finch.jl backend) on the subgraph-matching workloads.

Galley.jl is run from a local checkout with its own experiment loaders, so the
queries, data and true counts are the ones its paper used.  The timing follows
:mod:`finchtensorbench.benchmarks.subgraph_matching`, so the two CSVs line up:

* ``first``: one ``galley()`` call, compiling the query's kernels;
* ``exec``/``plan``: ``--runs`` more calls; ``exec`` is Galley.jl's execution
  time and ``plan_total`` its optimization plus execution time.  Galley.jl has
  no plan cache, so each call re-plans; its data statistics are built once at
  load and are not part of either time.

A query exceeding ``--timeout`` seconds, or a worker above
``--memory-limit-gb`` of resident memory, is killed and recorded; a fresh
worker continues with the next query.

Run with::

    pixi run python -m finchtensorbench.benchmarks.galley_jl_subgraph --dataset aids
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

# our dataset name -> Galley.jl WORKLOAD
WORKLOADS = {
    "aids": "aids",
    "human": "human",
    "yeast": "yeast_lite",
    "dblp": "dblp_lite",
    "youtube": "youtube_lite",
}
CONFIG = "galley.jl"
FIELDS = [
    "dataset", "config", "query", "family", "status", "ground_truth", "result",
    "correct", "first_s", "plan_total_s", "opt_s", "exec_median_s", "exec_min_s",
    "exec_mean_s", "exec_samples", "error", "cpu", "worker_query",
]
WORKER = Path(__file__).with_name("galley_jl_subgraph.jl")


def query_name(path: str) -> tuple[str, str]:
    """Galley.jl query path -> (our query name, family)."""
    p = Path(path)
    if p.parent.name.endswith("-lite"):
        family = re.fullmatch(r"query_(.+)_\d+", p.stem).group(1)
        return f"{family}/{p.stem}", family
    return f"{p.parent.name}/{p.stem}", p.parent.name


def true_count(args, path: str) -> str:
    """Galley.jl's true count for a query, as its loader reads it.

    Galley.jl looks for a "-lite" query's count under the "-lite" folder, which
    has none; the counts are under the full workload's folder.
    """
    p = Path(path.replace("Subgraph_Queries", "Subgraph_TrueCardinalities"))
    if p.parent.name.endswith("-lite"):
        p = p.parent.parent / p.parent.name.removesuffix("-lite") / p.name
    p = args.galley_root / p
    if not p.is_file():
        return ""
    lines = [line.split() for line in p.read_text().splitlines() if line.strip()]
    return lines[-1][0] if lines else ""


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

    def __init__(self, args, first: int, last: int):
        cmd = [
            "taskset", "-c", str(args.cpu), *args.julia.split(),
            f"--project={args.galley_root}", "--startup-file=no",
            str(WORKER), WORKLOADS[args.dataset], str(first), str(last), str(args.runs),
        ]
        self.log = open(args.log_dir / f"worker_{first}.log", "w")
        self.process = subprocess.Popen(
            cmd, cwd=args.galley_root, stdout=subprocess.PIPE,
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


def record(args, path, status, worker_query, *, expected="", count="", first="",
           opt=(), exec_=(), error=""):
    name, family = query_name(path)
    expected = expected or true_count(args, path)
    row = dict.fromkeys(FIELDS, "")
    row.update(dataset=args.dataset, config=CONFIG, query=name, family=family,
               status=status, ground_truth=expected, result=count, error=error,
               cpu=args.cpu, worker_query=worker_query)
    if status == "ok":
        row["correct"] = "" if expected == "" else str(int(count) == int(expected))
        if expected == "":
            row["status"] = "unchecked"
        elif int(count) != int(expected):
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
    parser.add_argument("--dataset", required=True, choices=sorted(WORKLOADS))
    parser.add_argument("--galley-root", type=Path, default=Path.home() / "Galley.jl")
    parser.add_argument("--julia", default="julia +1.11")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per query")
    parser.add_argument("--load-timeout", type=float, default=1800.0,
                        help="seconds a worker may take to load the workload")
    parser.add_argument("--max-queries-per-worker", type=int, default=50)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--limit", type=int, help="only the first N queries")
    parser.add_argument("--out-dir", type=Path, default=Path("graph/subgraph_matching"))
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_dir / f"{args.dataset}_galleyjl_{stamp}.csv"
    args.log_dir = args.out_dir / f"{args.dataset}_galleyjl_{stamp}_logs"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = None
    next_q = 1
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        def emit(row):
            rows.append(row)
            writer.writerow(row)
            file.flush()
            def secs(key):
                return f"{row[key]:.4g}s" if row[key] != "" else "-"

            print(f"  {row['status']:<9} {row['query']} exec {secs('exec_median_s')}"
                  f"  plan {secs('plan_total_s')}  first {secs('first_s')}"
                  f"  count {row['result'] or '-'} (gt {row['ground_truth'] or None})"
                  + (f"  {row['error'][:200]}" if row["error"] else ""), flush=True)

        while total is None or next_q <= total:
            last = next_q + args.max_queries_per_worker - 1
            if args.limit:
                last = min(last, args.limit)
            worker = Worker(args, next_q, last)
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
                        if total is None or next_q <= min(total, last):
                            worker.stop()
                            raise RuntimeError(
                                f"worker exited before a query; see {worker.log.name}")
                        break
                    code = worker.process.wait()
                    tail = Path(worker.log.name).read_text()[-500:].replace("\n", " ")
                    status = "oom" if code == -9 or "OutOfMemory" in tail else "crash"
                    emit(record(args, current[1], status, served, error=f"exit {code}: {tail}"))
                    next_q = current[0] + 1
                    break
                if msg and msg[0] == "@@READY":
                    total = int(msg[1]) if not args.limit else min(int(msg[1]), args.limit)
                elif msg and msg[0] == "@@BEGIN":
                    limit = args.timeout * (2 if served == 0 else 1)
                    current = (int(msg[1]), msg[2], limit)
                    deadline = now + limit
                elif msg and msg[0] == "@@RESULT":
                    _, i, path, status, expected, count, first, opt, exec_, err = msg
                    floats = lambda s: [float(x) for x in s.split(";") if x]
                    emit(record(args, path, status, served, expected=expected, count=count,
                                first=float(first), opt=floats(opt), exec_=floats(exec_),
                                error=err))
                    served += 1
                    next_q = int(i) + 1
                    current = None
                    deadline = float("inf")
                elif msg and msg[0] == "@@DONE":
                    worker.stop()
                    break
                if now > deadline or rss_gb(worker.process.pid) > args.memory_limit_gb:
                    over = now > deadline
                    worker.stop()
                    if current is None:
                        raise RuntimeError("worker failed to load the workload in time")
                    emit(record(args, current[1], "timeout" if over else "oom", served,
                                error=(f"exceeded {current[2]:g}s" if over
                                       else f"over {args.memory_limit_gb:g} GB")))
                    next_q = current[0] + 1
                    break

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
