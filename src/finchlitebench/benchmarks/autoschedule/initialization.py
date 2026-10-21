"""Runner for the autoscheduler benchmarks.

Query files live under ``queries/`` and ``queries_lite/``.  They describe
*what* to time.  This module describes *how* to time it:

1. Build a Finch scheduler with optional stage timers.
2. Call ``fl.compute(query(...), ctx=scheduler)``.
3. Isolate each loop/logic pair in a worker process.
4. Write raw samples and a summary CSV.

Run a registered workload with:
``python -m finchlitebench.benchmarks.autoschedule.initialization chain_lite``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Sequence

import finch as fl
from finch.autoschedule import default_schedulers
from finch.autoschedule.compiler import CompilerFormLowerer, LogicCompiler
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.factorizer.galley_factorizer.galley_optimize import (
    GalleyLogicFactorizer,
)
from finch.autoschedule.factorizer.optimize import DefaultLogicFactorizer
from finch.autoschedule.formatter import DefaultLogicFormatter
from finch.autoschedule.loop_orderer.loop_order_bnb import (
    BFSLoopOrderer,
    DFSLoopOrderer,
)
from finch.autoschedule.loop_orderer.loop_order_greedy import GreedyLoopOrderer
from finch.autoschedule.loop_orderer.loop_ordering import DefaultLoopOrderer
from finch.autoschedule.normalize import LogicNormalizer
from finch.autoschedule.tensor_stats import (
    DCStatsFactory,
    FDStatsFactory,
    UniformStatsFactory,
)
from finch.compile_jl.compiler import FinchJLCompiler
from finch.finch_logic import Aggregate, LogicSimplify, MockLogicLoader, Reorder

from .spec import (
    BENCHMARK_MODULES,
    BenchmarkCase as BenchmarkCase,
    BenchmarkSpec as BenchmarkSpec,
    cases_from_shapes as cases_from_shapes,
    load_benchmark,
    make_spec as make_spec,
)


LOOP_ORDERERS = {
    "default": DefaultLoopOrderer,
    "greedy": GreedyLoopOrderer,
    "bfs": BFSLoopOrderer,
    "dfs": DFSLoopOrderer,
}
LOGIC_ORDERERS = ("default", "galley-greedy", "galley-bfs", "galley-dfs")
STATS_FACTORIES = {"dc": DCStatsFactory, "uniform": UniformStatsFactory}
MODES = ("scheduler", "end-to-end")
RUNNER_MODULE = "finchlitebench.benchmarks.autoschedule.initialization"

# Use Julia for sparse statistics, with dense intermediate/output storage.
default_schedulers.NON_RECURSIVE_SCHEDULER = LogicNormalizer(
    LogicExecutor(
        DefaultLogicFactorizer(
            LogicSimplify(
                DefaultLoopOrderer(
                    DefaultLogicFormatter(
                        CompilerFormLowerer(LogicCompiler(FinchJLCompiler()))
                    )
                )
            )
        ),
        stats_factory=FDStatsFactory(),
        cache=True,
    )
)


# ---------------------------------------------------------------------------
# Scheduler construction
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    loop_orderer: float
    logic_orderer: float
    pipeline: float
    loop_idxs: str = "-"
    logic_idxs: str = "-"


class StageTimer:
    def __init__(self):
        self.times = {"loop_orderer": 0.0, "logic_orderer": 0.0}
        self.stack = []

    def reset(self):
        self.times = dict.fromkeys(self.times, 0.0)
        self.stack.clear()

    def start(self, name):
        self.stack.append((name, time.perf_counter()))

    def stop(self):
        name, started = self.stack.pop()
        self.times[name] += time.perf_counter() - started


class _TimedStage:
    """Time one stage, excluding the downstream stage it invokes."""

    def __init__(self, inner, timer, name):
        self.inner, self.timer, self.name = inner, timer, name

    def validate_inputs(self, *args):
        return self.inner.validate_inputs(*args)

    def __call__(self, *args):
        original_context = self.inner.ctx

        def paused_context(*context_args):
            self.timer.stop()
            try:
                return original_context(*context_args)
            finally:
                self.timer.start(self.name)

        self.inner.ctx = paused_context
        self.timer.start(self.name)
        try:
            return self.inner(*args)
        finally:
            self.timer.stop()
            self.inner.ctx = original_context

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _CaptureStage:
    def __init__(self, inner, slot, store):
        self.inner, self.slot, self.store = inner, slot, store

    def __call__(self, program, *args):
        self.store[self.slot] = program
        return self.inner(program, *args)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _NoOpLibrary:
    def __init__(self, output_count):
        self.output_count = output_count

    def main(self, *args):
        return tuple(args[-self.output_count :])


class _PlanningOnlyLoader(MockLogicLoader):
    """Return after loop ordering instead of evaluating the query."""

    def __init__(self, output_count):
        super().__init__()
        self.output_count = output_count

    def lower(self, *args):
        _, bindings, indexes, program = super().lower(*args)
        return _NoOpLibrary(self.output_count), bindings, indexes, program


def build_scheduler(
    loop_order="greedy",
    logic_order="default",
    stats="dc",
    execute=True,
    output_count=1,
):
    """Build the common Finch scheduler pipeline."""
    timer, captures = StageTimer(), {}
    downstream = (
        DefaultLogicFormatter(CompilerFormLowerer(LogicCompiler(FinchJLCompiler())))
        if execute
        else DefaultLogicFormatter(_PlanningOnlyLoader(output_count))
    )
    downstream = _CaptureStage(downstream, "loop_prgm", captures)
    loop_stage = LOOP_ORDERERS[loop_order](downstream)
    logic_downstream = _CaptureStage(
        LogicSimplify(_TimedStage(loop_stage, timer, "loop_orderer")),
        "logic_prgm",
        captures,
    )
    logic_stage = (
        DefaultLogicFactorizer(logic_downstream)
        if logic_order == "default"
        else GalleyLogicFactorizer(
            logic_downstream, optimizer=logic_order.removeprefix("galley-")
        )
    )
    context = LogicNormalizer(
        LogicExecutor(
            _TimedStage(logic_stage, timer, "logic_orderer"),
            stats_factory=STATS_FACTORIES[stats](),
        )
    )
    context.timer, context.captures = timer, captures
    return context


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _field_names(fields):
    return ",".join(getattr(field, "name", str(field)) for field in fields)


def _queries(program):
    bodies = getattr(program, "bodies", None)
    if bodies is None:
        yield program
    else:
        for body in bodies:
            yield from _queries(body) if hasattr(body, "bodies") else (body,)


def _orders(program, loop_order):
    orders = []
    for node in _queries(program) if program is not None else ():
        right = getattr(node, "rhs", None)
        if isinstance(right, Reorder):
            right = right.arg
        if not isinstance(right, Aggregate):
            continue
        if loop_order and isinstance(right.arg, Reorder):
            orders.append(_field_names(right.arg.idxs))
        elif not loop_order:
            orders.append(_field_names(right.idxs))
    return (" | " if loop_order else " -> ").join(orders) or "-"


def measure_samples(spec, inputs, context, warmups, runs):
    """Warm up and retain every measurement rather than only an average."""
    if warmups < 0 or runs < 1:
        raise ValueError("warmups must be nonnegative and runs must be positive")
    inputs = tuple(fl.defer(fl.asarray(value)) for value in inputs)

    def measure():
        context.timer.reset()
        started = time.perf_counter()
        fl.compute(spec.query(*inputs), ctx=context)
        pipeline = time.perf_counter() - started
        return BenchmarkResult(
            context.timer.times["loop_orderer"],
            context.timer.times["logic_orderer"],
            pipeline,
            _orders(context.captures.get("loop_prgm"), True),
            _orders(context.captures.get("logic_prgm"), False),
        )

    for _ in range(warmups):
        measure()
    return [measure() for _ in range(runs)]


# ---------------------------------------------------------------------------
# Records and summaries
# ---------------------------------------------------------------------------


RAW_FIELDS = (
    "mode",
    "workload",
    "shape",
    "pattern",
    "stats",
    "loop",
    "logic",
    "round",
    "sample",
    "pid",
    "seed",
    "loop_order",
    "logic_order",
    "logic_s",
    "loop_s",
    "pipeline_s",
)
GROUP_FIELDS = ("mode", "workload", "shape", "pattern", "stats", "loop", "logic")
TIMING_FIELDS = ("logic_s", "loop_s", "pipeline_s")
SUMMARY_FIELDS = (
    *GROUP_FIELDS,
    "samples",
    "loop_order",
    "logic_order",
    "plan_variants",
    *(
        f"{timing}_{stat}"
        for timing in TIMING_FIELDS
        for stat in ("median", "mean", "stdev", "min", "q1", "q3", "max")
    ),
)


def configuration_order(
    round_number,
    seed,
    loops=tuple(LOOP_ORDERERS),
    logics=LOGIC_ORDERERS,
):
    configurations = [(loop, logic) for logic in logics for loop in loops]
    random.Random(seed + round_number).shuffle(configurations)
    return configurations


def _worker_records(spec, args):
    records = []
    cases_by_mode = {
        "scheduler": args.scheduler_cases,
        "end-to-end": args.end_to_end_cases,
    }
    settings = {
        "scheduler": (args.scheduler_warmups, args.scheduler_runs),
        "end-to-end": (args.end_to_end_warmups, args.end_to_end_runs),
    }
    cases = spec.cases_by_name

    for mode in args.modes:
        warmups, runs = settings[mode]
        for case_name in cases_by_mode[mode]:
            case = cases[case_name]
            context = build_scheduler(
                args.loop_order,
                args.logic_order,
                args.stats,
                execute=mode == "end-to-end",
                output_count=spec.output_count,
            )
            samples = measure_samples(spec, spec.inputs(case), context, warmups, runs)
            for number, result in enumerate(samples, 1):
                records.append(
                    {
                        "mode": mode,
                        "workload": case.name,
                        "shape": "x".join(map(str, case.shape)),
                        "pattern": case.pattern,
                        "stats": args.stats,
                        "loop": args.loop_order,
                        "logic": args.logic_order,
                        "round": args.round,
                        "sample": number,
                        "pid": os.getpid(),
                        "seed": args.seed,
                        "loop_order": result.loop_idxs,
                        "logic_order": result.logic_idxs,
                        "logic_s": result.logic_orderer,
                        "loop_s": result.loop_orderer,
                        "pipeline_s": result.pipeline,
                    }
                )
    return records


def _percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[lower + 1] * weight


def summarize_records(records: Sequence[dict]) -> list[dict]:
    groups = {}
    for record in records:
        key = tuple(str(record[field]) for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(record)

    summaries = []
    for key, rows in sorted(groups.items()):
        summary = dict(zip(GROUP_FIELDS, key, strict=True))
        plans = {(row["loop_order"], row["logic_order"]) for row in rows}
        summary.update(
            samples=len(rows),
            loop_order=statistics.mode(row["loop_order"] for row in rows),
            logic_order=statistics.mode(row["logic_order"] for row in rows),
            plan_variants=len(plans),
        )
        for timing in TIMING_FIELDS:
            values = [float(row[timing]) for row in rows]
            summary.update(
                {
                    f"{timing}_median": statistics.median(values),
                    f"{timing}_mean": statistics.fmean(values),
                    f"{timing}_stdev": statistics.stdev(values)
                    if len(values) > 1
                    else 0.0,
                    f"{timing}_min": min(values),
                    f"{timing}_q1": _percentile(values, 0.25),
                    f"{timing}_q3": _percentile(values, 0.75),
                    f"{timing}_max": max(values),
                }
            )
        summaries.append(summary)
    return summaries


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path, fields, rows):
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writerows(rows)


# ---------------------------------------------------------------------------
# Parent process and CLI
# ---------------------------------------------------------------------------


def _run_worker(spec, args, loop, logic, round_number):
    with tempfile.TemporaryDirectory(prefix=f"{spec.name}-benchmark-") as folder:
        output = Path(folder) / "records.json"
        command = [
            sys.executable,
            "-m",
            RUNNER_MODULE,
            spec.name,
            "--worker-output",
            str(output),
            "--loop-order",
            loop,
            "--logic-order",
            logic,
            "--stats",
            args.stats,
            "--round",
            str(round_number),
            "--seed",
            str(args.seed),
            "--scheduler-warmups",
            str(args.scheduler_warmups),
            "--scheduler-runs",
            str(args.scheduler_runs),
            "--end-to-end-warmups",
            str(args.end_to_end_warmups),
            "--end-to-end-runs",
            str(args.end_to_end_runs),
            "--modes",
            *args.modes,
            "--scheduler-cases",
            *args.scheduler_cases,
            "--end-to-end-cases",
            *args.end_to_end_cases,
        ]
        environment = os.environ.copy()
        src = str(Path(__file__).resolve().parents[3])
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (src, environment.get("PYTHONPATH")) if value
        )
        result = subprocess.run(
            command, env=environment, text=True, capture_output=True
        )
        if result.returncode:
            raise RuntimeError(
                f"worker failed for {loop}/{logic}:\n{result.stdout}{result.stderr}"
            )
        return json.loads(output.read_text(encoding="utf-8"))


def run_benchmark(spec, args):
    spec.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = spec.results_dir / f"{spec.name}_{stamp}_raw.csv"
    summary_path = spec.results_dir / f"{spec.name}_{stamp}_summary.csv"
    records = []
    jobs = args.rounds * len(args.loops) * len(args.logics)
    completed = 0
    _write_csv(raw_path, RAW_FIELDS, ())

    for round_number in range(1, args.rounds + 1):
        for loop, logic in configuration_order(
            round_number, args.seed, args.loops, args.logics
        ):
            completed += 1
            print(
                f"[{completed}/{jobs}] round {round_number}: {loop}/{logic}",
                flush=True,
            )
            new_records = _run_worker(spec, args, loop, logic, round_number)
            records.extend(new_records)
            _append_csv(raw_path, RAW_FIELDS, new_records)

    _write_csv(summary_path, SUMMARY_FIELDS, summarize_records(records))
    print(f"wrote {raw_path}\nwrote {summary_path}")
    return raw_path, summary_path


def run_cli(spec=None, argv=None):
    """Parse the common CLI and run either the parent or worker process."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if spec is None:
        if not argv or argv[0] in {"-h", "--help"}:
            parser = argparse.ArgumentParser(
                description="Run a registered Finch autoscheduler benchmark"
            )
            parser.add_argument("benchmark", choices=tuple(BENCHMARK_MODULES))
            parser.print_help()
            return
        spec = load_benchmark(argv.pop(0))

    case_names = tuple(spec.cases_by_name)
    parser = argparse.ArgumentParser(description=f"Benchmark {spec.name}")
    parser.add_argument("--rounds", type=int, default=spec.rounds)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--stats", choices=tuple(STATS_FACTORIES), default="dc")
    parser.add_argument(
        "--loops", nargs="+", choices=tuple(LOOP_ORDERERS), default=list(LOOP_ORDERERS)
    )
    parser.add_argument(
        "--logics", nargs="+", choices=LOGIC_ORDERERS, default=list(LOGIC_ORDERERS)
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--scheduler-cases",
        nargs="+",
        choices=case_names,
        default=list(spec.scheduler_cases),
    )
    parser.add_argument(
        "--end-to-end-cases",
        nargs="+",
        choices=case_names,
        default=list(spec.end_to_end_cases),
    )
    parser.add_argument("--scheduler-warmups", type=int, default=spec.scheduler_warmups)
    parser.add_argument("--scheduler-runs", type=int, default=spec.scheduler_runs)
    parser.add_argument(
        "--end-to-end-warmups", type=int, default=spec.end_to_end_warmups
    )
    parser.add_argument("--end-to-end-runs", type=int, default=spec.end_to_end_runs)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--loop-order", choices=tuple(LOOP_ORDERERS), help=argparse.SUPPRESS
    )
    parser.add_argument("--logic-order", choices=LOGIC_ORDERERS, help=argparse.SUPPRESS)
    parser.add_argument("--round", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_output:
        if not args.loop_order or not args.logic_order:
            parser.error("worker mode needs --loop-order and --logic-order")
        args.worker_output.write_text(
            json.dumps(_worker_records(spec, args)), encoding="utf-8"
        )
        return
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    print(
        f"{args.rounds} shuffled rounds; "
        f"scheduler {args.scheduler_warmups}+{args.scheduler_runs}, "
        f"end-to-end {args.end_to_end_warmups}+{args.end_to_end_runs}"
    )
    run_benchmark(spec, args)


if __name__ == "__main__":
    run_cli()
