"""Time each loop/logic orderer pair on a benchmark; write one CSV row per sample."""

import csv
from datetime import datetime
import itertools
import sys
import time

import numpy as np

import finch as fl
from finch.autoschedule import LogicCapture, default_schedulers
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
from finch.autoschedule.loop_orderer.loop_ordering import (
    AbstractLoopOrderer,
    DefaultLoopOrderer,
)
from finch.autoschedule.normalize import LogicNormalizer
from finch.autoschedule.tensor_stats import FDStatsFactory
from finch.compile_jl.compiler import FinchJLCompiler
from finch.finch_logic import Aggregate, LogicSimplify, MockLogicLoader, Reorder

from finchlitebench.benchmarks.autoschedule.queries.common import density
from finchlitebench.benchmarks.autoschedule.spec import RESULTS_DIR, load_benchmark


LOOP_ORDERERS = {
    "default": DefaultLoopOrderer,
    "greedy": GreedyLoopOrderer,
    "bfs": BFSLoopOrderer,
    "dfs": DFSLoopOrderer,
}

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


def _time_lower(stage, times, key, sign):
    """Add ``sign`` times each ``stage.lower`` call's duration to ``times[key]``."""
    lower = stage.lower

    def timed(*args):
        started = time.perf_counter()
        try:
            return lower(*args)
        finally:
            times[key] += sign * (time.perf_counter() - started)

    stage.lower = timed


class _NoOpLibrary:
    """Stands in for a compiled kernel in planning-only runs."""

    def main(self, *args):
        # Planning-only results are discarded; zeros work for any output format.
        return (np.zeros(args[-1].shape),)


class _PlanningOnlyLoader(MockLogicLoader):
    """Return after loop ordering instead of evaluating the query."""

    def lower(self, *args):
        _, bindings, indexes, program = super().lower(*args)
        return _NoOpLibrary(), bindings, indexes, program


def _orders(node, loop_order):
    """Each aggregate's index order: its loop order, or else its reduced indices."""
    if node is None:
        return []
    if hasattr(node, "bodies"):
        return [order for body in node.bodies for order in _orders(body, loop_order)]
    right = getattr(node, "rhs", None)
    if isinstance(right, Reorder):
        right = right.arg
    if not isinstance(right, Aggregate):
        return []
    if not loop_order:
        idxs = right.idxs
    elif isinstance(right.arg, Reorder):
        idxs = right.arg.idxs
    else:
        return []
    return [",".join(getattr(idx, "name", str(idx)) for idx in idxs)]


FIELDS = (
    "mode",
    "workload",
    "shape",
    "pattern",
    "density",
    "stats",
    "pipeline",
    "loop",
    "logic",
    "sample",
    "loop_order",
    "logic_order",
    "logic_s",
    "loop_s",
    "pipeline_s",
)


def _name(value):
    """``COMPILE_JULIA_GALLEY`` for a Finch scheduler, else the class name."""
    for name, scheduler in vars(default_schedulers).items():
        if scheduler is value:
            return name
    return getattr(value, "__name__", type(value).__name__)


def run_benchmark(spec):
    """Time every loop/logic orderer combination and write a CSV."""
    stats, pipeline = _name(spec.stats), _name(spec.pipeline)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{spec.name}_{stamp}_raw.csv"
    cases = {case.name: case for case in spec.cases}
    settings = {
        "scheduler": (spec.cases, spec.scheduler_warmups, spec.scheduler_runs),
        "end-to-end": (
            [cases[name] for name in spec.end_to_end_cases],
            spec.end_to_end_warmups,
            spec.end_to_end_runs,
        ),
    }
    # Keep the pipeline's stages below its loop orderer (formatter and backend).
    formatter = spec.pipeline
    while not isinstance(formatter, AbstractLoopOrderer):
        formatter = formatter.ctx
    formatter = formatter.ctx

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for loop, logic, mode in itertools.product(
            spec.loops, spec.logics, spec.modes
        ):
            print(f"{loop}/{logic} {mode}", flush=True)
            mode_cases, warmups, runs = settings[mode]
            for case in mode_cases:
                downstream = (
                    formatter
                    if mode == "end-to-end"
                    else type(formatter)(_PlanningOnlyLoader())
                )
                # LogicCapture keeps the last program it saw in ``last_prgm``.
                loop_capture = LogicCapture(downstream)
                loop_stage = LOOP_ORDERERS[loop](loop_capture)
                logic_capture = LogicCapture(LogicSimplify(loop_stage))
                logic_stage = (
                    DefaultLogicFactorizer(logic_capture)
                    if logic == "default"
                    else GalleyLogicFactorizer(
                        logic_capture, optimizer=logic.removeprefix("galley-")
                    )
                )
                # Orderer time = its own lower() minus the stage it hands off to.
                times = {"logic": 0.0, "loop": 0.0}
                _time_lower(logic_stage, times, "logic", 1)
                _time_lower(logic_capture, times, "logic", -1)
                _time_lower(loop_stage, times, "loop", 1)
                _time_lower(loop_capture, times, "loop", -1)
                context = LogicNormalizer(
                    LogicExecutor(logic_stage, stats_factory=spec.stats())
                )
                inputs = tuple(
                    fl.defer(fl.asarray(value)) for value in spec.inputs(case)
                )

                # Samples numbered below 1 are warmups and are not written.
                for sample in range(1 - warmups, runs + 1):
                    times.update(logic=0.0, loop=0.0)
                    started = time.perf_counter()
                    fl.compute(spec.query(*inputs), ctx=context)
                    pipeline_s = time.perf_counter() - started
                    if sample < 1:
                        continue
                    loop_order = " | ".join(_orders(loop_capture.last_prgm, True))
                    logic_order = " -> ".join(_orders(logic_capture.last_prgm, False))
                    writer.writerow(
                        {
                            "mode": mode,
                            "workload": case.name,
                            "shape": "x".join(map(str, case.shape)),
                            "pattern": case.pattern,
                            "density": density(case),
                            "stats": stats,
                            "pipeline": pipeline,
                            "loop": loop,
                            "logic": logic,
                            "sample": sample,
                            "loop_order": loop_order or "-",
                            "logic_order": logic_order or "-",
                            "logic_s": times["logic"],
                            "loop_s": times["loop"],
                            "pipeline_s": pipeline_s,
                        }
                    )
            file.flush()

    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    run_benchmark(load_benchmark(sys.argv[1]))
