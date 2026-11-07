"""Compare loop-order cost with cached Julia execution for frontend expressions."""

import csv
import gc
import itertools
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr

import finch as fl
import finch.autoschedule.default_schedulers as default_schedulers
from finch.autoschedule.factorizer.galley_factorizer import insert_statistics
from finch.autoschedule.loop_orderer.loop_order_cost import (
    cost_of_reformat,
    loop_order_cost,
)
from finch.autoschedule.loop_orderer.loop_ordering import (
    AbstractLoopOrderer,
    drop_internal_reorders,
)
from finch.compile_jl.julia import get_jl
from finch.finch_logic import Aggregate, Alias, Plan, Query, QueryInto, Reorder, Table
from finch.symbolic import PostOrderDFS, gensym

from . import BenchmarkCase
from .microBenchmark import RECORDER as SHARED_RECORDER
from .microBenchmark import scheduler


def _loop_queries(plan):
    for query in plan.bodies[:-1]:
        match query:
            case Query(Table(), Aggregate()) | QueryInto(Table(), _, _):
                yield query
            case Query(Table(), Table()):
                pass
            case _:
                raise ValueError(f"Unsupported normalized query: {query}")


def _fields(query):
    match query:
        case Query(_, Aggregate(_, _, arg, red)):
            return tuple(dict.fromkeys((*arg.fields(), *query.lhs.idxs, *red)))
        case QueryInto(_, _, arg):
            return tuple(dict.fromkeys((*arg.fields(), *query.lhs.idxs)))


def _labels(count):
    alphabet = "ijklmnopqrstuvwxyzabcdefgh"
    return tuple(alphabet[i] if i < len(alphabet) else f"d{i}" for i in range(count))


def _describe(orders):
    return " | ".join(
        f"q{i}: " + " ".join(_labels(len(order))[j] for j in order)
        for i, order in enumerate(orders, 1)
    ) or "copy"


class _CapturedPlan(Exception):
    pass


def _discover(expression):
    def capture(plan, *_):
        raise _CapturedPlan(plan)

    try:
        fl.compute(expression, ctx=capture)
    except _CapturedPlan as captured:
        frontend_plan = captured.args[0]
    else:
        raise ValueError("Return a lazy Finch expression; defer inputs and omit fl.compute().")
    try:
        scheduler(capture)(frontend_plan)
    except _CapturedPlan as captured:
        plan = drop_internal_reorders(captured.args[0], keep_loop_orders=False)
        return frontend_plan, plan
    raise RuntimeError("The scheduler did not expose a normalized plan.")


def _permutations(query):
    fields = _fields(query)
    output = tuple(fields.index(f) for f in query.lhs.idxs)
    return [
        order
        for order in itertools.permutations(range(len(fields)))
        if not isinstance(query, QueryInto)
        or tuple(i for i in order if i in output) == output
    ]


class _FixedOrders(AbstractLoopOrderer):
    def __init__(self, recorder, orders):
        super().__init__(recorder)
        self.orders = orders

    def set_loop_orders(self, plan, stats, stats_factory):
        queries = []
        scheduled = iter(self.orders)
        expected = []
        for query in plan.bodies[:-1]:
            match query:
                case Query(Table(_, out) as lhs, Aggregate(op, init, arg, red)):
                    fields = _fields(query)
                    order = tuple(fields[i] for i in next(scheduled))
                    expected.append(order)
                    output_order = tuple(f for f in order if f in out)
                    rhs = Aggregate(op, init, Reorder(arg, order), red)
                    if output_order != out:
                        tmp = Table(Alias(gensym("benchmark")), output_order)
                        queries.extend((Query(tmp, rhs), Query(lhs, tmp)))
                        continue
                    query = Query(lhs, rhs)
                case QueryInto(lhs, op, arg):
                    fields = _fields(query)
                    order = tuple(fields[i] for i in next(scheduled))
                    expected.append(order)
                    query = QueryInto(lhs, op, Reorder(arg, order))
            queries.append(query)
        if next(scheduled, None) is not None:
            raise ValueError("The normalized query count changed during scheduling.")
        self.ctx.expected = expected
        return Plan((*queries, plan.bodies[-1]))


class _Recorder:
    def __call__(self, plan, bindings, stats, stats_factory):
        stats_bindings, cache = dict(stats), {}
        self.cost = 0.0
        actual = []
        for query in plan.bodies[:-1]:
            match query:
                case Query(_, Aggregate(_, _, Reorder(arg, order), _)) | QueryInto(
                    _, _, Reorder(arg, order)
                ):
                    actual.append(order)
                    self.cost += loop_order_cost(
                        arg, order, stats_factory, stats_bindings, query.lhs.idxs
                    )
                case Query(_, Table(tns, _)):
                    self.cost += cost_of_reformat(stats_bindings[tns])
                case _:
                    raise ValueError(f"Unsupported ordered query: {query}")
            insert_statistics(
                stats_factory, query, stats_bindings, replace=False, cache=cache
            )
        if actual != self.expected:
            raise RuntimeError(f"Requested orders {self.expected}, got {actual}")

        mod, *rest = SHARED_RECORDER.backend(plan, bindings, stats, stats_factory)

        def main(*args):
            self.execute = lambda: mod.main(*args)
            start = time.perf_counter()
            result = self.execute()
            self.initial_seconds = time.perf_counter() - start
            return result

        return SimpleNamespace(main=main), *rest


@contextmanager
def _input_statistics():
    # Frontend construction may evaluate input statistics. Keep previews independent
    # of Julia startup, and restore the caller's scheduler after preparing cases.
    previous = default_schedulers.NON_RECURSIVE_SCHEDULER
    previous_cache = default_schedulers.COMPILE_NUMBA.ctx.cache
    default_schedulers.NON_RECURSIVE_SCHEDULER = default_schedulers.COMPILE_NUMBA
    default_schedulers.COMPILE_NUMBA.ctx.cache = True
    try:
        yield
    finally:
        default_schedulers.NON_RECURSIVE_SCHEDULER = previous
        default_schedulers.COMPILE_NUMBA.ctx.cache = previous_cache


def _layout(query, number):
    fields = _fields(query)
    labels = dict(zip(fields, _labels(len(fields)), strict=True))
    arg = query.rhs.arg if isinstance(query.rhs, Aggregate) else query.rhs
    return {
        "query": f"q{number}",
        "indices": list(labels.values()),
        "output": [labels[f] for f in query.lhs.idxs],
        "inputs": [
            {"tensor": node.tns.name, "indices": [labels[f] for f in node.idxs]}
            for node in PostOrderDFS(arg) if isinstance(node, Table)
        ],
    }


def _prepare(cases, seed, dry_run):
    prepared, names = [], set()
    for case in cases:
        if not isinstance(case, BenchmarkCase):
            raise TypeError("build_cases() must yield BenchmarkCase objects")
        if not case.name or case.name in names:
            raise ValueError(f"Benchmark names must be nonempty and unique: {case.name!r}")
        names.add(case.name)
        outputs = (
            case.expression if isinstance(case.expression, tuple) else (case.expression,)
        )
        if not outputs or any(not isinstance(x, fl.LazyTensor) for x in outputs):
            raise TypeError(f"{case.name}: return a lazy Finch expression or tuple of them")
        frontend_plan, plan = _discover(case.expression)
        queries = list(_loop_queries(plan))
        count = math.prod(
            math.factorial(len(_fields(query))) // (
                math.factorial(len(query.lhs.idxs)) if isinstance(query, QueryInto) else 1
            )
            for query in queries
        )
        pools = [] if dry_run else [_permutations(query) for query in queries]
        rng = random.Random(seed)
        for pool in pools:
            rng.shuffle(pool)
        layout = [_layout(query, i) for i, query in enumerate(queries, 1)]
        metadata = json.dumps(case.metadata, sort_keys=True)
        print(
            f"{case.name}: {len(queries)} loop queries; {count} schedules; layout={layout}",
            flush=True,
        )
        prepared.append((case, frontend_plan, pools, count, json.dumps(layout), metadata))
    if not prepared:
        raise ValueError("build_cases() returned no benchmarks")
    return prepared


def run(cases, *, out, runs=20, warmup=1, seed=42, dry_run=False):
    """Enumerate all legal order combinations for each BenchmarkCase."""
    if runs < 1 or warmup < 0:
        raise ValueError("runs must be positive and warmup must be nonnegative")
    with _input_statistics():
        prepared = _prepare(cases, seed, dry_run)
    total = sum(item[3] for item in prepared)
    print(
        f"{total} schedules; {warmup} warmup; {runs} cached executions; seed={seed}",
        flush=True,
    )
    if dry_run:
        return

    jl = get_jl()
    if jl is None:
        raise RuntimeError("The Julia backend is required; install finch-tensor[julia].")
    fields = (
        "benchmark", "metadata", "query_layout", "loop_order", "executed_order",
        "est_cost", "seconds", "plan_seconds", "median_seconds",
        "filtered_mean_seconds", "filtered_run_count",
        *(f"run_{i}_seconds" for i in range(1, runs + 1)),
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        file.flush()
        progress = 0
        for case, frontend_plan, pools, _, layout, metadata in prepared:
            measurements = []
            for orders in itertools.product(*pools):
                progress += 1
                description = _describe(orders)
                print(f"[{progress}/{total}] {case.name} {description}", flush=True)
                recorder = _Recorder()
                ctx = scheduler(_FixedOrders(recorder, orders))
                print("    planning + initial execution", flush=True)
                start = time.perf_counter()
                ctx(frontend_plan)
                plan_seconds = time.perf_counter() - start - recorder.initial_seconds
                gc.collect()
                jl.seval("GC.gc()")
                for _ in range(warmup):
                    print("    warmup", flush=True)
                    recorder.execute()
                samples = []
                for i in range(1, runs + 1):
                    start = time.perf_counter()
                    result = recorder.execute()
                    samples.append(time.perf_counter() - start)
                    del result
                    print(
                        f"    timed run {i}/{runs}: {samples[-1] * 1000:.3f} ms",
                        flush=True,
                    )
                q1, q3 = np.percentile(samples, [25, 75])
                cutoff = max(3 * np.median(samples), q3 + 3 * (q3 - q1))
                filtered = [s for s in samples if s <= cutoff]
                measurements.append((recorder.cost, float(np.mean(samples))))
                writer.writerow(
                    {
                        "benchmark": case.name,
                        "metadata": metadata,
                        "query_layout": layout,
                        "loop_order": description,
                        "executed_order": description,
                        "est_cost": recorder.cost,
                        "seconds": float(np.mean(samples)),
                        "plan_seconds": plan_seconds,
                        "median_seconds": float(np.median(samples)),
                        "filtered_mean_seconds": float(np.mean(filtered)),
                        "filtered_run_count": len(filtered),
                        **{f"run_{i}_seconds": s for i, s in enumerate(samples, 1)},
                    }
                )
                file.flush()
                print(
                    f"    mean {np.mean(samples) * 1000:.3f} ms; cost {recorder.cost:.6g}",
                    flush=True,
                )
            finite = [
                (cost, seconds) for cost, seconds in measurements
                if np.isfinite(cost) and np.isfinite(seconds)
            ]
            rho = float("nan")
            if (
                len({cost for cost, _ in finite}) > 1
                and len({seconds for _, seconds in finite}) > 1
            ):
                rho = float(spearmanr(*zip(*finite, strict=True)).statistic)
            print(
                f"{case.name}: Spearman rho={rho:.6g} "
                f"({len(finite)}/{len(measurements)} finite cost/runtime pairs; raw means)",
                flush=True,
            )
    print(f"wrote {out}", flush=True)
