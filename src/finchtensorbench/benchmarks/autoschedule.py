"""Configurable Finch schedulers with per-stage timers.

Ported from the ``autoscheduler-benchmarks`` branch and updated for
finch-tensor-lite ``kbd/workspace-loop-orderer``.  A scheduler is assembled
from these choices:

* logic: ``default`` (DefaultLogicFactorizer) or ``galley-{greedy,bfs,dfs}``
* loop: ``default``, ``greedy``, ``bfs`` or ``dfs`` loop orderer
* formatter: ``fd`` (FDFormatter), ``galley`` (GalleyFormatter with cost-based
  protocols), ``cost`` (CostFormatter with cost-based protocols) or ``dense``
  (DefaultLogicFormatter)
* stats: ``dc`` (memoised degree constraints), ``lp`` (memoised LpBound
  statistics), ``sampling`` (memoised sampling statistics, randomly seeded),
  ``fd`` or ``uniform``
* workspace: whether a LogicWorkspacer sits between loop ordering and
  formatting (spelled as a trailing ``/ws``)

The two stock Julia pipelines are the presets ``default``
(``finch.autoschedule.default_schedulers.COMPILE_JULIA``) and ``galley``
(``COMPILE_JULIA_GALLEY``, which includes the workspacer).

Every stage is wrapped in a timer that records *exclusive* time, so a stage's
time excludes the downstream stages it invokes:

* ``stats_s``: computing statistics of the input tensors
* ``logic_s``: logical (join/aggregate) optimization, plus LogicSimplify
* ``loop_s``: loop ordering
* ``workspace_s``: workspace insertion (0 when there is no workspacer)
* ``lower_s``: formatting, protocol selection, code generation and, on a
  kernel-cache miss, Julia compilation
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from finch.autoschedule.compiler import CompilerFormLowerer, LogicCompiler
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.factorizer.galley_factorizer.galley_optimize import (
    GalleyLogicFactorizer,
)
from finch.autoschedule.factorizer.optimize import DefaultLogicFactorizer
from finch.autoschedule.formatter import (
    DefaultLogicFormatter,
    FDFormatter,
    GalleyFormatter,
)
from finch.autoschedule.formatter.smart_formatter import CostFormatter
from finch.autoschedule.loop_orderer.loop_order_bnb import (
    BFSLoopOrderer,
    DFSLoopOrderer,
)
from finch.autoschedule.loop_orderer.loop_order_greedy import GreedyLoopOrderer
from finch.autoschedule.loop_orderer.loop_ordering import DefaultLoopOrderer
from finch.autoschedule.normalize import LogicNormalizer
from finch.autoschedule.protocolizer import CostProtocolizer

try:  # wma/dropfill: drop the fills of aggregates over dense loops
    from finch.autoschedule.drop_fills import LogicFillDropper
except ImportError:  # finch-tensor without drop_fills
    LogicFillDropper = None
from finch.autoschedule.workspacer import LogicWorkspacer

try:  # finch-tensor kbd/subgraph-fixes: price workspaces with Galley's formats
    from finch.autoschedule.workspacer import galley_format_cost
except ImportError:  # older finch-tensor
    galley_format_cost = None
from finch.autoschedule.tensor_stats import (
    DCStatsFactory,
    FDStatsFactory,
    LPStatsFactory,
    SamplingStatsFactory,
    UniformStatsFactory,
)
from finch.autoschedule.tensor_stats.memo_stats import MemoStatsFactory
from finch.compile_jl.compiler import FinchJLCompiler
from finch.finch_logic import Aggregate, LogicSimplify, Reorder

LOGIC_ORDERERS = ("default", "galley-greedy", "galley-bfs", "galley-dfs")
LOOP_ORDERERS = {
    "default": DefaultLoopOrderer,
    "greedy": GreedyLoopOrderer,
    "bfs": BFSLoopOrderer,
    "dfs": DFSLoopOrderer,
}
FORMATTERS = ("fd", "galley", "cost", "dense")
STATS = {
    "dc": lambda: MemoStatsFactory(DCStatsFactory()),
    # LpBound: degree-sequence norms p in (0, 1, 2, 3, 4, 5, inf), LP-flow bound
    "lp": lambda: MemoStatsFactory(LPStatsFactory()),
    # defaults: sample_nnz=10_000, estimator="sh3"
    "sampling": lambda: MemoStatsFactory(SamplingStatsFactory()),
    "fd": FDStatsFactory,
    "uniform": UniformStatsFactory,
}
STAGES = ("stats", "logic", "loop", "workspace", "lower")


@dataclass(frozen=True)
class SchedulerConfig:
    logic: str
    loop: str
    formatter: str
    stats: str
    workspace: bool = False

    def __post_init__(self):
        for value, choices, axis in (
            (self.logic, LOGIC_ORDERERS, "logic"),
            (self.loop, tuple(LOOP_ORDERERS), "loop"),
            (self.formatter, FORMATTERS, "formatter"),
            (self.stats, tuple(STATS), "stats"),
        ):
            if value not in choices:
                raise ValueError(f"unknown {axis} {value!r}; choose from {choices}")

    @property
    def name(self) -> str:
        name = f"{self.logic}/{self.loop}/{self.formatter}/{self.stats}"
        return name + "/ws" if self.workspace else name


PRESETS = {
    "default": SchedulerConfig("default", "default", "fd", "fd"),
    "galley": SchedulerConfig("galley-bfs", "bfs", "galley", "dc", workspace=True),
    # galley with formats searched under the protocol cost model, as the
    # workspacer prices them, instead of chosen by Galley's rules
    "galley-cost": SchedulerConfig("galley-bfs", "bfs", "cost", "dc", workspace=True),
    # galley with LpBound statistics instead of degree constraints
    "galley-lp": SchedulerConfig("galley-bfs", "bfs", "galley", "lp", workspace=True),
    # galley with sampling statistics instead of degree constraints
    "galley-sampling": SchedulerConfig(
        "galley-bfs", "bfs", "galley", "sampling", workspace=True
    ),
}


def parse_config(text: str) -> SchedulerConfig:
    """A preset name or ``logic/loop/formatter/stats[/ws]``."""
    if text in PRESETS:
        return PRESETS[text]
    parts = text.split("/")
    workspace = parts[-1] == "ws"
    if workspace:
        parts = parts[:-1]
    if len(parts) != 4:
        raise ValueError(
            f"config {text!r} must be a preset {tuple(PRESETS)} or "
            "logic/loop/formatter/stats[/ws]"
        )
    logic, loop, formatter, stats = parts
    return SchedulerConfig(logic, loop, formatter, stats, workspace=workspace)


def config_label(config: SchedulerConfig) -> str:
    for name, preset in PRESETS.items():
        if preset == config:
            return name
    return config.name


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class StageTimer:
    """Exclusive stage timer: entering a stage pauses the enclosing one."""

    def __init__(self):
        self.times = dict.fromkeys(STAGES, 0.0)
        self._stack: list[str] = []
        self._mark = 0.0

    def reset(self):
        self.times = dict.fromkeys(STAGES, 0.0)
        self._stack.clear()

    def _charge(self, now):
        if self._stack:
            self.times[self._stack[-1]] += now - self._mark
        self._mark = now

    def start(self, name):
        self._charge(time.perf_counter())
        self._stack.append(name)

    def stop(self):
        self._charge(time.perf_counter())
        self._stack.pop()


class _TimedStage:
    def __init__(self, inner, timer, name):
        self.inner, self.timer, self.name = inner, timer, name

    def __call__(self, *args):
        self.timer.start(self.name)
        try:
            return self.inner(*args)
        finally:
            self.timer.stop()

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _CaptureStage:
    """Remember the last program a stage received, and report it (with the
    tensor formats bound so far) to ``on_program(slot, program, bindings)``
    before the stage runs, so a plan is visible even if its kernel never
    finishes."""

    def __init__(self, inner, slot, store, on_program=None):
        self.inner, self.slot, self.store = inner, slot, store
        self.on_program = on_program

    def __call__(self, program, *args):
        self.store[self.slot] = program
        if self.on_program is not None:
            self.on_program(self.slot, program, args[0] if args else {})
        return self.inner(program, *args)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _TimedExecutor(LogicExecutor):
    """Charge input-statistics computation (and cache lookup) to ``stats``."""

    timer: StageTimer

    def _load_cached(self, *args):
        self.timer.start("stats")
        try:
            return super()._load_cached(*args)
        finally:
            self.timer.stop()


PLAN_STAGES = {
    "logic": "after logic optimization (loop orderer input)",
    "loop": "after loop ordering",
    "workspace": "after workspacer (formatter input)",
    "compiler": "after formatting + protocols (compiler input)",
}


def build_scheduler(config: SchedulerConfig, cache: bool = True, on_program=None):
    """Assemble a scheduler; ``ctx.timer`` and ``ctx.captures`` expose timings.

    ``on_program(stage, program, bindings)``, if given, sees the program
    entering each stage of :data:`PLAN_STAGES` (``workspace`` only when the
    config has a workspacer); ``bindings`` maps aliases to formats.
    """
    timer, captures = StageTimer(), {}

    def tap(inner, slot):
        return _CaptureStage(inner, slot, captures, on_program)

    lowerer = tap(CompilerFormLowerer(LogicCompiler(FinchJLCompiler())), "compiler")
    if config.formatter == "fd":
        formatter = FDFormatter(lowerer)
    elif config.formatter in ("galley", "cost"):
        protocolizer = CostProtocolizer(lowerer)
        if LogicFillDropper is not None:
            # As COMPILE_JULIA_GALLEY: drop fills between formats and protocols.
            protocolizer = LogicFillDropper(protocolizer)
        if config.formatter == "galley":
            formatter = GalleyFormatter(protocolizer)
        else:
            formatter = CostFormatter(protocolizer)
    else:
        formatter = DefaultLogicFormatter(lowerer)
    formatter = _TimedStage(formatter, timer, "lower")
    if config.workspace:
        formatter = tap(formatter, "workspace")
        # Like COMPILE_JULIA_GALLEY, price workspaces with the formats the
        # formatter downstream chooses, where finch can.
        options = {}
        if config.formatter == "galley" and galley_format_cost is not None:
            options["format_cost"] = galley_format_cost
        formatter = _TimedStage(LogicWorkspacer(formatter, **options), timer, "workspace")
    formatter = tap(formatter, "loop")
    loop = _TimedStage(LOOP_ORDERERS[config.loop](formatter), timer, "loop")
    simplified = tap(LogicSimplify(loop), "logic")
    if config.logic == "default":
        logic = DefaultLogicFactorizer(simplified)
    else:
        logic = GalleyLogicFactorizer(
            simplified, optimizer=config.logic.removeprefix("galley-")
        )
    executor = _TimedExecutor(
        _TimedStage(logic, timer, "logic"),
        stats_factory=STATS[config.stats](),
        cache=cache,
    )
    executor.timer = timer
    context = LogicNormalizer(executor)
    context.timer, context.captures = timer, captures
    return context


# ---------------------------------------------------------------------------
# Plan description
# ---------------------------------------------------------------------------


def _field_names(fields):
    return ",".join(getattr(field, "name", str(field)) for field in fields)


def _queries(program):
    bodies = getattr(program, "bodies", None)
    if bodies is None:
        yield program
        return
    for body in bodies:
        if hasattr(body, "bodies"):
            yield from _queries(body)
        else:
            yield body


def describe_orders(program, loop_order: bool) -> str:
    """Aggregation indices of each query in *program* (logic) or loop orders."""
    orders = []
    for node in _queries(program) if program is not None else ():
        rhs = getattr(node, "rhs", None)
        if isinstance(rhs, Reorder):
            rhs = rhs.arg
        if not isinstance(rhs, Aggregate):
            continue
        if loop_order and isinstance(rhs.arg, Reorder):
            orders.append(_field_names(rhs.arg.idxs))
        elif not loop_order:
            orders.append(_field_names(rhs.idxs))
    return (" | " if loop_order else " -> ").join(orders) or "-"
