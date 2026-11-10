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
  ``vp`` (memoised value-propagation statistics), ``bu`` (memoised blocked
  uniform statistics), ``fd`` or ``uniform``
* workspace: whether a LogicWorkspacer sits between loop ordering and
  formatting, spelled as a trailing ``/ws`` (workspaces, then input
  reformats), ``/ws-only`` (workspaces only) or ``/rf`` (input reformats only)

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
from finch.autoschedule.workspacer.workspacer import (
    add_cost_effective_reformats,
    add_cost_effective_workspaces,
)

try:  # finch-tensor kbd/subgraph-fixes: price workspaces with Galley's formats
    from finch.autoschedule.workspacer import galley_format_cost
except ImportError:  # older finch-tensor
    galley_format_cost = None
from finch.autoschedule.tensor_stats import (
    BlockedUniformStatsFactory,
    DCStatsFactory,
    FDStatsFactory,
    LPStatsFactory,
    SamplingStatsFactory,
    UniformStatsFactory,
)
from finch.autoschedule.tensor_stats.memo_stats import MemoStatsFactory

try:  # value-propagation statistics
    from finch.autoschedule.tensor_stats import VPStatsFactory
except ImportError:  # finch-tensor without vp_stats
    VPStatsFactory = None
from finch.compile_jl.compiler import FinchJLCompiler
from finch.finch_logic import Aggregate, Fuse, LogicSimplify, Reorder

LOGIC_ORDERERS = ("default", "galley-greedy", "galley-bfs", "galley-dfs")
LOOP_ORDERERS = {
    "default": DefaultLoopOrderer,
    "greedy": GreedyLoopOrderer,
    "bfs": BFSLoopOrderer,
    "dfs": DFSLoopOrderer,
}
FORMATTERS = ("fd", "galley", "cost", "dense")
WORKSPACERS = ("ws", "ws-only", "rf")
STATS = {
    "dc": lambda: MemoStatsFactory(DCStatsFactory()),
    # LpBound: degree-sequence norms p in (0, 1, 2, 3, 4, 5, inf), LP-flow bound
    "lp": lambda: MemoStatsFactory(LPStatsFactory()),
    # defaults: sample_nnz=10_000, estimator="sh3"
    "sampling": lambda: MemoStatsFactory(SamplingStatsFactory()),
    "fd": FDStatsFactory,
    # value propagation (VPStatsFactory)
    "vp": lambda: MemoStatsFactory(VPStatsFactory()),
    "uniform": UniformStatsFactory,
    # blocked uniform: a 5x5 grid of block nnz counts (block_count=5,
    # block_width=5), uniform within each block
    "bu": lambda: MemoStatsFactory(BlockedUniformStatsFactory()),
}


def _symmetric(inner):
    """Memoised *inner* statistics wrapped to check each input for reverse
    symmetry, so loop ordering can read it reversed instead of transposing it
    (finch-tensor kbd/symmetric-stats: SymmetricStatsFactory)."""

    def factory():
        from finch.autoschedule.tensor_stats.symmetric_stats import (
            SymmetricStatsFactory,
        )

        return MemoStatsFactory(SymmetricStatsFactory(inner()))

    return factory


STATS |= {"dc+sym": _symmetric(DCStatsFactory), "lp+sym": _symmetric(LPStatsFactory)}
STAGES = ("stats", "logic", "loop", "workspace", "lower")


@dataclass(frozen=True)
class SchedulerConfig:
    logic: str
    loop: str
    formatter: str
    stats: str
    workspace: bool = False
    # with a workspace: "ws" stages workspaces then input reformats, as
    # LogicWorkspacer does; "ws-only" only workspaces; "rf" only reformats
    workspacer: str = "ws"

    def __post_init__(self):
        for value, choices, axis in (
            (self.logic, LOGIC_ORDERERS, "logic"),
            (self.loop, tuple(LOOP_ORDERERS), "loop"),
            (self.formatter, FORMATTERS, "formatter"),
            (self.stats, tuple(STATS), "stats"),
        ):
            if value not in choices:
                raise ValueError(f"unknown {axis} {value!r}; choose from {choices}")
        if self.workspacer not in WORKSPACERS:
            raise ValueError(
                f"unknown workspacer {self.workspacer!r}; choose from {WORKSPACERS}"
            )

    @property
    def name(self) -> str:
        name = f"{self.logic}/{self.loop}/{self.formatter}/{self.stats}"
        return f"{name}/{self.workspacer}" if self.workspace else name


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
    """A preset name or ``logic/loop/formatter/stats[/ws|/ws-only|/rf]``."""
    if text in PRESETS:
        return PRESETS[text]
    parts = text.split("/")
    workspacer = parts[-1] if parts[-1] in WORKSPACERS else None
    if workspacer:
        parts = parts[:-1]
    if len(parts) != 4:
        raise ValueError(
            f"config {text!r} must be a preset {tuple(PRESETS)} or "
            "logic/loop/formatter/stats[/ws|/ws-only|/rf]"
        )
    logic, loop, formatter, stats = parts
    return SchedulerConfig(
        logic,
        loop,
        formatter,
        stats,
        workspace=workspacer is not None,
        workspacer=workspacer or "ws",
    )


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


class _PartialWorkspacer(LogicWorkspacer):
    """LogicWorkspacer with its workspaces or its input reformats turned off."""

    def __init__(self, ctx, *, workspaces=True, reformats=True, **options):
        super().__init__(ctx, **options)
        self.workspaces, self.reformats = workspaces, reformats

    def lower(self, prgm, bindings, stats, stats_factory):
        args = (bindings, stats, stats_factory, self.cost_model, self.options,
                self.format_cost)  # fmt: skip
        if self.workspaces:
            prgm = add_cost_effective_workspaces(prgm, *args)
        if self.reformats:
            prgm = add_cost_effective_reformats(prgm, *args)
        return self.ctx(prgm, bindings, stats, stats_factory)


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
        workspacer = _PartialWorkspacer(
            formatter,
            workspaces=config.workspacer != "rf",
            reformats=config.workspacer != "ws-only",
            **options,
        )
        formatter = _TimedStage(workspacer, timer, "workspace")
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


def _queries(program, fused=()):
    """The queries of *program*, each with the fields of the ``Fuse`` loops
    around it, outermost first."""
    match program:
        case None:
            return
        case Fuse(idx, body):
            yield from _queries(body, (*fused, idx))
        case _ if hasattr(program, "bodies"):
            for body in program.bodies:
                yield from _queries(body, fused)
        case _:
            yield program, fused


def describe_orders(program, loop_order: bool) -> str:
    """The aggregated fields of each query in *program* (logic), or each
    query's loop order: its enclosing ``Fuse`` loops, then any loops of its
    own (a ``Reorder`` of the aggregated expression)."""
    orders = []
    for node, fused in _queries(program):
        rhs = getattr(node, "rhs", None)
        if isinstance(rhs, Reorder):
            rhs = rhs.arg
        if not isinstance(rhs, Aggregate):
            if loop_order and fused:
                orders.append(_field_names(fused))
            continue
        if loop_order:
            own = rhs.arg.idxs if isinstance(rhs.arg, Reorder) else ()
            orders.append(_field_names((*fused, *own)))
        else:
            orders.append(_field_names(rhs.idxs))
    return (" | " if loop_order else " -> ").join(orders) or "-"
