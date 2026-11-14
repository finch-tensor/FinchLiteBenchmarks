"""Inspect the actual Finch pipeline during an untimed benchmark warmup."""

from copy import copy

from finch.algebra.tensor import TensorFType
from finch.autoschedule import LogicExecutor, LogicNormalizer
from finch.autoschedule.executor import extract_tensors
from finch.autoschedule.workspacer import (
    LogicWorkspacer,
    add_cost_effective_reformats,
)

try:
    from finch.autoschedule.workspacer import add_accumulators
except ImportError:  # finch versions before workspace accumulators
    add_accumulators = None
from finch.compile_jl.compiler import FinchJLCompiler

from finchtensorbench.benchmarks.microbenchmarks.util.trace_stats import format_config, format_stats


class SchedulerTrace:
    def __init__(self):
        self.events = []

    def record_stage(self, stage, program, *args, **kwargs):
        sections = []
        config = format_config(stage)
        if config:
            sections.append(config)
        bindings = args[0] if args else kwargs.get("bindings", {})
        if isinstance(stage, (LogicNormalizer, LogicExecutor)):
            # Alias concrete inputs for display without printing their values or
            # changing the program that the scheduler actually receives.
            program, bindings = extract_tensors(program, dict(bindings or {}))
        sections.append(str(program))
        if bindings:
            lines = ["Tensor bindings / storage formats:"]
            for alias, tensor in bindings.items():
                shape = None if isinstance(tensor, TensorFType) else tensor.shape
                shape_text = f"shape={shape}, " if shape is not None else ""
                tensor_type = tensor if isinstance(tensor, TensorFType) else tensor.ftype
                lines.append(f"  {alias}: {shape_text}{tensor_type}")
            sections.append("\n".join(lines))
        if len(args) >= 2 and args[1]:
            sections.append("Planning statistics:\n" + format_stats(args[1]))
        self.events.append((f"Input to {type(stage).__name__}", "\n\n".join(sections)))

    def record_julia(self, library):
        for name, kernel in library.kernel_dict.items():
            self.events.append((f"Generated Julia: {name}", kernel.jl_code))

    def report(self, name, label):
        intro = [f"Scheduling report for {name}: {label}"]
        intro.append("Pipeline: " + " -> ".join(
            heading.removeprefix("Input to ")
            for heading, _ in self.events if heading.startswith("Input to ")
        ))
        intro.append(
            "Costs below are planner estimates, not benchmark seconds. "
            "Search entries describe candidates actually visited; pruned branches "
            "are not expanded into complete plans."
        )
        intro.append("The Julia section is generated source before Julia's native compilation.")
        sections = ["\n".join(intro)]
        for index, (heading, body) in enumerate(self.events, start=1):
            sections.append(f"[{index}] {heading}\n{body}")
        return "\n\n".join(sections) + "\n"


class CapturedStage:
    def __init__(self, stage, trace):
        self.stage = stage
        self.trace = trace

    def __getattr__(self, name):
        return getattr(self.stage, name)

    def __call__(self, program, *args, **kwargs):
        self.trace.record_stage(self.stage, program, *args, **kwargs)
        result = self.stage(program, *args, **kwargs)
        if isinstance(self.stage, FinchJLCompiler):
            self.trace.record_julia(result)
        return result


class ReformatOnlyWorkspacer(LogicWorkspacer):
    """LogicWorkspacer.lower with only add_cost_effective_workspaces skipped: the
    same input reformats and, where the workspacer adds them, the same
    accumulators, so the two modes differ only in workspace insertion."""

    def lower(self, program, bindings, stats, stats_factory):
        program = add_cost_effective_reformats(
            program,
            bindings,
            stats,
            stats_factory,
            self.cost_model,
            self.options,
            self.format_cost,
        )
        if add_accumulators is not None and getattr(self, "accumulators", False):
            program, formats = add_accumulators(program, bindings)
            bindings = {**bindings, **formats}
        return self.ctx(program, bindings, stats, stats_factory)


def configured_scheduler(preset, *, workspaces=None):
    """Clone a preset with fresh kernel caches and no trace instrumentation.

    None retains the preset. False skips only the workspacer's workspace
    insertion, keeping its reformats and accumulators. Loop search, fusion,
    statistics, cost models, and the input/output format planners keep the
    preset's configuration.
    """

    def clone(stage):
        copied = copy(stage)
        if isinstance(stage, LogicWorkspacer) and workspaces is not None and not workspaces:
            # The same workspacer, with every attribute it was configured with,
            # whose lower skips only workspace insertion.
            copied.__class__ = ReformatOnlyWorkspacer
        if isinstance(copied, LogicExecutor):
            copied.cached_kernels = {}
        for attr in ("ctx", "ctx_lower", "ctx_load"):
            child = getattr(stage, attr, None)
            if child is not None:
                setattr(copied, attr, clone(child))
        return copied

    return clone(preset)


def capture_scheduler(preset, *, workspaces=None):
    """Observe the same configured clone used by the timing runner."""
    scheduler = configured_scheduler(preset, workspaces=workspaces)
    trace = SchedulerTrace()

    def instrument(stage):
        if isinstance(stage, LogicWorkspacer) and workspaces is not None:
            state = "enabled" if workspaces else "disabled"
            trace.events.append((
                f"Workspace planning {state}",
                "Cost-based accumulator workspace insertion is " + state
                + "; cost-based input reformats remain enabled. "
                "Loop order remains the autoscheduler's choice.",
            ))
        for attr in ("ctx", "ctx_lower", "ctx_load"):
            child = getattr(stage, attr, None)
            if child is not None:
                setattr(stage, attr, instrument(child))
        return CapturedStage(stage, trace)

    return instrument(scheduler), trace


def finish_capture(stage):
    """Remove observers while preserving the warmed scheduler and its cache."""
    if isinstance(stage, CapturedStage):
        stage = stage.stage
    for attr in ("ctx", "ctx_lower", "ctx_load"):
        child = getattr(stage, attr, None)
        if child is not None:
            setattr(stage, attr, finish_capture(child))
    return stage
