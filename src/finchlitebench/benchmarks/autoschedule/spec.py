"""Benchmark specs: each file in queries/ and queries_lite/ defines a SPEC."""

from dataclasses import dataclass
import importlib
from pathlib import Path
import pkgutil
from typing import Callable

from finch.autoschedule import default_schedulers
from finch.autoschedule.tensor_stats import UniformStatsFactory


RESULTS_DIR = Path(__file__).resolve().parent / "results"

_LITE_RUN = {
    "scheduler_warmups": 1,
    "scheduler_runs": 3,
    "end_to_end_warmups": 1,
    "end_to_end_runs": 2,
}


@dataclass
class BenchmarkCase:
    name: str
    shape: tuple[int, ...]
    pattern: str
    seed: int = 0
    # Fraction of nonzeros; None uses the pattern's default (see queries/common.py).
    density: float | None = None


@dataclass
class BenchmarkSpec:
    name: str
    query: Callable
    inputs: Callable[[BenchmarkCase], tuple]
    cases: tuple[BenchmarkCase, ...]
    end_to_end_cases: tuple[str, ...]
    # A zero-argument stats factory, e.g. ``UniformStatsFactory``.
    stats: Callable = UniformStatsFactory
    # A Finch scheduler; only its formatter and backend are used.
    pipeline: object = default_schedulers.COMPILE_JULIA_GALLEY
    loops: tuple[str, ...] = ("default", "greedy", "bfs", "dfs")
    logics: tuple[str, ...] = ("default", "galley-greedy", "galley-bfs", "galley-dfs")
    modes: tuple[str, ...] = ("scheduler", "end-to-end")
    scheduler_warmups: int = 2
    scheduler_runs: int = 3
    end_to_end_warmups: int = 1
    end_to_end_runs: int = 2


def cases_from_shapes(shapes, patterns=("dense", "sparse"), seed=0):
    """One case per shape name and density pattern."""
    return tuple(
        BenchmarkCase(f"{name}-{pattern}", shape, pattern, seed=seed)
        for name, shape in shapes.items()
        for pattern in patterns
    )


def make_spec(
    name,
    query,
    inputs,
    cases,
    *,
    end_to_end_cases=None,
    lite=False,
    **overrides,
):
    """Build a BenchmarkSpec; end_to_end_cases defaults to every case."""
    if end_to_end_cases is None:
        end_to_end_cases = tuple(case.name for case in cases)
    settings = dict(_LITE_RUN) if lite else {}
    settings.update(overrides)
    return BenchmarkSpec(
        name=name,
        query=query,
        inputs=inputs,
        cases=tuple(cases),
        end_to_end_cases=tuple(end_to_end_cases),
        **settings,
    )


def load_benchmark(name: str) -> BenchmarkSpec:
    """Find ``name`` among every ``SPEC`` under ``queries/`` and ``queries_lite/``."""
    benchmarks = {}
    for package_name in (f"{__package__}.queries", f"{__package__}.queries_lite"):
        package = importlib.import_module(package_name)
        for info in pkgutil.walk_packages(package.__path__, f"{package_name}."):
            spec = getattr(importlib.import_module(info.name), "SPEC", None)
            if isinstance(spec, BenchmarkSpec):
                benchmarks[spec.name] = spec
    if name not in benchmarks:
        raise ValueError(f"unknown benchmark {name!r}; known: {sorted(benchmarks)}")
    return benchmarks[name]
