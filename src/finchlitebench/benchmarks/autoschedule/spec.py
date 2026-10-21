"""Workload descriptions for the autoscheduler benchmarks.

A query file only needs an einsum, an input builder, a case list, and
``SPEC = make_spec(...)``.  Timing, subprocesses, and CSV output live in
``initialization.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Callable


_PACKAGE = "finchlitebench.benchmarks.autoschedule"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SAPS_QUERIES = {
    "chain_4": "chain",
    "star_4": "star",
    "tree_5": "tree",
    "cycle_5": "cycle",
    "clique_6": "clique",
    "petal_6": "petal",
    "flower_8": "flower",
    "graph_7": "graph",
}

BENCHMARK_MODULES = {
    "chain": f"{_PACKAGE}.queries.chain",
    "diamond": f"{_PACKAGE}.queries.diamond",
    "star": f"{_PACKAGE}.queries.star",
    "tensor_contraction": f"{_PACKAGE}.queries.tensor_contraction",
    "triangle": f"{_PACKAGE}.queries.triangle",
    "chain_lite": f"{_PACKAGE}.queries_lite.chain",
    "diamond_lite": f"{_PACKAGE}.queries_lite.diamond",
    "star_lite": f"{_PACKAGE}.queries_lite.star",
    "tensor_contraction_lite": f"{_PACKAGE}.queries_lite.tensor_contraction",
    "triangle_lite": f"{_PACKAGE}.queries_lite.triangle",
    **{f"saps_{name}": f"{_PACKAGE}.queries.saps.{mod}" for name, mod in SAPS_QUERIES.items()},
    **{
        f"saps_{name}_lite": f"{_PACKAGE}.queries_lite.saps.{mod}"
        for name, mod in SAPS_QUERIES.items()
    },
}

_LITE_RUN = {
    "rounds": 1,
    "scheduler_warmups": 1,
    "scheduler_runs": 3,
    "end_to_end_warmups": 1,
    "end_to_end_runs": 3,
}


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    shape: tuple[int, ...]
    pattern: str
    seed: int = 0


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    query: Callable
    inputs: Callable[[BenchmarkCase], tuple]
    cases: tuple[BenchmarkCase, ...]
    scheduler_cases: tuple[str, ...]
    end_to_end_cases: tuple[str, ...]
    results_dir: Path = RESULTS_DIR
    output_count: int = 1
    rounds: int = 2
    scheduler_warmups: int = 2
    scheduler_runs: int = 10
    end_to_end_warmups: int = 1
    end_to_end_runs: int = 3

    @property
    def cases_by_name(self):
        return {case.name: case for case in self.cases}


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
    end_to_end_cases,
    scheduler_cases=None,
    lite=False,
    **overrides,
):
    """Fill the repeated spec fields so query files stay short."""
    if scheduler_cases is None:
        scheduler_cases = tuple(case.name for case in cases)
    settings = dict(_LITE_RUN) if lite else {}
    settings.update(overrides)
    return BenchmarkSpec(
        name=name,
        query=query,
        inputs=inputs,
        cases=tuple(cases),
        scheduler_cases=tuple(scheduler_cases),
        end_to_end_cases=tuple(end_to_end_cases),
        **settings,
    )


def load_benchmark(name: str) -> BenchmarkSpec:
    """Load one registered benchmark without importing every workload."""
    try:
        module_name = BENCHMARK_MODULES[name]
    except KeyError:
        choices = ", ".join(BENCHMARK_MODULES)
        raise ValueError(f"unknown benchmark {name!r}; choose from: {choices}") from None
    spec = importlib.import_module(module_name).SPEC
    if spec.name != name:
        raise ValueError(
            f"benchmark module {module_name!r} defines {spec.name!r}, expected {name!r}"
        )
    return spec
