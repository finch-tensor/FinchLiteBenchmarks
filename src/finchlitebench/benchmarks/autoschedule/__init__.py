"""Public API for the autoscheduler benchmark harness."""

from importlib import import_module

__all__ = [
    "BENCHMARK_MODULES",
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkSpec",
    "RAW_FIELDS",
    "cases_from_shapes",
    "configuration_order",
    "load_benchmark",
    "make_spec",
    "run_cli",
    "summarize_records",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.initialization"), name)
