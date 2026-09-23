"""Minute-scale version of the triangle benchmark."""

from pathlib import Path
import sys

if __package__:
    from ..initialization import BenchmarkCase, BenchmarkSpec, run_cli
    from ..queries.triangle import triangle_inputs, triangle_query
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from initialization import BenchmarkCase, BenchmarkSpec, run_cli
    from queries.triangle import triangle_inputs, triangle_query


CASES = (
    BenchmarkCase("dense-4", (4, 4), "dense"),
    BenchmarkCase("sparse-8", (8, 8), "sparse"),
    BenchmarkCase("skewed-8", (8, 8), "skewed"),
)

SPEC = BenchmarkSpec(
    name="triangle_lite",
    module="finchlitebench.benchmarks.autoschedule.queries_lite.triangle",
    query=triangle_query,
    inputs=triangle_inputs,
    cases=CASES,
    scheduler_cases=tuple(case.name for case in CASES),
    end_to_end_cases=("skewed-8",),
    results_dir=Path(__file__).resolve().parents[1] / "results",
    rounds=1,
    scheduler_warmups=1,
    scheduler_runs=3,
    end_to_end_warmups=1,
    end_to_end_runs=3,
)


if __name__ == "__main__":
    run_cli(SPEC)
