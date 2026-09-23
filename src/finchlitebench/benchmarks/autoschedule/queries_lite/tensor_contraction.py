"""Minute-scale version of the tensor-contraction benchmark."""

from pathlib import Path
import sys

if __package__:
    from ..initialization import BenchmarkCase, BenchmarkSpec, run_cli
    from ..queries.tensor_contraction import contraction_inputs, contraction_query
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from initialization import BenchmarkCase, BenchmarkSpec, run_cli
    from queries.tensor_contraction import contraction_inputs, contraction_query


CASES = (
    BenchmarkCase("balanced-dense", (6, 6, 6, 6), "dense"),
    BenchmarkCase("balanced-sparse", (8, 8, 8, 8), "sparse"),
    BenchmarkCase("narrow-j-sparse", (8, 2, 12, 8), "sparse"),
)

SPEC = BenchmarkSpec(
    name="tensor_contraction_lite",
    module=(
        "finchlitebench.benchmarks.autoschedule.queries_lite.tensor_contraction"
    ),
    query=contraction_query,
    inputs=contraction_inputs,
    cases=CASES,
    scheduler_cases=tuple(case.name for case in CASES),
    end_to_end_cases=("narrow-j-sparse",),
    results_dir=Path(__file__).resolve().parents[1] / "results",
    rounds=1,
    scheduler_warmups=1,
    scheduler_runs=3,
    end_to_end_warmups=1,
    end_to_end_runs=3,
)


if __name__ == "__main__":
    run_cli(SPEC)
