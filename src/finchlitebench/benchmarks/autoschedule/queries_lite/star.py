"""Minute-scale version of the star benchmark."""

from pathlib import Path
import sys

if __package__:
    from ..initialization import BenchmarkCase, BenchmarkSpec, run_cli
    from ..queries.star import star_inputs, star_query
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from initialization import BenchmarkCase, BenchmarkSpec, run_cli
    from queries.star import star_inputs, star_query


CASES = (
    BenchmarkCase("balanced-dense", (6, 6, 6, 6), "dense"),
    BenchmarkCase("balanced-sparse", (8, 8, 8, 8), "sparse"),
    BenchmarkCase("narrow-center-sparse", (2, 12, 12, 12), "sparse"),
)

SPEC = BenchmarkSpec(
    name="star_lite",
    module="finchlitebench.benchmarks.autoschedule.queries_lite.star",
    query=star_query,
    inputs=star_inputs,
    cases=CASES,
    scheduler_cases=tuple(case.name for case in CASES),
    end_to_end_cases=("narrow-center-sparse",),
    results_dir=Path(__file__).resolve().parents[1] / "results",
    rounds=1,
    scheduler_warmups=1,
    scheduler_runs=3,
    end_to_end_warmups=1,
    end_to_end_runs=3,
)


if __name__ == "__main__":
    run_cli(SPEC)
