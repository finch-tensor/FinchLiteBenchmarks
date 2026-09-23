"""Minute-scale version of the diamond benchmark."""

from pathlib import Path
import sys

if __package__:
    from ..initialization import BenchmarkCase, BenchmarkSpec, run_cli
    from ..queries.diamond import diamond_inputs, diamond_query
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from initialization import BenchmarkCase, BenchmarkSpec, run_cli
    from queries.diamond import diamond_inputs, diamond_query


CASES = (
    BenchmarkCase("balanced-dense", (6, 6, 6, 6), "dense"),
    BenchmarkCase("balanced-sparse", (8, 8, 8, 8), "sparse"),
    BenchmarkCase("narrow-k-sparse", (8, 8, 2, 12), "sparse"),
)

SPEC = BenchmarkSpec(
    name="diamond_lite",
    module="finchlitebench.benchmarks.autoschedule.queries_lite.diamond",
    query=diamond_query,
    inputs=diamond_inputs,
    cases=CASES,
    scheduler_cases=tuple(case.name for case in CASES),
    end_to_end_cases=("narrow-k-sparse",),
    results_dir=Path(__file__).resolve().parents[1] / "results",
    rounds=1,
    scheduler_warmups=1,
    scheduler_runs=3,
    end_to_end_warmups=1,
    end_to_end_runs=3,
)


if __name__ == "__main__":
    run_cli(SPEC)
