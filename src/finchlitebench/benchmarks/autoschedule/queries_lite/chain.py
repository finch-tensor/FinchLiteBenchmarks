"""Minute-scale version of the chain benchmark."""

from pathlib import Path
import sys

if __package__:
    from ..initialization import BenchmarkCase, BenchmarkSpec, run_cli
    from ..queries.chain import chain_inputs, chain_query
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from initialization import BenchmarkCase, BenchmarkSpec, run_cli
    from queries.chain import chain_inputs, chain_query


CASES = (
    BenchmarkCase("balanced-dense", (8, 8, 8, 8), "dense"),
    BenchmarkCase("balanced-sparse", (8, 8, 8, 8), "sparse"),
    BenchmarkCase("bottleneck-sparse", (8, 2, 16, 8), "sparse"),
)

SPEC = BenchmarkSpec(
    name="chain_lite",
    module="finchlitebench.benchmarks.autoschedule.queries_lite.chain",
    query=chain_query,
    inputs=chain_inputs,
    cases=CASES,
    scheduler_cases=tuple(case.name for case in CASES),
    end_to_end_cases=("bottleneck-sparse",),
    results_dir=Path(__file__).resolve().parents[1] / "results",
    rounds=1,
    scheduler_warmups=1,
    scheduler_runs=3,
    end_to_end_warmups=1,
    end_to_end_runs=3,
)


if __name__ == "__main__":
    run_cli(SPEC)
