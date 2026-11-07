"""Benchmark K4: six edge matrices connecting four indices a, b, c, d."""

from pathlib import Path

import numpy as np
import scipy.sparse as sps

import finch as fl
from finchlitebench.loopOrderCost import BenchmarkCase
from finchlitebench.loopOrderCost.runner import run

SIZE = 64
DENSITY = 0.2
RUNS = 20
WARMUP = 1
SEED = 42
OUT = Path("clique4.csv")
DRY_RUN = False


def build_cases():
    rng = np.random.default_rng(SEED)
    matrices = []
    for _ in range(6):
        matrix = sps.random(
            SIZE, SIZE, density=DENSITY, format="csr", random_state=rng,
            data_rvs=np.ones, dtype=np.float64,
        )
        matrix.indices = matrix.indices.astype(np.int64, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int64, copy=False)
        matrices.append(matrix)
    inputs = [fl.defer(fl.asarray(matrix)) for matrix in matrices]
    yield BenchmarkCase(
        "clique4",
        fl.einsum("ab,ac,ad,bc,bd,cd->", *inputs),
        {"size": SIZE, "density": DENSITY, "nnz": [m.nnz for m in matrices]},
    )


def main():
    if SIZE < 1:
        raise ValueError("SIZE must be positive")
    run(
        build_cases(), out=OUT, runs=RUNS, warmup=WARMUP,
        seed=SEED, dry_run=DRY_RUN,
    )


if __name__ == "__main__":
    main()
