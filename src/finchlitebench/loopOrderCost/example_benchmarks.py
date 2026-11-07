"""Copy this file and replace the yielded expressions with your benchmarks."""

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
OUT = Path("loop_order_cost.csv")
DRY_RUN = False


def build_cases():
    rng = np.random.default_rng(SEED)
    matrices = []
    for _ in range(3):
        matrix = sps.random(SIZE, SIZE, density=DENSITY, format="csr", random_state=rng)
        # Julia writes integer indices when constructing sparse results.
        matrix.indices = matrix.indices.astype(np.int64, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int64, copy=False)
        matrices.append(fl.defer(fl.asarray(matrix)))
    A, B, C = matrices
    metadata = {"size": SIZE, "density": DENSITY}
    yield BenchmarkCase("sddmm", fl.einsum("ij,jk,ik->ik", A, B, C), metadata)
    yield BenchmarkCase("matmul", A @ B, metadata)
    yield BenchmarkCase("row_sum", fl.sum(A * C, axis=1), metadata)


def main():
    if SIZE < 1:
        raise ValueError("SIZE must be positive")
    run(
        build_cases(), out=OUT, runs=RUNS, warmup=WARMUP,
        seed=SEED, dry_run=DRY_RUN,
    )


if __name__ == "__main__":
    main()
