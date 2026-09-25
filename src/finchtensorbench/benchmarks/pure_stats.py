import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import finch as ft
import numpy as np
import scipy.io
from finch.autoschedule import COMPILE_JULIA, with_default_scheduler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "tests" / "data"

MATRIX: dict[str, str] = {
    "ca-GrQc": "SNAP",
    "ca-HepPh": "SNAP",
    "ca-HepTh": "SNAP",
    "web-NotreDame": "SNAP",
    "bcsstk39": "Boeing",
    "ct20stif": "Boeing",
}

def act_hadamard(a, b):
    return ft.count_nonzero(ft.multiply(a, b))


def act_spgemm(a, b):
    return ft.count_nonzero(ft.matmul(a, b))


def act_spgemm2(a, b):
    return ft.count_nonzero(ft.matmul(ft.matmul(a, b), b))


def act_triangle(a):
    return ft.count_nonzero(ft.multiply(ft.matmul(a, a), a))



KERNELS: dict[str, tuple[Callable[..., int], int]] = {
    "hadamard": (act_hadamard, 2),
    "spgemm": (act_spgemm, 2),
    "spgemm-2": (act_spgemm2, 2),
    "triangle": (act_triangle, 1),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=MATRIX, default="ct20stif")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--pause", type=float, default=5.0, help="seconds between kernel tests")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable Finch's compiled-kernel cache, forcing a fresh Julia "
        "compile on every call instead of just the warm-up",
    )
    args = parser.parse_args()

    if args.no_cache:
        COMPILE_JULIA.ctx.cache = False

    path = DATA_DIR / f"{args.matrix}.mat"
    data = scipy.io.loadmat(path)
    matrix = data["Problem"]["A"][0, 0].tocsr()
    mat = matrix.astype(bool).astype(np.int64)
    mat.eliminate_zeros()
    mat.indices = mat.indices.astype(np.int64)
    mat.indptr = mat.indptr.astype(np.int64)

    print(
        f"matrix={MATRIX[args.matrix]}/{args.matrix} "
        f"shape={matrix.shape} nnz={matrix.nnz} "
        f"repeat={args.repeat}"
    )
    for index, (name, (function, operand_count)) in enumerate(KERNELS.items()):
        if index:
            time.sleep(args.pause)

        operands = [mat] * operand_count

        # Warm up
        if not args.no_cache:
            with with_default_scheduler(COMPILE_JULIA):
                function(*operands)

        result = None
        timings = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            with with_default_scheduler(COMPILE_JULIA):
                result = function(*operands)
            timings.append(time.perf_counter() - start)

        print(
            f"{name:10} result={result} "
            f"min={min(timings):.6f}s "
            f"mean={statistics.fmean(timings):.6f}s "
            f"median={statistics.median(timings):.6f}s"
        )


if __name__ == "__main__":
    main()
