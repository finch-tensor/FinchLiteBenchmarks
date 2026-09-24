import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import scipy.io

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


def actual_hadamard(a, b):
    return max(a.multiply(b).nnz, 1)


def actual_spgemm(a, b):
    return max((a @ b).nnz, 1)


def actual_spgemm2(a, b):
    return max(((a @ b) @ b).nnz, 1)


def actual_triangle(a):
    return max((a @ a).multiply(a).nnz, 1)


KERNELS: dict[str, tuple[Callable[..., int], int]] = {
    "hadamard": (actual_hadamard, 2),
    "spgemm": (actual_spgemm, 2),
    "spgemm-2": (actual_spgemm2, 2),
    "triangle": (actual_triangle, 1),
}


def load_matrix(name: str):
    path = DATA_DIR / f"{name}.mat"
    mat = scipy.io.loadmat(path)
    return mat["Problem"]["A"][0, 0].tocsr()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=MATRIX, default="ct20stif")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--pause", type=float, default=5.0, help="seconds between kernel tests")
    return parser.parse_args()


def main():
    args = parse_args()
    matrix = load_matrix(args.matrix)
    pattern = matrix.astype(bool)
    pattern.eliminate_zeros()

    print(
        f"matrix={MATRIX[args.matrix]}/{args.matrix} "
        f"shape={matrix.shape} nnz={matrix.nnz} "
        f"repeat={args.repeat}"
    )
    for index, (name, (function, operand_count)) in enumerate(KERNELS.items()):
        if index:
            time.sleep(args.pause)

        operands = [pattern] * operand_count
        result = 0
        timings = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            result = function(*operands)
            timings.append(time.perf_counter() - start)

        print(
            f"{name:10} result={result:<12} "
            f"min={min(timings):.6f}s "
            f"mean={statistics.fmean(timings):.6f}s "
            f"median={statistics.median(timings):.6f}s"
        )


if __name__ == "__main__":
    main()
