import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps
import torch

import scorch

from finchtensorbench.benchmarks.microbenchmarks.util import feature_matrix, paper_spgemm

ROWS = 5000
COLS = 5000
SEED = 42


def random_csr(shape, density, seed):
    return sps.csr_array(sps.random(
        *shape, density=density, format="csr",
        random_state=np.random.default_rng(seed), data_rvs=np.ones,
    ))


def to_scorch_csr(matrix, name, dtype):
    torch_csr = torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int32)),
        torch.from_numpy(matrix.indices.astype(np.int32)),
        torch.from_numpy(matrix.data).to(dtype),
        size=matrix.shape, device=torch.device("cpu"),
    )
    return scorch.from_csr(torch_csr, name)


def to_scorch_dense(array, name, dtype):
    return scorch.from_torch(torch.from_numpy(array).to(torch.device("cpu"), dtype), name)


######################### SDDMM ##########################

def sddmm(mask, a, b):
    return scorch.einsum("ij,ik,kj->ij", mask, a, b, format="dd")


def sddmm_inputs(k, mask, seed, dtype):
    rng = np.random.default_rng(seed + k)
    rows, cols = mask.shape
    a = rng.random((rows, k))
    b = rng.random((k, cols))
    return (
        to_scorch_csr(mask, "S", dtype),
        to_scorch_dense(a, "A", dtype),
        to_scorch_dense(b, "B", dtype),
    )


def sddmm_cases(dtype):
    results = []
    for density in (0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0):
        mask = random_csr((ROWS, COLS), density, SEED + 2)
        results.append((sddmm_inputs(128, mask, SEED, dtype), mask.nnz / (ROWS * COLS)))
    return results


######################### Chain Matmul ##########################

def matmul(a, b, c):
    return scorch.einsum("ij,jk,kl->il", a, b, c, format="dd")


def chain_cases(dtype):
    results = []
    shape = (ROWS, COLS)
    a = to_scorch_csr(random_csr(shape, 0.1, SEED), "A", dtype)
    b = to_scorch_csr(random_csr(shape, 0.1, SEED + 1), "B", dtype)
    for density in (0.00001, 0.0001, 0.001, 0.01, 0.1):
        c = random_csr(shape, density, SEED + 2)
        results.append(((a, b, to_scorch_csr(c, "C", dtype)), c.nnz / (ROWS * COLS)))
    return results


######################### Gustavson / Outer Product ##########################

def spgemm(at, bt):
    # C[i, j] = sum_k A[i, k] * B[j, k], i.e. A @ B.T; output rows are sparse.
    return scorch.einsum("ki,kj->ij", at, bt, format="ds")


def spgemm_cases(dtype):
    results = []
    for shared in (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536):
        a, b = paper_spgemm.make_inputs(shared)
        results.append(((to_scorch_csr(a.T, "A", dtype), to_scorch_csr(b.T, "B", dtype)), shared))
    return results


######################### Matrix Features ##########################

# Row Lp norms, p = 1..count. The values are nonnegative, so |x|^p = x^p. Scorch has no
# pow or root, so each version finishes with a PyTorch root on the length-R row sums.

def lp_root(total, p):
    total = total.to_torch()
    return total if p == 1 else total ** (1.0 / p)


def matrix_features_pow(powered, ones, count):
    """Version 1: the inputs carry x^p, precomputed on the stored values; Scorch sums rows."""
    return tuple(lp_root(scorch.einsum("ij,j->i", powered[p], ones, format="d"), p)
                 for p in feature_matrix.POWERS[:count])


def matrix_features_einsum(a, ones, count):
    """Version 2: Scorch computes sum_j x_ij^p as a p-way product einsum."""
    outputs = []
    for p in feature_matrix.POWERS[:count]:
        if p == 1:
            total = scorch.einsum("ij,j->i", a, ones, format="d")
        else:
            total = scorch.einsum(",".join(["ij"] * p) + "->i", *([a] * p), format="d")
        outputs.append(lp_root(total, p))
    return tuple(outputs)


def matrix_feature_cases(dtype):
    matrix = feature_matrix.make_matrix()
    ones = to_scorch_dense(np.ones(matrix.shape[1]), "x", dtype)
    a = to_scorch_csr(matrix, "A", dtype)
    powered = {p: to_scorch_csr(matrix.power(p), f"A{p}", dtype) for p in feature_matrix.POWERS}
    results = []
    for count in range(1, len(feature_matrix.POWERS) + 1):
        name = f"features_{count}"
        results.append((name, matrix_features_pow, (powered, ones, count), "pow on values"))
        results.append((name, matrix_features_einsum, (a, ones, count), "einsum product"))
    return results


######################### Main ##########################

def cases(dtype):
    benchmarks = {
        "sddmm": (sddmm, sddmm_cases),
        "chain": (matmul, chain_cases),
    }
    results = []
    for name, (function, make_cases) in benchmarks.items():
        for inputs, density in make_cases(dtype):
            results.append((name, function, inputs, density, ""))
    for inputs, shared_k in spgemm_cases(dtype):
        results.append(("spgemm", spgemm, inputs, shared_k, ""))
    for name, function, inputs, implementation in matrix_feature_cases(dtype):
        results.append((name, function, inputs, feature_matrix.DENSITY, implementation))
    return results


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("graph/scripts/scorch_result.csv"))
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("benchmark", "density", "active_k", "repeat", "seconds", "implementation"))
        for name, function, inputs, parameter, implementation in cases(torch.float32):
            is_spgemm = name == "spgemm"
            density = "" if is_spgemm else parameter
            active_k = parameter if is_spgemm else ""
            label = f"active_k={active_k}" if is_spgemm else f"density={density:.6g}"
            if implementation:
                label += f", {implementation}"
            print(f"Warming {name}: {label}", flush=True)
            function(*inputs)
            samples = []
            for trial in range(5):
                gc.collect()
                start = time.perf_counter()
                result = function(*inputs)
                seconds = time.perf_counter() - start
                del result
                writer.writerow((name, density, active_k, trial, seconds, implementation))
                output.flush()
                samples.append(seconds)
            median = float(np.median(samples))
            print(f"{name}: {label}, median={median:.6g}s", flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    run()
