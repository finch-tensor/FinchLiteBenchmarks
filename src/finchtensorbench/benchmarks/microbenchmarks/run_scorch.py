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
SDDMM_ROWS = 10000  # SDDMM uses larger matrices than chain matmul
SDDMM_COLS = 10000
SDDMM_DENSITIES = (0.00001, 0.0001, 0.001, 0.01)
CHAIN_AB_DENSITY = 0.01  # A and B in A @ B @ C
CHAIN_DENSITIES = (0.000001, 0.00001, 0.0001, 0.001, 0.01)  # C
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
    for density in SDDMM_DENSITIES:
        mask = random_csr((SDDMM_ROWS, SDDMM_COLS), density, SEED + 2)
        results.append((sddmm_inputs(128, mask, SEED, dtype), mask.nnz / (SDDMM_ROWS * SDDMM_COLS)))
    return results


######################### Chain Matmul ##########################

def matmul(a, b, c):
    return scorch.einsum("ij,jk,kl->il", a, b, c, format="dd")


def chain_cases(dtype):
    results = []
    shape = (ROWS, COLS)
    a = to_scorch_csr(random_csr(shape, CHAIN_AB_DENSITY, SEED), "A", dtype)
    b = to_scorch_csr(random_csr(shape, CHAIN_AB_DENSITY, SEED + 1), "B", dtype)
    for density in CHAIN_DENSITIES:
        c = random_csr(shape, density, SEED + 2)
        results.append(((a, b, to_scorch_csr(c, "C", dtype)), c.nnz / (ROWS * COLS)))
    return results


######################### Gustavson / Outer Product ##########################

def spgemm(a, b):
    # C = A @ B with Scorch's prebuilt SpGEMM, output rows sparse. Its einsum
    # ("ik,kj->ij") on these CSR inputs fails to compile in float64 and segfaults
    # in float32.
    return scorch.matmul(a, b)


def spgemm_cases(dtype):
    results = []
    for shared in (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536):
        a, b = paper_spgemm.benchmark_problem(shared)
        results.append(((to_scorch_csr(a, "A", dtype), to_scorch_csr(b, "B", dtype)), shared))
    return results


######################### Matrix Features ##########################

def power(values, p):
    # torch's CPU pow special-cases exponents 2 and 3; others take a generic path that
    # is ~40x slower here, so larger integer powers are computed by repeated squaring.
    if p in (2, 3):
        return values ** p
    result, base = None, values
    while p:
        if p & 1:
            result = base if result is None else result * base
        p >>= 1
        if p:
            base = torch.square(base)
    return result


def matrix_features(a, ones, count):
    """Row Lp norms, p = 1..count. Scorch has no pow or root: PyTorch raises the stored
    values to p (zeros stay zero, so the CSR index is reused), Scorch sums each row with
    its prebuilt CSR SpMV (matmul by a ones vector), and PyTorch takes the root of the
    length-R sums. The values are nonnegative, so |x|^p = x^p. All of it is timed."""
    outputs = []
    for p in feature_matrix.POWERS[:count]:
        powered = a if p == 1 else scorch.STensor(
            name=f"A{p}", shape=a.shape, index=a.index, value=power(a.values, p))
        total = scorch.matmul(powered, ones)
        outputs.append(total if p == 1 else total ** (1.0 / p))
    return tuple(outputs)


def matrix_feature_cases(dtype):
    matrix = feature_matrix.make_matrix()
    a = to_scorch_csr(matrix, "A", dtype)
    ones = to_scorch_dense(np.ones(matrix.shape[1]), "x", dtype)
    return [(f"features_{count}", (a, ones, count))
            for count in range(1, len(feature_matrix.POWERS) + 1)]


######################### Main ##########################

def cases(dtype):
    benchmarks = {
        "sddmm": (sddmm, sddmm_cases),
        "chain": (matmul, chain_cases),
    }
    results = []
    for name, (function, make_cases) in benchmarks.items():
        for inputs, density in make_cases(dtype):
            results.append((name, function, inputs, density))
    for inputs, shared_k in spgemm_cases(dtype):
        results.append(("spgemm", spgemm, inputs, shared_k))
    for name, inputs in matrix_feature_cases(dtype):
        results.append((name, matrix_features, inputs, feature_matrix.DENSITY))
    return results


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("graph/scripts/scorch_result.csv"))
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("benchmark", "density", "active_k", "repeat", "seconds"))
        for name, function, inputs, parameter in cases(torch.float32):
            is_spgemm = name == "spgemm"
            density = "" if is_spgemm else parameter
            active_k = parameter if is_spgemm else ""
            label = f"active_k={active_k}" if is_spgemm else f"density={density:.6g}"
            print(f"Warming {name}: {label}", flush=True)
            function(*inputs)
            samples = []
            for trial in range(5):
                gc.collect()
                start = time.perf_counter()
                result = function(*inputs)
                seconds = time.perf_counter() - start
                del result
                writer.writerow((name, density, active_k, trial, seconds))
                output.flush()
                samples.append(seconds)
            median = float(np.median(samples))
            print(f"{name}: {label}, median={median:.6g}s", flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    run()
