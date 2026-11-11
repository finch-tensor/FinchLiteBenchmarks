"""Input and reference output for the matrix-features benchmark, shared by every runner.

The benchmark computes row Lp norms, p = 1..count, of a 50,000 x 50,000 CSR matrix
with 1% of entries stored (25M nonzeros). Values are uniform in [0, 1), so
|x|^p = x^p and every backend can skip abs: the norm of row i is
(sum_j x_ij^p)^(1/p).
"""

import numpy as np
import scipy.sparse as sps

ROWS = 50_000
COLS = 50_000
DENSITY = 0.01
SEED = 42
POWERS = (1, 2, 3, 4)  # features_k computes the norms for POWERS[:k]


def make_matrix():
    matrix = sps.csr_array(sps.random(
        ROWS, COLS, density=DENSITY, format="csr",
        random_state=np.random.default_rng(SEED),
    ))
    matrix.sort_indices()
    return matrix


def reference_norms(matrix):
    """Row L1..L4 norms, in benchmark order."""
    return [np.asarray(matrix.power(p).sum(axis=1)).ravel() ** (1.0 / p) for p in POWERS]
