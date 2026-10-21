"""Diamond join: ``R[i] = sum_jkl A[i,j] B[j,k] C[j,l] D[k,l]``."""

import finch as fl

from ..spec import cases_from_shapes, make_spec
from .common import random_tensors


def diamond_query(A, B, C, D):
    return fl.einsum("ij,jk,jl,kl->i", A, B, C, D)


def diamond_inputs(case):
    i, j, k, l = case.shape
    return random_tensors(case, (i, j), (j, k), (j, l), (k, l))


CASES = cases_from_shapes(
    {
        "balanced": (24, 24, 24, 24),
        "narrow-k": (32, 32, 4, 64),
        "narrow-l": (32, 32, 64, 4),
    }
)

SPEC = make_spec(
    "diamond",
    diamond_query,
    diamond_inputs,
    CASES,
    end_to_end_cases=tuple(case.name for case in CASES if case.pattern == "sparse"),
)
