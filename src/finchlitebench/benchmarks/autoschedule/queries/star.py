"""Star join: ``R[j,k,l] = sum_i A[i,j] B[i,k] C[i,l]``."""

import finch as fl

from ..spec import cases_from_shapes, make_spec
from .common import random_tensors


def star_query(A, B, C):
    return fl.einsum("ij,ik,il->jkl", A, B, C)


def star_inputs(case):
    i, j, k, l = case.shape
    return random_tensors(case, (i, j), (i, k), (i, l))


CASES = cases_from_shapes(
    {
        "balanced": (32, 32, 32, 32),
        "narrow-center": (4, 64, 64, 64),
        "wide-center": (128, 16, 16, 16),
    }
)

SPEC = make_spec(
    "star",
    star_query,
    star_inputs,
    CASES,
    end_to_end_cases=("balanced-sparse", "wide-center-sparse"),
)
