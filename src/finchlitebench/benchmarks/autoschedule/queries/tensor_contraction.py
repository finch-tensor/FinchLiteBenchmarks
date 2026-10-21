"""Tensor contraction: ``R[i,l] = sum_jk A[i,j,k] B[j,k,l]``."""

import finch as fl

from ..spec import cases_from_shapes, make_spec
from .common import random_tensors


def contraction_query(A, B):
    return fl.einsum("ijk,jkl->il", A, B)


def contraction_inputs(case):
    i, j, k, l = case.shape
    return random_tensors(case, (i, j, k), (j, k, l))


CASES = cases_from_shapes(
    {
        "balanced": (24, 24, 24, 24),
        "narrow-j": (64, 4, 128, 64),
        "narrow-k": (64, 128, 4, 64),
    }
)

SPEC = make_spec(
    "tensor_contraction",
    contraction_query,
    contraction_inputs,
    CASES,
    end_to_end_cases=tuple(case.name for case in CASES if case.pattern == "sparse"),
)
