"""Chain join: ``R[i,l] = sum_jk A[i,j] B[j,k] C[k,l]``.

Tests join association and loop order without a cycle.  The rectangular
cases put the narrow dimension on opposite sides of B.
"""

import finch as fl

from ..spec import cases_from_shapes, make_spec
from .common import random_tensors


def chain_query(A, B, C):
    return fl.einsum("ij,jk,kl->il", A, B, C)


def chain_inputs(case):
    i, j, k, l = case.shape
    return random_tensors(case, (i, j), (j, k), (k, l))


CASES = cases_from_shapes(
    {
        "balanced": (32, 32, 32, 32),
        "left-bottleneck": (64, 4, 128, 64),
        "right-bottleneck": (64, 128, 4, 64),
    }
)

SPEC = make_spec(
    "chain",
    chain_query,
    chain_inputs,
    CASES,
    end_to_end_cases=(
        "balanced-sparse",
        "left-bottleneck-sparse",
        "right-bottleneck-sparse",
    ),
)
