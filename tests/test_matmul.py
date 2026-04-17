import numpy as np
import pytest

import finchlite as fl

from src.benchmarks.matmul import benchmark_matmul


@pytest.mark.parametrize(
    "A,B",
    [
        (
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
        ),
        (
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
        ),
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float32),
        ),
    ],
)
def test_benchmark_matmul(A, B):
    C_ref = A @ B
    C = benchmark_matmul(fl.get_default_scheduler(), A, B)
    assert np.allclose(np.asarray(C), C_ref)
