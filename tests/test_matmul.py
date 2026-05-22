import numpy as np
import pytest

from finchlitebench.benchmarks.matmul import benchmark_matmul
from finchlitebench.utils.exact_stats import ExactStatsFactory


@pytest.mark.parametrize("A,B", [
    (
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    ),
    (
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float64),
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
    ),
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float64),
    ),
])
def test_benchmark_matmul(A, B):
    expected = float(np.count_nonzero(A @ B))
    assert benchmark_matmul(ExactStatsFactory(), A, B) == expected
