import numpy as np
import pytest

from finchlitebench.benchmarks.mapjoin import (
    benchmark_elementwise_mul,
    benchmark_elementwise_add,
    benchmark_broadcast_mul,
    benchmark_broadcast_add,
    benchmark_join_mul,
    benchmark_join_add,
    benchmark_semiring,
    benchmark_semiring_tropical,
    benchmark_semiring_maxplus,
    benchmark_semiring_boolean,
    benchmark_semiring_maxmin,
)
from finchlitebench.utils.exact_stats import ExactStatsFactory



@pytest.mark.parametrize("A,B", [
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float64),
    ),
    (
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64),
    ),
    (
        np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=np.float64),
        np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64),
    ),
])
def test_elementwise_mul(A, B):
    assert benchmark_elementwise_mul(ExactStatsFactory(), A, B) == float(np.count_nonzero(A * B))


@pytest.mark.parametrize("A,B", [
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float64),
    ),
    (
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64),
    ),
    (
        np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=np.float64),
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
    ),
])
def test_elementwise_add(A, B):
    assert benchmark_elementwise_add(ExactStatsFactory(), A, B) == float(np.count_nonzero(A + B))


@pytest.mark.parametrize("A,b", [
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([1.0, 0.0, 1.0]),
    ),
    (
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float64),
        np.array([0.0, 1.0, 0.0]),
    ),
    (
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float64),
        np.array([1.0, 1.0, 1.0]),
    ),
])
def test_broadcast_mul(A, b):
    assert benchmark_broadcast_mul(ExactStatsFactory(), A, b) == float(np.count_nonzero(A * b[np.newaxis, :]))


@pytest.mark.parametrize("A,b", [
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([1.0, 0.0, 1.0]),
    ),
    (
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float64),
        np.array([0.0, 1.0, 0.0]),
    ),
    (
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
        np.array([1.0, 1.0, 1.0]),
    ),
])
def test_broadcast_add(A, b):
    assert benchmark_broadcast_add(ExactStatsFactory(), A, b) == float(np.count_nonzero(A + b[np.newaxis, :]))


@pytest.mark.parametrize("A,B", [
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float64),
    ),
    (
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64),
    ),
    (
        np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
    ),
])
def test_join_mul(A, B):
    assert benchmark_join_mul(ExactStatsFactory(), A, B) == float(np.count_nonzero(A[:, :, np.newaxis] * B[np.newaxis, :, :]))


@pytest.mark.parametrize("A,B", [
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float64),
    ),
    (
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64),
    ),
    (
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
    ),
])
def test_join_add(A, B):
    assert benchmark_join_add(ExactStatsFactory(), A, B) == float(np.count_nonzero(A[:, :, np.newaxis] + B[np.newaxis, :, :]))


@pytest.mark.parametrize("A,B,C,D", [
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64),
    ),
    (
        np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=np.float64),
        np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    ),
])
def test_semiring(A, B, C, D):
    assert benchmark_semiring(ExactStatsFactory(), A, B, C, D) == float(np.count_nonzero((A @ B) * (C @ D)))


@pytest.mark.parametrize("A,B,C,D", [
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64),
    ),
    (
        np.array([[2, 0, 3], [0, 1, 0], [4, 0, 2]], dtype=np.float64),
        np.array([[1, 2, 0], [0, 1, 3], [2, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64),
        np.array([[3, 0, 1], [0, 2, 0], [1, 0, 3]], dtype=np.float64),
    ),
])
def test_semiring_tropical(A, B, C, D):
    def tropical_mm(X, Y):
        return np.minimum.reduce(X[:, :, np.newaxis] + Y[np.newaxis, :, :], axis=1)
    assert benchmark_semiring_tropical(ExactStatsFactory(), A, B, C, D) == float(np.count_nonzero(tropical_mm(A, B) + tropical_mm(C, D)))


@pytest.mark.parametrize("A,B,C,D", [
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64),
    ),
    (
        np.array([[2, 0, 3], [0, 1, 0], [4, 0, 2]], dtype=np.float64),
        np.array([[1, 2, 0], [0, 1, 3], [2, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64),
        np.array([[3, 0, 1], [0, 2, 0], [1, 0, 3]], dtype=np.float64),
    ),
])
def test_semiring_maxplus(A, B, C, D):
    def max_plus_mm(X, Y):
        return np.maximum.reduce(X[:, :, np.newaxis] + Y[np.newaxis, :, :], axis=1)
    assert benchmark_semiring_maxplus(ExactStatsFactory(), A, B, C, D) == float(np.count_nonzero(max_plus_mm(A, B) + max_plus_mm(C, D)))


@pytest.mark.parametrize("A,B,C,D", [
    (
        np.array([[True, False, False], [False, True, False], [False, False, True]]),
        np.array([[True, True, False], [False, True, True], [False, False, True]]),
        np.array([[True, False, True], [False, False, False], [True, False, True]]),
        np.array([[False, True, False], [True, False, True], [False, True, False]]),
    ),
    (
        np.array([[True, True, False], [False, False, True], [True, False, True]]),
        np.array([[False, True, True], [True, False, False], [False, True, False]]),
        np.array([[True, False, False], [False, True, False], [False, False, True]]),
        np.array([[True, False, True], [False, True, False], [True, False, True]]),
    ),
])
def test_semiring_boolean(A, B, C, D):
    def bool_mm(X, Y):
        return np.any(X[:, :, np.newaxis] & Y[np.newaxis, :, :], axis=1)
    assert benchmark_semiring_boolean(ExactStatsFactory(), A, B, C, D) == float(np.count_nonzero(bool_mm(A, B) & bool_mm(C, D)))


@pytest.mark.parametrize("A,B,C,D", [
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64),
    ),
    (
        np.array([[0.8, 0.0, 0.5], [0.0, 0.9, 0.0], [0.3, 0.0, 0.7]], dtype=np.float64),
        np.array([[0.6, 0.4, 0.0], [0.0, 0.7, 0.5], [0.8, 0.0, 0.6]], dtype=np.float64),
        np.array([[0.0, 0.5, 0.0], [0.4, 0.0, 0.6], [0.0, 0.3, 0.0]], dtype=np.float64),
        np.array([[0.7, 0.0, 0.4], [0.0, 0.8, 0.0], [0.5, 0.0, 0.9]], dtype=np.float64),
    ),
])
def test_semiring_maxmin(A, B, C, D):
    def fuzzy_mm(X, Y):
        return np.maximum.reduce(np.minimum(X[:, :, np.newaxis], Y[np.newaxis, :, :]), axis=1)
    assert benchmark_semiring_maxmin(ExactStatsFactory(), A, B, C, D) == float(np.count_nonzero(np.minimum(fuzzy_mm(A, B), fuzzy_mm(C, D))))
