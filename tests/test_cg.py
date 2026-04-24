import numpy as np
import pytest

import finchlite as fl

from finchlitebench.benchmarks.cg import benchmark_cg


@pytest.mark.parametrize(
    "A,b,x",
    [
        (
            np.array([[6.0, -1.0, 0.0], [-1.0, 6.0, -1.0], [0.0, -1.0, 6.0]]),
            np.array([4.0, 8.0, 16.0]),
            np.zeros((3,)),
        ),
        (
            np.array([[7.0, 2.0, 1.0], [2.0, 6.0, -1.0], [1.0, -1.0, 5.0]]),
            np.array([13.0, -3.0, 8.0]),
            np.zeros((3,)),
        ),
        (
            np.array(
                [
                    [8.0, -1.0, 0.0, 0.0],
                    [-1.0, 8.0, -1.0, 0.0],
                    [0.0, -1.0, 8.0, -1.0],
                    [0.0, 0.0, -1.0, 8.0],
                ]
            ),
            np.array([8.0, -2.0, 6.0, 15.0]),
            np.zeros((4,)),
        ),
        (
            np.array([[12.0, 2.0, -1.0], [2.0, 10.0, 3.0], [-1.0, 3.0, 9.0]]),
            np.array([40.0, 10.0, -18.0]),
            np.zeros((3,)),
        ),
        (
            np.array([[120.0, -2.0, 0.0], [-2.0, 120.0, -2.0], [0.0, -2.0, 120.0]]),
            np.array([118.0, 116.0, 118.0]),
            np.zeros((3,)),
        ),
        (
            np.array(
                [
                    [15.0, -2.0, 0.0, 0.0, -1.0],
                    [-2.0, 14.0, -3.0, 0.0, 0.0],
                    [0.0, -3.0, 16.0, -2.0, 0.0],
                    [0.0, 0.0, -2.0, 15.0, -3.0],
                    [-1.0, 0.0, 0.0, -3.0, 17.0],
                ]
            ),
            np.array([27.0, -1.0, -18.0, 8.0, 46.0]),
            np.zeros((5,)),
        ),
    ],
)
def test_benchmark_cg(A, b, x):
    x_sol = benchmark_cg(fl.get_default_scheduler(), A, b, x)
    assert x_sol.shape == x.shape
