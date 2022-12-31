# -*- coding: utf-8 -*-
import numpy as np

from gabriel_nlopt import constants
from gabriel_nlopt.algorithms.armijo import step_size
from gabriel_nlopt.core.function import Function
from gabriel_nlopt.core.types import Array2x2, Float, Vector2


class ArmijoTestFunction(Function):
    def __call__(self, x: Vector2) -> Float:
        return np.float64(1 / 2 * (x[0] - 2) ** 2 + (x[1] - 1) ** 2)

    def gradient(self, x: Vector2) -> Vector2:
        gx = x[0] - 2
        gy = 2 * (x[1] - 1)
        return np.array([gx, gy], dtype=np.float64)

    def hessian(self, x: Vector2) -> Array2x2:
        return np.array([[1, 0], [0, 2]], dtype=np.float64)


def test_armijo():
    direction = np.array([3, 1], dtype=np.float64)
    x0 = np.array([1, 0], dtype=np.float64)
    eta = np.float64(1 / 4)
    gama = np.float64(0.8)
    assert (
        abs(
            step_size(eta, gama, x0, direction, ArmijoTestFunction()) - np.float64(0.64)
        )
        <= constants.EPSILON
    )
