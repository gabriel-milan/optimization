# -*- coding: utf-8 -*-
from gabriel_nlopt import constants, Function, minimize
import numpy as np

x0 = np.array([5, 5], dtype=np.float64)


class BasicFunction(Function):
    def __call__(self, x):
        return (np.power(x[0], 2) + np.power(x[1], 2)).astype(np.float64)

    def gradient(self, x):
        return np.array([2 * x[0], 2 * x[1]], dtype=np.float64)

    def hessian(self, x):
        return np.array([[2, 0], [0, 2]], dtype=np.float64)


def test_function():
    function = BasicFunction()
    objective = function(x0)
    assert (objective - 50) <= constants.EPSILON
    gradient = function.gradient(x0)
    assert (gradient[0] - 10) <= constants.EPSILON
    assert (gradient[1] - 10) <= constants.EPSILON
    hessian = function.hessian(x0)
    assert (hessian[0][0] - 2) <= constants.EPSILON
    assert (hessian[0][1] - 0) <= constants.EPSILON
    assert (hessian[1][0] - 0) <= constants.EPSILON
    assert (hessian[1][1] - 2) <= constants.EPSILON


def test_gradient():
    function = BasicFunction()
    x_min = minimize(function, x0, method="gradient")
    assert x_min[0] <= constants.EPSILON
    assert x_min[1] <= constants.EPSILON


def test_newton():
    function = BasicFunction()
    x_min = minimize(function, x0, method="newton")
    assert x_min[0] <= constants.EPSILON
    assert x_min[1] <= constants.EPSILON


def test_bfgs():
    function = BasicFunction()
    x_min = minimize(function, x0, method="bfgs")
    assert x_min[0] <= constants.EPSILON
    assert x_min[1] <= constants.EPSILON
