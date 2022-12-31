# -*- coding: utf-8 -*-
import numpy as np

from gabriel_nlopt import constants
from gabriel_nlopt.core.function import Function
from gabriel_nlopt.core.types import Float, Vector2


def step_size(
    eta: Float, gama: Float, x: Vector2, direction: Vector2, objective: Function
) -> Float:
    t: Float = np.float64(1)
    a: Float = objective(x + t * direction)
    b: Float = objective(x) + eta * t * np.dot(objective.gradient(x), direction)
    a = np.inf if np.isnan(a) else a
    while (a > b) and (t > constants.EPSILON):
        t *= gama
        t = 0 if t < constants.EPSILON else t
        a: Float = objective(x + t * direction)
        b: Float = objective(x) + eta * t * np.dot(objective.gradient(x), direction)
        a = np.inf if np.isnan(a) else a
    return t
