# -*- coding: utf-8 -*-
import numpy as np

from gabriel_nlopt import constants
from gabriel_nlopt.algorithms.armijo import step_size
from gabriel_nlopt.core.function import Function
from gabriel_nlopt.core.types import Float, Vector2


def gradient_method(
    objective: Function,
    x0: Vector2,
    eta: Float = constants.DEFAULT_ETA,
    gama: Float = constants.DEFAULT_GAMA,
    max_iterations: int = constants.DEFAULT_MAX_ITERATIONS_GRADIENT,
    no_improvement_limit: int = constants.DEFAULT_NO_IMPROVEMENT_LIMIT_GRADIENT,
    debug: bool = constants.DEFAULT_DEBUG,
) -> Vector2:
    current_objective = np.inf
    last_objective = objective(x0)
    no_improvement = 0
    best_x = x0
    while True:
        if max_iterations <= 0:
            print("Max iterations reached")
            break
        if (np.abs(objective.gradient(x0)) <= constants.EPSILON).all():
            print(f"Gradient is very close to zero: {objective.gradient(x0)}")
            break
        if no_improvement >= no_improvement_limit:
            print("No improvement limit reached")
            break
        if current_objective == -np.inf:
            best_x = x0
            print("Objective is -inf")
            break
        d = -objective.gradient(x0)
        t = step_size(eta, gama, x0, d, objective)
        x0 += t * d
        current_objective = objective(x0)
        if debug:
            print("=" * 80)
            print(f"d: {d}")
            print(f"t: {t}")
            print(f"x0: {x0}")
            print(f"current_objective: {current_objective}")
        if last_objective - current_objective < constants.EPSILON:
            no_improvement += 1
        else:
            no_improvement = 0
        if current_objective < objective(best_x):
            best_x = x0
        last_objective = current_objective
        max_iterations -= 1
    return best_x
