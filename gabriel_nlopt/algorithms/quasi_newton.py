# -*- coding: utf-8 -*-
import numpy as np

from gabriel_nlopt import constants
from gabriel_nlopt.algorithms.armijo import step_size
from gabriel_nlopt.core.function import Function
from gabriel_nlopt.core.types import Float, Vector2


def bfgs_method(
    objective: Function,
    x0: Vector2,
    eta: Float = constants.DEFAULT_ETA,
    gama: Float = constants.DEFAULT_GAMA,
    max_iterations: int = constants.DEFAULT_MAX_ITERATIONS_NEWTON,
    no_improvement_limit: int = constants.DEFAULT_NO_IMPROVEMENT_LIMIT_NEWTON,
    debug: bool = constants.DEFAULT_DEBUG,
) -> Vector2:
    current_objective = np.inf
    last_objective = objective(x0)
    no_improvement = 0
    best_x = x0
    # d = -objective.gradient(x0)
    # t = step_size(eta, gama, x0, d, objective)
    x1 = x0  # + t * d
    hk0 = np.eye(x0.shape[0], dtype=np.float64)
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
        pk = x1 - x0
        qk = objective.gradient(x1) - objective.gradient(x0)
        pk = pk[np.newaxis].T
        qk = qk[np.newaxis].T
        first_term = (1 + (qk.T @ hk0 @ qk) / (pk.T @ qk + constants.EPSILON)) * (
            (pk @ pk.T) / (pk.T @ qk + constants.EPSILON)
        )
        second_term = ((pk @ qk.T @ hk0) + (hk0 @ qk @ pk.T)) / (
            pk.T @ qk + constants.EPSILON
        )
        hk1 = hk0 + first_term - second_term
        if debug:
            print("=" * 80)
            print(f"pk: {pk}")
            print(f"qk: {qk}")
            print(f"hk0: {hk0}")
            print(f"first_term: {first_term}")
            print(f"second_term: {second_term}")
            print(f"hk1: {hk1}")
        d = -hk1 @ objective.gradient(x1)
        t = step_size(eta, gama, x1, d, objective)
        x0 = x1
        x1 += t * d
        current_objective = objective(x1)
        if debug:
            print("=" * 80)
            print(f"d: {d}")
            print(f"t: {t}")
            print(f"x0: {x0}")
            print(f"x1: {x1}")
            print(f"current_objective: {current_objective}")
        if last_objective - current_objective < constants.EPSILON:
            no_improvement += 1
        else:
            no_improvement = 0
        if current_objective < objective(best_x):
            best_x = x1
        last_objective = current_objective
        max_iterations -= 1
    return best_x
