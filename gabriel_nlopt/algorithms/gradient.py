import numpy as np

from gabriel_nlopt import constants
from gabriel_nlopt.algorithms.armijo import step_size
from gabriel_nlopt.core.function import Function
from gabriel_nlopt.core.types import Float, Vector2


def gradient_method(
    x0: Vector2,
    objective: Function,
    eta: Float = np.float64(0.1),
    gama: Float = np.float64(0.5),
    max_iterations: int = 1000,
    no_improvement_limit: int = 100,
    debug: bool = False,
) -> Vector2:
    start_objective = objective(x0)
    last_objective = objective(x0)
    no_improvement = 0
    best_x = x0
    while True:
        if max_iterations <= 0:
            if debug:
                print("Max iterations reached")
            break
        if (np.abs(objective.gradient(x0)) <= constants.EPSILON).all():
            if debug:
                print(f"Gradient is very close to zero: {objective.gradient(x0)}")
            break
        if no_improvement >= no_improvement_limit:
            if debug:
                print("No improvement limit reached")
            break
        d = -objective.gradient(x0)
        t = step_size(eta, gama, x0, d, objective)
        x0 += t * d
        current_objective = objective(x0)
        if last_objective - current_objective < constants.EPSILON:
            no_improvement += 1
        else:
            no_improvement = 0
        if current_objective < objective(best_x):
            best_x = x0
        last_objective = current_objective
        max_iterations -= 1
    return best_x
