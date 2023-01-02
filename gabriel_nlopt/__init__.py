from .core.constants import constants
from .core.function import Function
from .core.types import Float, Vector2


def minimize(
    objective: Function,
    x0: Vector2,
    method: str = constants.DEFAULT_MINIMIZE_METHOD,
    eta: Float = constants.DEFAULT_ETA,
    gama: Float = constants.DEFAULT_GAMA,
    max_iterations: int = constants.DEFAULT_MAX_ITERATIONS_GRADIENT,
    no_improvement_limit: int = constants.DEFAULT_NO_IMPROVEMENT_LIMIT_GRADIENT,
    debug: bool = constants.DEFAULT_DEBUG,
) -> Vector2:
    """
    Minimize a function using a given method.

    Args:
        objective (gabriel_nlopt.core.function.Function): The function to minimize.
        x0 (gabriel_nlopt.core.types.Vector2): The starting point.
        method (str): The method to use. One of "gradient", "newton" or "bfgs".
        eta (gabriel_nlopt.core.types.Float): Armijo's parameter.
        gama (gabriel_nlopt.core.types.Float): Armijo's parameter.
        max_iterations (int): The maximum number of iterations.
        no_improvement_limit (int): The number of iterations without improvement.
        debug (bool): Whether to print debug information.

    Returns:
        gabriel_nlopt.core.types.Vector2: The minimum point.

    Raises:
        AssertionError: If any of the input data is invalid.
    """
    from gabriel_nlopt.algorithms.gradient import gradient_method
    from gabriel_nlopt.algorithms.newton import newton_method
    from gabriel_nlopt.algorithms.quasi_newton import bfgs_method

    # Assert all input data is valid.
    assert isinstance(objective, Function), f"{objective} is not a Function."
    assert isinstance(method, str), f"{method} is not a str."
    assert method in ["gradient", "newton", "bfgs"], f"{method} is not a valid method."
    assert isinstance(max_iterations, int), f"{max_iterations} is not an int."
    assert isinstance(
        no_improvement_limit, int
    ), f"{no_improvement_limit} is not an int."
    assert isinstance(debug, bool), f"{debug} is not a bool."

    # Run the algorithm.
    if method == "gradient":
        return gradient_method(
            objective=objective,
            x0=x0,
            eta=eta,
            gama=gama,
            max_iterations=max_iterations,
            no_improvement_limit=no_improvement_limit,
            debug=debug,
        )
    elif method == "newton":
        return newton_method(
            objective=objective,
            x0=x0,
            eta=eta,
            gama=gama,
            max_iterations=max_iterations,
            no_improvement_limit=no_improvement_limit,
            debug=debug,
        )
    elif method == "bfgs":
        return bfgs_method(
            objective=objective,
            x0=x0,
            eta=eta,
            gama=gama,
            max_iterations=max_iterations,
            no_improvement_limit=no_improvement_limit,
            debug=debug,
        )
    else:
        raise ValueError(f"{method} is not a valid method.")
