import numpy as np

from gabriel_nlopt.core.types import Float


class Constants:
    __slots__ = ()
    DEFAULT_DEBUG: bool = False
    DEFAULT_ETA: Float = np.float64(1 / 4)
    DEFAULT_GAMA: Float = np.float64(0.8)
    DEFAULT_MAX_ITERATIONS_GRADIENT: int = int(1e6)
    DEFAULT_MAX_ITERATIONS_NEWTON: int = int(1e6)
    DEFAULT_NO_IMPROVEMENT_LIMIT_GRADIENT: int = int(1e3)
    DEFAULT_NO_IMPROVEMENT_LIMIT_NEWTON: int = int(1e3)
    EPSILON: Float = np.finfo(np.float64).eps


constants = Constants()
