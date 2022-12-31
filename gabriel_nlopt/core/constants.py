import numpy as np

from gabriel_nlopt.core.types import Float


class Constants:
    __slots__ = ()
    EPSILON: Float = np.finfo(np.float64).eps


constants = Constants()
