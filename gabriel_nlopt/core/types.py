# -*- coding: utf-8 -*-
from typing import Annotated, Literal, TypeVar

import numpy as np
import numpy.typing as npt

Float = TypeVar("Float", bound=np.float64)

Vector2 = Annotated[npt.NDArray[Float], Literal[2]]
Array2x2 = Annotated[npt.NDArray[Float], Literal[2, 2]]
