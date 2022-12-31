# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

from gabriel_nlopt.core.types import Array2x2, Float, Vector2


class Function(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, x: Vector2) -> Float:
        pass

    @abstractmethod
    def gradient(self, x: Vector2) -> Vector2:
        pass

    @abstractmethod
    def hessian(self, x: Vector2) -> Array2x2:
        pass
