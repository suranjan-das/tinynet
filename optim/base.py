from abc import ABC, abstractmethod
from collections.abc import Iterable
from ..tensor import tensor

class Optimizer(ABC):
    def __init__(self, *parameters):
        self.parameters = list(self._flatten(parameters))

    def _flatten(self, items):
        for item in items:
            if isinstance(item, tensor):
                yield item
            elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                yield from self._flatten(item)
            else:
                raise TypeError(f"Unsupported parameter type: {type(item)}")

    @abstractmethod
    def step(self):
        """Update parameters"""
        pass

    def zero_grad(self):
        """Set gradients of all parameters to zero"""
        for param in self.parameters:
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad = None
