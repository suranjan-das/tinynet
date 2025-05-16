from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, *parameters):
        self.parameters = list(parameters)

    @abstractmethod
    def step(self):
        """Update parameters"""
        pass

    def zero_grad(self):
        """Set gradients of all parameters to zero"""
        for param in self.parameters:
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad = None
