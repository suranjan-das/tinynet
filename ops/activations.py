from .base import Operation
from ..backend import get_xp


# Sigmoid operation
class Sigmoid(Operation):
    def forward(self, x):
        xp = get_xp(x.device)
        self.s = 1 / (1 + xp.exp(-x.data))
        return self.s

    def backward(self, grad, x):
        return (grad.data * self.s * (1 - self.s),)

# ReLU operation
class ReLU(Operation):
    def forward(self, x):
        self.mask = x.data > 0
        return x.data * self.mask

    def backward(self, grad, x):
        return (grad.data * self.mask,)
