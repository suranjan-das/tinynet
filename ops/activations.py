from .base import Operation


# Sigmoid operation
class Sigmoid(Operation):
    def forward(self, x):
        self.s = 1 / (1 + x.xp.exp(-x.data))
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
