from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                # Initialize velocity if not already done
                if param not in self.velocities:
                    self.velocities[param] = param.xp.zeros_like(param.data)

                # Update velocity
                v = self.velocities[param]
                v *= self.momentum
                v -= self.lr * param.grad.data
                self.velocities[param] = v

                # Update parameter
                param.data += v

