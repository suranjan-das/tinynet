from .base import Operation

class Exp(Operation):
    def forward(self, x):
        self.out = x.device.xp.exp(x.data)
        return self.out

    def backward(self, grad, x):
        return (grad.data * self.out,)

class Log(Operation):
    def forward(self, x):
        self.x_data = x.data
        return x.device.xp.log(x.data)

    def backward(self, grad, x):
        return (grad.data / self.x_data,)

class Sqrt(Operation):
    def forward(self, x):
        self.x_data = x.data
        self.out = x.device.xp.sqrt(x.data)
        return self.out

    def backward(self, grad, x):
        return (grad.data / (2 * self.out),)
    
class LogSoftmax(Operation):
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x):
        self.x = x
        xp = x.device.xp
        self.max_val = xp.max(x.data, axis=self.axis, keepdims=True)
        self.shifted = x.data - self.max_val
        self.logsumexp = xp.log(xp.sum(xp.exp(self.shifted), axis=self.axis, keepdims=True))
        return self.shifted - self.logsumexp

    def backward(self, grad, x):
        xp = x.device.xp
        softmax = xp.exp(self.shifted - self.logsumexp)  # softmax(x)
        sum_grad = xp.sum(grad.data, axis=self.axis, keepdims=True)
        return (grad.data - softmax * sum_grad,)  # dL/dx

    
