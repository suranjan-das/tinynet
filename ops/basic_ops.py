from .base import Operation
from ..core.utils import unbroadcast, expand_grad

class Neg(Operation):
    def forward(self, x):
        return -x.data

    def backward(self, grad, x):
        return (-grad.data,)  # dL/dX = -dL/dC

# Addition operation
class Add(Operation):
    def forward(self, a, b):
        self.a_shape = a.data.shape
        self.b_shape = b.data.shape
        return a.data + b.data

    def backward(self, grad, a, b):
        grad_a = unbroadcast(grad.data, self.a_shape)
        grad_b = unbroadcast(grad.data, self.b_shape)
        return grad_a, grad_b

# Subtraction operation
class Subtract(Operation):
    def forward(self, a, b):
        self.a_shape = a.data.shape
        self.b_shape = b.data.shape
        return a.data - b.data

    def backward(self, grad, a, b):
        grad_a = unbroadcast(grad.data, self.a_shape)
        grad_b = unbroadcast(-grad.data, self.b_shape)
        return grad_a, grad_b

# Element-wise multiplication
class Multiply(Operation):
    def forward(self, a, b):
        self.a = a
        self.b = b
        self.a_shape = a.data.shape
        self.b_shape = b.data.shape
        return a.data * b.data

    def backward(self, grad, a, b):
        grad_a = unbroadcast(grad.data * b.data, self.a_shape)
        grad_b = unbroadcast(grad.data * a.data, self.b_shape)
        return grad_a, grad_b

# Element-wise division
class Divide(Operation):
    def forward(self, a, b):
        self.a = a
        self.b = b
        self.a_shape = a.data.shape
        self.b_shape = b.data.shape
        return a.data / b.data

    def backward(self, grad, a, b):
        grad_a = unbroadcast(grad.data / b.data, self.a_shape)
        grad_b = unbroadcast(-grad.data * a.data / (b.data ** 2), self.b_shape)
        return grad_a, grad_b
    
class Pow(Operation):
    def forward(self, a, b):
        self.a_data = a.data
        self.b_data = b.data
        self.out = self.a_data ** self.b_data
        self.a_shape = self.a_data.shape
        self.b_shape = self.b_data.shape
        return self.out

    def backward(self, grad, a, b):
        xp = a.xp
        grad_data = grad.data
        grad_a = unbroadcast(grad_data * self.b_data * (self.a_data ** (self.b_data - 1)), self.a_shape)
        grad_b = unbroadcast(grad_data * self.out * xp.log(self.a_data), self.b_shape)
        return grad_a, grad_b
    
class MatMul(Operation):
    def forward(self, a, b):
        self.a_shape = a.data.shape
        self.b_shape = b.data.shape

        # Reshape 1D to 2D for consistent matmul
        a_data = a.data.reshape(1, -1) if a.data.ndim == 1 else a.data
        b_data = b.data.reshape(-1, 1) if b.data.ndim == 1 else b.data

        self.a_data = a_data
        self.b_data = b_data

        result = a_data @ b_data
        return result

    def backward(self, grad, a, b):
        grad_data = grad.data
        if grad_data.ndim == 0:
            grad_data = grad_data.reshape(1, 1)
        elif grad_data.ndim == 1:
            grad_data = grad_data.reshape(-1, 1) if self.b_data.shape[1] == 1 else grad_data.reshape(1, -1)

        grad_a = grad_data @ self.b_data.T
        grad_b = self.a_data.T @ grad_data

        # Reshape gradients to match original input shapes
        grad_a = grad_a.reshape(self.a_shape)
        grad_b = grad_b.reshape(self.b_shape)

        return grad_a, grad_b    


# Transpose operation
class Transpose(Operation):
    def forward(self, x):
        return x.data.T

    def backward(self, grad, x):
        return (grad.data.T,)

# Reshape operation    
class Reshape(Operation):
    def __init__(self, shape=None):
        self.new_shape = shape
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.data.shape
        return x.data.reshape(self.new_shape)

    def backward(self, grad, x):
        return (grad.data.reshape(self.original_shape),)
    
class GetItem(Operation):
    def __init__(self, idx):
        self.idx = idx

    def forward(self, x):
        return x.data[self.idx]

    def backward(self, grad, x):
        grad_data = x.xp.zeros_like(x.data)
        grad_data[self.idx] = grad.data
        return (grad_data,)
    
# Sum operation
class Sum(Operation):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return x.data.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad, x):
        if self.axis is None and not self.keepdims:
            grad.data = grad.data.reshape(1).repeat(x.data.size, axis=0).reshape(x.data.shape)
        elif self.axis is not None and not self.keepdims:
            grad.data = expand_grad(grad, x.shape, self.axis)
        return (grad.data,)

# Mean operation
class Mean(Operation):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return x.data.mean(axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad, x):
        count = x.data.size if self.axis is None else x.shape[self.axis]
        grad.data = grad.data / count
        if self.axis is None and not self.keepdims:
            grad.data = grad.data.reshape(1).repeat(x.data.size, axis=0).reshape(x.data.shape)
        elif self.axis is not None and not self.keepdims:
            grad.data = expand_grad(grad, x.data.shape, self.axis)
        return (grad.data,)
    
# Scalar addition operation (scalar + tensor or tensor + scalar)
class ScalarAdd(Operation):
    def __init__(self, scalar, is_scalar_first=False):
        self.scalar = scalar
        self.is_scalar_first = is_scalar_first

    def forward(self, x):
        return self.scalar + x.data if self.is_scalar_first else x.data + self.scalar

    def backward(self, grad, x):
        return (grad.data,)  # dL/dX = dL/dC

# Scalar subtraction operation (scalar - tensor or tensor - scalar)
class ScalarSubtract(Operation):
    def __init__(self, scalar, is_scalar_first=False):
        self.scalar = scalar
        self.is_scalar_first = is_scalar_first

    def forward(self, x):
        return self.scalar - x.data if self.is_scalar_first else x.data - self.scalar

    def backward(self, grad, x):
        return (-grad.data,) if self.is_scalar_first else (grad.data,)  # dL/dX = -dL/dC or dL/dC

# Scalar multiplication operation (scalar * tensor or tensor * scalar)
class ScalarMultiply(Operation):
    def __init__(self, scalar, is_scalar_first=None):
        self.scalar = scalar
        self.is_scalar_first = is_scalar_first
        
    def forward(self, x):
        return self.scalar * x.data

    def backward(self, grad, x):
        return (grad.data * self.scalar,)  # dL/dX = dL/dC * scalar

# Scalar division operation (scalar / tensor or tensor / scalar)
class ScalarDivide(Operation):
    def __init__(self, scalar, is_scalar_first=False):
        self.scalar = scalar
        self.is_scalar_first = is_scalar_first

    def forward(self, x):
        return self.scalar / x.data if self.is_scalar_first else x.data / self.scalar

    def backward(self, grad, x):
        if self.is_scalar_first:
            return (-grad.data * self.scalar / (x.data * x.data),)  # dL/dX = -dL/dC * scalar / X^2
        return (grad.data / self.scalar,)  # dL/dX = dL/dC / scalar
    
class ScalarPow(Operation):
    def __init__(self, scalar, is_scalar_first=False):
        self.scalar = scalar
        self.is_scalar_first = is_scalar_first

    def forward(self, x):
        return self.scalar ** x.data if self.is_scalar_first else x.data ** self.scalar

    def backward(self, grad, x):
        xp = x.xp
        if self.is_scalar_first:
            grad_input = grad.data * xp.log(self.scalar) * (self.scalar ** x.data)
        else:
            grad_input = grad.data * self.scalar * (x.data ** (self.scalar - 1))
        return (grad_input,)
