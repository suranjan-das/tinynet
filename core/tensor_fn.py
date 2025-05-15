from ..ops.basic_ops import *
from ..ops.math_ops import *
from ..core.base_fn import binary_op, unary_op, scalar_op


# Binary operations
def add(a, b):
    return binary_op(a, b, Add)

def subtract(a, b):
    return binary_op(a, b, Subtract)

def multiply(a, b):
    return binary_op(a, b, Multiply)

def divide(a, b):
    return binary_op(a, b, Divide)

def pow(a, b):
    return binary_op(a, b, Pow)

def matmul(a, b):
    return binary_op(a, b, MatMul)


# Scalar operations
def scalar_add(scalar, x, is_scalar_first=False):
    return scalar_op(scalar, x, ScalarAdd, is_scalar_first)

def scalar_subtract(scalar, x, is_scalar_first=False):
    return scalar_op(scalar, x, ScalarSubtract, is_scalar_first)

def scalar_multiply(scalar, x, is_scalar_first=False):
    return scalar_op(scalar, x, ScalarMultiply, is_scalar_first)

def scalar_divide(scalar, x, is_scalar_first=False):
    return scalar_op(scalar, x, ScalarDivide, is_scalar_first)

def scalar_pow(scalar, x, is_scalar_first=False):
    return scalar_op(scalar, x, ScalarPow, is_scalar_first)


# Unary operations
def neg(x):
    return unary_op(x, Neg)

def transpose(x):
    return unary_op(x, Transpose)

def reshape(x, shape):
    return unary_op(x, Reshape, shape=shape)

def getitem(x, idx):
    return unary_op(x, GetItem, idx=idx)

def sum(x, axis=None, keepdims=False):
    return unary_op(x, Sum, axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=False):
    return unary_op(x, Mean, axis=axis, keepdims=keepdims)

def exp(x):
    return unary_op(x, Exp)

def log(x):
    return unary_op(x, Log)

def sqrt(x):
    return unary_op(x, Sqrt)

def log_softmax(x, axis=-1):
    return unary_op(x, LogSoftmax, axis=axis)
