from ..ops.basic_ops import *
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


# Unary operations
def transpose(x):
    return unary_op(x, Transpose)

def sum(x, axis=None, keepdims=False):
    return unary_op(x, Sum, axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=False):
    return unary_op(x, Mean, axis=axis, keepdims=keepdims)
