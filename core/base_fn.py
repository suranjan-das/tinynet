def binary_op(a, b, OpClass, **kwargs):
    assert hasattr(a, 'data') and hasattr(b, 'data'), f"Invalid inputs to binary_op: {a}, {b}"
    requires_grad = a.requires_grad or b.requires_grad

    if requires_grad:
        op = OpClass(**kwargs)
        data = op.forward(a, b)
    else:
        op = None
        data = OpClass(**kwargs).forward(a, b)  # Could be optimized further if backend supports nograd ops

    return data, requires_grad, op


def unary_op(x, OpClass, **kwargs):
    assert hasattr(x, 'data'), f"Invalid input to unary_op: {x}"
    requires_grad = x.requires_grad

    if requires_grad:
        op = OpClass(**kwargs)
        data = op.forward(x)
    else:
        op = None
        data = OpClass(**kwargs).forward(x)

    return data, requires_grad, op


def scalar_op(scalar, x, OpClass, is_scalar_first=False):
    assert hasattr(x, 'data'), f"Invalid input to scalar_op: {x}"
    requires_grad = x.requires_grad

    if requires_grad:
        op = OpClass(scalar, is_scalar_first)
        data = op.forward(x)
    else:
        op = None
        data = OpClass(scalar, is_scalar_first).forward(x)

    return data, requires_grad, op
