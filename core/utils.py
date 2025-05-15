# core/utils.py

def unbroadcast(grad, shape):
    """
    Reduce grad to match the original shape by summing along broadcasted dimensions.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)

def expand_grad(grad, target_shape, axis):
    # Expand grad to match input shape
    target = list(grad.data.shape)
    for ax in (axis,) if isinstance(axis, int) else axis:
        target.insert(ax, 1)
    grad.data = grad.data.reshape(target)
    return grad.data.repeat(target_shape[ax], axis=ax)
