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
