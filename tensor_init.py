from .tensor import tensor
from .backend import get_xp

def _process_shape(shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    else:
        shape = tuple(shape)
    if not all(isinstance(x, int) for x in shape):
        raise TypeError("Shape must be integers or a tuple/list of integers")
    return shape

def zeros(*shape, requires_grad=False, dtype=None, device='cpu'):
    shape = _process_shape(shape)
    xp = get_xp(device)
    data = xp.zeros(shape, dtype=dtype)
    return tensor(data, requires_grad=requires_grad, device=device, dtype=dtype)

def ones(*shape, requires_grad=False, dtype=None, device='cpu'):
    shape = _process_shape(shape)
    xp = get_xp(device)
    data = xp.ones(shape, dtype=dtype)
    return tensor(data, requires_grad=requires_grad, device=device, dtype=dtype)

def full(*shape, fill_value, requires_grad=False, dtype=None, device='cpu'):
    shape = _process_shape(shape)
    xp = get_xp(device)
    data = xp.full(shape, fill_value, dtype=dtype)
    return tensor(data, requires_grad=requires_grad, device=device, dtype=dtype)

def rand(*shape, seed=None, requires_grad=False, dtype=None, device='cpu'):
    shape = _process_shape(shape)
    xp = get_xp(device)
    xp.random.seed(seed)  # Set the random seed for reproducibility
    data = xp.random.rand(*shape).astype(dtype or xp.float32)
    return tensor(data, requires_grad=requires_grad, device=device, dtype=dtype)

def randn(*shape, seed=None, requires_grad=False, dtype=None, device='cpu'):
    shape = _process_shape(shape)
    xp = get_xp(device)
    xp.random.seed(seed)  # Set the random seed for reproducibility
    data = xp.random.randn(*shape).astype(dtype or xp.float32)
    return tensor(data, requires_grad=requires_grad, device=device, dtype=dtype)

def arange(start, stop=None, step=1, *, dtype=None, requires_grad=False, device='cpu'):
    xp = get_xp(device)
    if stop is None:
        # Only one argument given: arange(stop)
        start, stop = 0, start
    data = xp.arange(start, stop, step, dtype=dtype)
    return tensor(data, requires_grad=requires_grad, device=device, dtype=dtype)

def linspace(start, stop, num=10, *, dtype=None, requires_grad=False, device='cpu'):
    xp = get_xp(device)
    data = xp.linspace(start, stop, num=num, dtype=dtype)
    return tensor(data, requires_grad=requires_grad, device=device, dtype=dtype)
