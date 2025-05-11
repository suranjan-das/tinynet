from .tensor import tensor
from .device import Device

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
    dev = Device(device)
    data = dev.xp.zeros(shape, dtype=dtype)
    return tensor(data, requires_grad=requires_grad, device=dev, dtype=dtype)

def ones(*shape, requires_grad=False, dtype=None, device='cpu'):
    shape = _process_shape(shape)
    dev = Device(device)
    data = dev.xp.ones(shape, dtype=dtype)
    return tensor(data, requires_grad=requires_grad, device=dev, dtype=dtype)

def full(*shape, fill_value, requires_grad=False, dtype=None, device='cpu'):
    shape = _process_shape(shape)
    dev = Device(device)
    data = dev.xp.full(shape, fill_value, dtype=dtype)
    return tensor(data, requires_grad=requires_grad, device=dev, dtype=dtype)

def rand(*shape, seed=None, requires_grad=False, dtype=None, device='cpu'):
    shape = _process_shape(shape)
    dev = Device(device)
    dev.xp.random.seed(seed)  # Set the random seed for reproducibility
    data = dev.xp.random.rand(*shape).astype(dtype or dev.xp.float32)
    return tensor(data, requires_grad=requires_grad, device=dev, dtype=dtype)

def randn(*shape, seed=None, requires_grad=False, dtype=None, device='cpu'):
    shape = _process_shape(shape)
    dev = Device(device)
    dev.xp.random.seed(seed)  # Set the random seed for reproducibility
    data = dev.xp.random.randn(*shape).astype(dtype or dev.xp.float32)
    return tensor(data, requires_grad=requires_grad, device=dev, dtype=dtype)

def arange(start, stop=None, step=1, *, dtype=None, requires_grad=False, device='cpu'):
    dev = Device(device)
    if stop is None:
        # Only one argument given: arange(stop)
        start, stop = 0, start
    data = dev.xp.arange(start, stop, step, dtype=dtype)
    return tensor(data, requires_grad=requires_grad, device=dev, dtype=dtype)

def linspace(start, stop, num=50, *, dtype=None, requires_grad=False, device='cpu'):
    dev = Device(device)
    data = dev.xp.linspace(start, stop, num=num, dtype=dtype)
    return tensor(data, requires_grad=requires_grad, device=dev, dtype=dtype)
