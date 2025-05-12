# tensor class with device support
from .device import Device
from .core.tensor_fn import *
        
class tensor:
    def __init__(self, data, requires_grad=False, parents=None, op=None, device='cpu', dtype=None):
        self.device = Device(device) if not isinstance(device, Device) else device
        self.data = self.device.array(data) if dtype is None else self.device.array(data, dtype=dtype)
        self.dtype = self.data.dtype
        self.requires_grad = requires_grad
        self.grad = None
        if requires_grad:
            self.grad = tensor(
                self.device.zeros_like(self.data),
                requires_grad=False,
                device=self.device,
                dtype=self.dtype
            )
        self.parents = parents or []
        self.op = op

    def to(self, device):
        if self.device.device_str == device:
            return self
        new_device = Device(device)
        new_tensor = tensor(
            new_device.array(self.data.tolist(), dtype=self.dtype),
            requires_grad=self.requires_grad if self.requires_grad else False,
            device=new_device,
            dtype=self.dtype
        )
        return new_tensor

    def backward(self, grad=None, visited=None):
        if not self.requires_grad:
            return
        if visited is None:
            visited = set()
        if id(self) in visited:
            return
        visited.add(id(self))

        if self.data.size != 1 and grad is None:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")

        if grad is None:
            grad = tensor(
                self.device.ones_like(self.data),
                requires_grad=False,
                device=self.device,
                dtype=self.dtype
            )

        grad = grad if isinstance(grad, tensor) else tensor(
            grad,
            requires_grad=False,
            device=self.device,
            dtype=self.dtype
        )

        self.grad = self.grad + grad if self.grad is not None else grad

        if self.op:
            parent_grads = self.op.backward(grad, *self.parents)
            for parent, parent_grad in zip(self.parents, parent_grads):
                parent.backward(tensor(
                    parent_grad,
                    requires_grad=False,
                    device=parent.device,
                    dtype=parent.dtype
                ), visited=visited)

    def to_cpu(self):
        """Return data and grad as NumPy arrays for plotting."""
        data = self.device.asnumpy(self.data)
        grad = self.device.asnumpy(self.grad.data) if self.grad is not None else None
        return data, grad

    def __repr__(self):
        return self.data.__repr__().replace('array', 'tensor')

    @property
    def shape(self):
        return self.data.shape
    
    def __len__(self):
        return len(self.data)


    def _apply_op(self, other, op_fn, scalar=False, is_scalar_first=False, unary=False, **kwargs):
        if unary:
            data, requires_grad, op = op_fn(self, **kwargs)
            parents = [self]
        else:
            if scalar:
                data, requires_grad, op = op_fn(other, self, is_scalar_first=is_scalar_first)
                parents = [self]
            else:
                data, requires_grad, op = op_fn(self, other, **kwargs)
                parents = [self, other]
        return tensor(data, requires_grad, parents=parents, op=op, device=self.device, dtype=data.dtype)
    
    @property
    def T(self):
        return self._apply_op(None, transpose, unary=True)
    
    def reshape(self, *shape):
        return self._apply_op(None, reshape, unary=True, shape=shape)
    
    def __getitem__(self, idx):
        return self._apply_op(None, getitem, unary=True, idx=idx)

    def __add__(self, other):
        return self._apply_op(other, scalar_add if isinstance(other, (int, float)) else add, scalar=isinstance(other, (int, float)))

    def __radd__(self, other):
        return self._apply_op(other, scalar_add if isinstance(other, (int, float)) else add, scalar=isinstance(other, (int, float)), is_scalar_first=True)

    def __sub__(self, other):
        return self._apply_op(other, scalar_subtract if isinstance(other, (int, float)) else subtract, scalar=isinstance(other, (int, float)))

    def __rsub__(self, other):
        return self._apply_op(other, scalar_subtract if isinstance(other, (int, float)) else subtract, scalar=isinstance(other, (int, float)), is_scalar_first=True)

    def __mul__(self, other):
        return self._apply_op(other, scalar_multiply if isinstance(other, (int, float)) else multiply, scalar=isinstance(other, (int, float)))

    def __rmul__(self, other):
        return self._apply_op(other, scalar_multiply if isinstance(other, (int, float)) else multiply, scalar=isinstance(other, (int, float)))

    def __truediv__(self, other):
        return self._apply_op(other, scalar_divide if isinstance(other, (int, float)) else divide, scalar=isinstance(other, (int, float)))

    def __rtruediv__(self, other):
        return self._apply_op(other, scalar_divide if isinstance(other, (int, float)) else divide, scalar=isinstance(other, (int, float)), is_scalar_first=True)
    
    def __pow__(self, other):
        return self._apply_op(other, scalar_pow if isinstance(other, (int, float)) else pow, scalar=isinstance(other, (int, float)))
    
    def __rpow__(self, other):
        return self._apply_op(other, scalar_pow if isinstance(other, (int, float)) else pow, scalar=isinstance(other, (int, float)), is_scalar_first=True)

    def __matmul__(self, other):
        return self._apply_op(other, matmul)
    
    def sum(self, axis=None, keepdims=False):
        return self._apply_op(None, sum, unary=True, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return self._apply_op(None, mean, unary=True, axis=axis, keepdims=keepdims)
    
    def exp(self):
        return self._apply_op(None, exp, unary=True)
    
    def log(self):
        return self._apply_op(None, log, unary=True)
    
    def sqrt(self):
        return self._apply_op(None, sqrt, unary=True)
    
    def log_softmax(self, axis=-1):
        return self._apply_op(None, log_softmax, unary=True, axis=axis)
