# tensor class with device support
from .backend import get_xp
from .core.tensor_fn import *
        
class tensor:
    def __init__(self, data, requires_grad=False, parents=None, op=None, device='cpu', dtype=None):
        self.device = device
        self.xp = get_xp(device) # get the appropriate array library
        if isinstance(data, tensor):
            data = data.data
        self.data = self.xp.array(data) if dtype is None else self.xp.array(data, dtype=dtype)
        self.dtype = self.data.dtype
        self.requires_grad = requires_grad
        self.grad = None
        self.parents = parents or []
        self.op = op
        self.is_leaf = (not self.parents and self.op is None)

    def to(self, device):
        if self.device == device:
            return self
        new_tensor = tensor(
            self.xp.array(self.data, dtype=self.dtype),
            requires_grad=self.requires_grad,
            device=device,
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

        # Implicit gradient only allowed for scalar outputs
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            grad = tensor(
                self.xp.ones_like(self.data),
                requires_grad=False,
                device=self.device,
                dtype=self.dtype
            )
        elif not isinstance(grad, tensor):
            grad = tensor(
                grad,
                requires_grad=False,
                device=self.device,
                dtype=self.dtype
            )

        # Accumulate gradient only for leaf tensors
        if self.is_leaf:
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        # Propagate gradients
        if self.op:
            parent_grads = self.op.backward(grad, *self.parents)
            for parent, parent_grad in zip(self.parents, parent_grads):
                parent.backward(tensor(
                    parent_grad,
                    requires_grad=False,
                    device=parent.device,
                    dtype=parent.dtype
                ), visited=visited)

        # Free graph memory
        self.op = None
        self.parents = None

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
    
    # helper function to process tensor type index
    def _process_index(self, idx):
        if isinstance(idx, tensor):
            return idx.data
        elif isinstance(idx, slice):
            return slice(
                self._process_index(idx.start),
                self._process_index(idx.stop),
                self._process_index(idx.step)
            )
        elif isinstance(idx, (list, tuple)):
            return type(idx)(self._process_index(i) for i in idx)
        else:
            return idx

    def __getitem__(self, idx):
        processed_idx = self._process_index(idx)
        return self._apply_op(None, getitem, unary=True, idx=processed_idx)
    
    def __neg__(self):
        return self._apply_op(None, neg, unary=True)

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
    
    def argmax(self, axis=None):
        return tensor(self.xp.argmax(self.data, axis=axis), device=self.device, dtype=self.dtype)
    
    def argmin(self, axis=None):
        return tensor(self.xp.argmin(self.data, axis=axis), device=self.device, dtype=self.dtype)
