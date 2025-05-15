from ..tensor import tensor
from ..backend import get_xp
from ..tensor_init import *

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def register_parameter(self, name, param):
        if isinstance(param, tensor) and param.requires_grad:
            self._parameters[name] = param

    def add_module(self, name, module):
        if isinstance(module, Module):
            self._modules[name] = module

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix=''):
        for name, param in self._parameters.items():
            yield f"{prefix}{name}", param
        for name, module in self._modules.items():
            yield from module.named_parameters(f"{prefix}{name}.")

    def train(self):
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self):
        self.training = False
        for module in self._modules.values():
            module.eval()

    def __setattr__(self, name, value):
        if isinstance(value, tensor):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        object.__setattr__(self, name, value)


class Linear(Module):
    def __init__(self, in_features, out_features, *, bias=True, gain=1.0, device='cpu', dtype=None):
        super().__init__()
        xp = get_xp(device=device)
        fan_in, fan_out = in_features, out_features
        std = gain * xp.sqrt(2.0 / (fan_in + fan_out))
        self.weight = tensor(xp.random.normal(0.0, std, size=(fan_in, fan_out)), requires_grad=True, device=device, dtype=dtype)
        if bias:
            self.bias = tensor(xp.random.normal(0.0, std, size=(fan_out,)), requires_grad=True, device=device, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x):
        return x @ self.weight + self.bias

