from ..tensor import tensor
from ..ops.activations import Sigmoid, ReLU

def acivation_op(x, OpClass):
    op = OpClass()
    data = op.forward(x)
    return tensor(data, x.requires_grad, parents=[x], op=op, device=x.device.device_str, dtype=x.dtype)

sigmoid = lambda x: acivation_op(x, Sigmoid)
relu = lambda x: acivation_op(x, ReLU)