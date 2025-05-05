# Operation base class
class Operation:
    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad, *inputs):
        raise NotImplementedError