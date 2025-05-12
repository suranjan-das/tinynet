class Loss:
    def __call__(self, pred, target):
        raise NotImplementedError

class MSELoss(Loss):
    def __call__(self, pred, target):
        diff = pred - target
        return (diff * diff).mean()

class CrossEntropyLoss(Loss):
    def __call__(self, logits, target):
        # logits: (N, C), target: (N,)
        # Apply softmax and compute negative log likelihood
        log_probs = logits.log_softmax(axis=1)  # Assuming you implement log_softmax
        nll = -1 * log_probs[range(len(target)), target]
        return nll.mean()
