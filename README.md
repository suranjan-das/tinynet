# TinyNet

**TinyNet** is a minimalist deep learning library built for **fun and educational purposes** 🎓.  
It mimics the PyTorch API and supports both **CPU (via NumPy)** and **GPU (via CuPy)**.  
This project is great for understanding how deep learning frameworks work under the hood!

---

## ✨ Features

- `tensor` class with automatic differentiation (autograd)
- Modular `nn.Module` system like PyTorch
- Common layers and activations: `Linear`, `Sigmoid`, etc.
- Loss functions: `CrossEntropyLoss` and more
- Optimizers: `SGD` with momentum
- Device support: **CPU (NumPy)** and **GPU (CuPy)**
- Designed to be small, readable, and educational

---

## 🚀 Installation

Clone the repo:
```bash
git clone https://github.com/suranjan-das/tinynet.git
```

## Example
```python
import tinynet as tn
import tinynet.nn as nn
import tinynet.optim as optim

# Generate synthetic input and target
input = tn.randn(100, 15).to('cuda')
target = tn.arange(100).to('cuda')

# Define a simple feedforward model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(15, 25, gain=0.1)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(25, 100, gain=0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x

# Instantiate model, loss function, and optimizer
model = Model().to('cuda')
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.99)

# Training loop
for epoch in range(100):
    pred = model(input)
    loss = loss_fn(pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

```

## Why Use TinyNet?
This repo is perfect if you:

- Want to learn how deep learning frameworks work internally
- Are building a custom autograd engine for experimentation
- Prefer a tiny, hackable codebase
- Enjoy writing things from scratch for deeper understanding

