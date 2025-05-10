tinynet/
├── __init__.py
├── backend.py                # Backend logic (NumPy/CuPy switching)
├── device.py                 # Device abstraction (cpu/cuda)
├── tensor.py                 # Tensor class and core data structure
├── tensor_init.py           # Tensor creation utilities (zeros, ones, randn, etc.)
├── core/
│   ├── base_fn.py            # Possibly core math ops or foundational functions
│   ├── tensor_fn.py          # Tensor manipulation functions (reshape, transpose, etc.)
│   └── utils.py              # Helpers
├── functional/
│   ├── __init__.py
│   └── activations.py        # Stateless functional layers like ReLU, Sigmoid, etc.
├── nn/                      # Leave as a module for future layers (Linear, Conv, etc.)
├── ops/
│   ├── activations.py        # Op-level implementations
│   ├── base.py               # Abstract operation class?
│   └── basic_ops.py          # Add, Mul, MatMul, etc.
├── optim/                   # Optimizers (SGD, Adam, etc.)
├── tests/                   # Unit tests
├── README.md
├── dir_structure.md         # (Nice touch — remove or rename to `docs/` if this grows)
