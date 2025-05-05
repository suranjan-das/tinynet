# backend.py

_backend = "numpy"  # Default
USE_CUPY = False

try:
    import cupy
    if cupy.cuda.runtime.getDeviceCount() > 0:
        USE_CUPY = True
except (ImportError, RuntimeError):
    USE_CUPY = False

import numpy

def get_backend():
    return _backend

def get_xp():
    if _backend == "cupy":
        return cupy
    return numpy

def set_backend(name):
    global _backend
    name = name.lower()
    if name == "cupy":
        if not USE_CUPY:
            raise RuntimeError("CuPy not available or no CUDA device found.")
        _backend = "cupy"
    elif name == "numpy":
        _backend = "numpy"
    else:
        raise ValueError("Backend must be 'numpy' or 'cupy'")
