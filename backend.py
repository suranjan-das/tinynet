# backend.py
# This module provides a way to handle both NumPy and CuPy arrays, depending on the availability of CUDA.
# It defines a function to get the appropriate array library based on the device specified.

# It also includes a check for the availability of CuPy and CUDA devices.
_CUPY_AVAILABLE = False

import numpy
try:
    import cupy
    if cupy.cuda.runtime.getDeviceCount() > 0:
        _CUPY_AVAILABLE = True
        cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)
except (ImportError, RuntimeError):
    _CUPY_AVAILABLE = False


def get_xp(device="cpu"):
    if device.startswith("cuda") :
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available or no CUDA device found.")
        return cupy
    elif device == "cpu":
        return numpy
    else:
        raise ValueError("Device must be 'cpu' or 'cuda[:id]'")