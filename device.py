from .backend import get_backend, get_xp, set_backend

class Device:
    def __init__(self, device_str: str = "cpu"):
        self.device_str = device_str.lower()
        self.backend = get_backend()
        self.xp = get_xp()

        if self.device_str == "cpu":
            self.is_cuda = False
        elif self.device_str.startswith("cuda"):
            set_backend("cupy")
            self.xp = get_xp()
            if self.backend != "cupy":
                raise RuntimeError("CuPy backend required for CUDA device, but current backend is NumPy.")

            self.is_cuda = True
            device_id = int(self.device_str.split(":")[-1]) if ":" in self.device_str else 0
            self.device_id = device_id
            self.xp.cuda.Device(device_id).use()
        else:
            raise ValueError("Device must be 'cpu' or 'cuda[:id]'")

    def array(self, data, dtype=None):
        return self.xp.array(data, dtype=dtype) if dtype else self.xp.array(data)

    def zeros_like(self, data):
        return self.xp.zeros_like(data)

    def ones_like(self, data):
        return self.xp.ones_like(data)

    def asnumpy(self, data):
        # Only CuPy arrays need conversion
        if self.backend == "cupy":
            return self.xp.asnumpy(data)
        return data