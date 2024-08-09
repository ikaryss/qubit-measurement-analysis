import numpy as np
import scipy


class ArrayModule:
    def __init__(self, device="cpu"):
        self.device = device
        if device == "cpu":
            self.np = np
            self.scipy = scipy
            self.array_transport = self.asarray
        elif device.startswith("cuda"):
            import cupy as cp
            import cupyx.scipy

            self.np = cp
            self.scipy = cupyx.scipy
            self.array_transport = cp.asnumpy
            cuda_device = int(device.split(":")[1]) if ":" in device else 0
            cp.cuda.Device(cuda_device).use()

        else:
            raise ValueError(f"Unsupported device: {device}")

    def __getattr__(self, name):
        return getattr(self.np, name)

    def __repr__(self) -> str:
        return f"{self.device} array module with {self.np} as numpy and {self.scipy} as scipy"

    def free_all_blocks(self):
        if self.device.startswith("cuda"):
            self.get_default_memory_pool().free_all_blocks()
            self.get_default_pinned_memory_pool().free_all_blocks()

    @property
    def dtype(self):
        return self.complex64
