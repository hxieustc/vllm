# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA driver ordering contract after native writes to a GPU destination.

Enum values are from the CUDA13 cuda.h shipped with the serving image.
Native completion is required before complete_reads; this is not a DMA fence.
"""

import ctypes


class GPUVisibility:
    def __init__(self, ordering, options, flush):
        self.ordering = int(ordering)
        self.options = int(options)
        self._flush = flush
        if self.ordering < 100 and not self.options & 1:
            raise RuntimeError("GPU cannot prove owner visibility for GPUDirect writes")

    @classmethod
    def for_device(cls, ordinal):
        library = ctypes.CDLL("libcuda.so.1")
        get = library.cuDeviceGetAttribute
        get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        get.restype = ctypes.c_int

        def attribute(number):
            value = ctypes.c_int()
            code = get(ctypes.byref(value), number, int(ordinal))
            if code:
                raise RuntimeError(f"cuDeviceGetAttribute({number}) failed: {code}")
            return value.value

        def flush():
            function = library.cuFlushGPUDirectRDMAWrites
            function.argtypes = [ctypes.c_int, ctypes.c_int]
            function.restype = ctypes.c_int
            code = function(0, 100)  # CURRENT_CTX, TO_OWNER
            if code:
                raise RuntimeError(f"cuFlushGPUDirectRDMAWrites failed: {code}")

        return cls(attribute(118), attribute(117), flush)

    def complete_reads(self):
        if self.ordering < 100:
            self._flush()

    def proof(self):
        return dict(
            gpudirect_write_ordering=self.ordering,
            gpudirect_flush_options=self.options,
            visibility="native_owner_ordering"
            if self.ordering >= 100
            else "cuFlushGPUDirectRDMAWrites_TO_OWNER",
        )
