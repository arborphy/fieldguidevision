"""Thread lock for Metal/MPS operations in PyTorch."""

import threading

# PyTorch's MPS backend is not thread-safe when multiple threads submit Metal command buffers concurrently.
# This lock ensures only one thread accesses PyTorch MPS models at a time.
MPS_LOCK = threading.Lock()
