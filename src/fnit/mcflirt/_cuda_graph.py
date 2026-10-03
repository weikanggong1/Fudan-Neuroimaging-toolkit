"""CUDA graph policy for the project's fixed PyTorch CUDA allocator."""

import os


def cuda_graph_capture_enabled():
    """Match PyTorch 2.5.1's presence-based uncached-allocator flag.

    Set allocator configuration before the first CUDA allocation. The upstream
    allocator treats even an empty value or ``"0"`` as disabling its cache;
    uncached cudaMalloc allocations cannot occur during graph capture.
    """
    return "PYTORCH_NO_CUDA_MEMORY_CACHING" not in os.environ
