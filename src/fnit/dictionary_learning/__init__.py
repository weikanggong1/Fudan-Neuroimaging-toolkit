"""Standalone dictionary learning for post analysis.

``fit_dictionary_learning`` fits array inputs with scikit-learn on the CPU.
``fit_dictionary_learning_streaming`` fits HDF5 inputs with PyTorch on CUDA.
Both return a mapping of modality names to atom by feature dictionaries.
"""

from .cpu import fit_dicl
from .torch_backend import GPU_ALGORITHM_VERSION, fit_dicl_gpu_streaming

# Descriptive names preserve the mature functions' arguments and results.
fit_dictionary_learning = fit_dicl
fit_dictionary_learning_streaming = fit_dicl_gpu_streaming

__all__ = [
    "fit_dictionary_learning",
    "fit_dictionary_learning_streaming",
    "fit_dicl",
    "fit_dicl_gpu_streaming",
    "GPU_ALGORITHM_VERSION",
]
