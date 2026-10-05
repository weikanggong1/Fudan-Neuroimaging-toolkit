"""Scoped CUDA policy and scalar diagnostics for the SynthSeg parcel chain."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .._dmri import configure_device


def validate_cudnn_tf32(policy):
    if policy is not None and not isinstance(policy, bool):
        raise ValueError("cudnn_tf32 must be True, False or None")


def check_cached_policy(component, policy, name):
    """Reject an explicitly supplied cache with another declared policy."""
    if component is None:
        return
    cached = getattr(component, "cudnn_tf32", True)
    validate_cudnn_tf32(cached)
    if cached is not policy:
        raise ValueError(
            f"{name} cached cudnn_tf32={cached!r} does not match "
            f"requested cudnn_tf32={policy!r}; construct matching models"
        )


def cuda_precision_state():
    return {"matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32)}


@contextmanager
def cuda_tf32_scope(device, policy, trace=None):
    """Use the existing SynthSeg matmul policy, then restore both CUDA flags.

    True/False declares cuDNN TF32; None inherits the caller's cuDNN flag.
    CUDA matmul TF32 remains enabled inside the scope, as in Segmenter.
    CPU does not write either flag. Caller autocast and all other backends
    are untouched. This process-global CUDA policy requires serial calls
    or separate processes for calls with different policies.
    """
    validate_cudnn_tf32(policy)
    previous = cuda_precision_state()
    location = (configure_device(None, configure_precision=False) if device is None
                else torch.device(device))
    cuda = location.type == "cuda"
    try:
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            if policy is not None:
                torch.backends.cudnn.allow_tf32 = policy
        if trace is not None:
            trace["cuda_precision_before"] = previous
            trace["cuda_precision_active"] = cuda_precision_state()
        yield
    finally:
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
            torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]
        if trace is not None:
            trace["cuda_precision_after"] = cuda_precision_state()
            trace["cuda_precision_restored"] = trace["cuda_precision_after"] == previous


def tensor_precision(tensor, *, operation, model=None):
    """Read actual entry state without hooks, tensor copies or synchronization."""
    def autocast_state(device_type):
        try:
            enabled = torch.is_autocast_enabled(device_type)
        except TypeError:  # Torch 2.1 compatibility.
            enabled = (torch.is_autocast_enabled() if device_type == "cuda"
                       else torch.is_autocast_cpu_enabled())
        if hasattr(torch, "get_autocast_dtype"):
            dtype = torch.get_autocast_dtype(device_type)
        else:
            dtype = (torch.get_autocast_gpu_dtype() if device_type == "cuda"
                     else torch.get_autocast_cpu_dtype())
        return {"enabled": bool(enabled), "dtype": str(dtype)}

    row = {"operation": operation, "device": str(tensor.device),
           "input_dtype": str(tensor.dtype), **cuda_precision_state(),
           "autocast": {name: autocast_state(name) for name in ("cpu", "cuda")}}
    if model is not None:
        row["model_dtypes"] = sorted({str(value.dtype) for value in model.parameters()})
    return row
