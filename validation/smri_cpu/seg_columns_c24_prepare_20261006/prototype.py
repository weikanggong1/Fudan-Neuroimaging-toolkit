"""Isolated C24/32-plane prototype. No production integration or automatic compute.

The already accepted CPU provider guard supplies the same loaded Torch SGEMM;
the new private DSO exports only its independent C24 names. No old binary is
recompiled or overwritten by this diagnostic.
"""
from pathlib import Path
import ctypes
import hashlib

import torch
from torch.nn import functional as F
from fnit.synthseg_parc.cpu_columns import _Columns, ordinary_tensor_without_forward_ad


_WEIGHT_SHA = "080e9e82b43ebd3eb35fa991a5d264b073c35e0c027aea9534510470e552d699"
_BIAS_SHA = "e5c6f0bafe28b340b599bd6b52faaec3a1c462edd6676103c0efddfbd6a73944"


def parameter_sha(value):
    return hashlib.sha256(memoryview(value.detach().numpy()).cast("B")).hexdigest()


class ColumnsC24(_Columns):
    def __init__(self, library, provider_sha256, *, allow_compute=False, allow_bounded_contracts=False):
        # Metadata construction is CPU-only and never allocates a tensor.
        # The accepted base with library=None only binds the already loaded provider.
        super().__init__(None, provider_sha256, allow_compute=allow_compute,
                         allow_bounded_contracts=allow_bounded_contracts)
        import os
        self.library = ctypes.CDLL(str(Path(library).resolve()), mode=os.RTLD_LOCAL | os.RTLD_NOW)
        self.library.fnit_columns_c24_abi_description.argtypes = []
        self.library.fnit_columns_c24_abi_description.restype = ctypes.c_int
        if self.library.fnit_columns_c24_abi_description() != 10404:
            raise RuntimeError("C24 F32/int32 LP64 ABI mismatch")
        self.copy = self.library.fnit_copy_columns_c24_f32
        self.copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int64] * 5
        self.copy.restype = ctypes.c_int
        self.gemm = self.library.fnit_same_provider_sgemm_c24_f32
        self.gemm.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64]
        self.gemm.restype = ctypes.c_int

    def forward(self, layer, image):
        # Early foreign-device return does not inspect parameters or provider.
        if getattr(getattr(image, "device", None), "type", None) != "cpu":
            return layer(image)
        if not self.allow_compute:
            raise RuntimeError("C24 numerical contracts and MRI are not authorized")
        from fnit.synthseg_parc.cpu_conv import CPUInferenceConv3d, cpu_autocast_enabled
        if (type(layer) is not CPUInferenceConv3d
                or not all(ordinary_tensor_without_forward_ad(v) for v in (image, layer.weight, layer.bias))):
            return layer(image)
        if not (image.ndim == 5 and tuple(image.shape[:2]) == (1, 24)
                and image.layout == torch.strided and image.dtype == torch.float32 and not image.requires_grad
                and layer.weight.device.type == layer.bias.device.type == "cpu"
                and layer.weight.dtype == layer.bias.dtype == torch.float32
                and tuple(layer.weight.shape) == (24, 24, 3, 3, 3) and tuple(layer.bias.shape) == (24,)
                and layer.weight.is_contiguous() and layer.bias.is_contiguous()
                and not layer.training and not torch.is_grad_enabled() and not cpu_autocast_enabled()
                and not torch.backends.mkldnn.enabled and torch.get_num_threads() == 8
                and layer.stride == layer.dilation == (1, 1, 1) and layer.padding == (1, 1, 1)
                and layer.groups == 1 and layer.padding_mode == "zeros"
                and not layer._forward_hooks and not layer._forward_pre_hooks and not layer._backward_hooks
                and not layer._backward_pre_hooks and not torch.jit.is_tracing() and not torch.jit.is_scripting()):
            return layer(image)
        from torch.nn.modules import module
        if (module._global_forward_hooks or module._global_forward_pre_hooks or module._global_backward_hooks
                or getattr(module, "_global_backward_pre_hooks", {})):
            return layer(image)
        if tuple(image.shape) != (1, 24, 192, 224, 256) and not self.allow_bounded_contracts:
            return layer(image)
        if parameter_sha(layer.weight) != _WEIGHT_SHA or parameter_sha(layer.bias) != _BIAS_SHA:
            return layer(image)
        self._provider_still_matches()
        _, _, depth, height, width = image.shape
        plane = height * width
        maximum_m = min(32, depth) * plane
        if min(depth, height, width) < 1 or maximum_m > 2**31 - 1:
            raise ValueError("invalid original C24 slab matrix geometry")
        # Exactly one layer-call lifetime. Nothing is cached on self or the model.
        columns_storage = image.new_empty((648 * maximum_m,))
        output_storage = image.new_empty((24 * maximum_m,))
        output = image.new_empty((1, 24, depth, height, width))
        weight = layer.weight.contiguous()
        for start in range(0, depth, 32):
            stop = min(start + 32, depth)
            lower, upper = max(0, start - 1), min(depth, stop + 1)
            chunk = image[:, :, lower:upper]
            before, after = max(0, 1 - start), max(0, stop + 1 - depth)
            if before or after:
                chunk = F.pad(chunk, (0, 0, 0, 0, before, after))
            chunk = chunk.contiguous()
            slab_depth, m = stop - start, (stop - start) * plane
            # Last partial slab is compact K x actual-M, never max-M row stride.
            columns = columns_storage[:648 * m].view(648, m)
            slab = output_storage[:24 * m].view(24, m)
            code = self.copy(chunk.data_ptr(), columns.data_ptr(), chunk.shape[2], height, width,
                             slab_depth, columns.numel())
            if code:
                raise RuntimeError("copy-only im2col rejected C24 input: " + str(code))
            slab.copy_(layer.bias.reshape(24, 1))
            code = self.gemm(self.sgemm_address, columns.data_ptr(), weight.data_ptr(), slab.data_ptr(), m)
            if code:
                raise RuntimeError("same-provider SGEMM rejected C24 geometry: " + str(code))
            output[:, :, start:stop].copy_(slab.view(1, 24, slab_depth, height, width))
        self._provider_still_matches()
        return output
