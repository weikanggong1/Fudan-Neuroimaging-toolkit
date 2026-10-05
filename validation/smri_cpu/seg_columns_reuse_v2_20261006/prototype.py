"""Private v2 guard revision; reuse frozen v1 binary; no automatic numerical execution."""
import ctypes
import hashlib
import os
from pathlib import Path

import torch
from torch.nn import functional as F


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


def ordinary_tensor_without_forward_ad(value):
    # Parameter itself is supported; subclasses and dual tensors keep the old path.
    if type(value) not in (torch.Tensor, torch.nn.Parameter):
        return False
    return torch.autograd.forward_ad.unpack_dual(value).tangent is None


class DlInfo(ctypes.Structure):
    _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                ("symbol", ctypes.c_char_p), ("symbol_address", ctypes.c_void_p)]


class ColumnsReuse:
    def __init__(self, library, provider_sha256, *, allow_compute=False, allow_bounded_contracts=False):
        self.allow_compute = bool(allow_compute)
        self.allow_bounded_contracts = bool(allow_bounded_contracts)
        torch_cpu = Path(torch.__file__).resolve().parent / "lib/libtorch_cpu.so"
        # NOLOAD refuses to introduce a new BLAS provider. Keep this handle alive.
        self.torch_handle = ctypes.CDLL(str(torch_cpu), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
        self.sgemm = getattr(self.torch_handle, "sgemm_")
        self.sgemm_address = ctypes.cast(self.sgemm, ctypes.c_void_p).value
        process = ctypes.CDLL(None)
        self.dladdr = process.dladdr
        self.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        self.dladdr.restype = ctypes.c_int
        info = DlInfo()
        if self.dladdr(ctypes.c_void_p(self.sgemm_address), ctypes.byref(info)) != 1:
            raise RuntimeError("SGEMM public provider identity could not be established")
        self.provider = Path(info.filename.decode()).resolve()
        if not self.provider.name.startswith("libmkl_intel_lp64.so"):
            raise RuntimeError("SGEMM is not the recorded Torch MKL LP64 provider")
        if file_sha(self.provider) != provider_sha256:
            raise RuntimeError("SGEMM provider bytes differ from the frozen interface report")
        self.provider_sha256 = provider_sha256
        self.library = ctypes.CDLL(str(Path(library).resolve()), mode=os.RTLD_LOCAL | os.RTLD_NOW)
        self.library.fnit_columns_abi_description.argtypes = []
        self.library.fnit_columns_abi_description.restype = ctypes.c_int
        if self.library.fnit_columns_abi_description() != 10404:
            raise RuntimeError("F32/int32 ABI mismatch")
        self.copy = self.library.fnit_copy_columns_f32
        self.copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int64] * 5
        self.copy.restype = ctypes.c_int
        self.gemm = self.library.fnit_same_provider_sgemm_f32
        self.gemm.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64]
        self.gemm.restype = ctypes.c_int

    def _provider_still_matches(self):
        info = DlInfo()
        address = ctypes.cast(getattr(self.torch_handle, "sgemm_"), ctypes.c_void_p).value
        if address != self.sgemm_address or self.dladdr(ctypes.c_void_p(address), ctypes.byref(info)) != 1:
            raise RuntimeError("public SGEMM address changed")
        if ctypes.cast(ctypes.CDLL(None).sgemm_, ctypes.c_void_p).value != address:
            raise RuntimeError("global and loaded Torch SGEMM addresses diverged")
        if Path(info.filename.decode()).resolve() != self.provider or file_sha(self.provider) != self.provider_sha256:
            raise RuntimeError("public SGEMM provider changed")

    def forward(self, layer, image):
        if not self.allow_compute:
            raise RuntimeError("numerical contracts and MRI are not authorized for this frozen prototype")
        from fnit.synthseg_parc.cpu_conv import CPUInferenceConv3d, cpu_autocast_enabled
        if (type(layer) is not CPUInferenceConv3d
                or not all(ordinary_tensor_without_forward_ad(value)
                           for value in (image, layer.weight, layer.bias))):
            return layer(image)
        if not (image.ndim == 5 and image.shape[:2] == (1, 72)
                and image.device.type == "cpu" and image.dtype == torch.float32 and not image.requires_grad
                and layer.weight.device.type == "cpu" and layer.weight.dtype == torch.float32
                and tuple(layer.weight.shape) == (24, 72, 3, 3, 3)
                and layer.bias is not None and layer.bias.device.type == "cpu" and layer.bias.dtype == torch.float32
                and tuple(layer.bias.shape) == (24,) and not layer.training and not torch.is_grad_enabled()
                and not torch.backends.mkldnn.enabled and not cpu_autocast_enabled()
                and layer.stride == layer.dilation == (1, 1, 1) and layer.padding == (1, 1, 1)
                and layer.groups == 1 and layer.padding_mode == "zeros"
                and not layer._forward_hooks and not layer._forward_pre_hooks and not layer._backward_hooks):
            return layer(image)
        from torch.nn.modules import module
        if (module._global_forward_hooks or module._global_forward_pre_hooks or module._global_backward_hooks
                or getattr(module, "_global_backward_pre_hooks", {}) or layer._backward_pre_hooks):
            return layer(image)
        if tuple(image.shape) != (1, 72, 192, 224, 256) and not self.allow_bounded_contracts:
            return layer(image)
        self._provider_still_matches()
        _, _, depth, height, width = image.shape
        plane = height * width
        maximum_m = min(14, depth) * plane
        if min(depth, height, width) < 1 or maximum_m > 2**31 - 1:
            raise ValueError("invalid original slab matrix geometry")
        # Workspace lives only inside one layer call; never retained by self/model.
        columns_storage = image.new_empty((1944 * maximum_m,))
        output_storage = image.new_empty((24 * maximum_m,))
        output = image.new_empty((1, 24, depth, height, width))
        weight = layer.weight.contiguous()
        for start in range(0, depth, 14):
            stop = min(start + 14, depth)
            lower, upper = max(0, start - 1), min(depth, stop + 1)
            chunk = image[:, :, lower:upper]
            pad_before, pad_after = max(0, 1 - start), max(0, stop + 1 - depth)
            if pad_before or pad_after:
                chunk = F.pad(chunk, (0, 0, 0, 0, pad_before, pad_after))
            chunk = chunk.contiguous()
            slab_depth = stop - start
            m = slab_depth * plane
            # Compact flattened prefix preserves the LAST slab's actual row stride M.
            columns = columns_storage[:1944 * m].view(1944, m)
            slab = output_storage[:24 * m].view(24, m)
            code = self.copy(chunk.data_ptr(), columns.data_ptr(), chunk.shape[2], height, width, slab_depth, columns.numel())
            if code:
                raise RuntimeError("copy-only im2col rejected input: " + str(code))
            slab.copy_(layer.bias.reshape(24, 1))
            code = self.gemm(self.sgemm_address, columns.data_ptr(), weight.data_ptr(), slab.data_ptr(), m)
            if code:
                raise RuntimeError("same-provider SGEMM rejected geometry: " + str(code))
            output[:, :, start:stop].copy_(slab.view(1, 24, slab_depth, height, width))
        self._provider_still_matches()
        return output
