"""Optional exact-shape CPU columns reuse; CUDA never imports this module."""
import ctypes
import hashlib
import os
import subprocess
import threading
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
    # Raw-pointer copy must not ignore lazy negative/conjugate view metadata.
    if value.is_neg() or value.is_conj():
        return False
    return torch.autograd.forward_ad.unpack_dual(value).tangent is None


class DlInfo(ctypes.Structure):
    _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                ("symbol", ctypes.c_char_p), ("symbol_address", ctypes.c_void_p)]


class _Columns:
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
        if library is None:
            # Runtime/provider metadata gate before compiler/cache work.
            return
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
            return None
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
            return None
        from torch.nn.modules import module
        if (module._global_forward_hooks or module._global_forward_pre_hooks or module._global_backward_hooks
                or getattr(module, "_global_backward_pre_hooks", {}) or layer._backward_pre_hooks):
            return None
        if tuple(image.shape) != (1, 72, 192, 224, 256) and not self.allow_bounded_contracts:
            return None
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


_WEIGHT_SHA = "a489a4a212aa2b6ba53d386b6a3ebba90e03e13f544f34c646ea67d7a71935dd"
_BIAS_SHA = "a7203077944d5eb6afc212a7f58eeffc77f2481f1921621b37f08466bcb709bc"
_ENGINES = {}
_ENGINE_MUTEX = threading.RLock()


def _parameter_sha(value):
    return hashlib.sha256(memoryview(value.detach().numpy()).cast("B")).hexdigest()


def _eligible(layer, image):
    """Reject all unsupported states before loader/compiler/provider work."""
    # CUDA/foreign image returns without even reading parameters or global flags.
    if getattr(getattr(image, "device", None), "type", None) != "cpu":
        return False
    from .cpu_conv import CPUInferenceConv3d, cpu_autocast_enabled
    if type(layer) is not CPUInferenceConv3d or not getattr(layer, "_fnit_columns_reuse", False):
        return False
    if not all(ordinary_tensor_without_forward_ad(value) for value in (image, layer.weight, layer.bias)):
        return False
    if not (tuple(image.shape) == (1, 72, 192, 224, 256) and image.layout == torch.strided and image.dtype == torch.float32
            and not image.requires_grad and not layer.training and not torch.is_grad_enabled()
            and layer.weight.device.type == layer.bias.device.type == "cpu"
            and layer.weight.dtype == layer.bias.dtype == torch.float32
            and tuple(layer.weight.shape) == (24, 72, 3, 3, 3) and tuple(layer.bias.shape) == (24,)
            and layer.weight.is_contiguous() and layer.bias.is_contiguous()
            and not torch.backends.mkldnn.enabled and not cpu_autocast_enabled()
            and torch.get_num_threads() == 8 and not torch.jit.is_tracing() and not torch.jit.is_scripting()
            and layer.stride == layer.dilation == (1, 1, 1) and layer.padding == (1, 1, 1)
            and layer.groups == 1 and layer.padding_mode == "zeros"
            and not layer._forward_hooks and not layer._forward_pre_hooks and not layer._backward_hooks
            and not layer._backward_pre_hooks):
        return False
    from torch.nn.modules import module
    if (module._global_forward_hooks or module._global_forward_pre_hooks or module._global_backward_hooks
            or getattr(module, "_global_backward_pre_hooks", {})):
        return False
    # Only the accepted real model's parameter bytes are covered by this narrow release.
    # Checking values also catches .data mutations that do not increment Tensor._version.
    for value, expected in ((layer.weight, _WEIGHT_SHA), (layer.bias, _BIAS_SHA)):
        actual = _parameter_sha(value)
        if actual != expected:
            return False
    return True


def _engine():
    from ._cpu_columns_build import build_artifact
    with _ENGINE_MUTEX:
        artifact = build_artifact(torch)
        key = (artifact["key"], artifact["library"])
        if key not in _ENGINES:
            _ENGINES[key] = _Columns(artifact["library"], artifact["provider_sha256"], allow_compute=True)
        return _ENGINES[key]


def try_columns_reuse(layer, image):
    """Return one guarded CPU result or None before numerical work begins.

    Unsupported states/build/provider failures retain the mature slabs. Once
    copy/SGEMM can start, exceptions propagate without a hidden second solve.
    No caller precision, allocator, environment or thread settings are written.
    """
    if not _eligible(layer, image):
        return None
    try:
        engine = _engine()
    except (OSError, RuntimeError, ValueError, TimeoutError, AttributeError, subprocess.SubprocessError):
        # Cache failure accounting is handled by the build helper's identity key.
        return None
    # Do not catch numerical errors: executing old convolution after a partial
    # candidate call would conceal failure and duplicate work.
    return engine.forward(layer, image)
