"""CPU eligibility and layout wrapper; serial kernel math retained separately."""
import sys

def try_cpu_centroid(values):
    import torch
    if isinstance(values, torch.Tensor) and values.device.type != 'cpu':
        return None
    if (type(values) is not torch.Tensor or values.dtype != torch.float32
            or values.ndim != 3 or values.layout != torch.strided
            or values.is_nested or values.requires_grad
            or values.is_neg() or values.is_conj() or sys.byteorder != 'little'
            or torch._C._functorch.is_functorch_wrapped_tensor(values)
            or torch.autograd.forward_ad.unpack_dual(values).tangent is not None):
        return None
    if not bool(torch.isfinite(values).all()):
        return None
    mass = values.double()
    total = mass.sum()
    if float(total) <= 0:
        return None
    import numpy as np
    from ._cpu_centroid_serial import centroid_serial
    packed = np.array(values.detach().numpy().ravel(order='F'), dtype=np.float32, copy=True)
    packed.setflags(write=False)
    result = centroid_serial(packed, *map(int, values.shape), np.float64(0.0))
    return np.asarray(result, dtype=np.float64)
