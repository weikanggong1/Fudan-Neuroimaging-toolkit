"""Saved-state CPU projection diagnostic; no production or CUDA changes."""

def project_source_float_division(torch, voxel_gradient, voxel_sizes):
    """Project one accepted Float32 CPU saved gradient with three header divisors.

    The independent NumPy reference must first be checked word for word,
    including signed zero, before interpreting any later gradient prefix.
    This single broadcast method does not claim an ATen instruction trace.
    """
    if (voxel_gradient.device.type != 'cpu' or voxel_gradient.dtype != torch.float32
            or tuple(voxel_gradient.shape) != (3,24,28,24)
            or voxel_gradient.requires_grad or len(voxel_sizes) != 3):
        raise ValueError('one accepted Float32 CPU gradient shape only')
    divisors = torch.tensor(tuple(voxel_sizes), dtype=torch.float32, device='cpu')
    divisors = divisors.reshape(3,1,1,1)
    if not bool(torch.isfinite(divisors).all()) or not bool((divisors > 0).all()):
        raise ValueError('three positive finite Float32 header divisors required')
    return voxel_gradient / divisors
