"""CPU batching of FLIRT's existing float/double affine composition."""

import torch


def _matrix_product(first, second):
    # Small CPU bmm uses unfused arithmetic, unlike the reference CPU mm.
    # addcmul preserves mm's left-to-right double FMA accumulation.
    result = first[:, :, 0, None] * second[:, None, 0, :]
    for index in range(1, first.shape[2]):
        result = torch.addcmul(result, first[:, :, index, None],
                              second[:, None, index, :])
    return result


def _homogeneous(linear, offset):
    matrix = torch.eye(4, dtype=linear.dtype).repeat(linear.shape[0], 1, 1)
    matrix[:, :3, :3] = linear
    matrix[:, :3, 3] = offset
    return matrix


def _axis_rotation(angles, axis, centre):
    vectors = torch.zeros((len(angles), 3), dtype=angles.dtype)
    vectors[:, axis] = angles
    theta = torch.linalg.vector_norm(vectors, dim=1).float()
    # The scalar reference compares float(theta) against a Python double.
    active = ~(theta.double() < 1e-8)
    direction = vectors / torch.where(active, theta, 1).to(angles.dtype)[:, None]
    x2 = torch.stack((-direction[:, 1], direction[:, 0], torch.zeros_like(angles)), dim=1)
    x2_norm = torch.linalg.vector_norm(x2, dim=1)
    replacement = torch.tensor([1., 0., 0.], dtype=angles.dtype)
    x2 = torch.where((x2_norm <= 0)[:, None], replacement, x2)
    x2 = x2 / torch.linalg.vector_norm(x2, dim=1)[:, None]
    x3 = torch.linalg.cross(direction, x2, dim=1)
    x3_norm = torch.linalg.vector_norm(x3, dim=1)
    x3 = x3 / torch.where(active, x3_norm, 1)[:, None]
    basis = torch.stack((x2, x3, direction), dim=2)
    cosine, sine = torch.cos(theta).to(angles.dtype), torch.sin(theta).to(angles.dtype)
    core = torch.eye(3, dtype=angles.dtype).repeat(len(angles), 1, 1)
    core[:, 0, 0] = core[:, 1, 1] = cosine
    core[:, 0, 1] = sine
    core[:, 1, 0] = -sine
    linear = _matrix_product(_matrix_product(basis, core), basis.transpose(1, 2))
    rotation = _homogeneous(linear, centre - linear @ centre)
    identity = torch.eye(4, dtype=angles.dtype)
    return torch.where(active[:, None, None], rotation, identity)


def fsl_affine_from_parameters_batch(parameters, centre, dof=12):
    """Compose CPU float64 input-to-reference scaled-mm matrices ``[B,4,4]``.

    The Euler basis, float32 angle/sine/cosine and double matrix composition
    follow ``core.fsl_affine_from_parameters``. No global precision settings
    are modified. Parameter columns are rotations, translations, scales and
    xy/xz/yz skews; seven DOF uses a common scale.
    """
    if isinstance(parameters, torch.Tensor) and parameters.device.type != "cpu":
        raise ValueError("affine parameter batching expects CPU parameters")
    if isinstance(centre, torch.Tensor) and centre.device.type != "cpu":
        raise ValueError("affine parameter batching expects a CPU centre")
    parameters = torch.as_tensor(parameters, dtype=torch.float64)
    centre = torch.as_tensor(centre, dtype=torch.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 12 or centre.shape != (3,):
        raise ValueError("parameters and centre must have shapes [B,12] and [3]")
    if dof not in (6, 7, 9, 10, 11, 12):
        raise ValueError("dof must be one of 6, 7, 9, 10, 11, or 12")
    batch = len(parameters)
    affine = torch.eye(4, dtype=parameters.dtype).repeat(batch, 1, 1)
    for axis in range(3):
        affine = _matrix_product(affine, _axis_rotation(parameters[:, axis], axis, centre))
    affine[:, :3, 3] += parameters[:, 3:6]
    if dof == 6:
        return affine
    scales = parameters[:, 6:9] if dof >= 9 else parameters[:, 6:7].expand(-1, 3)
    scale_linear = torch.diag_embed(scales)
    scale = _homogeneous(scale_linear, centre - scale_linear @ centre)
    skew_linear = torch.eye(3, dtype=parameters.dtype).repeat(batch, 1, 1)
    if dof >= 10:
        skew_linear[:, 0, 1] = parameters[:, 9]
    if dof >= 11:
        skew_linear[:, 0, 2] = parameters[:, 10]
    if dof >= 12:
        skew_linear[:, 1, 2] = parameters[:, 11]
    skew = _homogeneous(skew_linear, centre - skew_linear @ centre)
    return _matrix_product(_matrix_product(affine, skew), scale)
