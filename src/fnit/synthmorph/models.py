"""PyTorch inference for FreeSurfer 8.2 SynthMorph pretrained networks.

Port of VoxelMorph's VxmAffineFeatureDetector and HyperVxmJoint (Apache-2.0).
Original authors: Malte Hoffmann, Andrew Hoopes, Adrian Dalca and collaborators.
HDF5 weights are read directly; TensorFlow is not imported. A hypernetwork is
evaluated once per regularization value, and its resulting convolutions are
retained for all image pairs processed by this instance.
"""

from pathlib import Path
import re

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .spatial import affine_to_dense, compose, integrate, transform


def _dataset_groups(h5, prefix, required):
    """Find numbered Keras layers regardless of the outer model name."""
    groups = {}
    pattern = re.compile(rf"^{re.escape(prefix)}(?:_(\d+))?$")

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        parts = name.split("/")
        if len(parts) < 2:
            return
        match = pattern.fullmatch(parts[-2])
        key = parts[-1].split(":")[0]
        if match and key in required:
            index = int(match.group(1) or 0)
            if key in groups.setdefault(index, {}):
                raise ValueError(f"Multiple {prefix} layers with index {index} in {h5.filename}")
            groups[index][key] = obj

    h5.visititems(visit)
    out = [groups[k] for k in sorted(groups)]
    if any(set(group) != set(required) for group in out):
        raise ValueError(f"Incomplete {prefix} weights in {h5.filename}")
    return out


def _conv(kernel, bias):
    """Keras (D,H,W,in,out) -> torch (out,in,D,H,W), no spatial flip."""
    kernel = torch.as_tensor(kernel, dtype=torch.float32)
    bias = torch.as_tensor(bias, dtype=torch.float32)
    layer = nn.Conv3d(kernel.shape[-2], kernel.shape[-1], 3, padding=1)
    layer.weight = nn.Parameter(kernel.permute(4, 3, 0, 1, 2).contiguous(), requires_grad=False)
    layer.bias = nn.Parameter(bias.contiguous(), requires_grad=False)
    return layer


class FeatureDetector(nn.Module):
    """Shared 4-level encoder, four coarse convolutions, 64 feature maps."""

    def __init__(self, weights):
        super().__init__()
        with h5py.File(weights, "r") as h5:
            groups = _dataset_groups(h5, "conv3d", ("kernel", "bias"))
            if len(groups) != 9:
                raise ValueError(f"Expected 9 affine convolution layers, found {len(groups)}")
            self.layers = nn.ModuleList([_conv(g["kernel"][...], g["bias"][...]) for g in groups])

    def forward(self, x, *, cpu_joint_inference=False):
        if (cpu_joint_inference and x.device.type == 'cpu' and x.dtype == torch.float32
                and not torch.is_grad_enabled() and torch.backends.mkldnn.is_available()
                and torch.backends.mkldnn.enabled):
            from ._cpu_features import detector_features, supported_inference
            if supported_inference(self, x):
                return detector_features(self, x)
        for layer in self.layers[:4]:
            x = F.max_pool3d(F.leaky_relu(layer(x), 0.2), 2)
        for layer in self.layers[4:8]:
            x = F.leaky_relu(layer(x), 0.2)
        return F.relu(self.layers[8](x))


def barycenter(features, full_shape):
    """Neurite's centered, unit-extent feature barycenters, then voxel scaling."""
    mass = features.sum(dim=(2, 3, 4))
    centers = []
    for axis, (size, full) in enumerate(zip(features.shape[2:], full_shape), start=2):
        coord = (torch.arange(size, device=features.device, dtype=features.dtype)
                 - (size - 1) / 2) / size
        shape = [1] * features.ndim
        shape[axis] = size
        moment = (features * coord.reshape(shape)).sum(dim=(2, 3, 4))
        centers.append(torch.where(mass != 0, moment / mass, 0) * full)
    return torch.stack(centers, dim=-1), mass


def _cpu_joint_barycenter_legacy(features, full_shape):
    """Keep the published CPU reduction semantics for training and observers."""
    mass = features.permute(0, 2, 3, 4, 1).contiguous().sum((1, 2, 3))
    coordinates = [
        (torch.arange(size, dtype=features.dtype, device=features.device)
         - (size - 1) / 2) / size for size in features.shape[2:]
    ]
    grid = torch.stack(torch.meshgrid(*coordinates, indexing='ij'), -1)
    values = features.contiguous().unsqueeze(-1)
    denominator = values.sum((2, 3, 4))
    moment = (values * grid).sum((2, 3, 4))
    centers = torch.where(denominator != 0, moment / denominator, 0)
    centers *= torch.as_tensor(full_shape, dtype=features.dtype, device=features.device)
    return centers, mass


def _cpu_joint_inference_enabled(module, moving, fixed, mid_space):
    if (not mid_space or module.training or moving.device.type != 'cpu'
            or fixed.device.type != 'cpu' or moving.dtype != torch.float32
            or fixed.dtype != torch.float32 or torch.is_grad_enabled()):
        return False
    from ._cpu_features import supported_inference
    return supported_inference(module.detector, moving) and supported_inference(module.detector, fixed)


def _cpu_inner_sum(values):
    """FP32 CPU reference order for a contiguous innermost reduction.

    The reference CPU kernel accumulates four streams of eight values,
    combines streams from left to right, and halves the packet for its
    horizontal sum. Explicit additions preserve that order without using
    a reference runtime or changing CUDA reductions.
    """
    count = values.shape[-1]
    packets = [values.new_zeros((*values.shape[:-1], 8)) for _ in range(4)]
    end = count // 32 * 32
    for position in range(0, end, 32):
        for stream in range(4):
            packets[stream] += values[..., position + stream * 8:position + (stream + 1) * 8]
    merged = ((packets[0] + packets[1]) + packets[2]) + packets[3]
    position = end
    while position + 8 <= count:
        merged += values[..., position:position + 8]
        position += 8
    tail = values.new_zeros(values.shape[:-1])
    for index in range(position, count):
        tail += values[..., index]
    while merged.shape[-1] > 1:
        middle = merged.shape[-1] // 2
        merged = merged[..., :middle] + merged[..., middle:]
    return tail + merged[..., 0]


def _cpu_joint_barycenter(features, full_shape):
    """Match CPU joint confidence masses, XYZ moments and denominators.

    Confidence masses preserve the feature channels and use four staggered
    spatial streams. XYZ moments accumulate each spatial point in order;
    their separate denominator uses the contiguous inner reduction above.
    These distinct FP32 orders matter before the affine square roots.
    """
    values = features.contiguous().flatten(2)
    count = values.shape[-1]
    streams = [values.new_zeros(values.shape[:-1]) for _ in range(4)]
    end = count // 4 * 4
    for position in range(0, end, 4):
        for stream in range(4):
            streams[stream] += values[..., position + stream]
    mass = ((streams[0] + streams[1]) + streams[2]) + streams[3]
    for position in range(end, count):
        mass += values[..., position]
    coordinates = [
        (torch.arange(size, dtype=features.dtype, device=features.device)
         - (size - 1) / 2) / size for size in features.shape[2:]
    ]
    grid = torch.stack(torch.meshgrid(*coordinates, indexing='ij'), -1).reshape(-1, 3)
    moment = values.new_zeros((*values.shape[:-1], 3))
    for position in range(count):
        moment += values[..., position, None] * grid[position]
    denominator = _cpu_inner_sum(values).unsqueeze(-1)
    centers = torch.where(denominator != 0, moment / denominator, 0)
    centers *= torch.as_tensor(full_shape, dtype=features.dtype, device=features.device)
    return centers, mass


def fit_affine(source, target, weights):
    """Match the original weighted normal equations (target -> source)."""
    x = torch.cat((target, torch.ones_like(target[..., :1])), dim=-1)
    xt = x.transpose(-1, -2) * weights.unsqueeze(-2)
    beta = torch.linalg.inv(xt @ x) @ xt @ source
    matrix = torch.eye(4, dtype=x.dtype, device=x.device).expand(*x.shape[:-2], 4, 4).clone()
    matrix[..., :3, :] = beta.transpose(-1, -2)
    return matrix


def _cpu_joint_inverse(matrix):
    """Four-by-four CPU partial-pivot LU with the reference solve order."""
    if matrix.shape[-2:] != (4, 4):
        raise ValueError('CPU joint inverse requires four-by-four matrices')
    lu = matrix
    right = torch.eye(4, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
    for column in range(4):
        pivot = lu[..., column:, column].abs().argmax(-1) + column
        order = torch.arange(4, device=matrix.device).expand(*matrix.shape[:-2], 4).clone()
        order[..., column] = pivot
        order.scatter_(-1, pivot.unsqueeze(-1), column)
        indices = order.unsqueeze(-1).expand_as(matrix)
        lu = lu.gather(-2, indices)
        right = right.gather(-2, indices)
        diagonal = lu[..., column, column]
        if torch.any(diagonal == 0):
            raise torch.linalg.LinAlgError('CPU joint inverse: matrix is singular')
        lower = lu[..., column + 1:, column] / diagonal.unsqueeze(-1)
        corner = (lu[..., column + 1:, column + 1:]
                  - lower.unsqueeze(-1) * lu[..., column:column + 1, column + 1:])
        bottom = torch.cat((lu[..., column + 1:, :column], lower.unsqueeze(-1), corner), -1)
        lu = torch.cat((lu[..., :column + 1, :], bottom), -2)
    lower_rows = []
    for row in range(4):
        value = right[..., row, :]
        for column in range(row):
            value = value - lu[..., row, column, None] * lower_rows[column]
        lower_rows.append(value)
    upper_rows = [None] * 4
    for row in range(3, -1, -1):
        value = lower_rows[row]
        for column in range(row + 1, 4):
            value = value - lu[..., row, column, None] * upper_rows[column]
        upper_rows[row] = value * (1 / lu[..., row, row, None])
    return torch.stack(upper_rows, -2)


def _cpu_joint_fit_affine(source, target, weights):
    """CPU joint normal equations with contiguous weighted transpose."""
    x = torch.cat((target, torch.ones_like(target[..., :1])), -1)
    xt = (x.transpose(-1, -2) * weights.unsqueeze(-2)).contiguous()
    beta = _cpu_joint_inverse(xt @ x) @ xt @ source
    matrix = torch.eye(4, dtype=x.dtype, device=x.device).expand(*x.shape[:-2], 4, 4).clone()
    matrix[..., :3, :] = beta.transpose(-1, -2)
    return matrix


def _rigid(matrix):
    """Discard scale/shear with the original Cholesky/Euler convention."""
    mat = matrix[..., :3, :3]
    upper = torch.linalg.cholesky(mat.transpose(-1, -2) @ mat).transpose(-1, -2)
    scale = upper.diagonal(dim1=-2, dim2=-1).clone()
    scale[..., 0] *= torch.sign(torch.linalg.det(mat))
    shear = torch.diag_embed(1 / scale) @ upper
    # The source reconstructs shear with ones on the diagonal.
    shear = torch.triu(shear, diagonal=1) + torch.eye(3, device=mat.device, dtype=mat.dtype)
    rotation = mat @ torch.linalg.inv(torch.diag_embed(scale) @ shear)
    clip = lambda value: value.clamp(-1, 1)
    angle2 = torch.asin(clip(rotation[..., 0, 2]))
    c2 = torch.cos(angle2)
    angle1 = torch.atan2(clip(-rotation[..., 1, 2] / c2), clip(rotation[..., 2, 2] / c2))
    angle3 = torch.atan2(clip(-rotation[..., 0, 1] / c2), clip(rotation[..., 0, 0] / c2))
    locked = (angle2.abs() - np.pi / 2).abs() < 1e-6
    angle1 = torch.where(locked, 0, angle1)
    angle3 = torch.where(locked, torch.atan2(clip(rotation[..., 1, 0]), clip(rotation[..., 1, 1])), angle3)
    c1, s1 = torch.cos(angle1), torch.sin(angle1)
    c2, s2 = torch.cos(angle2), torch.sin(angle2)
    c3, s3 = torch.cos(angle3), torch.sin(angle3)
    out = matrix.clone()
    out[..., :3, :3] = torch.stack((
        c2*c3, -c2*s3, s2,
        s1*s2*c3+c1*s3, -s1*s2*s3+c1*c3, -s1*c2,
        -c1*s2*c3+s1*s3, c1*s2*s3+s1*c3, c1*c2,
    ), dim=-1).reshape(*mat.shape)
    return out


def matrix_sqrt(matrix):
    """Principal affine square root by double-precision Denman--Beavers."""
    y = matrix.to(torch.float64)
    z = torch.eye(4, device=y.device, dtype=y.dtype).expand_as(y).clone()
    for _ in range(32):
        next_y = 0.5 * (y + torch.linalg.inv(z))
        z = 0.5 * (z + torch.linalg.inv(y))
        if torch.max(torch.abs(next_y - y)).item() < 1e-12:
            y = next_y
            break
        y = next_y
    residual = torch.linalg.matrix_norm(y @ y - matrix.double())
    if not torch.all(torch.isfinite(y)) or torch.any(residual > 1e-6):
        raise ValueError("Affine transform has no converged real principal square root")
    return y.to(matrix.dtype)


def _cpu_joint_matrix_sqrt(matrix):
    """CPU joint FP32 Schur root; retain Torch's differentiable route."""
    if matrix.requires_grad or matrix.dtype != torch.float32:
        return matrix_sqrt(matrix)
    from ._cpu_eigen import affine_sqrt
    root = torch.from_numpy(affine_sqrt(matrix.detach().numpy()))
    residual = torch.linalg.matrix_norm(root.double() @ root.double() - matrix.double())
    # The Schur output is already rounded to FP32. Check its backward error
    # against that precision; the public image/field accuracy gates are unchanged.
    bound = 32 * torch.finfo(matrix.dtype).eps * torch.linalg.matrix_norm(matrix.double())
    if not torch.all(torch.isfinite(root)) or torch.any(residual > bound):
        raise ValueError('Affine transform has no converged real principal square root')
    return root


def _cpu_joint_center_affine(matrix, full_shape):
    """Compose the declared 3x4 half-affine in centered voxel coordinates."""
    center = torch.eye(4, device=matrix.device, dtype=matrix.dtype)
    center[:3, 3] = -(torch.as_tensor(full_shape, device=matrix.device) - 1) * 0.5
    uncenter = center.clone()
    uncenter[:3, 3] = -center[:3, 3]
    # The original ComposeTransform consumes 3x4 affines and reconstructs
    # the homogeneous row at each step. Its reverse traversal evaluates
    # uncenter @ (half @ center), with the exact positive center translation.
    row = center[3:].expand(*matrix.shape[:-2], 1, 4)
    square = torch.cat((matrix[..., :3, :], row), dim=-2)
    inner = square @ center
    return uncenter @ torch.cat((inner[..., :3, :], row), dim=-2)


class AffineNetwork(nn.Module):
    def __init__(self, weights, rigid=False):
        super().__init__()
        self.detector = FeatureDetector(weights)
        self.rigid = rigid

    def forward(self, moving, fixed, half_res=True, mid_space=False, return_features=False):
        full_shape = moving.shape[2:]
        if half_res:
            moving = moving[..., ::2, ::2, ::2]
            fixed = fixed[..., ::2, ::2, ::2]
        cpu_inference = _cpu_joint_inference_enabled(self, moving, fixed, mid_space)
        if cpu_inference:
            feat1, feat2 = (self.detector(moving, cpu_joint_inference=True),
                            self.detector(fixed, cpu_joint_inference=True))
        else:
            feat1, feat2 = self.detector(moving), self.detector(fixed)
        center_function = (_cpu_joint_barycenter
                           if cpu_inference else
                           (_cpu_joint_barycenter_legacy if mid_space and moving.device.type == 'cpu' else barycenter))
        cen1, mass1 = center_function(feat1, full_shape)
        cen2, mass2 = center_function(feat2, full_shape)
        if cpu_inference:
            weights = (mass1 / _cpu_inner_sum(mass1).unsqueeze(-1)) * (mass2 / _cpu_inner_sum(mass2).unsqueeze(-1))
        else:
            weights = (mass1 / mass1.sum(-1, keepdim=True)) * (mass2 / mass2.sum(-1, keepdim=True))
        fit_function = (_cpu_joint_fit_affine
                        if cpu_inference else fit_affine)
        inverse_function = (_cpu_joint_inverse
                            if cpu_inference else torch.linalg.inv)
        sqrt_function = (_cpu_joint_matrix_sqrt
                         if cpu_inference else matrix_sqrt)
        affine1 = fit_function(cen1, cen2, weights)
        affine2 = fit_function(cen2, cen1, weights)
        affine1 = (affine1 + inverse_function(affine2)) * 0.5
        if self.rigid:
            affine1 = _rigid(affine1)
        affine2 = inverse_function(affine1)
        if mid_space:
            affine1, affine2 = sqrt_function(affine1), sqrt_function(affine2)
        center = torch.eye(4, device=moving.device, dtype=moving.dtype)
        center[:3, 3] = -(torch.as_tensor(full_shape, device=moving.device) - 1) * 0.5
        if cpu_inference:
            affine1 = _cpu_joint_center_affine(affine1, full_shape)
            affine2 = _cpu_joint_center_affine(affine2, full_shape)
        else:
            affine1 = torch.linalg.inv(center) @ affine1 @ center
            affine2 = torch.linalg.inv(center) @ affine2 @ center
        out = affine1[0], affine2[0]
        return (*out, feat1, feat2) if return_features else out


class DeformNetwork(nn.Module):
    """HyperVxmJoint's UNet, materialized for one user-selected hyper value."""

    def __init__(self, weights, hyper=0.5):
        super().__init__()
        self.weights_path = str(Path(weights))
        self.hyper = None
        self.layers = nn.ModuleList()
        self.set_hyper(hyper)

    @torch.no_grad()
    def set_hyper(self, hyper):
        hyper = float(hyper)
        if not 0 < hyper < 1:
            raise ValueError("Regularization strength must lie in the open interval (0, 1)")
        if self.hyper == hyper:
            return
        device = next(self.parameters()).device if self.layers else torch.device("cpu")
        with h5py.File(self.weights_path, "r") as h5:
            dense = _dataset_groups(h5, "dense", ("kernel", "bias"))
            groups = _dataset_groups(h5, "hyper_conv_from_dense", (
                "hyperkernel_kernel", "hyperkernel_bias", "hyperbias_kernel", "hyperbias_bias"))
            if len(dense) != 4 or len(groups) != 13:
                raise ValueError("Weights do not contain the expected 4-layer hypernetwork and 13 convolutions")
            value = torch.tensor([[hyper]], dtype=torch.float32)
            for g in dense:
                value = F.relu(value @ torch.from_numpy(g["kernel"][...]) + torch.from_numpy(g["bias"][...]))
            layers = []
            for g in groups:
                # Only one large hyperkernel is resident at a time (no TF dependency).
                kernel = value @ torch.from_numpy(g["hyperkernel_kernel"][...])
                kernel += torch.from_numpy(g["hyperkernel_bias"][...])
                bias = value @ torch.from_numpy(g["hyperbias_kernel"][...])
                bias += torch.from_numpy(g["hyperbias_bias"][...])
                out_channels = bias.numel()
                kernel = kernel.reshape(3, 3, 3, -1, out_channels)
                layers.append(_conv(kernel, bias.reshape(-1)))
        self.layers = nn.ModuleList(layers).to(device)
        self.hyper = hyper

    def forward(self, moving, fixed):
        x = torch.cat((moving, fixed), dim=1)
        skips = []
        for layer in self.layers[:4]:
            x = F.leaky_relu(layer(x), 0.2)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        for layer in self.layers[4:8]:
            x = F.leaky_relu(layer(x), 0.2)
            x = torch.cat((F.interpolate(x, scale_factor=2, mode="nearest"), skips.pop()), dim=1)
        for layer in self.layers[8:12]:
            x = F.leaky_relu(layer(x), 0.2)
        return self.layers[12](x)


class SynthMorphNetwork(nn.Module):
    """Single-pair inference; tensors are (1,C,I,J,K), vectors ordered I,J,K.

    ``weights`` maps ``affine``, ``rigid``, and/or ``deform`` to official HDF5
    files. Outputs are pull transforms mapping fixed voxel coordinates to
    moving coordinates, followed by the reverse transform. Affine outputs
    are homogeneous 4x4 matrices; nonlinear outputs are (1,3,I,J,K) shifts.
    """

    def __init__(self, weights, model="joint", hyper=0.5, int_steps=7, device="cpu"):
        super().__init__()
        if model not in ("joint", "deform", "affine", "rigid"):
            raise ValueError(f"Unknown registration model: {model}")
        self.model = model
        self.int_steps = int_steps
        if model != "deform":
            self.affine = AffineNetwork(weights["rigid" if model == "rigid" else "affine"], rigid=model == "rigid")
        if model in ("joint", "deform"):
            self.deform = DeformNetwork(weights["deform"], hyper)
        self.eval().to(device)

    @torch.inference_mode()
    def forward(self, moving, fixed, return_intermediates=False, *, compute_inverse=True):
        if moving.shape != fixed.shape or moving.ndim != 5 or moving.shape[:2] != (1, 1):
            raise ValueError("Expected matching single-channel tensors of shape (1,1,I,J,K)")
        if any(size % 32 for size in moving.shape[2:]):
            raise ValueError("Network spatial dimensions must be multiples of 32")
        if self.model in ("affine", "rigid"):
            return self.affine(moving, fixed)
        full_shape = moving.shape[2:]
        half_shape = tuple(size // 2 for size in full_shape)
        half1 = moving[..., ::2, ::2, ::2]
        half2 = fixed[..., ::2, ::2, ::2]
        scale2 = torch.diag(moving.new_tensor([2, 2, 2, 1]))
        scale_half = torch.diag(moving.new_tensor([0.5, 0.5, 0.5, 1]))
        if self.model == "joint":
            affine1, affine2 = self.affine(half1, half2, half_res=False, mid_space=True)
            affine1, affine2 = scale2 @ affine1, scale2 @ affine2
            mov1 = transform(moving, affine1, shape=half_shape, fill_value=0)
            mov2 = transform(fixed, affine2, shape=half_shape, fill_value=0)
        else:
            affine1 = affine2 = scale2
            mov1, mov2 = half1, half2
        velocity = (self.deform(mov1, mov2) - self.deform(mov2, mov1)) * 0.5
        deform1 = integrate(velocity, self.int_steps)
        deform2 = integrate(-velocity, self.int_steps) if compute_inverse else None
        total1 = [affine1, deform1]
        total2 = [affine2, deform2] if compute_inverse else None
        if self.model == "joint":
            total1.extend((scale_half, affine1))
            if compute_inverse:
                total2.extend((scale_half, affine2))
        total1 = compose(total1)
        total2 = compose(total2) if compute_inverse else None
        down = affine_to_dense(scale_half, full_shape)
        forward = compose((total1, down))
        backward = compose((total2, down)) if compute_inverse else None
        if return_intermediates:
            return forward, backward, {"velocity": velocity, "deform_forward": deform1,
                                       "deform_backward": deform2, "affine_forward": affine1,
                                       "affine_backward": affine2, "half_moving": mov1, "half_fixed": mov2}
        return forward, backward
