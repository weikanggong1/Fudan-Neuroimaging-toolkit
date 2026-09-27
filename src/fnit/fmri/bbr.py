"""Rigid EPI-to-T1 boundary registration using the FLIRT BBR cost.

FSL's no-fieldmap BBR samples EPI intensities 2 mm either side of the T1
white-matter boundary. Matrices here map EPI to T1 in FLIRT scaled-mm space.
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

from ..flirt import TorchFLIRT
from ..flirt.coordinates import flirt_to_world_affine, voxel_to_fsl_scaled_mm


def _image(value, name):
    image = nib.load(str(value)) if isinstance(value, (str, Path)) else value
    if not isinstance(image, nib.spatialimages.SpatialImage) or image.ndim != 3:
        raise ValueError(f"{name} must be a 3D NIfTI image or path")
    return image


def _boundary(wm, reference):
    segmentation = np.asarray(wm.dataobj, dtype=np.float32)
    if segmentation.shape != reference.shape or not np.allclose(
        wm.affine, reference.affine, atol=1e-4, rtol=0
    ):
        raise ValueError("wmseg must have the same voxel grid and affine as t1")
    mask = segmentation > 0.5
    neighbours = convolve(mask.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant")
    points = np.argwhere(mask & (neighbours < 27))
    if len(points) < 100:
        raise ValueError("wmseg has too few white-matter boundary points")
    spacing = np.asarray(reference.header.get_zooms()[:3], dtype=np.float64)
    smoothed = gaussian_filter(mask.astype(np.float32), 2.0 / spacing)
    gradient = np.stack(np.gradient(smoothed), axis=-1)[tuple(points.T)]
    reference_fsl = voxel_to_fsl_scaled_mm(reference.affine, reference.shape, spacing)
    normal = -gradient / spacing
    normal *= np.sign(np.diag(reference_fsl)[:3])
    length = np.linalg.norm(normal, axis=1)
    valid = length > 1e-8
    points = points[valid]
    normal = normal[valid] / length[valid, None]
    surface = points @ reference_fsl[:3, :3].T + reference_fsl[:3, 3]
    return (surface + 2.0 * normal).astype(np.float32), (surface - 2.0 * normal).astype(np.float32), surface.mean(axis=0)


def _rigid(parameters, centre):
    angles = parameters[:3]
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rotation = rz @ ry @ rx
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = centre - rotation @ centre + parameters[3:]
    return matrix


class _BBRCost:
    def __init__(self, epi, grey_points, white_points, device):
        self.device = torch.device(device)
        data = np.asarray(epi.dataobj, dtype=np.float32)
        self.image = torch.as_tensor(np.transpose(data, (2, 1, 0)).copy(), device=self.device)[None, None]
        self.shape = data.shape
        self.input_fsl = voxel_to_fsl_scaled_mm(epi.affine, epi.shape, epi.header.get_zooms()[:3])
        self.points = torch.as_tensor(np.stack((grey_points, white_points)), device=self.device)

    def __call__(self, matrix, step=2):
        transform = np.linalg.inv(self.input_fsl) @ np.linalg.inv(matrix)
        points = self.points[:, ::step].reshape(-1, 3)
        linear = torch.as_tensor(transform[:3, :3], dtype=torch.float32, device=self.device)
        offset = torch.as_tensor(transform[:3, 3], dtype=torch.float32, device=self.device)
        voxels = points @ linear.T + offset
        grid = torch.stack([2.0 * voxels[:, k] / (self.shape[k] - 1) - 1.0 for k in range(3)], -1)
        values = F.grid_sample(
            self.image, grid[None, None, None], mode="bilinear",
            padding_mode="border", align_corners=True,
        ).reshape(2, -1)
        grey, white = values
        total = grey + white
        denominator = torch.where(total.abs() > 1e-6, total, torch.ones_like(total))
        difference = torch.where(total.abs() > 1e-6, 200.0 * (grey - white) / denominator, 0.0)
        return float((1.0 + torch.tanh(-0.5 * difference)).mean())


def _resample(epi, t1, matrix, device):
    output = np.empty(t1.shape, dtype=np.float32)
    input_fsl = voxel_to_fsl_scaled_mm(epi.affine, epi.shape, epi.header.get_zooms()[:3])
    reference_fsl = voxel_to_fsl_scaled_mm(t1.affine, t1.shape, t1.header.get_zooms()[:3])
    voxel_matrix = np.linalg.inv(input_fsl) @ np.linalg.inv(matrix) @ reference_fsl
    source = torch.as_tensor(np.transpose(np.asarray(epi.dataobj, dtype=np.float32), (2, 1, 0)).copy(), device=device)[None, None]
    for z in range(0, t1.shape[2], 16):
        stop = min(z + 16, t1.shape[2])
        coords = np.indices((t1.shape[0], t1.shape[1], stop - z), dtype=np.float32)
        coords[2] += z
        voxels = np.einsum("ij,j...->i...", voxel_matrix[:3, :3], coords) + voxel_matrix[:3, 3, None, None, None]
        grid = torch.as_tensor(np.stack([
            2.0 * voxels[k] / (epi.shape[k] - 1) - 1.0 for k in range(3)
        ], -1).transpose(2, 1, 0, 3).astype(np.float32, copy=True), device=device)[None]
        sampled = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        output[:, :, z:stop] = sampled[0, 0].permute(2, 1, 0).cpu().numpy()
    header = t1.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    return nib.Nifti1Image(output, t1.affine, header)


@dataclass(frozen=True)
class BBRResult:
    moved: nib.Nifti1Image
    matrix: np.ndarray
    moving_to_fixed_world: np.ndarray
    initial_cost: float
    final_cost: float
    boundary_points: int
    runtime_seconds: float

    def save(self, output=None, omat=None):
        if output is not None:
            nib.save(self.moved, str(output))
        if omat is not None:
            np.savetxt(omat, self.matrix, fmt="%.12g")
        return self


def register_bbr(epi, t1, wmseg, *, init=None, device=None, grid_search=True):
    """Register one EPI reference volume to T1 using a T1 white-matter mask.

    ``init`` is an EPI-to-T1 FLIRT scaled-mm matrix. If absent, the existing
    FNIT 6-DOF normalized-mutual-information FLIRT provides initial alignment.
    ``wmseg`` must be binary and aligned to ``t1``. No fieldmap is applied.
    """
    start = time.perf_counter()
    epi_input, t1_input = epi, t1
    epi, t1, wmseg = _image(epi, "epi"), _image(t1, "t1"), _image(wmseg, "wmseg")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if init is None:
        if not isinstance(epi_input, (str, Path)) or not isinstance(t1_input, (str, Path)):
            raise ValueError("file paths are required for automatic FLIRT initialization")
        initial = TorchFLIRT(device=str(device), dof=6, cost="normmi")(
            epi_input, t1_input
        ).matrix
    elif isinstance(init, (str, Path)):
        initial = np.loadtxt(init)
    else:
        initial = np.asarray(init)
    if initial.shape != (4, 4) or not np.isfinite(initial).all():
        raise ValueError("init must be a finite 4x4 FLIRT matrix")
    grey, white, centre = _boundary(wmseg, t1)
    cost = _BBRCost(epi, grey, white, device)
    initial_cost = cost(initial)
    offsets = np.zeros(6)
    if grid_search:
        offsets = min(
            (np.array((*angle, *shift))
             for angle in product((-0.07, 0.0, 0.07), repeat=3)
             for shift in product((-4.0, 0.0, 4.0), repeat=3)),
            key=lambda p: cost(_rigid(p, centre) @ initial, step=200),
        )
    for step, iterations in ((200, 8), (2, 8)):
        result = minimize(
            lambda p: cost(_rigid(p, centre) @ initial, step=step),
            offsets, method="Powell", bounds=[(-0.15, 0.15)] * 3 + [(-8.0, 8.0)] * 3,
            options={"maxiter": iterations, "xtol": 1e-4, "ftol": 1e-5},
        )
        offsets = result.x
    matrix = _rigid(offsets, centre) @ initial
    final_cost = cost(matrix)
    world = flirt_to_world_affine(
        matrix, epi.affine, t1.affine, epi.shape, t1.shape,
        epi.header.get_zooms()[:3], t1.header.get_zooms()[:3],
    )
    return BBRResult(
        moved=_resample(epi, t1, matrix, device), matrix=matrix,
        moving_to_fixed_world=world, initial_cost=initial_cost,
        final_cost=final_cost, boundary_points=len(grey),
        runtime_seconds=time.perf_counter() - start,
    )


__all__ = ["BBRResult", "register_bbr"]
