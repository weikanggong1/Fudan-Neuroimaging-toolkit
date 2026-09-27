"""SynthMorph deformable registration and VBM Jacobian modulation."""

from dataclasses import dataclass

import nibabel as nib
import numpy as np
import torch

from .._nib import FNITNifti1Image, new_image
from .._transforms import same_geometry


@dataclass
class SynthMorphVBMResult:
    """Template-space images and the target-to-source SynthMorph pull warp."""

    moved: FNITNifti1Image
    pull_transform: nib.Nifti1Image
    full_pull_jacobian: FNITNifti1Image
    nonlinear_jacobian: FNITNifti1Image
    modulated_gm: FNITNifti1Image
    affine_pull_determinant: float
    qc: dict


def _determinant(matrix):
    """Determinant of a 3x3 matrix field stored as ``[3, 3, ...]``."""
    return (
        matrix[0, 0]
        * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1]
        * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2]
        * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )


def pull_jacobian_determinants(
    displacement_ras,
    target_vox2world,
    moving_to_fixed_world,
    *,
    device="cpu",
):
    """Compute full and nonlinear-only determinants of a pull warp.

    ``displacement_ras`` is indexed on the fixed grid and stores
    ``source_world(target) - target_world``. ``moving_to_fixed_world`` is the
    forward affine supplied to SynthMorph. The full pull determinant therefore
    contains ``det(inv(moving_to_fixed_world))``; VBM modulation removes that
    affine factor.
    """
    device = torch.device(device)
    field = torch.as_tensor(
        np.array(displacement_ras, dtype=np.float32, copy=True), device=device
    )
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError("displacement_ras must have shape (X, Y, Z, 3)")
    if any(size < 2 for size in field.shape[:3]):
        raise ValueError("all displacement dimensions must contain at least two voxels")
    if not bool(torch.isfinite(field).all()):
        raise ValueError("displacement_ras contains NaN or infinity")

    target = torch.as_tensor(target_vox2world, dtype=torch.float32, device=device)
    affine = torch.as_tensor(
        moving_to_fixed_world, dtype=torch.float32, device=device
    )
    if target.shape not in ((3, 3), (4, 4)):
        raise ValueError("target_vox2world must have shape 3x3 or 4x4")
    if affine.shape not in ((3, 3), (4, 4)):
        raise ValueError("moving_to_fixed_world must have shape 3x3 or 4x4")
    target = target[:3, :3]
    affine = affine[:3, :3]
    if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(affine).all()):
        raise ValueError("affine matrices must contain only finite values")
    target_det = torch.linalg.det(target)
    affine_det = torch.linalg.det(affine)
    if abs(float(target_det)) < 1e-8:
        raise ValueError("target_vox2world must be invertible")
    if abs(float(affine_det)) < 1e-8:
        raise ValueError("moving_to_fixed_world must be invertible")

    # Component first: [world component, target voxel axis, X, Y, Z].
    field = field.movedim(-1, 0)
    voxel_gradient = torch.stack(
        torch.gradient(field, dim=(1, 2, 3), edge_order=1), dim=1
    )
    world_gradient = torch.einsum(
        "abxyz,bc->acxyz", voxel_gradient, torch.linalg.inv(target)
    )
    pull_gradient = world_gradient + torch.eye(
        3, dtype=field.dtype, device=device
    )[:, :, None, None, None]
    full = _determinant(pull_gradient)
    affine_pull_determinant = affine_det.reciprocal()
    nonlinear = full / affine_pull_determinant
    return full, nonlinear, affine_pull_determinant


def _world_affine(initial, moving, fixed):
    source_geometry = getattr(initial, "source", None)
    target_geometry = getattr(initial, "target", None)
    if source_geometry is not None and not same_geometry(
        source_geometry, moving, tolerance=1e-3
    ):
        raise ValueError("moving_to_fixed source geometry does not match moving")
    if target_geometry is not None and not same_geometry(
        target_geometry, fixed, tolerance=1e-3
    ):
        raise ValueError("moving_to_fixed target geometry does not match fixed")
    matrix = np.asarray(initial, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("moving_to_fixed must contain a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError("moving_to_fixed must be a homogeneous affine matrix")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-8:
        raise ValueError("moving_to_fixed must be invertible")
    return matrix


class SynthMorphDeformRegistration:
    """Reuse the package SynthMorph deform model for template registration."""

    def __init__(
        self,
        *,
        weights=None,
        device="cpu",
        extent=256,
        hyper=0.5,
        steps=7,
        synthmorph=None,
    ):
        self.device = torch.device(device)
        if synthmorph is None:
            from ..synthmorph import SynthMorph

            synthmorph = SynthMorph(
                weights=weights,
                device=self.device,
                model="deform",
                extent=extent,
                hyper=hyper,
                steps=steps,
            )
        self.synthmorph = synthmorph

    def __call__(self, moving, fixed, moving_to_fixed):
        """Register GM after a moving-to-fixed affine initialization."""
        if not isinstance(moving, nib.spatialimages.SpatialImage) or not isinstance(
            fixed, nib.spatialimages.SpatialImage
        ):
            raise TypeError("moving and fixed must be nibabel spatial images")
        initial = _world_affine(moving_to_fixed, moving, fixed)
        result = self.synthmorph(
            moving,
            fixed,
            init=initial,
            mid_space=False,
        )
        if not isinstance(result.moved, nib.spatialimages.SpatialImage):
            raise TypeError("SynthMorph moved output must be a nibabel spatial image")
        if not isinstance(result.transform, nib.spatialimages.SpatialImage):
            raise TypeError("SynthMorph deform output must be a nibabel spatial image")
        pull = result.transform
        source_geometry = getattr(pull, "source", None)
        target_geometry = getattr(pull, "target", None)
        if source_geometry is not None and not same_geometry(
            source_geometry, moving, tolerance=1e-3
        ):
            raise ValueError("SynthMorph pull source geometry does not match moving")
        if target_geometry is not None and not same_geometry(
            target_geometry, fixed, tolerance=1e-3
        ):
            raise ValueError("SynthMorph pull target geometry does not match fixed")
        if not np.allclose(pull.affine, fixed.affine, atol=1e-3, rtol=0):
            raise ValueError("SynthMorph pull target geometry does not match fixed")
        if not np.allclose(result.moved.affine, fixed.affine, atol=1e-3, rtol=0):
            raise ValueError("SynthMorph moved output does not use the fixed grid")

        full, nonlinear, affine_pull = pull_jacobian_determinants(
            np.asanyarray(pull.dataobj),
            fixed.affine,
            initial,
            device=self.device,
        )
        moved = np.asarray(result.moved.dataobj, dtype=np.float32)
        if moved.ndim == 4 and moved.shape[-1] == 1:
            moved = moved[..., 0]
        if moved.shape != tuple(fixed.shape[:3]):
            raise ValueError(
                "SynthMorph moved output must contain one fixed-grid frame"
            )
        if not np.isfinite(moved).all():
            raise ValueError("SynthMorph moved output contains NaN or infinity")

        full_array = full.detach().cpu().numpy().astype(np.float32, copy=False)
        nonlinear_array = nonlinear.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        moved_volume = new_image(moved, fixed)
        full_volume = new_image(full_array, fixed)
        nonlinear_volume = new_image(nonlinear_array, fixed)
        modulated = new_image(moved * nonlinear_array, fixed)
        qc = {
            "warp_convention": (
                "fixed-grid target-to-source disp-ras: "
                "source_world(target)-target_world"
            ),
            "full_pull_jacobian_convention": (
                "det(d source_world / d target_world)"
            ),
            "output_jacobian_convention": (
                "full pull determinant divided by affine pull determinant"
            ),
            "affine_pull_jacobian_determinant": float(affine_pull),
            "full_pull_jacobian_min": float(full.min()),
            "full_pull_jacobian_max": float(full.max()),
            "nonlinear_jacobian_min": float(nonlinear.min()),
            "nonlinear_jacobian_max": float(nonlinear.max()),
            "nonpositive_full_pull_jacobian_voxels": int((full <= 0).sum()),
            "nonpositive_nonlinear_jacobian_voxels": int((nonlinear <= 0).sum()),
        }
        return SynthMorphVBMResult(
            moved=moved_volume,
            pull_transform=pull,
            full_pull_jacobian=full_volume,
            nonlinear_jacobian=nonlinear_volume,
            modulated_gm=modulated,
            affine_pull_determinant=float(affine_pull),
            qc=qc,
        )


__all__ = [
    "SynthMorphVBMResult",
    "SynthMorphDeformRegistration",
    "pull_jacobian_determinants",
]
