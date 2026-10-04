"""Common affine, nonlinear-warp, and modulation chain for FastVBM."""

from dataclasses import dataclass
import hashlib

import nibabel as nib
import numpy as np
import torch

from .._nib import FNITNifti1Image, load_image, new_image
from .._transforms import same_geometry
from ..applywarp import TorchApplyWarp
from ..flirt import TorchFLIRT
from ..flirt.coordinates import voxel_to_fsl_scaled_mm
from .synthmorph_backend import SynthMorphDeformRegistration


@dataclass
class VBMRegistrationResult:
    """Gray-matter maps on the template grid and registration diagnostics."""

    warped_gm: FNITNifti1Image
    jacobian: FNITNifti1Image
    modulated_gm: FNITNifti1Image
    pull_world_affine: np.ndarray
    fit_score: float
    maximum_displacement_mm: float
    qc: dict


@dataclass(frozen=True)
class _PreparedRegistration:
    """Backend-independent state passed to either nonlinear estimator."""

    moving_data: np.ndarray
    fixed_data: np.ndarray
    moving_affine: np.ndarray
    fixed_affine: np.ndarray
    flirt_matrix: np.ndarray
    moving_to_fixed_world: np.ndarray
    fixed_to_moving_world: np.ndarray
    initial: np.ndarray
    reference_mask: FNITNifti1Image
    reference_mask_source: str
    linear_qc: dict
    signature: dict


def _array(volume, name):
    if not isinstance(volume, nib.spatialimages.SpatialImage):
        raise TypeError(f"{name} must be a nibabel spatial image")
    data = np.asanyarray(volume.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{name} must contain one 3D frame")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if not np.any(data > 0):
        raise ValueError(f"{name} GM image is empty")
    return np.asarray(data, dtype=np.float32)


def _affine(volume, name):
    affine = np.asarray(volume.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{name} affine must be a finite 4x4 matrix")
    if abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError(f"{name} affine must be invertible")
    return affine


def _normalized_correlation(first, second):
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 2:
        return float("nan")
    first = first[valid] - first[valid].mean()
    second = second[valid] - second[valid].mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first.dot(second) / denominator) if denominator > 0 else 0.0


def _displacement_qc(pull, fixed, pull_affine):
    displacement = np.asarray(pull.dataobj)
    shape = tuple(fixed.shape[:3])
    if displacement.shape != (*shape, 3):
        raise ValueError("pull displacement must match the fixed grid")
    fixed_affine = np.asarray(fixed.affine, dtype=np.float64)
    pull_affine = np.asarray(pull_affine, dtype=np.float64)
    pull_linear_delta = pull_affine[:3, :3] - np.eye(3)
    voxel_coefficients = pull_linear_delta @ fixed_affine[:3, :3]
    offset = (
        pull_linear_delta @ fixed_affine[:3, 3] + pull_affine[:3, 3]
    )
    y = np.arange(shape[1], dtype=np.float64)[None, :, None, None]
    z = np.arange(shape[2], dtype=np.float64)[None, None, :, None]
    maximum_total_squared = 0.0
    maximum_nonlinear_squared = 0.0
    for start in range(0, shape[0], 8):
        stop = min(start + 8, shape[0])
        x = np.arange(start, stop, dtype=np.float64)[:, None, None, None]
        affine_displacement = (
            offset
            + x * voxel_coefficients[:, 0]
            + y * voxel_coefficients[:, 1]
            + z * voxel_coefficients[:, 2]
        )
        chunk = np.asarray(displacement[start:stop], dtype=np.float64)
        nonlinear = chunk - affine_displacement
        maximum_total_squared = max(
            maximum_total_squared,
            float(np.einsum("...i,...i->...", chunk, chunk).max()),
        )
        maximum_nonlinear_squared = max(
            maximum_nonlinear_squared,
            float(np.einsum("...i,...i->...", nonlinear, nonlinear).max()),
        )
    return {
        "maximum_total_pull_displacement_mm": maximum_total_squared**0.5,
        "maximum_nonlinear_displacement_mm": maximum_nonlinear_squared**0.5,
    }


def _implementation_name(value):
    if isinstance(value, SynthMorphDeformRegistration):
        return "fnit.synthmorph.SynthMorph"
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"


def _array_sha256(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _pre_nonlinear_signature(
    moving_data,
    fixed_data,
    moving_affine,
    fixed_affine,
    flirt_matrix,
    reference_mask,
):
    """Hash every value shared by the two nonlinear branches."""
    values = {
        "moving_gm_sha256": _array_sha256(moving_data),
        "fixed_template_sha256": _array_sha256(fixed_data),
        "moving_vox2world_sha256": _array_sha256(moving_affine),
        "fixed_vox2world_sha256": _array_sha256(fixed_affine),
        "flirt_matrix_sha256": _array_sha256(flirt_matrix),
        "reference_mask_sha256": _array_sha256(reference_mask),
    }
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode("ascii"))
        digest.update(values[name].encode("ascii"))
    values["combined_sha256"] = digest.hexdigest()
    return values


def _reference_mask(value, fixed, fixed_data):
    if value is None:
        data = np.asarray(fixed_data > 0, dtype=np.uint8)
        return new_image(data, fixed), data, "derived-fixed-positive-non-fsl-exact"
    value = load_image(value, "reference_mask")
    data = np.asanyarray(value.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError("reference_mask must contain one finite 3D frame")
    if tuple(data.shape) != tuple(fixed.shape[:3]) or not np.allclose(
        value.affine,
        fixed.affine,
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("reference_mask must use the fixed template grid")
    data = np.asarray(data > 0, dtype=np.uint8)
    if not np.any(data):
        raise ValueError("reference_mask is empty")
    return new_image(data, fixed), data, "explicit"


def _prepare_registration(
    moving,
    fixed,
    moving_data,
    fixed_data,
    moving_affine,
    fixed_affine,
    *,
    device,
    reference_mask,
):
    """Run the one shared FSL-coordinate affine preparation stage."""
    linear = TorchFLIRT(device=device)(moving, fixed)
    moving_to_fixed = np.asarray(
        linear.moving_to_fixed_world, dtype=np.float64
    )
    pull_affine = np.asarray(linear.fixed_to_moving_world, dtype=np.float64)
    flirt_matrix = np.asarray(linear.matrix, dtype=np.float64)
    linear_qc = dict(linear.qc)
    initial = moving_to_fixed.copy()
    reference_mask, reference_mask_data, reference_mask_source = _reference_mask(
        reference_mask, fixed, fixed_data
    )
    signature = _pre_nonlinear_signature(
        moving_data,
        fixed_data,
        moving_affine,
        fixed_affine,
        flirt_matrix,
        reference_mask_data,
    )
    return _PreparedRegistration(
        moving_data=moving_data,
        fixed_data=fixed_data,
        moving_affine=moving_affine,
        fixed_affine=fixed_affine,
        flirt_matrix=flirt_matrix,
        moving_to_fixed_world=moving_to_fixed,
        fixed_to_moving_world=pull_affine,
        initial=initial,
        reference_mask=reference_mask,
        reference_mask_source=reference_mask_source,
        linear_qc=linear_qc,
        signature=signature,
    )


def _validate_pull_warp(pull, moving, fixed):
    if not isinstance(pull, nib.spatialimages.SpatialImage):
        raise TypeError("nonlinear estimator must return a nibabel spatial image")
    source_geometry = getattr(pull, "source", None)
    target_geometry = getattr(pull, "target", None)
    if source_geometry is not None and not same_geometry(
        source_geometry, moving, tolerance=1e-3
    ):
        raise ValueError("nonlinear pull source geometry does not match moving")
    if target_geometry is not None and not same_geometry(
        target_geometry, fixed, tolerance=1e-3
    ):
        raise ValueError("nonlinear pull target geometry does not match fixed")
    if not np.allclose(pull.affine, fixed.affine, atol=1e-3, rtol=0):
        raise ValueError("nonlinear pull target geometry does not match fixed")
    displacement = np.asarray(pull.dataobj, dtype=np.float32)
    expected = (*tuple(fixed.shape[:3]), 3)
    if displacement.shape != expected:
        raise ValueError(f"nonlinear pull must have shape {expected}")
    if not np.isfinite(displacement).all():
        raise ValueError("nonlinear pull contains NaN or infinity")
    return pull, displacement


def _synthmorph_pull(model, moving, fixed, initial):
    """Run only the SynthMorph nonlinear estimator, before shared VBM steps."""
    result = model.synthmorph(moving, fixed, init=initial, mid_space=False)
    if not hasattr(result, "transform"):
        raise TypeError("SynthMorph result must contain a transform")
    pull, _ = _validate_pull_warp(result.transform, moving, fixed)
    return (
        pull,
        {
            "backend": "pytorch-synthmorph-deform",
            "model": "deform",
            "mid_space": False,
        },
        None,
    )


def _estimate_nonlinear(
    model,
    backend,
    moving,
    fixed,
    prepared,
):
    """Return only a fixed-grid RAS pull; all later work is shared."""
    if isinstance(model, SynthMorphDeformRegistration):
        if backend != "synthmorph":
            raise ValueError(
                "a SynthMorph deform model requires registration_backend='synthmorph'"
            )
        return _synthmorph_pull(model, moving, fixed, prepared.initial)

    # Import lazily because FNIRT itself imports FastVBM coordinate helpers.
    from ..fnirt.registration import TorchFNIRT

    if isinstance(model, TorchFNIRT):
        if backend != "fnirt":
            raise ValueError(
                "a TorchFNIRT model requires registration_backend='fnirt'"
            )
        result = model(
            moving,
            fixed,
            prepared.initial,
            reference_mask=prepared.reference_mask,
        )
        pull, _ = _validate_pull_warp(result.pull_transform, moving, fixed)
        native_jacobian = np.asarray(
            result.nonlinear_jacobian.dataobj, dtype=np.float32
        )
        return pull, dict(result.qc), native_jacobian

    result = model(moving, fixed, prepared.initial)
    if not hasattr(result, "pull_transform"):
        raise TypeError("custom nonlinear result must contain pull_transform")
    pull, _ = _validate_pull_warp(result.pull_transform, moving, fixed)
    native_jacobian = getattr(result, "nonlinear_jacobian", None)
    if isinstance(native_jacobian, nib.spatialimages.SpatialImage):
        native_jacobian = np.asarray(native_jacobian.dataobj, dtype=np.float32)
    else:
        native_jacobian = None
    return pull, dict(getattr(result, "qc", {})), native_jacobian


def _pull_ras_to_fsl_fields(pull, moving, fixed, flirt_matrix, *, device):
    """Convert a fixed-grid RAS pull to FNIRT residual and dense FSL fields.

    SynthMorph stores ``source_world(target) - target_world`` on the fixed
    grid.  FSL uses scaled-mm axes.  With FLIRT matrix ``A`` (input to
    reference), FNIRT's nonlinear residual is

    ``source_fsl - inv(A) @ target_fsl``.

    The dense field consumed by ``applywarp`` is
    ``source_fsl - target_fsl``.  Both nonlinear estimators pass through this
    same conversion.
    """
    pull, displacement = _validate_pull_warp(pull, moving, fixed)
    shape = tuple(int(value) for value in fixed.shape[:3])
    moving_world = _affine(moving, "moving")
    fixed_world = _affine(fixed, "fixed")
    moving_fsl = voxel_to_fsl_scaled_mm(
        moving_world,
        tuple(moving.shape[:3]),
        np.linalg.norm(moving_world[:3, :3], axis=0),
    )
    fixed_fsl = voxel_to_fsl_scaled_mm(
        fixed_world,
        shape,
        np.linalg.norm(fixed_world[:3, :3], axis=0),
    )

    axes = torch.meshgrid(
        *(
            torch.arange(size, dtype=torch.float32, device=device)
            for size in shape
        ),
        indexing="ij",
    )
    target_voxels = torch.stack(axes).reshape(3, -1)
    fixed_world_tensor = torch.as_tensor(
        fixed_world, dtype=torch.float32, device=device
    )
    target_world = (
        fixed_world_tensor[:3, :3] @ target_voxels
        + fixed_world_tensor[:3, 3:4]
    )
    displacement_tensor = torch.as_tensor(
        np.moveaxis(displacement, -1, 0).copy(),
        dtype=torch.float32,
        device=device,
    ).reshape(3, -1)
    source_world = target_world + displacement_tensor

    moving_world_to_voxel = torch.as_tensor(
        np.linalg.inv(moving_world), dtype=torch.float32, device=device
    )
    source_voxels = (
        moving_world_to_voxel[:3, :3] @ source_world
        + moving_world_to_voxel[:3, 3:4]
    )
    moving_fsl_tensor = torch.as_tensor(
        moving_fsl, dtype=torch.float32, device=device
    )
    fixed_fsl_tensor = torch.as_tensor(
        fixed_fsl, dtype=torch.float32, device=device
    )
    source_fsl = (
        moving_fsl_tensor[:3, :3] @ source_voxels
        + moving_fsl_tensor[:3, 3:4]
    )
    target_fsl = (
        fixed_fsl_tensor[:3, :3] @ target_voxels
        + fixed_fsl_tensor[:3, 3:4]
    )
    affine_pull = torch.linalg.inv(
        torch.as_tensor(flirt_matrix, dtype=torch.float32, device=device)
    )
    affine_source_fsl = (
        affine_pull[:3, :3] @ target_fsl + affine_pull[:3, 3:4]
    )
    residual = (source_fsl - affine_source_fsl).reshape(3, *shape)
    dense_relative = (source_fsl - target_fsl).reshape(3, *shape)
    return (
        residual.movedim(0, -1).contiguous(),
        dense_relative.movedim(0, -1).contiguous(),
        fixed_fsl,
    )


def _fsl_dense_nonlinear_jacobian(residual_fsl, fixed_fsl):
    """FSL dense-field finite-difference Jacobian, on CPU or CUDA."""
    if residual_fsl.ndim != 4 or residual_fsl.shape[-1] != 3:
        raise ValueError("residual_fsl must have shape (X, Y, Z, 3)")
    if any(size < 2 for size in residual_fsl.shape[:3]):
        raise ValueError("Jacobian dimensions must each contain at least two voxels")
    field = residual_fsl.movedim(-1, 0)
    gradient_voxel = torch.stack(
        torch.gradient(field, dim=(1, 2, 3), edge_order=1), dim=1
    )
    inverse_fixed_linear = torch.linalg.inv(
        torch.as_tensor(
            np.asarray(fixed_fsl)[:3, :3],
            dtype=field.dtype,
            device=field.device,
        )
    )
    derivative = torch.einsum(
        "caxyz,ab->cbxyz", gradient_voxel, inverse_fixed_linear
    )
    matrix = derivative + torch.eye(
        3, dtype=field.dtype, device=field.device
    )[:, :, None, None, None]
    return (
        matrix[0, 0]
        * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1]
        * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2]
        * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )


def _nifti(data, affine):
    image = nib.Nifti1Image(np.asarray(data), np.asarray(affine, dtype=np.float64))
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    return image


def _common_applywarp(
    moving_data,
    fixed_data,
    moving_affine,
    fixed_affine,
    dense_relative_fsl,
    *,
    device,
):
    """Resample through the public GPU applywarp implementation."""
    moving_image = _nifti(moving_data, moving_affine)
    fixed_image = _nifti(fixed_data, fixed_affine)
    warp_image = _nifti(
        dense_relative_fsl.detach().cpu().numpy().astype(np.float32),
        fixed_affine,
    )
    # Intent 2006 unambiguously marks an FSL relative displacement field.
    warp_image.header["intent_code"] = 2006
    result = TorchApplyWarp(device=device)(
        moving_image,
        fixed_image,
        warp=warp_image,
        interpolation="trilinear",
        warp_convention="relative",
        output_dtype="float",
    )
    return np.asarray(result.image.dataobj, dtype=np.float32), result.qc


def _register_gm(
    moving,
    fixed,
    *,
    device="cpu",
    reference_mask=None,
    synthmorph_weights=None,
    synthmorph_extent=256,
    synthmorph_hyper=0.5,
    synthmorph_steps=7,
    registration_backend="synthmorph",
    fnirt_strides=(4, 2, 1, 1),
    fnirt_steps=(5, 5, 10, 5),
    fnirt_input_fwhm_mm=(6.0, 4.0, 2.0, 2.0),
    fnirt_reference_fwhm_mm=(4.0, 2.0, 0.0, 0.0),
    fnirt_warp_resolution_mm=10.0,
    fnirt_regularization=(150.0, 75.0, 50.0, 30.0),
    deform_model=None,
):
    """Register one GM PVE image to a GM template and compute modulation.

    Both public backends use the same GM arrays, FSL-correlation-ratio FLIRT
    stage, reference grid, RAS-to-FSL warp conversion, GPU ``applywarp``
    resampling and modulation. CPU FNIRT uses the estimator's analytic
    coefficient Jacobian, matching FSL ``fnirt --jout``. CUDA FNIRT and
    SynthMorph retain the dense nonlinear-only Jacobian. Nonlinear pull
    estimation uses SynthMorph deform or :class:`~fnit.fnirt.TorchFNIRT`.

    The affine preparation stage always runs the package TorchFLIRT
    implementation.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if registration_backend not in ("synthmorph", "fnirt"):
        raise ValueError("registration_backend must be 'synthmorph' or 'fnirt'")
    moving_data = _array(moving, "moving")
    fixed_data = _array(fixed, "fixed")
    moving_affine = _affine(moving, "moving")
    fixed_affine = _affine(fixed, "fixed")

    prepared = _prepare_registration(
        moving,
        fixed,
        moving_data,
        fixed_data,
        moving_affine,
        fixed_affine,
        device=device,
        reference_mask=reference_mask,
    )
    if deform_model is None:
        if registration_backend == "synthmorph":
            deform_model = SynthMorphDeformRegistration(
                weights=synthmorph_weights,
                device=device,
                extent=synthmorph_extent,
                hyper=synthmorph_hyper,
                steps=synthmorph_steps,
            )
        else:
            from ..fnirt.registration import GMFNIRTConfig, TorchFNIRT

            if np.isscalar(fnirt_warp_resolution_mm):
                warp_resolution = (float(fnirt_warp_resolution_mm),) * 3
            else:
                warp_resolution = tuple(
                    float(value) for value in fnirt_warp_resolution_mm
                )
            level_count = len(tuple(fnirt_strides))
            config = GMFNIRTConfig(
                subsampling=tuple(int(value) for value in fnirt_strides),
                maximum_iterations=tuple(int(value) for value in fnirt_steps),
                input_fwhm_mm=tuple(
                    float(value) for value in fnirt_input_fwhm_mm
                ),
                reference_fwhm_mm=tuple(
                    float(value) for value in fnirt_reference_fwhm_mm
                ),
                regularization=tuple(
                    float(value) for value in fnirt_regularization
                ),
                estimate_intensity=tuple(
                    index < level_count - 1 for index in range(level_count)
                ),
                apply_reference_mask=tuple(
                    index == level_count - 1 for index in range(level_count)
                ),
                warp_resolution_mm=warp_resolution,
                implicit_reference_mask=False,
                implicit_input_mask=False,
            )
            deform_model = TorchFNIRT(
                device=device,
                config=config,
            )
    from ..fnirt.registration import TorchFNIRT

    package_synthmorph = isinstance(
        deform_model, SynthMorphDeformRegistration
    )
    package_fnirt = isinstance(deform_model, TorchFNIRT)
    pull, estimator_qc, native_jacobian = _estimate_nonlinear(
        deform_model,
        registration_backend,
        moving,
        fixed,
        prepared,
    )
    residual_fsl, dense_relative_fsl, fixed_fsl = _pull_ras_to_fsl_fields(
        pull,
        moving,
        fixed,
        prepared.flirt_matrix,
        device=device,
    )
    jacobian_tensor = _fsl_dense_nonlinear_jacobian(
        residual_fsl, fixed_fsl
    )
    moved_data, applywarp_qc = _common_applywarp(
        moving_data,
        fixed_data,
        moving_affine,
        fixed_affine,
        dense_relative_fsl,
        device=device,
    )
    jacobian = (
        jacobian_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    native_jacobian_comparison = None
    if native_jacobian is not None:
        if native_jacobian.shape != jacobian.shape:
            raise ValueError(
                "nonlinear estimator Jacobian does not use the fixed grid"
            )
        difference = jacobian.astype(np.float64) - native_jacobian.astype(
            np.float64
        )
        native_jacobian_comparison = {
            "reference": "estimator analytic nonlinear-only Jacobian",
            "common_dense_mean_absolute_error": float(
                np.abs(difference).mean()
            ),
            "common_dense_maximum_absolute_error": float(
                np.abs(difference).max()
            ),
            "common_dense_correlation": _normalized_correlation(
                jacobian, native_jacobian
            ),
        }
    use_analytic_jacobian = package_fnirt and device.type == "cpu"
    if use_analytic_jacobian:
        if native_jacobian is None or not np.isfinite(native_jacobian).all():
            raise ValueError("CPU FNIRT requires a finite analytic nonlinear Jacobian")
        # TorchFNIRT already evaluated the final spline coefficients with
        # their physical voxel sizes. Avoid replacing that FSL --jout
        # definition with finite differences of the sampled pull field.
        jacobian = native_jacobian.copy()
    warped_gm = new_image(moved_data, fixed)
    jacobian_volume = new_image(jacobian, fixed)
    modulated_gm = new_image(moved_data * jacobian, fixed)
    displacement_qc = _displacement_qc(
        pull, fixed, prepared.fixed_to_moving_world
    )
    residual_array = residual_fsl.detach().cpu().numpy()
    maximum_fsl_residual = float(
        torch.linalg.vector_norm(residual_fsl, dim=-1).max().detach().cpu()
    )
    nonlinear_backend = (
        "pytorch-synthmorph-deform"
        if package_synthmorph
        else (
            "pytorch-fnirt-gm-config"
            if package_fnirt
            else "custom-deform-model"
        )
    )
    qc = {
        "linear": prepared.linear_qc,
        "pre_nonlinear_signature": prepared.signature,
        "reference_mask_source": prepared.reference_mask_source,
        "fsl_reference_mask_exact": (
            False
            if prepared.reference_mask_source.startswith("derived-")
            else None
        ),
        "reference_mask_parity_requirement": (
            "FSL parity requires the same explicit --refmask used by FNIRT; "
            "an arbitrary caller-supplied mask is not automatically equivalent"
        ),
        "shared_pre_nonlinear_inputs": (
            "moving GM, fixed GM template, FSL FLIRT matrix, and reference grid"
        ),
        "reference_mask_role": (
            "recorded in the common registration context and passed to FNIRT; "
            "the SynthMorph deform model has no reference-mask input"
        ),
        "only_backend_specific_stage": (
            "nonlinear pull-field estimation, including estimator-specific "
            "objective and mask use"
        ),
        "nonlinear_backend": nonlinear_backend,
        "nonlinear_estimator_qc": estimator_qc,
        "synthmorph_implementation": (
            _implementation_name(deform_model) if package_synthmorph else None
        ),
        "custom_deform_implementation": (
            None
            if package_synthmorph or package_fnirt
            else _implementation_name(deform_model)
        ),
        "synthmorph_warp_convention": (
            "fixed-grid target-to-source disp-ras: "
            "source_world(target)-target_world"
            if package_synthmorph
            else None
        ),
        "synthmorph_model": "deform" if package_synthmorph else None,
        "synthmorph_extent": synthmorph_extent if package_synthmorph else None,
        "synthmorph_hyper": synthmorph_hyper if package_synthmorph else None,
        "synthmorph_integration_steps": (
            synthmorph_steps if package_synthmorph else None
        ),
        "synthmorph_mid_space": False if package_synthmorph else None,
        "fnirt_implementation": (
            _implementation_name(deform_model) if package_fnirt else None
        ),
        "fnirt_style": False,
        "fsl_fnirt_numerically_equivalent": (
            estimator_qc.get("fsl_fnirt_numerically_equivalent", False)
            if package_fnirt
            else None
        ),
        "warp_conversion": (
            "fixed-grid RAS pull -> FSL scaled-mm residual relative to "
            "inv(FLIRT) -> full dense FSL relative field"
        ),
        "resampling": "fnit.applywarp.TorchApplyWarp",
        "applywarp": applywarp_qc,
        "jacobian_convention": (
            "det(I + d residual_fsl / d fixed_fsl), affine excluded"
        ),
        "jacobian_method": (
            "FSL coefficient analytic spline derivatives"
            if use_analytic_jacobian
            else "FSL dense-field centred finite differences with one-sided edges"
        ),
        "native_jacobian_comparison": native_jacobian_comparison,
        "maximum_fsl_nonlinear_residual_mm": maximum_fsl_residual,
        "residual_fsl_sha256": _array_sha256(residual_array),
        "affine_jacobian_determinant": float(
            np.linalg.det(prepared.fixed_to_moving_world[:3, :3])
        ),
        "jacobian_min": float(jacobian.min()),
        "jacobian_max": float(jacobian.max()),
        "nonpositive_jacobian_voxels": int((jacobian <= 0).sum()),
        **displacement_qc,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return VBMRegistrationResult(
        warped_gm=warped_gm,
        jacobian=jacobian_volume,
        modulated_gm=modulated_gm,
        pull_world_affine=prepared.fixed_to_moving_world.copy(),
        fit_score=_normalized_correlation(moved_data, fixed_data),
        maximum_displacement_mm=maximum_fsl_residual,
        qc=qc,
    )


__all__ = [
    "VBMRegistrationResult",
]
