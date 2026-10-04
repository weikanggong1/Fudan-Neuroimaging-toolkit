"""End-to-end GPU FAST VBM pipeline."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .._nib import FNITNifti1Image, load_image, new_image
from ..fast import FASTResult, TorchFAST
from .registration import VBMRegistrationResult, _register_gm
from .synthmorph_backend import SynthMorphDeformRegistration


OUTPUT_FILENAMES = {
    "brain": "T1_brain.nii.gz",
    "brain_mask": "brain_mask.nii.gz",
    "pve_csf": "T1_brain_pve_0.nii.gz",
    "pve_gm": "T1_brain_pve_1.nii.gz",
    "pve_wm": "T1_brain_pve_2.nii.gz",
    "hard_segmentation": "T1_brain_seg.nii.gz",
    "pve_segmentation": "T1_brain_pveseg.nii.gz",
    "mixel_type": "T1_brain_mixeltype.nii.gz",
    "bias_field": "T1_brain_bias.nii.gz",
    "restored": "T1_brain_restore.nii.gz",
    "warped_gm": "T1_GM_to_template_GM.nii.gz",
    "jacobian": "T1_GM_JAC_nl.nii.gz",
    "modulated_gm": "T1_GM_to_template_GM_mod.nii.gz",
}


def _load_volume(value, name):
    value = load_image(value, name)
    data = np.asanyarray(value.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
        value = new_image(data, value)
    if data.ndim != 3:
        raise ValueError(f"{name} must contain one 3D frame")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return value, data


def _same_grid(image, mask):
    return image.shape[:3] == mask.shape[:3] and np.allclose(
        image.affine,
        mask.affine,
        atol=1e-5,
        rtol=0,
    )


def _validate_geometry(volume, name):
    affine = np.asarray(volume.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{name} affine must be a finite 4x4 matrix")
    linear = affine[:3, :3]
    voxel_size = np.linalg.norm(linear, axis=0)
    if np.any(voxel_size <= 0) or abs(np.linalg.det(linear)) < 1e-8:
        raise ValueError(f"{name} affine must be invertible with positive voxel sizes")


@dataclass
class FastVBMResult:
    """Brain extraction, FAST tissue maps, and template-space VBM outputs."""

    brain: FNITNifti1Image
    brain_mask: FNITNifti1Image
    fast: FASTResult
    registration: VBMRegistrationResult
    settings: dict
    timing_sec: dict

    @property
    def pve_gm(self):
        return self.fast.pve_gm

    @property
    def warped_gm(self):
        return self.registration.warped_gm

    @property
    def jacobian(self):
        return self.registration.jacobian

    @property
    def modulated_gm(self):
        return self.registration.modulated_gm

    def volumes(self):
        """Return every image output keyed by its stable API name."""
        return {
            "brain": self.brain,
            "brain_mask": self.brain_mask,
            "pve_csf": self.fast.pve_csf,
            "pve_gm": self.fast.pve_gm,
            "pve_wm": self.fast.pve_wm,
            "hard_segmentation": self.fast.hard_segmentation,
            "pve_segmentation": self.fast.pve_segmentation,
            "mixel_type": self.fast.mixel_type,
            "bias_field": self.fast.bias_field,
            "restored": self.fast.restored,
            "warped_gm": self.registration.warped_gm,
            "jacobian": self.registration.jacobian,
            "modulated_gm": self.registration.modulated_gm,
        }

    def report(self):
        """Return JSON-serializable settings, timing, and registration QC."""
        mask = np.asanyarray(self.brain_mask.dataobj) > 0
        bias = np.asanyarray(self.fast.bias_field.dataobj)
        nonlinear_backend = self.settings["registration_backend"]
        nonlinear_label = (
            "PyTorch SynthMorph deform"
            if nonlinear_backend == "synthmorph"
            else "PyTorch FNIRT GM-config cubic B-spline"
        )
        jacobian_convention = (
            "FSL nonlinear-only: det(I + d residual_fsl / d fixed_fsl)"
        )
        registration_timing = (
            "linear registration, selected nonlinear backend, Jacobian, "
            "modulation, and CPU output materialization"
        )
        if nonlinear_backend == "synthmorph":
            registration_timing += (
                "; the first call includes lazy checkpoint loading"
            )
        linear_qc = self.registration.qc.get("linear", {})
        flirt_equivalent = linear_qc.get("validated_fsl_equivalent")
        fnirt_equivalent = (
            self.registration.qc.get("fsl_fnirt_numerically_equivalent", False)
            if nonlinear_backend == "fnirt"
            else None
        )
        return {
            "status": "experimental",
            "method": (
                "TorchFAST GM; FSL-default correlation-ratio/Brent FLIRT; "
                f"{nonlinear_label}; common FSL warp conversion, GPU "
                "applywarp, nonlinear Jacobian, and modulation"
            ),
            "fnirt_equivalent": fnirt_equivalent,
            "fsl_fnirt_numerically_equivalent": fnirt_equivalent,
            "fsl_flirt_equivalent": flirt_equivalent,
            "settings": self.settings,
            "timing_sec": self.timing_sec,
            "timing_definition": {
                "total": (
                    "API call entry through CPU output materialization; includes "
                    "input reads and the first call's lazy SynthStrip load; excludes "
                    "FastVBMResult.save and output NIfTI writes"
                ),
                "brain_extraction": (
                    "brain extraction only; the first call includes lazy SynthStrip "
                    "weight loading, later calls reuse the loaded model"
                ),
                "fast": "TorchFAST inference and GPU-to-CPU output materialization",
                "registration_jacobian_modulation": registration_timing,
            },
            "fast": {
                "tissue_means": list(self.fast.tissue_means),
                "tissue_variances": list(self.fast.tissue_variances),
                "bias_range_inside_mask": [
                    float(bias[mask].min()),
                    float(bias[mask].max()),
                ],
            },
            "registration": {
                "fit_score_normalized_correlation": self.registration.fit_score,
                "pull_world_affine": self.registration.pull_world_affine.tolist(),
                "maximum_displacement_mm": (
                    self.registration.maximum_displacement_mm
                ),
                **self.registration.qc,
                "jacobian_convention": jacobian_convention,
            },
            "outputs": OUTPUT_FILENAMES,
        }

    def save(self, output_dir, *, overwrite=False, report=True):
        """Save all NIfTI outputs atomically per file and return their paths."""
        output_dir = Path(output_dir)
        paths = {
            name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()
        }
        report_path = output_dir / "fast_vbm_report.json"
        candidates = [*paths.values(), report_path]
        existing = [path for path in candidates if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"output exists: {existing[0]}; pass overwrite=True to replace it"
            )
        if overwrite and report_path.exists():
            report_path.unlink()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name or 'fast-vbm'}.tmp-", dir=output_dir.parent
        ) as directory:
            stage = Path(directory)
            for name, volume in self.volumes().items():
                volume.save(stage / OUTPUT_FILENAMES[name])
            if report:
                (stage / report_path.name).write_text(
                    json.dumps(self.report(), indent=2, allow_nan=False) + "\n"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            for name, path in paths.items():
                os.replace(stage / OUTPUT_FILENAMES[name], path)
            if report:
                os.replace(stage / report_path.name, report_path)
        return {name: str(path) for name, path in paths.items()}


class FastVBM:
    """Reusable raw-T1 to VBM pipeline using SynthStrip and TorchFAST.

    SynthStrip is loaded lazily and is not required when ``brain_mask`` is
    supplied. Linear registration uses this package's FSL-default FLIRT port.
    Nonlinear registration uses either PyTorch SynthMorph or PyTorch FNIRT.
    Both branches share every VBM stage except nonlinear pull-field estimation.
    """

    def __init__(
        self,
        *,
        device="cpu",
        threads=None,
        synthstrip_weights=None,
        synthmorph_weights=None,
        bias_correction=True,
        fast_execution="tensor",
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
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if threads is not None:
            if not isinstance(threads, int) or threads < 1:
                raise ValueError("threads must be a positive integer")
            torch.set_num_threads(threads)
        self.synthstrip_weights = synthstrip_weights
        self.synthmorph_weights = synthmorph_weights
        if registration_backend not in ("synthmorph", "fnirt"):
            raise ValueError(
                "registration_backend must be 'synthmorph' or 'fnirt'"
            )
        self.registration_backend = registration_backend
        if fast_execution not in ("tensor", "fsl"):
            raise ValueError("fast_execution must be 'tensor' or 'fsl'")
        self.fast_execution = fast_execution
        self.threads = threads
        self.extractor = None
        self.deform_model = None
        self.fast = TorchFAST(
            device=self.device,
            threads=threads,
            bias_fwhm_mm=20.0 if bias_correction else 0.0,
            execution=fast_execution,
        )
        self.bias_correction = bias_correction
        self.synthmorph_extent = synthmorph_extent
        self.synthmorph_hyper = synthmorph_hyper
        self.synthmorph_steps = synthmorph_steps
        self.fnirt_strides = tuple(fnirt_strides)
        self.fnirt_steps = tuple(fnirt_steps)
        self.fnirt_input_fwhm_mm = tuple(fnirt_input_fwhm_mm)
        self.fnirt_reference_fwhm_mm = tuple(fnirt_reference_fwhm_mm)
        self.fnirt_warp_resolution_mm = fnirt_warp_resolution_mm
        self.fnirt_regularization = tuple(fnirt_regularization)

    def _extract(self, image):
        if self.extractor is None:
            from ..synthstrip import SynthStrip

            self.extractor = SynthStrip(
                weights=self.synthstrip_weights,
                device=self.device,
                threads=self.threads,
            )
        result = self.extractor(image)
        return result.image, result.mask

    def _deform_model(self):
        if self.deform_model is None:
            if self.registration_backend == "synthmorph":
                self.deform_model = SynthMorphDeformRegistration(
                    weights=self.synthmorph_weights,
                    device=self.device,
                    extent=self.synthmorph_extent,
                    hyper=self.synthmorph_hyper,
                    steps=self.synthmorph_steps,
                )
            else:
                from ..fnirt.registration import GMFNIRTConfig, TorchFNIRT

                if np.isscalar(self.fnirt_warp_resolution_mm):
                    warp_resolution = (
                        float(self.fnirt_warp_resolution_mm),
                    ) * 3
                else:
                    warp_resolution = tuple(self.fnirt_warp_resolution_mm)
                level_count = len(self.fnirt_strides)
                config = GMFNIRTConfig(
                    subsampling=self.fnirt_strides,
                    maximum_iterations=self.fnirt_steps,
                    input_fwhm_mm=self.fnirt_input_fwhm_mm,
                    reference_fwhm_mm=self.fnirt_reference_fwhm_mm,
                    warp_resolution_mm=warp_resolution,
                    regularization=self.fnirt_regularization,
                    estimate_intensity=tuple(
                        index < level_count - 1
                        for index in range(level_count)
                    ),
                    apply_reference_mask=tuple(
                        index == level_count - 1
                        for index in range(level_count)
                    ),
                    implicit_reference_mask=False,
                    implicit_input_mask=False,
                )
                self.deform_model = TorchFNIRT(
                    device=self.device,
                    config=config,
                )
        return self.deform_model

    def __call__(
        self,
        image,
        template,
        *,
        brain_mask=None,
        reference_mask=None,
    ):
        """Run one T1 image; all returned template maps use ``template``'s grid."""
        total_started = time.perf_counter()
        image, image_data = _load_volume(image, "image")
        template, template_data = _load_volume(template, "template")
        _validate_geometry(image, "image")
        _validate_geometry(template, "template")
        if not np.any(template_data > 0):
            raise ValueError("template must contain positive GM values")

        extraction_started = time.perf_counter()
        if brain_mask is None:
            brain, mask = self._extract(image)
            mask_source = "synthstrip"
        else:
            mask, mask_data = _load_volume(brain_mask, "brain_mask")
            if not _same_grid(image, mask):
                raise ValueError("brain_mask must use the same shape and geometry as image")
            binary = np.asarray(mask_data > 0, dtype=np.uint8)
            if not np.any(binary):
                raise ValueError("brain_mask is empty")
            mask = new_image(binary, image)
            brain_data = image_data.copy()
            brain_data[binary == 0] = min(float(image_data.min()), 0.0)
            brain = new_image(brain_data, image)
            mask_source = "explicit"
        mask_data = np.asanyarray(mask.dataobj)
        if not np.isfinite(mask_data).all() or not np.any(mask_data > 0):
            raise ValueError("brain mask must be finite and nonempty")
        if not _same_grid(image, mask):
            raise ValueError("brain mask must use the same shape and geometry as image")
        extraction_sec = time.perf_counter() - extraction_started

        fast_started = time.perf_counter()
        fast_result = self.fast(brain, mask=mask)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        fast_sec = time.perf_counter() - fast_started

        registration_started = time.perf_counter()
        registration = _register_gm(
            fast_result.pve_gm,
            template,
            device=self.device,
            reference_mask=reference_mask,
            synthmorph_weights=self.synthmorph_weights,
            synthmorph_extent=self.synthmorph_extent,
            synthmorph_hyper=self.synthmorph_hyper,
            synthmorph_steps=self.synthmorph_steps,
            registration_backend=self.registration_backend,
            fnirt_strides=self.fnirt_strides,
            fnirt_steps=self.fnirt_steps,
            fnirt_input_fwhm_mm=self.fnirt_input_fwhm_mm,
            fnirt_reference_fwhm_mm=self.fnirt_reference_fwhm_mm,
            fnirt_warp_resolution_mm=self.fnirt_warp_resolution_mm,
            fnirt_regularization=self.fnirt_regularization,
            deform_model=self._deform_model(),
        )
        registration_sec = time.perf_counter() - registration_started
        timing = {
            "brain_extraction": extraction_sec,
            "fast": fast_sec,
            "registration_jacobian_modulation": registration_sec,
            "total": time.perf_counter() - total_started,
        }
        settings = {
            "device": str(self.device),
            "tf32": {
                "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn": bool(torch.backends.cudnn.allow_tf32),
                "reduced_precision_tensor_dtype": False,
            },
            "mask_source": mask_source,
            "bias_correction": self.bias_correction,
            "linear_backend": "pytorch-fsl-flirt-default",
            "linear_cost": "FSL correlation ratio",
            "linear_optimizer": "MISCMATHS Brent coordinate search",
            "linear_schedule": "FSL default 8/4/2/1 mm",
            "linear_forward_convention": "moving-to-fixed-world-ras",
            "linear_pull_convention": "fixed-to-moving-world-ras",
            "registration_backend": self.registration_backend,
            "registration_reference_mask_source": registration.qc.get(
                "reference_mask_source", "unreported"
            ),
            "nonlinear_backend": (
                "pytorch-synthmorph-deform"
                if self.registration_backend == "synthmorph"
                else "pytorch-fnirt-gm-config"
            ),
            "nonlinear_implementation": (
                "fnit.synthmorph.SynthMorph"
                if self.registration_backend == "synthmorph"
                else "fnit.fnirt.TorchFNIRT"
            ),
            "synthmorph_implementation": (
                "fnit.synthmorph.SynthMorph"
                if self.registration_backend == "synthmorph"
                else None
            ),
            "synthmorph_extent": (
                self.synthmorph_extent
                if self.registration_backend == "synthmorph"
                else None
            ),
            "synthmorph_hyper": (
                self.synthmorph_hyper
                if self.registration_backend == "synthmorph"
                else None
            ),
            "synthmorph_integration_steps": (
                self.synthmorph_steps
                if self.registration_backend == "synthmorph"
                else None
            ),
            "synthmorph_mid_space": (
                False if self.registration_backend == "synthmorph" else None
            ),
            "synthmorph_warp_convention": (
                "fixed-grid-target-to-source-disp-ras"
                if self.registration_backend == "synthmorph"
                else None
            ),
            "warp_convention": (
                "fixed-grid FSL scaled-mm relative displacement after common "
                "RAS-pull conversion"
            ),
            "shared_registration_chain": (
                "FAST GM -> FSL FLIRT -> nonlinear estimator -> FSL warp "
                "conversion -> GPU applywarp -> nonlinear Jacobian -> modulation"
            ),
            "only_backend_specific_stage": (
                "nonlinear pull-field estimation, including estimator-specific "
                "objective and mask use"
            ),
            "reference_mask_role": (
                "recorded for both backends and consumed by the FNIRT "
                "estimator; SynthMorph has no mask input"
            ),
            "fnirt_style": False,
            "fsl_fnirt_numerically_equivalent": registration.qc.get(
                "fsl_fnirt_numerically_equivalent", False
            ) if self.registration_backend == "fnirt" else None,
            "fnirt_strides": (
                list(self.fnirt_strides)
                if self.registration_backend == "fnirt"
                else None
            ),
            "fnirt_steps": (
                list(self.fnirt_steps)
                if self.registration_backend == "fnirt"
                else None
            ),
            "fnirt_input_fwhm_mm": (
                list(self.fnirt_input_fwhm_mm)
                if self.registration_backend == "fnirt"
                else None
            ),
            "fnirt_reference_fwhm_mm": (
                list(self.fnirt_reference_fwhm_mm)
                if self.registration_backend == "fnirt"
                else None
            ),
            "fnirt_warp_resolution_mm": (
                self.fnirt_warp_resolution_mm
                if self.registration_backend == "fnirt"
                else None
            ),
            "fnirt_regularization": (
                list(self.fnirt_regularization)
                if self.registration_backend == "fnirt"
                else None
            ),
            "torch_threads": torch.get_num_threads(),
            "fast": asdict(self.fast.config),
        }
        return FastVBMResult(
            brain=brain,
            brain_mask=mask,
            fast=fast_result,
            registration=registration,
            settings=settings,
            timing_sec=timing,
        )

    def run(
        self,
        image,
        template,
        output_dir,
        *,
        brain_mask=None,
        reference_mask=None,
        overwrite=False,
    ):
        """Run one T1 and save all outputs under ``output_dir``."""
        result = self(
            image,
            template,
            brain_mask=brain_mask,
            reference_mask=reference_mask,
        )
        result.save(output_dir, overwrite=overwrite)
        return result


__all__ = ["FastVBMResult", "FastVBM", "OUTPUT_FILENAMES"]
