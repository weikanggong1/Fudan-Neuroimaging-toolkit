"""UK Biobank v1.5 single-subject TBSS registration and map propagation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion
import surfa as sf

from .._dmri import configure_device
from ..applywarp import TorchApplyWarp
from ..flirt import TorchFLIRT
from ..fnirt import GMFNIRTConfig, TorchFNIRT


UKB_PIPELINE_COMMIT = "0e39a7f7eb76b55437942bfa3073512506b6c8fa"


@dataclass(frozen=True)
class TBSSConfig:
    """UKB ``oxford_s1/s2/s3.cnf`` combined into one continuous schedule."""

    skeleton_threshold: float = 2000.0
    fnirt: GMFNIRTConfig = GMFNIRTConfig(
        subsampling=(8, 4, 2, 2, 1, 1),
        maximum_iterations=(5, 5, 5, 5, 50, 25),
        input_fwhm_mm=(12.0, 8.0, 4.0, 4.0, 1.0, 1.0),
        reference_fwhm_mm=(12.0, 8.0, 4.0, 4.0, 1.0, 1.0),
        regularization=(300.0, 75.0, 50.0, 40.0, 100.0, 30.0),
        estimate_intensity=(True, True, True, False, False, False),
        apply_reference_mask=(False, False, False, False, False, False),
        warp_resolution_mm=(10.0, 10.0, 10.0),
        warp_resolution_schedule_mm=(
            (10.0, 10.0, 10.0),
            (10.0, 10.0, 10.0),
            (10.0, 10.0, 10.0),
            (10.0, 10.0, 10.0),
            (2.0, 2.0, 2.0),
            (2.0, 2.0, 2.0),
        ),
        jacobian_range=(0.01, 100.0),
        ssd_weighted_lambda=True,
    )


@dataclass(frozen=True)
class TBSSResult:
    standard_maps: dict[str, nib.Nifti1Image]
    skeleton_maps: dict[str, nib.Nifti1Image]
    affine: np.ndarray
    coefficients: nib.Nifti1Image
    jacobian: nib.Nifti1Image
    preprocessed_fa: nib.Nifti1Image
    registration_weight: nib.Nifti1Image
    qc: dict

    def save(self, output_dir, *, overwrite=False):
        output_dir = Path(output_dir).expanduser()
        standard_dir = output_dir / "standard"
        skeleton_dir = output_dir / "skeleton"
        fixed = {
            output_dir / "dti_FA_preprocessed.nii.gz": self.preprocessed_fa,
            output_dir / "dti_FA_registration_weight.nii.gz": self.registration_weight,
            output_dir / "dti_FA_to_MNI_warp.nii.gz": self.coefficients,
            output_dir / "dti_FA_to_MNI_jacobian.nii.gz": self.jacobian,
        }
        maps = {
            **{standard_dir / f"{name}.nii.gz": image for name, image in self.standard_maps.items()},
            **{skeleton_dir / f"{name}.nii.gz": image for name, image in self.skeleton_maps.items()},
        }
        matrix_path = output_dir / "dti_FA_to_MNI_affine.mat"
        report_path = output_dir / "tbss_report.json"
        existing = [path for path in (*fixed, *maps, matrix_path, report_path) if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"output exists: {existing[0]}; pass overwrite=True")
        standard_dir.mkdir(parents=True, exist_ok=True)
        skeleton_dir.mkdir(parents=True, exist_ok=True)
        for path, image in {**fixed, **maps}.items():
            nib.save(image, str(path))
        np.savetxt(matrix_path, self.affine, fmt="%.12g")
        report_path.write_text(json.dumps(self.qc, indent=2) + "\n", encoding="utf-8")
        written = (*fixed, *maps)
        return {
            **{
                str(path.relative_to(output_dir)): path
                for path in written
            },
            matrix_path.name: matrix_path,
            report_path.name: report_path,
        }


def _image_like(data, reference, *, dtype=np.float32):
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    return nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header)


def preprocess_fa(image):
    """Port ``bb_tbss_1_preproc`` for one FA image."""
    source = nib.load(os.fspath(image)) if isinstance(image, (str, os.PathLike)) else image
    values = np.asarray(source.dataobj, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("FA must be one 3D image")
    values = np.minimum(values, 1.0)
    support = binary_erosion(
        values != 0, structure=np.ones((3, 3, 3), dtype=bool), border_value=0
    )
    values *= support
    for axis in range(3):
        start = [slice(None)] * 3
        end = [slice(None)] * 3
        start[axis] = 0
        end[axis] = -1
        values[tuple(start)] = 0
        values[tuple(end)] = 0
    brain = values != 0
    dilated = binary_dilation(
        brain, structure=np.ones((3, 3, 3), dtype=bool), iterations=2
    )
    # Exact fslmaths expression: abs(dilated - 1) + original mask.
    weight = np.abs(dilated.astype(np.int16) - 1) + brain.astype(np.int16)
    return _image_like(values, source), _image_like(weight, source, dtype=np.uint8)


class TorchTBSS:
    """Map UKB DTI/NODDI maps to FMRIB58_FA_1mm with one FA warp."""

    def __init__(self, device=None, *, config=None):
        self.device = configure_device(device)
        self.config = TBSSConfig() if config is None else config

    def __call__(self, maps, fa_reference, fa_skeleton):
        missing = [name for name in ("FA", "L1", "L2", "L3", "MO", "MD", "ICVF", "OD", "ISOVF") if name not in maps]
        if missing:
            raise ValueError(f"missing diffusion map: {missing[0]}")
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        preprocessed, weight = preprocess_fa(maps["FA"])
        reference = nib.load(os.fspath(fa_reference))
        skeleton = nib.load(os.fspath(fa_skeleton))
        if reference.shape[:3] != skeleton.shape[:3] or not np.allclose(
            reference.affine, skeleton.affine, atol=1e-5, rtol=0
        ):
            raise ValueError("FA reference and skeleton must use the same grid")
        moving_volume = sf.Volume(
            np.asarray(preprocessed.dataobj, dtype=np.float32),
            geometry=sf.ImageGeometry(preprocessed.shape[:3], vox2world=preprocessed.affine),
        )
        reference_volume = sf.Volume(
            np.asarray(reference.dataobj, dtype=np.float32),
            geometry=sf.ImageGeometry(reference.shape[:3], vox2world=reference.affine),
        )
        weight_volume = sf.Volume(
            np.asarray(weight.dataobj, dtype=np.float32),
            geometry=sf.ImageGeometry(weight.shape[:3], vox2world=weight.affine),
        )
        linear = TorchFLIRT(device=self.device)(
            moving_volume, reference_volume, inweight=weight_volume
        )
        initial = sf.Affine(
            linear.moving_to_fixed_world,
            source=moving_volume,
            target=reference_volume,
            space="world",
        )
        nonlinear = TorchFNIRT(device=self.device or "cpu", config=self.config.fnirt)(
            moving_volume, reference_volume, initial
        )
        apply = TorchApplyWarp(device=self.device)
        standard = {}
        for name, image in maps.items():
            source = preprocessed if name == "FA" else image
            warped = apply(
                source,
                reference,
                warp=nonlinear.coefficient_image,
                interpolation="trilinear",
                warp_convention="relative",
            ).image
            values = np.asarray(warped.dataobj, dtype=np.float32)
            if values.ndim == 4 and values.shape[3] == 1:
                values = values[..., 0]
            if values.ndim != 3:
                raise ValueError(f"{name} must be a 3D map or one-frame 4D image")
            standard[name] = _image_like(values, reference)
        valid_mask = (
            np.asarray(standard["FA"].dataobj, dtype=np.float32) != 0
        ) & (np.asarray(reference.dataobj, dtype=np.float32) != 0)
        standard["FA"] = _image_like(
            np.asarray(standard["FA"].dataobj, dtype=np.float32) * valid_mask,
            reference,
        )
        skeleton_mask = (
            np.asarray(skeleton.dataobj, dtype=np.float32)
            >= self.config.skeleton_threshold
        ) & valid_mask
        skeletonised = {
            name: _image_like(
                np.asarray(image.dataobj, dtype=np.float32) * skeleton_mask,
                reference,
            )
            for name, image in standard.items()
        }
        elapsed = time.perf_counter() - started
        return TBSSResult(
            standard,
            skeletonised,
            linear.matrix,
            nonlinear.coefficient_image,
            _image_like(np.asarray(nonlinear.nonlinear_jacobian.data), reference),
            preprocessed,
            weight,
            {
                "device": str(self.device),
                "dtype": "float32 images / float64 FNIRT coefficients",
                "tf32": bool(str(self.device).startswith("cuda")),
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else 0
                ),
                "reference_implementation": "UK Biobank pipeline v1.5 bb_tbss",
                "reference_commit": UKB_PIPELINE_COMMIT,
                "standard_grid": "FMRIB58_FA_1mm (MNI152 space)",
                "maps": list(standard),
                "skeleton_threshold": self.config.skeleton_threshold,
                "ukb_preprocessing_contract": True,
                "ukb_output_contract": True,
                "official_oxford_three_stage_config": True,
                "official_flirt_input_weight_used": True,
                "official_config_sha256": {
                    "oxford_s1.cnf": "3df6320e97300f9cef8a77b6c7c6306b8256351d5a7022d9966c076f24747227",
                    "oxford_s2.cnf": "b2fe418169f8ce1dde5e277a09e8edcdb6a0dc3f3065972ae69404020ef639ec",
                    "oxford_s3.cnf": "9d304dfd224b8e48f20364dcce77c7618722cf3a9bcd40d764a6d25ee8020eaf",
                },
                "ukb_numerically_equivalent": False,
                "known_difference": (
                    "the official schedules are combined continuously; stage-2/3 "
                    "SCG is executed by the packaged FNIRT LM/PCG optimiser"
                ),
                "flirt": linear.qc,
                "fnirt": nonlinear.qc,
                "elapsed_seconds": elapsed,
            },
        )

    def run(self, *args, output_dir, overwrite=False, **kwargs):
        result = self(*args, **kwargs)
        result.save(output_dir, overwrite=overwrite)
        return result


__all__ = ["TBSSConfig", "TBSSResult", "TorchTBSS", "preprocess_fa"]
