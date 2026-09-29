"""Validate saved full-length fMRI volume outputs from the FNIRT branch."""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-output", type=Path, required=True)
    parser.add_argument("--mni-template", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    output = args.pipeline_output
    bold_path = output / "filtered_func_data_clean_MNI152_2mm.nii.gz"
    mask_path = output / "masks/brain_MNI152_2mm.nii.gz"
    pull_path = output / "reg/MNI152_2mm_to_T1_pull_ras.nii.gz"
    for path in (bold_path, mask_path, pull_path):
        with gzip.open(path, "rb") as stream:
            while stream.read(8 * 1024 * 1024):
                pass
    bold = nib.load(str(bold_path))
    mask_image = nib.load(str(mask_path))
    pull_image = nib.load(str(pull_path))
    template = nib.load(str(args.mni_template))
    report = json.loads((output / "pipeline_report.json").read_text())
    if report["registration_backend"] != "fnirt":
        raise ValueError("pipeline did not select FNIRT")
    if report["t1_to_mni_qc"]["global_intensity_model"] != "global_non_linear_with_bias":
        raise ValueError("pipeline did not use the T1 intensity model")
    if report["t1_to_mni_qc"]["t1_intensity_fitting"] != "joint LM polynomial, cubic bias and deformation estimation":
        raise ValueError("pipeline did not use joint T1 intensity optimisation")
    if bold.shape[:3] != template.shape or bold.shape[3] < 2:
        raise ValueError("BOLD dimensions do not match the template")
    if mask_image.shape != template.shape or pull_image.shape != (*template.shape, 3):
        raise ValueError("mask or pull field has the wrong shape")
    for image in (bold, mask_image, pull_image):
        if not np.allclose(image.affine, template.affine, atol=1e-4):
            raise ValueError("output affine differs from MNI template")
    if bold.get_data_dtype() != np.dtype("float32"):
        raise ValueError("BOLD output is not float32")
    if not np.isclose(bold.header.get_zooms()[3], report["tr_seconds"], atol=1e-5):
        raise ValueError("BOLD TR differs from the input report")
    if bold.header.get_xyzt_units()[1] != "sec":
        raise ValueError("BOLD time unit is not seconds")
    mask = np.asarray(mask_image.dataobj) > 0.5
    if not mask.any():
        raise ValueError("MNI mask is empty")
    pull = np.asarray(pull_image.dataobj, dtype=np.float32)
    if not np.isfinite(pull).all():
        raise ValueError("pull field is non-finite")
    values = np.asarray(bold.dataobj, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("BOLD contains non-finite values")
    outside_max = 0.0
    inside_nonzero = 0
    for start in range(0, bold.shape[3], 8):
        block = values[..., start:start + 8]
        outside_max = max(outside_max, float(np.abs(block[~mask]).max()))
        inside_nonzero += int(np.count_nonzero(block[mask]))
    if outside_max != 0:
        raise ValueError("BOLD is nonzero outside its MNI mask")
    log = args.run_log.read_text()
    match = re.search(r"wall_seconds=([0-9.]+) max_rss_kb=([0-9]+)", log)
    if match is None:
        raise ValueError("process wall and RSS were not recorded")
    core = report["t1_to_mni_qc"]
    summary = {
        "schema_version": 1,
        "whole_process_exit_status": 0,
        "registration_backend": report["registration_backend"],
        "t1_intensity_model": core["global_intensity_model"],
        "t1_level_count": len(core["levels"]),
        "t1_fsl_numerically_equivalent": core["fsl_fnirt_numerically_equivalent"],
        "input_bold_shape": report["input_bold_shape"],
        "output_bold_shape": list(bold.shape),
        "output_dtype": str(bold.get_data_dtype()),
        "tr_seconds": float(bold.header.get_zooms()[3]),
        "mni_template_grid_matches": True,
        "mask_voxels": int(mask.sum()),
        "finite_output_values": int(values.size),
        "nonfinite_output_values": 0,
        "inside_mask_nonzero_fraction": inside_nonzero / (int(mask.sum()) * bold.shape[3]),
        "outside_mask_max_abs": outside_max,
        "gzip_crc_passed": True,
        "ica_components": report["ica_components"],
        "ica_converged": report["ica_converged"],
        "aroma_noise_components": report["aroma_noise_components"],
        "timing_seconds": report["timing_seconds"],
        "whole_process_wall_seconds": float(match.group(1)),
        "whole_process_max_cpu_rss_kb": int(match.group(2)),
        "cuda_peak_allocated_gb": report["cuda_peak_allocated_gb"],
        "cuda_peak_reserved_gb": report["cuda_peak_reserved_gb"],
        "source_sha256": {
            name: sha256(args.source_root / path)
            for name, path in {
                "fnirt_registration.py": "src/fnit/fnirt/registration.py",
                "fmri_normalization.py": "src/fnit/fmri/normalization.py",
                "fmri_end_to_end.py": "src/fnit/fmri/end_to_end.py",
            }.items()
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
