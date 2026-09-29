"""Compare the T1 FNIRT fMRI registration branch with same-input FSL outputs."""

import argparse
import gzip
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.fmri.normalization import register_t1_to_mni, resample_world


def image(path):
    with gzip.open(path, "rb") as stream:
        while stream.read(8 * 1024 * 1024):
            pass
    source = nib.load(str(path))
    data = np.asarray(source.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite image: {path}")
    return source, data


def correlation(first, second, mask):
    return float(np.corrcoef(first[mask], second[mask])[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-mask", required=True)
    parser.add_argument("--fsl-warped", required=True)
    parser.add_argument("--fsl-pull-x", required=True)
    parser.add_argument("--fsl-pull-y", required=True)
    parser.add_argument("--fsl-pull-z", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.cuda.init()
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    registration = register_t1_to_mni(
        t1_brain=args.moving,
        mni_brain=args.reference,
        output_dir=output,
        backend="fnirt",
        reference_mask=args.reference_mask,
        device=args.device,
    )
    registration_seconds = time.perf_counter() - started
    resample_world(
        source=args.moving,
        reference=args.reference,
        reference_to_source_world=np.eye(4),
        output=output / "T1_in_MNI.nii.gz",
        pre_affine_pull_ras=registration.pull_ras,
        batch_size=1,
        device=args.device,
    )
    total_seconds = time.perf_counter() - started

    reference_image, reference = image(args.reference)
    _, mask = image(args.reference_mask)
    result_image, result = image(output / "T1_in_MNI.nii.gz")
    official_image, official = image(args.fsl_warped)
    pull_image, pull = image(registration.pull_ras)
    if any(item.shape != reference_image.shape for item in (
        result_image, official_image, pull_image.slicer[..., 0]
    )):
        raise ValueError("output and FSL reference grids differ")
    brain = mask > 0.5
    support_result = result > 0.05 * np.percentile(result[brain & (result > 0)], 99)
    support_official = official > 0.05 * np.percentile(official[brain & (official > 0)], 99)
    valid = brain & support_result & support_official
    grid = np.indices(reference_image.shape, dtype=np.float32).reshape(3, -1)
    world = (
        reference_image.affine[:3, :3].astype(np.float32) @ grid
        + reference_image.affine[:3, 3:4].astype(np.float32)
    ).T.reshape((*reference_image.shape, 3))
    official_world = np.stack([
        image(path)[1] for path in (
            args.fsl_pull_x, args.fsl_pull_y, args.fsl_pull_z
        )
    ], axis=-1)
    difference = np.linalg.norm(world + pull - official_world, axis=-1)[valid]
    report = {
        "registration_backend": "fnirt",
        "intensity_model": registration.qc["global_intensity_model"],
        "intensity_fitting": registration.qc["t1_intensity_fitting"],
        "fsl_fnirt_numerically_equivalent": False,
        "registration_seconds": registration_seconds,
        "registration_and_resample_seconds": total_seconds,
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated(torch.device(args.device)) / 1e9,
        "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved(torch.device(args.device)) / 1e9,
        "output_shape": list(result.shape),
        "output_grid_matches_reference": bool(np.allclose(result_image.affine, reference_image.affine)),
        "gzip_crc_and_finite": True,
        "pearson_vs_fsl_same_input": correlation(result, official, brain),
        "support_dice_vs_fsl_same_input": float(
            2 * np.count_nonzero(support_result & support_official)
            / (support_result.sum() + support_official.sum())
        ),
        "pull_difference_median_mm": float(np.median(difference)),
        "pull_difference_p95_mm": float(np.percentile(difference, 95)),
        "pull_compared_voxels": int(valid.sum()),
        "pearson_to_mni_template": correlation(result, reference, brain),
        "level_intensity": [
            {
                "level": level["level"],
                "polynomial": level["t1_polynomial"],
                "bias_range": level["t1_bias_range"],
                "bias_pcg": level["t1_bias_pcg"],
            }
            for level in registration.qc["levels"]
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
