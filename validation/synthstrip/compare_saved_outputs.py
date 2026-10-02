#!/usr/bin/env python3
"""Compare complete saved SynthStrip 3D outputs without rerunning inference.

Only anonymous scalar metrics and hashes are reported. Original child-process
and FNIT subfunction timings retain their separate measured boundaries.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_volume(path):
    import nibabel as nib
    import numpy as np
    image = nib.load(str(path))
    array = np.asarray(image.dataobj)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("all compared inputs must be finite complete 3D NIfTI volumes")
    return image, array


def same_grid(left, right):
    import numpy as np
    if left.shape != right.shape or not np.array_equal(left.affine, right.affine):
        raise ValueError("compared images must have exactly the same grid and affine")


def mask_metrics(candidate_path, reference_path):
    import numpy as np
    candidate_image, candidate = load_volume(candidate_path)
    reference_image, reference = load_volume(reference_path)
    same_grid(candidate_image, reference_image)
    if not np.isin(candidate, (0, 1)).all() or not np.isin(reference, (0, 1)).all():
        raise ValueError("saved brain masks must be binary")
    candidate, reference = candidate != 0, reference != 0
    c, r = int(candidate.sum()), int(reference.sum())
    intersection = int((candidate & reference).sum())
    metrics = {
        "shape": list(candidate.shape), "complete_3d": True,
        "candidate_voxels": c, "reference_voxels": r,
        "dice": 2 * intersection / (c + r) if c + r else 1.0,
        "changed_voxels": int((candidate != reference).sum()),
        "candidate_only_voxels": int((candidate & ~reference).sum()),
        "reference_only_voxels": int((reference & ~candidate).sum()),
        "exact_equal": bool(np.array_equal(candidate, reference)),
        "candidate_sha256": sha256(candidate_path),
        "reference_sha256": sha256(reference_path),
    }
    return metrics, candidate | reference, candidate_image


def intensity_metrics(candidate_path, reference_path, region, region_image):
    import numpy as np
    candidate_image, candidate = load_volume(candidate_path)
    reference_image, reference = load_volume(reference_path)
    same_grid(candidate_image, reference_image)
    same_grid(candidate_image, region_image)

    def measure(c, r):
        c, r = c.astype(np.float64, copy=False).reshape(-1), r.astype(np.float64, copy=False).reshape(-1)
        error = c - r
        cc, rr = c - c.mean(), r - r.mean()
        denominator = float(np.linalg.norm(cc) * np.linalg.norm(rr))
        return {"samples": int(c.size), "rmse": float(np.sqrt(np.mean(error * error))),
                "max_abs": float(np.abs(error).max()),
                "spatial_pearson_r": float(cc @ rr / denominator) if denominator else None,
                "exact_equal": bool(np.array_equal(c, r)),
                "changed_voxels": int(np.count_nonzero(c != r))}

    return {"shape": list(candidate.shape), "complete_3d": True,
            "candidate_dtype": str(candidate.dtype), "reference_dtype": str(reference.dtype),
            "whole_image": measure(candidate, reference),
            "brain_union": measure(candidate[region], reference[region]),
            "candidate_sha256": sha256(candidate_path), "reference_sha256": sha256(reference_path)}


def comparison(args):
    api = json.loads(args.api_report.read_text())
    native = json.loads(args.native_report.read_text())
    if api["source_revision"] != args.source_revision:
        raise ValueError("source revision differs from the captured current API report")
    source_hash = sha256(args.source_file)
    weights_hash = sha256(args.weights)
    captured_hash = api["source_sha256"]["src/fnit/synthstrip/pipeline.py"]
    if source_hash != captured_hash or args.expected_source_sha256 and source_hash != args.expected_source_sha256:
        raise ValueError("live source hash differs from captured/expected runtime source")
    native_weight_hash = native["input_sha256"]["synthstrip_weights"]
    if weights_hash != native_weight_hash or args.expected_weights_sha256 and weights_hash != args.expected_weights_sha256:
        raise ValueError("checkpoint hash differs from original/expected checkpoint")
    for role in ("sbref", "t1w"):
        if api["input_sha256"][role] != native["input_sha256"][role]:
            raise ValueError("current and original input file hashes differ")
    native_steps = {item["name"]: item for item in native["steps"]}
    pipeline_times = api["timing_seconds"]["pipeline_stages"]
    report = {
        "source_revision": args.source_revision,
        "synthstrip_source_sha256": source_hash,
        "checkpoint_sha256": weights_hash, "checkpoint_bytes": int(args.weights.stat().st_size),
        "runtime_source_matches_capture": source_hash == captured_hash,
        "same_checkpoint_as_original": weights_hash == native_weight_hash,
        "inputs_sha256": {role: api["input_sha256"][role] for role in ("sbref", "t1w")},
        "current_api_report_sha256": sha256(args.api_report),
        "original_anatomy_report_sha256": sha256(args.native_report),
        "original_program_sha256": native["component_sha256"]["mri_synthstrip_source"],
        "timing": {
            "current": {"epi_seconds": pipeline_times["epi_synthstrip"],
                        "t1_seconds": pipeline_times["t1_synthstrip"],
                        "boundary": "Reused-model FNIT subfunction stage: image processing, inference and stage output saving; model construction is outside both stages."},
            "original": {"epi_seconds": native_steps["epi_synthstrip"]["process_wall_seconds"],
                         "t1_seconds": native_steps["t1_synthstrip"]["process_wall_seconds"],
                         "boundary": native["timing_boundary"]},
            "matched_boundaries": False,
            "controlled_speedup_claim": False,
        },
        "environment": api["environment"],
        "scope": "Latest full-run EPI/T1 extraction vs saved unmodified FreeSurfer outputs on the same real input files; complete original 3D grids. No inference/original software rerun.",
        "privacy": "Scalars and hashes only; no paths, affines, coordinates or voxel arrays are published.",
    }
    epi_mask, epi_region, epi_grid = mask_metrics(args.epi_mask, args.native_epi_mask)
    t1_mask, t1_region, t1_grid = mask_metrics(args.t1_mask, args.native_t1_mask)
    report["epi"] = {"mask": epi_mask}
    report["t1"] = {"mask": t1_mask,
                    "brain": intensity_metrics(args.t1_brain, args.native_t1_brain, t1_region, t1_grid)}
    if args.epi_brain is not None and args.native_epi_brain is not None:
        report["epi"]["brain"] = intensity_metrics(args.epi_brain, args.native_epi_brain, epi_region, epi_grid)
    else:
        report["epi"]["brain"] = {"compared": False, "reason": "Current pipeline capture preserves the binary EPI mask, not a standalone SynthStrip EPI brain image."}
    if args.frozen_t1_mask is not None and args.frozen_t1_brain is not None:
        frozen_mask, region, grid = mask_metrics(args.t1_mask, args.frozen_t1_mask)
        report["current_vs_frozen_t1"] = {
            "mask": frozen_mask,
            "brain": intensity_metrics(args.t1_brain, args.frozen_t1_brain, region, grid),
        }
        report["existing_fixed_warp_figure_preserved"] = bool(
            frozen_mask["exact_equal"] and report["current_vs_frozen_t1"]["brain"]["whole_image"]["exact_equal"])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("epi-mask", "native-epi-mask", "t1-mask", "native-t1-mask", "t1-brain",
                 "native-t1-brain", "source-file", "weights", "api-report", "native-report"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("epi-brain", "native-epi-brain", "frozen-t1-mask", "frozen-t1-brain"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--expected-source-sha256")
    p.add_argument("--expected-weights-sha256")
    p.add_argument("--report", required=True, help="Anonymous output JSON; '-' prints JSON to stdout")
    args = p.parse_args()
    if (args.frozen_t1_mask is None) != (args.frozen_t1_brain is None):
        p.error("both frozen T1 files are required for the figure-preservation check")
    if (args.epi_brain is None) != (args.native_epi_brain is None):
        p.error("both EPI brain files are required for intensity comparison")
    output = comparison(args)
    contents = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.report == "-":
        print(contents, end="")
    else:
        path = Path(args.report)
        if path.exists():
            raise FileExistsError("choose a fresh report file")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)


if __name__ == "__main__":
    main()
