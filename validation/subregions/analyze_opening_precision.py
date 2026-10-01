"""Validation-only hippocampal alignment opening on an existing precision driver.

Forward the usual analyze_raw_precision.py arguments and optionally add
--alignment-opening. The control changes only the binary target supplied to
the initial affine estimator. Coarse labels used by later fits are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from time import monotonic

import nibabel as nib
import numpy as np
from scipy import ndimage


PUBLIC_INPUT_SHA256 = "f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _opening_estimator(original, records: list[dict]):
    axis = np.arange(-1, 2)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    structure = x * x + y * y + z * z <= 1

    def estimate(atlas_image, target_image, target_labels, target_label_ids, **kwargs):
        started = monotonic()
        target = np.isin(target_labels, target_label_ids)
        opened = ndimage.binary_dilation(
            ndimage.binary_erosion(target, structure=structure, border_value=1),
            structure=structure)
        if np.count_nonzero(opened) < 4:
            raise ValueError("alignment opening leaves fewer than four target voxels")
        before = nib.affines.apply_affine(target_image.affine, np.argwhere(target).mean(0))
        after = nib.affines.apply_affine(target_image.affine, np.argwhere(opened).mean(0))
        record = {
            "target_label_ids": [int(label) for label in target_label_ids],
            "target_shape": [int(size) for size in target.shape],
            "target_affine": np.asarray(target_image.affine).tolist(),
            "target_voxel_sizes_mm": np.linalg.norm(target_image.affine[:3, :3], axis=0).tolist(),
            "before_voxels": int(target.sum()), "after_voxels": int(opened.sum()),
            "changed_voxels": int(np.count_nonzero(target != opened)),
            "removed_voxels": int(np.count_nonzero(target & ~opened)),
            "added_voxels": int(np.count_nonzero(~target & opened)),
            "before_world_centroid_mm": before.tolist(), "after_world_centroid_mm": after.tolist(),
            "centroid_shift_mm": float(np.linalg.norm(after - before)),
            "before_mask_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
            "opened_mask_sha256": hashlib.sha256(opened.tobytes()).hexdigest(),
            "opening_cpu_seconds": monotonic() - started,
        }
        matrix, score = original(atlas_image, target_image, opened.astype(np.uint8), (1,), **kwargs)
        record.update({"alignment_dice": float(score), "atlas_to_target_voxel": np.asarray(matrix).tolist()})
        records.append(record)
        return matrix, score

    return estimate


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument("--alignment-opening", action="store_true")
    wrapper_parser.add_argument("--precision-driver", type=Path,
                                default=Path(__file__).with_name("analyze_raw_precision.py"))
    wrapper_parser.add_argument("--expected-driver-sha256")
    wrapper_args, forwarded = wrapper_parser.parse_known_args(argv)
    selection_parser = argparse.ArgumentParser(add_help=False)
    selection_parser.add_argument("--root", type=Path)
    selection_parser.add_argument("--output", type=Path)
    selection_parser.add_argument("--structure", default="thalamus")
    selection, _ = selection_parser.parse_known_args(forwarded)
    help_requested = "--help" in forwarded or "-h" in forwarded
    if wrapper_args.alignment_opening and not help_requested and selection.structure not in (
            "hippo-amygdala-left", "hippo-amygdala-right"):
        wrapper_parser.error("--alignment-opening requires a hippo-amygdala structure")
    if not help_requested and (selection.root is None or selection.output is None):
        wrapper_parser.error("--root and --output are required")
    driver = wrapper_args.precision_driver.resolve()
    driver_hash = sha256(driver)
    if wrapper_args.expected_driver_sha256 is not None and driver_hash != wrapper_args.expected_driver_sha256:
        raise ValueError("precision driver checksum differs from --expected-driver-sha256")
    input_record = None
    if not help_requested:
        raw = selection.root.resolve().parent.parent / "examples/data/sub-01_T1w.nii.gz"
        if raw.stat().st_size != 3847853 or sha256(raw) != PUBLIC_INPUT_SHA256:
            raise ValueError("public subject input differs from the verified CC0 benchmark")
        input_record = {"path": str(raw), "bytes": raw.stat().st_size,
                        "sha256": PUBLIC_INPUT_SHA256, "license": "CC0"}
    prior_argv, prior_path = sys.argv, list(sys.path)
    sys.path.insert(0, str(driver.parent))
    sys.argv = [str(driver), *forwarded]
    records = []
    base_module = None
    original_estimator = None
    try:
        if help_requested:
            print("Additional validation arguments: --alignment-opening, --precision-driver PATH, "
                  "--expected-driver-sha256 SHA256", flush=True)
        namespace = runpy.run_path(str(driver), run_name="_frozen_precision_driver")
        from fnit.gems.recipes import base as base_module
        original_estimator = base_module.estimate_mask_affine
        if wrapper_args.alignment_opening:
            base_module.estimate_mask_affine = _opening_estimator(original_estimator, records)
        namespace["main"]()
        report_path = selection.output / "analysis.json"
        report = json.loads(report_path.read_text())
        report["alignment_opening_control"] = {
            "enabled": wrapper_args.alignment_opening,
            "scope": "initial affine binary target only",
            "morphology": "radius-one voxel sphere; erosion border_value=1 then dilation border_value=0",
            "grid_rule": "driver processing grid; no extra resampling",
            "official_reference": "FreeSurfer v8.2.0 python/gems/subregions/core.py lines 250-253",
            "public_input": input_record, "records": records,
            "wrapper_sha256": sha256(Path(__file__)),
            "precision_driver_path": str(driver), "precision_driver_sha256": driver_hash,
            "expected_driver_sha256": wrapper_args.expected_driver_sha256,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"alignment_opening": wrapper_args.alignment_opening,
                          "changed_voxels": [record["changed_voxels"] for record in records]}), flush=True)
    finally:
        if base_module is not None and original_estimator is not None:
            base_module.estimate_mask_affine = original_estimator
        sys.argv = prior_argv
        sys.path[:] = prior_path


if __name__ == "__main__":
    main()
