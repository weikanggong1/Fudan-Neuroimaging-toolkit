"""逐阶段比较两例真实 T1 的 recon-all 体积前段，不以相关系数代替标签 Dice。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


STAGES = (
    "orig/001.mgz", "rawavg.mgz", "orig.mgz", "nu.mgz", "T1.mgz",
    "brainmask.mgz", "norm.mgz", "brain.mgz", "wm.seg.mgz",
    "wm.asegedit.mgz", "wm.mgz", "filled.mgz",
)
LABELS = {"filled.mgz"}
FOREGROUND = {"brainmask.mgz", "wm.seg.mgz", "wm.asegedit.mgz", "wm.mgz"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dice(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref, cand = reference.ravel().astype(np.int64), candidate.ravel().astype(np.int64)
    if np.any(ref < 0) or np.any(cand < 0):
        raise ValueError("discrete labels must be nonnegative")
    n = int(max(ref.max(), cand.max())) + 1
    count_ref, count_cand = np.bincount(ref, minlength=n), np.bincount(cand, minlength=n)
    common = np.bincount(ref[ref == cand], minlength=n)
    labels = np.flatnonzero(count_ref + count_cand)
    labels = labels[labels != 0]
    result = {str(int(label)): float(2 * common[label] /
                                     (count_ref[label] + count_cand[label]))
              for label in labels}
    foreground_ref, foreground_cand = ref != 0, cand != 0
    total = int(foreground_ref.sum() + foreground_cand.sum())
    result["foreground"] = float(2 * np.count_nonzero(foreground_ref & foreground_cand)
                                 / total) if total else 1.0
    return result


def compare_volume(reference_path: Path, candidate_path: Path, labels: bool) -> dict:
    """读取 MGH 强度或离散标签，报告网格、类型、体素误差及可选 Dice。"""
    reference_image, candidate_image = (nib.load(str(path)) for path in
                                        (reference_path, candidate_path))
    reference, candidate = (np.asarray(image.dataobj) for image in
                            (reference_image, candidate_image))
    row = {"reference_sha256": _sha256(reference_path),
           "candidate_sha256": _sha256(candidate_path),
           "reference_shape": [int(axis) for axis in reference.shape],
           "candidate_shape": [int(axis) for axis in candidate.shape],
           "reference_dtype": str(reference.dtype), "candidate_dtype": str(candidate.dtype),
           "affine_max_absolute_mm": float(np.max(np.abs(reference_image.affine -
                                                         candidate_image.affine)))}
    if reference.shape != candidate.shape:
        return row
    error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    row.update(different_voxels=int(np.count_nonzero(error)),
               maximum_absolute_voxel_difference=float(error.max()),
               p99_absolute_voxel_difference=float(np.percentile(error, 99)))
    if labels:
        row["dice_by_label"] = _dice(reference, candidate)
    elif reference_path.name in FOREGROUND:
        ref_foreground, cand_foreground = reference != 0, candidate != 0
        total = int(ref_foreground.sum() + cand_foreground.sum())
        row["foreground_dice"] = (float(2 * np.count_nonzero(
            ref_foreground & cand_foreground) / total) if total else 1.0)
    return row


def compare_lta(reference_path: Path, candidate_path: Path) -> dict:
    """比较 talairach.lta 的 4×4 数值；文件头和体积几何仍保留原文件哈希。"""
    def matrix(path: Path) -> np.ndarray:
        lines = path.read_text().splitlines()
        begin = lines.index("1 4 4") + 1
        return np.asarray([[float(value) for value in line.split()]
                           for line in lines[begin:begin + 4]])
    reference, candidate = matrix(reference_path), matrix(candidate_path)
    return {"reference_sha256": _sha256(reference_path),
            "candidate_sha256": _sha256(candidate_path),
            "different_elements": int(np.count_nonzero(reference != candidate)),
            "maximum_absolute_element_difference": float(np.max(np.abs(reference - candidate))),
            "reference_matrix": reference.tolist(), "candidate_matrix": candidate.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-subject", type=Path, required=True)
    parser.add_argument("--candidate-subject", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--scope", choices=("frozen_stage", "connected_prefix", "raw_t1_full"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for stage in STAGES:
        reference_path = args.reference_subject / "mri" / stage
        candidate_path = args.candidate_subject / "mri" / stage
        rows[stage] = (compare_volume(reference_path, candidate_path, stage in LABELS)
                       if reference_path.is_file() and candidate_path.is_file()
                       else {"reference_exists": reference_path.is_file(),
                             "candidate_exists": candidate_path.is_file()})
    name = "transforms/talairach.lta"
    reference_path, candidate_path = (root / "mri" / name for root in
                                      (args.reference_subject, args.candidate_subject))
    rows[name] = (compare_lta(reference_path, candidate_path)
                  if reference_path.is_file() and candidate_path.is_file()
                  else {"reference_exists": reference_path.is_file(),
                        "candidate_exists": candidate_path.is_file()})
    report = {"code_commit": args.code_commit, "scope": args.scope,
              "reference_subject": str(args.reference_subject),
              "candidate_subject": str(args.candidate_subject), "stages": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: row.get("different_voxels", row.get("different_elements"))
                      for name, row in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
