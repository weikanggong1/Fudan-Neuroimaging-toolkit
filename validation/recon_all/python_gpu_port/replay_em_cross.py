"""在隔离目录做 nu/brainmask 四组 GCA 注册及相同后续归一化。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import nibabel as nib
import numpy as np

from fnit.recon_all.ca_normalize_python import run_ca_normalize


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lta(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = lines.index("1 4 4") + 1
    return np.asarray([[float(value) for value in line.split()]
                       for line in lines[start:start + 4]])


def _volume_difference(reference: Path, candidate: Path) -> dict:
    a, b = (np.asarray(nib.load(str(path)).dataobj) for path in (reference, candidate))
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"different_voxels": int(np.count_nonzero(error)),
            "max_absolute_difference": float(error.max()),
            "p99_absolute_difference": float(np.percentile(error, 99))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference_nu", "candidate_nu", "reference_mask", "candidate_mask",
                 "binary", "assets", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    atlas = args.assets / "average/RB_all_2020-01-02.gca"
    pairs = (
        ("reference_nu_reference_mask", args.reference_nu, args.reference_mask),
        ("candidate_nu_reference_mask", args.candidate_nu, args.reference_mask),
        ("reference_nu_candidate_mask", args.reference_nu, args.candidate_mask),
        ("candidate_nu_candidate_mask", args.candidate_nu, args.candidate_mask),
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("diagnostic output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"code_commit": args.code_commit, "host": platform.node(),
              "binary_sha256": _hash(args.binary), "atlas_sha256": _hash(atlas),
              "input_sha256": {name: _hash(getattr(args, name)) for name in
                               ("reference_nu", "candidate_nu",
                                "reference_mask", "candidate_mask")},
              "input_differences": {
                  "nu": _volume_difference(args.reference_nu, args.candidate_nu),
                  "brainmask": _volume_difference(args.reference_mask, args.candidate_mask),
              }, "groups": {}, "scope": "isolated_cross_input_diagnostic_only"}
    baseline_lta, baseline_norm = None, None
    for name, nu, mask in pairs:
        directory = args.output_dir / name
        (directory / "transforms").mkdir(parents=True)
        (directory / "nu.mgz").symlink_to(nu.resolve())
        (directory / "brainmask.mgz").symlink_to(mask.resolve())
        lta = directory / "transforms/talairach.lta"
        norm, ctrl = directory / "norm.mgz", directory / "ctrl_pts.mgz"
        command = [str(args.binary), "-uns", "3", "-mask", "brainmask.mgz",
                   "nu.mgz", str(atlas), "transforms/talairach.lta"]
        env = {**os.environ, "FREESURFER_HOME": str(args.assets),
               "OMP_NUM_THREADS": "4"}
        started = time.perf_counter()
        with (directory / "mri_em_register.log").open("w") as stream:
            subprocess.run(command, cwd=directory, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        em_seconds = time.perf_counter() - started
        started = time.perf_counter()
        run_ca_normalize(nu, mask, atlas, lta, norm, ctrl)
        norm_seconds = time.perf_counter() - started
        row = {"nu": str(nu), "brainmask": str(mask),
               "lta_sha256": _hash(lta), "norm_sha256": _hash(norm),
               "em_seconds": em_seconds, "ca_normalize_seconds": norm_seconds,
               "lta_matrix": _lta(lta).tolist()}
        if baseline_lta is not None:
            delta = np.abs(_lta(lta) - baseline_lta)
            row["lta_vs_reference_pair"] = {
                "different_elements": int(np.count_nonzero(delta)),
                "max_absolute_difference": float(delta.max())}
            row["norm_vs_reference_pair"] = _volume_difference(baseline_norm, norm)
        else:
            baseline_lta, baseline_norm = _lta(lta), norm
        report["groups"][name] = row
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, em_seconds, norm_seconds, flush=True)


if __name__ == "__main__":
    main()
