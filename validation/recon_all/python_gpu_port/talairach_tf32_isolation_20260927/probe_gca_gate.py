"""Continue a saved candidate Talairach prefix through GCA, norm and aseg.presurf."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import surfa as sf

from fnit.recon_all.ca_normalize_python import run_ca_normalize
from fnit.recon_all.native_free import _run_native_em_register, _segment_callosum


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def volume_check(candidate: Path, official: Path) -> dict:
    a, b = nib.load(str(candidate)), nib.load(str(official))
    x, y = np.asarray(a.dataobj), np.asarray(b.dataobj)
    return {
        "different_voxels": int(np.count_nonzero(x != y)),
        "total_voxels": int(x.size),
        "dtype_equal": x.dtype == y.dtype,
        "affine_exact": bool(np.array_equal(a.affine, b.affine)),
        "header_284_exact": a.header.binaryblock == b.header.binaryblock,
        "candidate_sha256": sha256(candidate),
    }


def affine_check(candidate: Path, official: Path) -> dict:
    a, b = sf.load_affine(str(candidate)), sf.load_affine(str(official))
    return {
        "matrix_max_abs_vs_official": float(np.max(np.abs(a.matrix - b.matrix))),
        "matrix_exact": bool(np.array_equal(a.matrix, b.matrix)),
        "candidate_sha256": sha256(candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate_subject", "saved_candidate_synthseg", "official_subject",
                 "assets_dir", "mri_em_register_binary"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    candidate = args.candidate_subject.resolve()
    official = args.official_subject.resolve()
    if candidate == official or official in candidate.parents:
        raise ValueError("candidate_subject must not be inside official_subject")
    saved_seg = args.saved_candidate_synthseg.resolve()
    if official == saved_seg or official in saved_seg.parents:
        raise ValueError("saved SynthSeg input must not come from official subject")
    mri, reference = candidate / "mri", official / "mri"
    if not all((mri / f"{name}.mgz").is_file() for name in ("nu", "brainmask")):
        raise FileNotFoundError("candidate nu/brainmask prefix is incomplete")
    if any((mri / name).exists() for name in
           ("norm.mgz", "aseg.presurf.mgz", "transforms/talairach.lta")):
        raise ValueError("candidate downstream outputs already exist")
    atlas = args.assets_dir / "average/RB_all_2020-01-02.gca"
    output = candidate / "gca_gate_report.json"
    report = {
        "status": "running",
        "scope": "saved candidate FP32 Talairach prefix through GCA/norm/aseg.presurf; no WM or surfaces",
        "input_sha256": {
            "candidate_nu": sha256(mri / "nu.mgz"),
            "candidate_brainmask": sha256(mri / "brainmask.mgz"),
            "saved_candidate_synthseg": sha256(saved_seg),
            "gca_atlas": sha256(atlas),
            "mri_em_register_binary": sha256(args.mri_em_register_binary),
        },
        "volumes": {}, "transforms": {}, "seconds": {},
    }

    def save() -> None:
        output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        tick = time.perf_counter()
        _run_native_em_register(args.mri_em_register_binary, mri, atlas,
                                args.assets_dir)
        report["seconds"]["mri_em_register"] = time.perf_counter() - tick
        report["transforms"]["talairach_lta"] = affine_check(
            mri / "transforms/talairach.lta",
            reference / "transforms/talairach.lta")
        save()

        tick = time.perf_counter()
        run_ca_normalize(mri / "nu.mgz", mri / "brainmask.mgz", atlas,
                         mri / "transforms/talairach.lta",
                         mri / "norm.mgz", mri / "ctrl_pts.mgz")
        report["seconds"]["mri_ca_normalize"] = time.perf_counter() - tick
        for name in ("norm", "ctrl_pts"):
            report["volumes"][name] = volume_check(
                mri / f"{name}.mgz", reference / f"{name}.mgz")
        save()

        shutil.copyfile(saved_seg, mri / "synthseg.rca.mgz")
        report["volumes"]["synthseg.rca"] = volume_check(
            mri / "synthseg.rca.mgz", reference / "synthseg.rca.mgz")
        tick = time.perf_counter()
        _segment_callosum(mri)
        report["seconds"]["mri_cc_and_copies"] = time.perf_counter() - tick
        for name in ("aseg.auto_noCCseg", "aseg.auto", "aseg.presurf"):
            report["volumes"][name] = volume_check(
                mri / f"{name}.mgz", reference / f"{name}.mgz")
        report["transforms"]["cc_up"] = affine_check(
            mri / "transforms/cc_up.lta", reference / "transforms/cc_up.lta")
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        save()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
