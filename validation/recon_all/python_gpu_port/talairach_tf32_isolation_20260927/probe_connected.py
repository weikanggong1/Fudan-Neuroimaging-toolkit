"""Replay T1 import through brainmask with local FP32 SynthMorph inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import surfa as sf
import torch

from fnit.recon_all.input_talairach_chain import run_input_talairach_chain
from fnit.recon_all.mri_mask_gpu import mask_volume
from fnit.recon_all.n4_wrapper import make_nu
from fnit.recon_all.normalization import normalize_t1
from fnit.recon_all.sclimbic import _etiv_from_lta
from fnit.synthmorph import SynthMorph


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("t1", "subject_dir", "weights_dir", "assets_dir",
                 "saved_nu0", "official_subject"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.subject_dir.exists() and any(args.subject_dir.iterdir()):
        raise ValueError("subject_dir must be empty")
    torch.set_num_threads(args.threads)
    original_call = SynthMorph.__call__
    applied = []

    def local_fp32(self, *positional, **keyword):
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        applied.append((torch.backends.cuda.matmul.allow_tf32,
                        torch.backends.cudnn.allow_tf32))
        try:
            return original_call(self, *positional, **keyword)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = before[0]
            torch.backends.cudnn.allow_tf32 = before[1]

    SynthMorph.__call__ = local_fp32
    try:
        start = time.perf_counter()
        input_report = run_input_talairach_chain(
            args.t1, args.subject_dir, args.weights_dir, args.assets_dir,
            device=args.device, threads=args.threads)
        input_seconds = time.perf_counter() - start
    finally:
        SynthMorph.__call__ = original_call
    mri = args.subject_dir / "mri"
    start = time.perf_counter()
    make_nu(mri / "orig.mgz", args.saved_nu0,
            mri / "transforms/talairach.xfm", mri / "nu.mgz")
    nu_seconds = time.perf_counter() - start
    start = time.perf_counter()
    normalize_t1(mri / "nu.mgz", mri / "transforms/talairach.xfm",
                 mri / "T1.mgz", device=args.device)
    t1_seconds = time.perf_counter() - start
    start = time.perf_counter()
    mask_volume(mri / "T1.mgz", mri / "synthstrip.mgz",
                mri / "brainmask.mgz", device=args.device)
    mask_seconds = time.perf_counter() - start
    official_mri = args.official_subject / "mri"
    volumes = {}
    for name in ("orig", "synthstrip", "nu", "T1", "brainmask"):
        a, b = (nib.load(str(root / f"{name}.mgz")) for root in (mri, official_mri))
        x, y = np.asarray(a.dataobj), np.asarray(b.dataobj)
        volumes[name] = {
            "different_voxels": int(np.count_nonzero(x != y)),
            "total_voxels": int(x.size),
            "dtype_equal": x.dtype == y.dtype,
            "affine_exact": bool(np.array_equal(a.affine, b.affine)),
            "header_284_exact": a.header.binaryblock == b.header.binaryblock,
            "candidate_sha256": sha256(mri / f"{name}.mgz"),
        }
    transforms = {}
    for name, file in (("aff", "transforms/synthmorph.mni305/aff.lta"),
                       ("voxel", "transforms/talairach.xfm.lta")):
        candidate, official = (sf.load_affine(str(root / file))
                               for root in (mri, official_mri))
        transforms[name] = {
            "matrix_max_abs_vs_official": float(np.max(np.abs(candidate.matrix - official.matrix))),
            "candidate_sha256": sha256(mri / file),
        }
    transforms["voxel"]["etiv_candidate_mm3"] = _etiv_from_lta(
        mri / "transforms/talairach.xfm.lta")
    transforms["voxel"]["etiv_official_mm3"] = _etiv_from_lta(
        official_mri / "transforms/talairach.xfm.lta")
    report = {
        "status": "complete",
        "scope": "T1 through brainmask; saved candidate N4 nu0 reused; no GCA or later stages",
        "input_sha256": {
            "t1": sha256(args.t1), "saved_candidate_nu0": sha256(args.saved_nu0),
            "synthstrip_weight": sha256(args.weights_dir / "synthstrip.1.pt"),
            "synthmorph_affine_weight": sha256(args.weights_dir / "synthmorph.affine.2.h5"),
            "mni305_template": sha256(args.assets_dir / "average/mni305.cor.stripped.mgz"),
        },
        "local_synthmorph_tf32_flags": applied,
        "device": args.device,
        "threads": args.threads,
        "seconds": {
            "input_talairach": input_seconds,
            "synthstrip": input_report["synthstrip_seconds"],
            "talairach": input_report["talairach_seconds"],
            "nu_from_saved_nu0": nu_seconds,
            "T1_normalize": t1_seconds,
            "brainmask": mask_seconds,
        },
        "volumes": volumes,
        "transforms": transforms,
    }
    (args.subject_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
