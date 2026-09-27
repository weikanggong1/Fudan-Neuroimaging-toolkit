"""Paired real-T1 probe for the connected MNI152 and auxiliary stages."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import surfa as sf
import torch

from fnit.recon_all.aux_seg import mri_mcadura_seg, mri_vsinus_seg
from fnit.recon_all.finalsurfs_python import run_finalsurfs
from fnit.recon_all.mni_aux_chain import register_mni152_affine


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    first = next(i + 1 for i, line in enumerate(lines) if line.strip() == "1 4 4")
    return np.asarray([[float(v) for v in line.split()]
                       for line in lines[first:first + 4]])


def paired_volume(candidate: Path, official: Path, *, label_counts: bool = False) -> dict:
    a, b = nib.load(str(candidate)), nib.load(str(official))
    av, bv = np.asarray(a.dataobj), np.asarray(b.dataobj)
    result = {
        "candidate_sha256": sha256(candidate),
        "official_sha256": sha256(official),
        "shape": list(av.shape),
        "mismatched_voxels": int(np.count_nonzero(av != bv)),
        "dtype_candidate": str(a.get_data_dtype()),
        "dtype_official": str(b.get_data_dtype()),
        "affine_max_abs": float(np.max(np.abs(a.affine - b.affine))),
    }
    if label_counts:
        labels = sorted(set(np.unique(av).tolist()) | set(np.unique(bv).tolist()))
        result["labels"] = {
            str(int(label)): {
                "candidate": int(np.count_nonzero(av == label)),
                "official": int(np.count_nonzero(bv == label)),
                "intersection": int(np.count_nonzero((av == label) & (bv == label))),
            } for label in labels if label != 0
        }
    with gzip.open(candidate, "rb") as x, gzip.open(official, "rb") as y:
        result["mgh_header_284_equal"] = x.read(284) == y.read(284)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("reference_crop", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    subject, official = args.subject, args.official
    report = {"device": args.device,
              "input_sha256": {name: sha256(subject / "mri" / name)
                               for name in ("orig.mgz", "nu.mgz", "synthseg.rca.mgz")},
              "timings_seconds": {}}
    start = time.perf_counter()
    lta = register_mni152_affine(subject, args.weights, args.assets, device=args.device)
    report["timings_seconds"]["mni152_affine"] = time.perf_counter() - start
    reference_lta = official / "mri/transforms/synthmorph.1.0mm.1.0mm/reg.targ_to_invol.lta"
    reference_matrix, candidate_matrix = matrix(reference_lta), matrix(lta)
    corners = np.array([[x, y, z, 1] for x in (0, 96, 192)
                        for y in (0, 114, 228) for z in (0, 96, 192)]).T
    report["lta"] = {
        "matrix_max_abs_voxel": float(np.max(np.abs(candidate_matrix - reference_matrix))),
        "mapped_27_point_mean_mm": float(np.mean(np.linalg.norm(
            (candidate_matrix @ corners - reference_matrix @ corners)[:3], axis=0))),
        "mapped_27_point_max_mm": float(np.max(np.linalg.norm(
            (candidate_matrix @ corners - reference_matrix @ corners)[:3], axis=0))),
        "affine_world_max_abs": float(np.max(np.abs(
            sf.load_affine(str(lta.parent / "aff.lta")).convert(space="world").matrix -
            sf.load_affine(str(reference_lta.parent / "aff.lta")).convert(space="world").matrix))),
        "candidate_sha256": sha256(lta),
        "official_sha256": sha256(reference_lta),
    }
    crop = nib.load(str(lta.parent / "invol.crop.nii.gz"))
    reference_crop = nib.load(str(args.reference_crop))
    report["crop"] = {
        "shape": list(crop.shape),
        "mismatched_voxels": int(np.count_nonzero(
            np.asarray(crop.dataobj) != np.asarray(reference_crop.dataobj))),
        "affine_max_abs": float(np.max(np.abs(crop.affine - reference_crop.affine))),
    }
    start = time.perf_counter()
    mca = mri_mcadura_seg(subject / "mri/nu.mgz", subject / "mri/mca-dura.mgz",
                          lta.parent, args.assets, device=args.device, weights_dir=args.weights)
    report["timings_seconds"]["mca_dura"] = time.perf_counter() - start
    report["mca_dura"] = paired_volume(mca, official / "mri/mca-dura.mgz", label_counts=True)
    start = time.perf_counter()
    vsinus = mri_vsinus_seg(subject / "mri/nu.mgz", subject / "mri/vsinus.mgz",
                            lta.parent, args.assets,
                            ctxseg_path=subject / "mri/synthseg.rca.mgz",
                            stats_path=subject / "stats/vsinus.stats",
                            talairach_lta=subject / "mri/transforms/talairach.xfm.lta",
                            device=args.device, weights_dir=args.weights)
    report["timings_seconds"]["vsinus"] = time.perf_counter() - start
    report["vsinus"] = paired_volume(vsinus, official / "mri/vsinus.mgz", label_counts=True)
    start = time.perf_counter()
    final = run_finalsurfs(subject, device=args.device)
    report["timings_seconds"]["finalsurfs"] = time.perf_counter() - start
    report["finalsurfs"] = paired_volume(final, official / "mri/brain.finalsurfs.mgz")
    report["timings_seconds"]["sum_stages"] = sum(report["timings_seconds"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key]["mismatched_voxels"]
                      for key in ("mca_dura", "vsinus", "finalsurfs")}, indent=2))


if __name__ == "__main__":
    main()
