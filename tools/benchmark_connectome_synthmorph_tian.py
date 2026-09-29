"""同一真实 T1、MNI 模板与 Tian S1/S4 的 FNIT/官方 SynthMorph 对照。"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from connectome_benchmark_common import _sha256
from fnit.connectome.atlas_tian import synthmorph_tian_to_t1
from fnit.weights import WEIGHT_FILES, resolve_weights, verify_file


def compare(candidate, reference):
    actual = np.asarray(candidate.dataobj)
    expected = np.asarray(reference.dataobj)
    if actual.shape != expected.shape or not np.allclose(candidate.affine, reference.affine, atol=1e-5):
        raise ValueError("candidate/reference Tian T1 grids differ")
    labels = sorted(set(np.unique(actual).tolist()) | set(np.unique(expected).tolist()))
    foreground = (actual > 0) | (expected > 0)
    dice = {}
    for label in labels:
        if label == 0:
            continue
        a, b = actual == label, expected == label
        dice[str(label)] = float(2 * (a & b).sum() / (a.sum() + b.sum())) if a.any() or b.any() else 1.
    return {
        "voxel_xor": int(np.count_nonzero(actual != expected)),
        "voxel_count": int(actual.size),
        "foreground_dice": float(2 * ((actual > 0) & (expected > 0)).sum() /
                                 ((actual > 0).sum() + (expected > 0).sum())),
        "per_label_dice": dice,
        "per_label_dice_mean": float(np.mean(list(dice.values()))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("t1-brain", "mni-template", "tian-s1", "tian-s4", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--official-s1", type=Path)
    parser.add_argument("--official-s4", type=Path)
    parser.add_argument("--official-warp", type=Path,
                        help="独立官方形变；隔离标签重采样误差")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    weight_files = {}
    for name in ("synthmorph.affine.2.h5", "synthmorph.deform.3.h5"):
        path = resolve_weights(name, explicit=args.weights)
        _, size, sha256 = WEIGHT_FILES[name]
        if not verify_file(path, size, sha256):
            raise ValueError(f"SynthMorph weight failed size/SHA-256 verification: {name}")
        weight_files[name] = {"bytes": size, "sha256": sha256}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.empty(1, device=args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    start = time.perf_counter()
    native_s1, transformation = synthmorph_tian_to_t1(
        t1_brain=args.t1_brain, mni_template=args.mni_template, tian_mni=args.tian_s1,
        device=args.device, weights=args.weights,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    registration_and_s1_seconds = time.perf_counter() - start
    start = time.perf_counter()
    native_s4, _ = synthmorph_tian_to_t1(
        t1_brain=args.t1_brain, mni_template=args.mni_template, tian_mni=args.tian_s4,
        device=args.device, transform=transformation,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    s4_apply_seconds = time.perf_counter() - start
    nib.save(native_s1, str(args.output_dir / "tian_s1_t1.nii.gz"))
    nib.save(native_s4, str(args.output_dir / "tian_s4_t1.nii.gz"))
    transformation.save(args.output_dir / "mni_to_t1_fnit.nii.gz")
    report = {
        "input_sha256": {name: _sha256(path) for name, path in (
            ("t1_brain", args.t1_brain), ("mni_template", args.mni_template),
            ("tian_s1", args.tian_s1), ("tian_s4", args.tian_s4))},
        "verified_weight_files": weight_files,
        "model": "FNIT PyTorch SynthMorph joint, MNI moving to native T1 fixed",
        "device": args.device,
        "registration_and_s1_apply_seconds": registration_and_s1_seconds,
        "s4_reuse_transform_apply_seconds": s4_apply_seconds,
        "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated(args.device) / 2**30
                                     if args.device.startswith("cuda") else None),
        "atlases": {},
    }
    if args.official_warp:
        official_warp = nib.load(str(args.official_warp))
        official_field = np.asarray(official_warp.dataobj)
        if official_field.ndim == 5 and official_field.shape[3] == 1:
            official_field = official_field[:, :, :, 0, :]
        candidate_field = np.asarray(transformation.dataobj)
        if candidate_field.shape != official_field.shape or not np.allclose(
            transformation.affine, official_warp.affine, atol=1e-5
        ):
            raise ValueError("FNIT and official SynthMorph warp grids differ")
        displacement_error = np.linalg.norm(
            candidate_field.astype(np.float64) - official_field.astype(np.float64), axis=-1
        )
        report["warp_error_mm"] = {
            "mean": float(displacement_error.mean()),
            "p95": float(np.percentile(displacement_error, 95)),
            "max": float(displacement_error.max()),
        }
    for name, candidate, reference_path in (
        ("s1", native_s1, args.official_s1), ("s4", native_s4, args.official_s4),
    ):
        report["atlases"][name] = {
            "output_shape": candidate.shape,
            "output_labels": np.unique(np.asarray(candidate.dataobj)).astype(int).tolist(),
            "reference": compare(candidate, nib.load(str(reference_path))) if reference_path else None,
        }
        if args.official_warp and reference_path:
            from fnit.synthmorph import apply_transform
            from fnit._transforms import DenseWarp
            warp_image = nib.load(str(args.official_warp))
            warp_data = np.asarray(warp_image.dataobj)
            if warp_data.ndim == 5 and warp_data.shape[3] == 1:
                warp_data = warp_data[:, :, :, 0, :]
            atlas_path = {"s1": args.tian_s1, "s4": args.tian_s4}[name]
            fixed_transform = DenseWarp(
                warp_data, source=nib.load(str(atlas_path)), target=warp_image,
            )
            start = time.perf_counter()
            fixed_warp = apply_transform(
                image=atlas_path, transformation=fixed_transform,
                method="nearest", dtype="int16",
            )
            report["atlases"][name]["fixed_official_warp"] = {
                **compare(fixed_warp, nib.load(str(reference_path))),
                "fnit_apply_seconds": time.perf_counter() - start,
            }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
