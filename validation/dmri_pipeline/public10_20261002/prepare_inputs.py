"""Concatenate the three acquired shells without spatial resampling or cropping."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(values, image, path):
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image(values, image.affine, header), path)


def prepare(manifest, source_root, output_dir, index):
    subject = manifest["subjects"][index - 1]
    selected = [r for r in manifest["files"] if r["path"].split("/")[0] == subject]
    if len(selected) != 20:
        raise ValueError("incomplete frozen subject")
    for row in selected:
        path = source_root / row["path"]
        if not path.is_file() or path.stat().st_size != row["size"]:
            raise ValueError("missing or incomplete downloaded file")
        if row.get("content_digest_algorithm"):
            h = hashlib.new(row["content_digest_algorithm"])
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != row["content_digest"]:
                raise ValueError("upstream image digest mismatch")
        elif digest(path) != row["sha256"]:
            raise ValueError("pinned metadata digest mismatch")
    target = output_dir / f"case{index:02d}"
    if target.exists():
        raise FileExistsError(target)
    raw = target / "raw"
    raw.mkdir(parents=True, mode=0o700)
    images, arrays, bvals, bvecs, metadata = [], [], [], [], []
    dwi_root = source_root / subject / "ses-1" / "dwi"
    for shell in (1, 2, 3):
        stem = dwi_root / f"{subject}_ses-1_acq-shell{shell}_dir-AP_run-1_dwi"
        image = nib.load(str(stem) + ".nii.gz")
        values = np.asarray(image.dataobj, dtype=np.float32)
        b = np.loadtxt(str(stem) + ".bval").reshape(-1)
        g = np.loadtxt(str(stem) + ".bvec")
        if values.ndim != 4 or values.shape[3] != b.size or g.shape != (3, b.size):
            raise ValueError("DWI and gradients have inconsistent dimensions")
        if not np.isfinite(values).all() or not np.isfinite(g).all():
            raise ValueError("non-finite acquired values")
        if images and (image.shape[:3] != images[0].shape[:3]
                       or not np.allclose(image.affine, images[0].affine, atol=5e-4, rtol=0)):
            raise ValueError("shell spatial grids differ")
        images.append(image); arrays.append(values); bvals.append(b); bvecs.append(g)
        metadata.append(json.loads(Path(str(stem) + ".json").read_text()))
    for key in ("PhaseEncodingDirection", "TotalReadoutTime"):
        if len({m[key] for m in metadata}) != 1:
            raise ValueError("shell acquisition parameters differ")
    values = np.concatenate(arrays, axis=3)
    b = np.concatenate(bvals); g = np.concatenate(bvecs, axis=1)
    save(values, images[0], raw / "AP.nii.gz")
    np.savetxt(raw / "AP.bval", b[None], fmt="%.17g")
    np.savetxt(raw / "AP.bvec", g, fmt="%.17g")
    echo_times = [float(m["EchoTime"]) for m in metadata]
    (raw / "AP.json").write_text(json.dumps({
        "PhaseEncodingDirection": metadata[0]["PhaseEncodingDirection"],
        "TotalReadoutTime": metadata[0]["TotalReadoutTime"],
        "MergedShellEchoTimes": echo_times, "MergedShellVolumeCounts": [int(x.size) for x in bvals],
        "Preparation": "concatenate acquired float32-decoded volumes; no interpolation, crop or intensity normalization",
    }, indent=2) + "\n")
    fmap = source_root / subject / "ses-1" / "fmap" / f"{subject}_ses-1_acq-shell1_dir-PA_run-1_epi"
    image = nib.load(str(fmap) + ".nii.gz")
    pa = np.asarray(image.dataobj, dtype=np.float32)
    if pa.ndim == 3:
        pa = pa[..., None]
    if pa.shape != (*images[0].shape[:3], 1) or not np.allclose(image.affine, images[0].affine, atol=5e-4, rtol=0):
        raise ValueError("reverse b0 does not share DWI grid")
    reverse_metadata = json.loads(Path(str(fmap) + ".json").read_text())
    if metadata[0]["PhaseEncodingDirection"] != "j-" or reverse_metadata["PhaseEncodingDirection"] != "j":
        raise ValueError("unexpected frozen phase encoding")
    save(pa, image, raw / "PA.nii.gz")
    np.savetxt(raw / "PA.bval", np.zeros((1, 1)), fmt="%g")
    (raw / "PA.json").write_text(json.dumps({k: reverse_metadata[k] for k in
                                             ("PhaseEncodingDirection", "TotalReadoutTime", "EchoTime")}, indent=2) + "\n")
    t1 = source_root / subject / "ses-1" / "anat" / f"{subject}_ses-1_T1w.nii.gz"
    (target / "T1w.nii.gz").symlink_to(t1.resolve())
    # Verify the saved common input, including all acquired intensities and gradients.
    assert np.array_equal(np.asarray(nib.load(raw / "AP.nii.gz").dataobj, dtype=np.float32), values)
    assert np.array_equal(np.asarray(nib.load(raw / "PA.nii.gz").dataobj, dtype=np.float32), pa)
    assert np.array_equal(np.loadtxt(raw / "AP.bval").reshape(-1), b)
    assert np.array_equal(np.loadtxt(raw / "AP.bvec"), g)
    report = {"case": f"case{index:02d}", "dataset": manifest["dataset"], "version": manifest["version"],
              "source_subject": subject, "shape": list(values.shape), "echo_times": echo_times,
              "b_values_and_counts": {str(x): int(np.count_nonzero(b == x)) for x in np.unique(b)},
              "raw_files": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(raw.iterdir())},
              "t1": {"bytes": t1.stat().st_size, "sha256": digest(t1)},
              "maximum_shell_affine_difference_mm": float(max(np.max(np.abs(x.affine - images[0].affine)) for x in images)),
              "decoded_acquired_values_preserved": True, "spatial_interpolation_or_crop": False,
              "model_limit": "acquired echo time differs by shell; compare algorithms, not biological ground truth"}
    (target / "input_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (target / "input.ready").write_text("0\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-index", type=int, choices=range(1, 11), required=True)
    args = parser.parse_args()
    report = prepare(json.loads(args.manifest.read_text()), args.source_root, args.output_dir, args.case_index)
    print(json.dumps({"case": report["case"], "shape": report["shape"], "input_ready": True}), flush=True)


if __name__ == "__main__":
    main()
