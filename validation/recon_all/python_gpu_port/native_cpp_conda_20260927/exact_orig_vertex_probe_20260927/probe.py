"""Compare FNIT white/pial/vertex maps on same-index, exact official orig meshes.

Validation only: `--orig-dir` supplies independently reconstructed FNIT orig
surfaces; `--prefix-subject` supplies the same-T1 MRI prefix. No official
surface is used as a candidate input. All outputs go into --output-subject.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import nibabel as nib
from nibabel.freesurfer import io as fs
import numpy as np

from fnit.recon_all.label_cortex_python import label_cortex
from fnit.recon_all.native_free import (_pial_from_cortex, _replace_vertices,
                                         _run_native_surface_metrics)
from fnit.recon_all.smooth_surface_python import smooth_surface
from fnit.recon_all.surface_area_gpu import mid_area_map
from fnit.recon_all.surface_roi_gpu import vertex_volume_map


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def errors(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        return {"same_shape": False, "reference_shape": list(reference.shape),
                "candidate_shape": list(candidate.shape)}
    distance = np.abs(np.asarray(reference, np.float64) - np.asarray(candidate, np.float64))
    return {"same_shape": True, "elements": int(reference.size),
            "exact": int(np.count_nonzero(reference == candidate)),
            "mean_abs": float(distance.mean()),
            "p50_abs": float(np.quantile(distance, .5)),
            "p95_abs": float(np.quantile(distance, .95)),
            "p99_abs": float(np.quantile(distance, .99)),
            "max_abs": float(distance.max(initial=0)),
            "reference_mean": float(np.mean(reference)),
            "candidate_mean": float(np.mean(candidate))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig-dir", type=Path, required=True)
    parser.add_argument("--prefix-subject", type=Path, required=True)
    parser.add_argument("--official-subject", type=Path, required=True)
    parser.add_argument("--output-subject", type=Path, required=True)
    parser.add_argument("--native-bin", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    orig_dir, prefix, official, subject = (p.resolve() for p in
                                            (args.orig_dir, args.prefix_subject,
                                             args.official_subject, args.output_subject))
    if subject.exists():
        raise FileExistsError(subject)
    subject.mkdir(parents=True)
    (subject / "mri").symlink_to(prefix / "mri", target_is_directory=True)
    for name in ("surf", "label", "scripts"):
        (subject / name).mkdir()
    surf, labels = subject / "surf", subject / "label"
    aseg_image = nib.load(str(prefix / "mri/aseg.mgz"))
    aseg = np.asarray(nib.load(str(prefix / "mri/aseg.auto.mgz")).dataobj).astype(np.int16)
    report = {"status": "running", "real_t1_sha256":
              "f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a",
              "aseg_sha256": sha256(prefix / "mri/aseg.mgz"), "hemispheres": {}}
    report_path = subject / "exact-orig-vertex-probe.json"
    for hemi in ("lh", "rh"):
        started = time.perf_counter()
        orig = orig_dir / f"{hemi}.orig.fnit_sqrt"
        shutil.copyfile(orig, surf / f"{hemi}.orig")
        reference_orig, reference_faces = fs.read_geometry(str(official / "surf" / f"{hemi}.orig"))
        candidate_orig, candidate_faces = fs.read_geometry(str(surf / f"{hemi}.orig"))
        if not (np.array_equal(reference_orig, candidate_orig)
                and np.array_equal(reference_faces, candidate_faces)):
            raise ValueError(f"{hemi} orig is not same-index exact")
        smooth_surface(surf / f"{hemi}.orig", surf / f"{hemi}.smoothwm", device="cpu")
        white = surf / f"{hemi}.white"
        shutil.copyfile(surf / f"{hemi}.smoothwm", white)
        shutil.copyfile(white, surf / f"{hemi}.white.preaparc")
        white_xyz, faces = fs.read_geometry(str(white))
        pial_xyz, _ = _pial_from_cortex(white_xyz, faces, aseg_image, aseg, hemi)
        pial = surf / f"{hemi}.pial"
        _replace_vertices(white, pial, pial_xyz)
        shutil.copyfile(pial, surf / f"{hemi}.pial.T1")
        geometry_seconds = time.perf_counter() - started
        metric_started = time.perf_counter()
        metric_times = _run_native_surface_metrics(
            args.native_bin.resolve(), subject, hemi, args.assets.resolve())
        mid_area_map(surf / f"{hemi}.area", surf / f"{hemi}.area.pial",
                     surf / f"{hemi}.area.mid", device="cpu")
        cortex = labels / f"{hemi}.cortex.label"
        label_cortex(white, prefix / "mri/aseg.mgz", cortex)
        vertex_volume_map(white, pial, cortex, surf / f"{hemi}.volume", device="cpu")
        metric_seconds = time.perf_counter() - metric_started
        row = {"orig_sha256": sha256(orig), "vertices": len(white_xyz),
               "faces": len(faces), "orig_ordered_exact": True,
               "geometry_seconds": geometry_seconds,
               "metric_seconds": metric_seconds,
               "native_metric_seconds": metric_times, "surfaces": {}, "maps": {}}
        for name in ("smoothwm", "white", "pial"):
            a, af = fs.read_geometry(str(official / "surf" / f"{hemi}.{name}"))
            b, bf = fs.read_geometry(str(surf / f"{hemi}.{name}"))
            row["surfaces"][name] = {"ordered_faces_exact": bool(np.array_equal(af, bf)),
                                     "coordinate_mm": errors(a, b)}
        for name in ("thickness", "area", "area.pial", "area.mid",
                     "volume", "curv", "curv.pial"):
            a = fs.read_morph_data(str(official / "surf" / f"{hemi}.{name}"))
            b = fs.read_morph_data(str(surf / f"{hemi}.{name}"))
            row["maps"][name] = errors(a, b)
        report["hemispheres"][hemi] = row
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "complete"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
