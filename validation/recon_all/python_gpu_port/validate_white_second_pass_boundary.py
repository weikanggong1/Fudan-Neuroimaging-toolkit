"""Validate white pass-two rip and target preparation on a real frozen T1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from compare_white_second_pass_boundary import VERTEX_DTYPE
from fnit.recon_all.place_surface_border import compute_border_values_first_pass
from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
from fnit.recon_all.place_surface_rip import rip_white_preaparc_pass
from fnit.recon_all.place_surface_smoothing import average_marked_values
from fnit.recon_all.place_surface_volume import prepare_placement_volume


def _field_error(candidate: np.ndarray, reference: np.ndarray) -> dict:
    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return {
        "exact_components": int(np.count_nonzero(candidate == reference)),
        "components": int(candidate.size),
        "max_abs": float(difference.max()),
        "p99_abs": float(np.percentile(difference, 99)),
        "count_gt_1e4": int(np.count_nonzero(difference > 1e-4)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--first-raw", type=Path, required=True)
    parser.add_argument("--second-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    first = np.frombuffer(args.first_raw.read_bytes(), dtype=VERTEX_DTYPE)
    second = np.frombuffer(args.second_raw.read_bytes(), dtype=VERTEX_DTYPE)
    vertices, faces, metadata = nib.freesurfer.read_geometry(
        str(args.subject / "surf/lh.orig"), read_metadata=True)
    assert len(first) == len(second) == len(vertices) == 106622
    brain_image = nib.load(str(args.subject / "mri/brain.finalsurfs.mgz"))
    seg_image = nib.load(str(args.subject / "mri/aseg.presurf.mgz"))
    wm_image = nib.load(str(args.subject / "mri/wm.mgz"))
    stats = dict(line.split()[:2] for line in
                 (args.subject / "surf/autodet.gw.stats.lh.dat").read_text().splitlines()
                 if len(line.split()) >= 2)
    volume, _ = prepare_placement_volume(
        np.asarray(brain_image.dataobj), np.asarray(wm_image.dataobj),
        surface="white", mid_gray=float(stats["MID_GRAY"]),
    )
    seg = np.asarray(seg_image.dataobj)
    prepared_at = time.perf_counter()
    ripped, values = rip_white_preaparc_pass(
        first["xyz"], first["normal"], faces, seg, volume,
        surface_ras_to_voxel(seg_image.header, metadata),
        hemisphere="lh", ripped=(first["rip"] != 0), values=first["val"],
    )
    ripped_at = time.perf_counter()
    thresholds = np.array([float(stats[f"white_{name}"]) for name in
                           ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")])
    border = compute_border_values_first_pass(
        volume, seg, first["xyz"], first["normal"], first["orig"],
        ripped, values, surface_ras_to_voxel(brain_image.header, metadata),
        thresholds, hemisphere="lh", surface="white", sigma=1.0,
    )
    border_at = time.perf_counter()
    averaged = average_marked_values(border[0], border[4], ripped, faces, 5)
    finish = time.perf_counter()
    arrays_path = args.out.with_suffix(".npz")
    np.savez_compressed(arrays_path, ripped=ripped, values=averaged,
                        distances=border[1], mean=border[2], target=border[3],
                        marked=border[4], sigma=border[5])
    report = {
        "subject": str(args.subject),
        "candidate_arrays": str(arrays_path),
        "candidate_arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.first_raw, args.second_raw,
                                      args.subject / "surf/lh.orig",
                                      args.subject / "surf/autodet.gw.stats.lh.dat",
                                      args.subject / "mri/brain.finalsurfs.mgz",
                                      args.subject / "mri/wm.mgz",
                                      args.subject / "mri/aseg.presurf.mgz")},
        "ripped_count": int(np.count_nonzero(ripped)),
        "reference_ripped_count": int(np.count_nonzero(second["rip"])),
        "ripped_mismatch": int(np.count_nonzero(ripped != (second["rip"] != 0))),
        "ripped_mismatch_vertices": np.flatnonzero(ripped != (second["rip"] != 0)).astype(int).tolist(),
        "marked_mismatch": int(np.count_nonzero(border[4] != second["marked"])),
        "fields": {
            "value_after_average": _field_error(averaged, second["val"]),
            "distance": _field_error(border[1], second["d"]),
            "gradient_magnitude": _field_error(border[2], second["mean"]),
            "target": _field_error(border[3], second["target"]),
            "sigma": _field_error(border[5], second["val2"]),
        },
        "active_fields": {
            "gradient_magnitude": _field_error(border[2][ripped == 0], second["mean"][ripped == 0]),
            "sigma": _field_error(border[5][ripped == 0], second["val2"][ripped == 0]),
        },
        "seconds": {"prepare": prepared_at - started, "rip": ripped_at - prepared_at,
                    "border": border_at - ripped_at, "average": finish - border_at,
                    "total": finish - started},
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("ripped_mismatch_vertices", "input_sha256")}, indent=2))


if __name__ == "__main__":
    main()
