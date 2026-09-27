"""Compare Python white third-pass targets with frozen installed FreeSurfer RAM."""

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
    mismatch = np.flatnonzero((candidate != reference).reshape(-1))
    return {
        "exact_components": int(candidate.size - len(mismatch)),
        "components": int(candidate.size),
        "max_abs": float(difference.max()),
        "count_gt_1e4": int(np.count_nonzero(difference > 1e-4)),
        "first_mismatch_components": mismatch[:10].astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--second-final-raw", type=Path, required=True)
    parser.add_argument("--third-boundary-raw", type=Path, required=True)
    parser.add_argument("--python-second-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    previous = np.frombuffer(args.second_final_raw.read_bytes(), dtype=VERTEX_DTYPE)
    reference = np.frombuffer(args.third_boundary_raw.read_bytes(), dtype=VERTEX_DTYPE)
    with np.load(args.python_second_npz) as prior:
        python_final = prior["step26"]
    _, faces, metadata = nib.freesurfer.read_geometry(
        str(args.subject / "surf/lh.orig"), read_metadata=True)
    assert len(previous) == len(reference) == len(python_final) == 106622
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
        previous["xyz"], previous["normal"], faces, seg, volume,
        surface_ras_to_voxel(seg_image.header, metadata),
        hemisphere="lh", ripped=(previous["rip"] != 0), values=previous["val"],
    )
    ripped_at = time.perf_counter()
    thresholds = np.array([float(stats[f"white_{name}"]) for name in
                           ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")])
    border = compute_border_values_first_pass(
        volume, seg, previous["xyz"], previous["normal"], previous["orig"],
        ripped, values, surface_ras_to_voxel(brain_image.header, metadata),
        thresholds, hemisphere="lh", surface="white", sigma=0.5,
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
        "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (args.second_final_raw, args.third_boundary_raw,
                                args.python_second_npz,
                                args.subject / "surf/lh.orig",
                                args.subject / "surf/autodet.gw.stats.lh.dat",
                                args.subject / "mri/brain.finalsurfs.mgz",
                                args.subject / "mri/wm.mgz",
                                args.subject / "mri/aseg.presurf.mgz")},
        "python_arrays": str(arrays_path),
        "python_arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        "python_second_final_xyz": _field_error(python_final, previous["xyz"]),
        "official_second_final_to_third_entry": {
            name: _field_error(previous[name], reference[name])
            for name in ("xyz", "orig", "normal")},
        "ripped_count": int(np.count_nonzero(ripped)),
        "reference_ripped_count": int(np.count_nonzero(reference["rip"])),
        "ripped_mismatch": int(np.count_nonzero(ripped != (reference["rip"] != 0))),
        "ripped_mismatch_vertices": np.flatnonzero(ripped != (reference["rip"] != 0)).astype(int).tolist(),
        "marked_mismatch": int(np.count_nonzero(border[4] != reference["marked"])),
        "fields": {
            "value_after_average": _field_error(averaged, reference["val"]),
            "distance": _field_error(border[1], reference["d"]),
            "gradient_magnitude": _field_error(border[2], reference["mean"]),
            "target": _field_error(border[3], reference["target"]),
            "sigma": _field_error(border[5], reference["val2"]),
        },
        "active_fields": {
            "gradient_magnitude": _field_error(border[2][ripped == 0], reference["mean"][ripped == 0]),
            "sigma": _field_error(border[5][ripped == 0], reference["val2"][ripped == 0]),
        },
        "seconds": {"prepare": prepared_at - started, "rip": ripped_at - prepared_at,
                    "border": border_at - ripped_at, "average": finish - border_at,
                    "total": finish - started},
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("inputs", "ripped_mismatch_vertices")}, indent=2))


if __name__ == "__main__":
    main()
