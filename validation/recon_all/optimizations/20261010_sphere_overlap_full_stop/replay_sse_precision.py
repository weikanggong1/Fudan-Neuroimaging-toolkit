"""在真实冻结一步中分离sqrt浮点重载对球面SSE/步长的影响。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import nibabel.freesurfer.io as fsio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--python-directory", type=Path, required=True)
    parser.add_argument("--smoothwm", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--native-after", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_directory))
    from fnit.recon_all.sphere_standard_line_search import first_epoch_sse, first_epoch_line_search, _distance_sse
    from fnit.recon_all.sphere_standard_unfold import _face_geometry
    trace = json.loads((args.python_directory / "trace.json").read_text())
    before = np.load(args.python_directory / f"step-{args.update:04d}.before.npz")
    forces = np.load(args.python_directory / f"step-{args.update:04d}.forces.npz")
    metric, metric_faces = fsio.read_geometry(args.smoothwm)
    faces, xyz, gradient = before["faces"], before["coordinates"], before["gradient"]
    if not np.array_equal(metric_faces, faces):
        raise ValueError("same-index diagnostic requires identical ordered faces")
    original_area = np.abs(_face_geometry(metric.astype(np.float32), faces)[0])
    original_total = np.float32(np.sum(original_area, dtype=np.float64))
    offsets, neighbors, distances = forces["offsets"], forces["neighbors"], forces["original_distances"]
    weight = trace["trace"][args.update]["integration_state"]["weight"]

    def rounded_sqrt_objective(positions):
        result = first_epoch_sse(positions, faces, offsets, neighbors, distances,
                                 original_area, original_total, weight)
        source_scale = float(np.float32(result["distance_scale"]))
        distance = _distance_sse(positions, offsets, neighbors, distances, source_scale)
        result.update(total=float(np.float32(weight))*distance+result["negative_area"],
                      weighted_distance=float(np.float32(weight))*distance,
                      distance_scale=source_scale)
        return result

    result = first_epoch_line_search(xyz, gradient, faces, offsets, neighbors,
                                     distances, original_area, original_total, weight,
                                     objective=rounded_sqrt_objective)
    report = {"status": "complete", "scope": "isolated same-input expression diagnosis, not a production patch",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "update": args.update, "old": trace["trace"][args.update]["search"],
              "distance_sqrt_float32": result}
    if args.native_after is not None:
        from fnit.recon_all.sphere_python import project_radially
        native, native_faces = fsio.read_geometry(args.native_after)
        if not np.array_equal(native_faces, faces):
            raise ValueError("native after-update ordered faces differ")
        changed = (xyz.astype(np.float64) + result["selected_dt"] * gradient.astype(np.float64)).astype(np.float32)
        changed = project_radially(changed, already_sphere=True)
        error = changed.astype(np.float64) - native
        report["after_coordinates"] = {"different_elements": int(np.count_nonzero(error)),
                                       "max_abs_mm": float(np.abs(error).max(initial=0)),
                                       "max_distance_mm": float(np.linalg.norm(error, axis=1).max(initial=0)),
                                       "native_after_sha256": hashlib.sha256(args.native_after.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("start",float(result["starting_sse"]["total"]).hex())
    for entry in result["initial_candidates"]:
        print("trial",float(entry["dt"]).hex(),float(entry["sse"]).hex())
    print("quadratic",{key:float(value).hex() for key,value in result["quadratic"].items()})
    print("selected_dt",float(result["selected_dt"]).hex())


if __name__ == "__main__":
    main()
