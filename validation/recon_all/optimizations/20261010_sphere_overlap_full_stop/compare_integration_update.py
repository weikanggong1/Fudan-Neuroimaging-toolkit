"""比较实际同输入积分更新的原参考与FNIT状态；不向算法反馈参考。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import nibabel.freesurfer.io as fsio
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(left, right):
    if left.shape != right.shape:
        return {"shape_equal": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    error = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"shape_equal": True, "different_elements": int(np.count_nonzero(left != right)),
            "max_abs": float(error.max(initial=0)), "p99_abs": float(np.percentile(error, 99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--native-baseline", type=Path, required=True)
    parser.add_argument("--python-directory", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_directory))
    from fnit.recon_all.place_surface_normals import initial_vertex_normals
    from fnit.recon_all.sphere_standard_unfold import _face_geometry

    py_trace = json.loads((args.python_directory / "trace.json").read_text())
    before_path = args.python_directory / f"step-{args.update:04d}.before.npz"
    forces_path = args.python_directory / f"step-{args.update:04d}.forces.npz"
    before, forces = np.load(before_path), np.load(forces_path)
    coordinates, faces = before["coordinates"], before["faces"]
    norms = initial_vertex_normals(coordinates, faces)
    areas, _ = _face_geometry(coordinates, faces)
    report = {"status": "complete", "scope": "isolated actual-update observation; reference never fed back",
              "update": args.update, "script_sha256": sha(Path(__file__)),
              "python_trace_sha256": sha(args.python_directory / "trace.json"),
              "input_arrays_sha256": {before_path.name: sha(before_path), forces_path.name: sha(forces_path)},
              "state": {}, "reference_transparency": [], "search": py_trace["trace"][args.update]["search"]}
    for index in range(args.update + 2):
        name = f"lh.sphere{index:04d}"
        observed, observed_faces = fsio.read_geometry(args.native_directory / "surf" / name)
        baseline, baseline_faces = fsio.read_geometry(args.native_baseline / "surf" / name)
        report["reference_transparency"].append({"index": index, "coordinates": compare(observed, baseline),
                                                "ordered_faces_equal": bool(np.array_equal(observed_faces, baseline_faces))})
    for phase, candidate in [("distance", forces["distance"]),
                             ("area-before-average", forces["combined"]),
                             ("averaged", before["gradient"])]:
        vertex_path = args.native_directory / "observer" / (phase + ".vertices.f32")
        face_path = args.native_directory / "observer" / (phase + ".faces.f32")
        native = np.fromfile(vertex_path, np.float32).reshape(len(coordinates), 9)
        native_faces = np.fromfile(face_path, np.float32).reshape(len(faces), 2)
        report["state"][phase] = {"coordinates": compare(native[:, :3], coordinates),
                                  "normals": compare(native[:, 3:6], norms),
                                  "gradient": compare(native[:, 6:9], candidate),
                                  "areas": compare(native_faces[:, 0], areas),
                                  "vertices_sha256": sha(vertex_path), "faces_sha256": sha(face_path)}
    log = (args.native_directory / "native.log").read_text()
    report["native_search_hex"] = [line for line in log.splitlines() if line.startswith("FNIT_")]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"state": report["state"], "native_search_hex": report["native_search_hex"]}, indent=2))


if __name__ == "__main__":
    main()
