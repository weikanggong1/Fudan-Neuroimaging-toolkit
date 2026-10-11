"""球面首步隔离重放：参考初态仅用于诊断，不进入生产调用。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import nibabel.freesurfer.io as fsio
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def comparison(left, right):
    if left.shape != right.shape:
        return {"shape_equal": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    error = np.abs(left.astype(np.float64) - right.astype(np.float64))
    row = {"shape_equal": True, "different_elements": int(np.count_nonzero(left != right)),
           "max_abs": float(error.max(initial=0)), "p99_abs": float(np.percentile(error, 99)),
           "left_sha256": array_sha(left), "right_sha256": array_sha(right)}
    if left.ndim == 2 and left.shape[1] == 3:
        displacement = np.linalg.norm(left.astype(np.float64) - right.astype(np.float64), axis=1)
        row.update(mean_distance_mm=float(displacement.mean()), max_distance_mm=float(displacement.max()),
                   p99_distance_mm=float(np.percentile(displacement, 99)))
    return row


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--native-before", type=Path, required=True)
    parser.add_argument("--native-after", type=Path, required=True)
    parser.add_argument("--python-before", type=Path, required=True)
    parser.add_argument("--python-after", type=Path, required=True)
    parser.add_argument("--smoothwm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_directory))
    from fnit.recon_all.sphere_standard_metric import sample_standard_metric_matrix, average_standard_metric
    from fnit.recon_all.sphere_standard_nonlinear import one_ring_metric
    from fnit.recon_all.sphere_standard_unfold import first_epoch_gradient, _face_geometry
    from fnit.recon_all.sphere_standard_average import average_standard_gradient
    from fnit.recon_all.sphere_standard_line_search import first_epoch_line_search
    from fnit.recon_all.sphere_python import project_radially
    from fnit.recon_all.place_surface_normals import FaceNormalTopology, initial_vertex_normals
    import numba

    reference_before, faces = fsio.read_geometry(args.native_before)
    reference_after, after_faces = fsio.read_geometry(args.native_after)
    metric, metric_faces = fsio.read_geometry(args.smoothwm)
    python_before = np.load(args.python_before)
    python_after = np.load(args.python_after)
    if not (np.array_equal(faces, after_faces) and np.array_equal(faces, metric_faces)
            and np.array_equal(faces, python_before["faces"]) and np.array_equal(faces, python_after["faces"])):
        raise ValueError("ordered faces differ; same-index diagnosis is invalid")
    faces = faces.astype(np.int32)
    metric = metric.astype(np.float32)
    offsets, neighbors, raw, sampled = sample_standard_metric_matrix(metric, faces, seed=1234)
    distances, reciprocal_pairs, reciprocal_seconds = average_standard_metric(offsets, neighbors, raw)
    local_offsets, local_neighbors, _, _ = one_ring_metric(faces, len(metric), offsets, neighbors, distances)
    original_area, _ = _face_geometry(metric, faces)
    original_area = np.abs(original_area)
    original_total = np.float32(np.sum(original_area, dtype=np.float64))
    topology = FaceNormalTopology(faces, len(metric))
    report = {"status": "running", "scope": "isolated diagnostic native checkpoint cross-input; never producer",
              "native_snapshot_phase": "after initial projection/metric properties/starting SSE; before first gradient",
              "script_sha256": sha(Path(__file__)), "inputs_sha256": {}, "threads": numba.get_num_threads(),
              "affinity": sorted(os.sched_getaffinity(0)), "seed": 1234, "averaging_rounds": 1024,
              "distance_weight": 1e-6, "source_files_sha256": {}, "cases": [],
              "metric_sampling": sampled, "reciprocal_pairs": reciprocal_pairs,
              "reciprocal_seconds": reciprocal_seconds,
              "sampled_metric_sha256": {"offsets": array_sha(offsets), "neighbors": array_sha(neighbors),
                  "raw_distances": array_sha(raw), "symmetric_distances": array_sha(distances)},
              "initial_state_comparison": comparison(reference_before.astype(np.float32), python_before["coordinates"])}
    for name in ["native_before", "native_after", "python_before", "python_after", "smoothwm"]:
        report["inputs_sha256"][name] = sha(getattr(args, name))
    for name, before in [("python-initial-control", python_before["coordinates"]),
                         ("native-initial-diagnostic", reference_before.astype(np.float32))]:
        tick = time.perf_counter()
        area, face_normals = _face_geometry(before, faces)
        normal = initial_vertex_normals(before, faces, topology=topology)
        distance, area_gradient, gradient, info = first_epoch_gradient(
            before, faces, metric, offsets, neighbors, distances, distance_weight=1e-6,
            normal_topology=topology, original_metric=(original_area, original_total))
        averaged = average_standard_gradient(gradient, local_offsets, local_neighbors, 1024)
        search = first_epoch_line_search(before, averaged, faces, offsets, neighbors, distances,
                                         original_area, original_total, distance_weight=1e-6)
        shifted = (before.astype(np.float64) + search["selected_dt"] * averaged.astype(np.float64)).astype(np.float32)
        after = project_radially(shifted, already_sphere=True)
        np.savez(args.output / (name + ".npz"), coordinates_before=before, coordinates_after=after,
                 area=area, face_normals=face_normals, vertex_normals=normal, distance_gradient=distance,
                 area_gradient=area_gradient, gradient=gradient, averaged_gradient=averaged,
                 faces=faces)
        row = {"name": name, "seconds_including_jit": time.perf_counter() - tick,
               "gradient_report": info, "search": search,
               "compare_native_after": comparison(after, reference_after.astype(np.float32)),
               "compare_python_after": comparison(after, python_after["coordinates"]),
               "compare_recorded_python_average": comparison(averaged, python_before["gradient"])}
        report["cases"].append(row)
        (args.output / "report.json").write_text(json.dumps(serializable(report), indent=2) + "\n")
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("fnit.recon_all") and path and Path(path).suffix == ".py":
            report["source_files_sha256"][name] = sha(Path(path))
    report.update(status="complete", total_seconds_including_io=time.perf_counter() - started)
    (args.output / "report.json").write_text(json.dumps(serializable(report), indent=2) + "\n")


if __name__ == "__main__":
    main()
