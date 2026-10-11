"""标准球面完整终态/全部积分轨迹同输入回归，不降低质量门。"""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(left, right):
    delta = left.astype(np.float64) - right.astype(np.float64)
    lengths = np.linalg.norm(delta, axis=1)
    return {"different_coordinate_elements": int(np.count_nonzero(delta)),
            "max_abs_coordinate_mm": float(np.abs(delta).max(initial=0)),
            "mean_distance_mm": float(lengths.mean()),
            "p99_distance_mm": float(np.percentile(lengths, 99)),
            "max_distance_mm": float(lengths.max(initial=0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace_path = args.candidate_directory / "trace.json"
    trace = json.loads(trace_path.read_text())
    if trace["status"] != "complete":
        raise ValueError("complete stage output required; a prefix is not a terminal result")
    candidate_path = args.candidate_directory / args.reference.name
    candidate, faces = fsio.read_geometry(candidate_path)
    reference, reference_faces = fsio.read_geometry(args.reference)
    if candidate.shape != reference.shape or not np.array_equal(faces, reference_faces):
        raise ValueError("ordered vertex/face correspondence does not hold")
    rows = trace["trace"]
    comparisons = [row.get("diagnostic_native_after") for row in rows]
    compared = [row for row in comparisons if row is not None and row.get("status") == "compared"]
    report = {"status": "complete", "scope": "frozen same-input full standard sphere stage, not raw T1",
              "script_sha256": sha(Path(__file__)), "candidate_trace_sha256": sha(trace_path),
              "candidate_surface_sha256": sha(candidate_path), "reference_surface_sha256": sha(args.reference),
              "decoded_final": compare(candidate, reference), "ordered_faces_equal": True,
              "integral_updates": len(rows), "reference_checkpoint_comparisons": len(compared),
              "all_compared_after_coordinates_exact": len(compared) == len(rows) and all(row["different_coordinates"] == 0 for row in compared),
              "candidate_execution_complete": True,
              "candidate_quality": trace["api"].get("overlap_diagnostics"),
              "candidate_api_seconds_including_observer_io": trace["api"]["total_seconds_including_io"],
              "candidate_total_observer_wall_seconds": trace["total_trace_wall_seconds"],
              "candidate_torch_target_memory": trace.get("torch_target_memory"),
              "allocator_environment": trace.get("allocator_environment"),
              "precision": trace["precision"], "kwargs": trace["kwargs"],
              "threads": trace["threads"], "affinity": trace["affinity"],
              "source_files_sha256": trace["source_files_sha256"], "inputs_sha256": trace["inputs_sha256"]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["decoded_final", "integral_updates", "all_compared_after_coordinates_exact", "candidate_quality", "candidate_api_seconds_including_observer_io", "candidate_torch_target_memory"]}, indent=2))


if __name__ == "__main__":
    main()
