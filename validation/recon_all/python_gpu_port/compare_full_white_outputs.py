"""完整white.preaparc报告的同输入网格/体积对照；不执行或修补候选流程。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(first, second):
    a, af = fs.read_geometry(first)
    b, bf = fs.read_geometry(second)
    ordered = a.shape == b.shape and np.array_equal(af, bf)
    result = {"same_vertex_count_and_ordered_faces": ordered,
              "first_sha256": sha(first), "second_sha256": sha(second)}
    if ordered:
        d = np.linalg.norm(a-b, axis=1)
        result.update(different_coordinate_elements=int(np.count_nonzero(a != b)),
            different_vertices=int(np.count_nonzero(d)), mean_distance_mm=float(d.mean()),
            p99_distance_mm=float(np.percentile(d, 99)), maximum_distance_mm=float(d.max()),
            vertices_over_0_1_mm=int(np.count_nonzero(d > .1)))
    else:
        result["same_index_comparison"] = "not_applicable; shape or ordered faces differ"
    return result


def volume(first, second):
    ai, bi = nib.load(first), nib.load(second)
    a, b = np.asarray(ai.dataobj), np.asarray(bi.dataobj)
    same_grid = a.shape == b.shape and np.array_equal(ai.affine, bi.affine)
    result = {"same_shape_and_affine": same_grid, "same_dtype": a.dtype == b.dtype,
        "first_shape": list(a.shape), "second_shape": list(b.shape),
        "first_dtype": str(a.dtype), "second_dtype": str(b.dtype),
        "first_sha256": sha(first), "second_sha256": sha(second)}
    if same_grid:
        delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
        result.update(different_voxels=int(np.count_nonzero(delta)), maximum_absolute_error=float(delta.max()),
                      p99_absolute_error=float(np.percentile(delta, 99)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--native-reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = json.loads(args.control_report.read_text())
    native = json.loads(args.native_reference_report.read_text())
    if control["status"] != "complete" or native["status"] != "complete":
        raise ValueError("complete reports required; running or failed results are separate evidence")
    candidate = json.loads(args.candidate_report.read_text()) if args.candidate_report else None
    if candidate is not None and candidate["status"] != "complete":
        raise ValueError("candidate has not completed")
    if native["input_sha256"] != control["input_sha256"] or (
        candidate is not None and candidate["input_sha256"] != control["input_sha256"]):
        raise ValueError("reports must bind exactly the same five stage input hashes")
    a = control["python_runs"]["cpu"]
    report = {"scope": "frozen_same_input_complete_white_preaparc_only_not_final_white_or_recon_all",
        "control_report_sha256": sha(args.control_report), "native_reference_report_sha256": sha(args.native_reference_report),
        "candidate_report_sha256": sha(args.candidate_report) if args.candidate_report else None,
        "script_sha256": sha(__file__), "input_sha256": control["input_sha256"],
        "control_wall_seconds": a["wall_seconds"], "native_comparisons": {},
        "overall_metric_equivalence": "not_assessed", "strict_official_reproduction": "not_run_when_only_Conda_reference_is_present"}
    if candidate is not None:
        b = candidate["python_runs"]["torch"]
        report.update(candidate_wall_seconds=b["wall_seconds"], same_pass_trial_coordinate_trace=a["trace"] == b["trace"],
            backend_geometry=geometry(a["stage"]["output"], b["stage"]["output"]),
            backend_volume=volume(a["stage"]["output_volume"], b["stage"]["output_volume"]),
            control_stage_seconds=a["stage"]["stage_seconds"], candidate_stage_seconds=b["stage"]["stage_seconds"],
            speed_ratio_observation=a["wall_seconds"]/b["wall_seconds"],
            wall_reduction_percent_observation=100*(1-b["wall_seconds"]/a["wall_seconds"]))
    for name, group in native["native_runs"].items():
        rows = []
        for run in group["runs"]:
            command = run["command"]
            native_volume = command[command.index("--outvol")+1]
            row = {"native_binary_sha256": group["binary_sha256"], "native_wall_seconds": run["wall_seconds"],
                   "control_surface": geometry(a["stage"]["output"], run["output"]),
                   "control_volume": volume(a["stage"]["output_volume"], native_volume)}
            if candidate is not None:
                row.update(candidate_surface=geometry(b["stage"]["output"], run["output"]),
                           candidate_volume=volume(b["stage"]["output_volume"], native_volume))
            rows.append(row)
        report["native_comparisons"][name] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
