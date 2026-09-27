"""Summarize one real-T1 FNIT v3 run against its archived FreeSurfer subject."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import nibabel.freesurfer.io as fs
import numpy as np
from scipy.spatial import cKDTree


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def process_wall_seconds(path: Path) -> float:
    line = next(line for line in path.read_text().splitlines()
                if "Elapsed (wall clock) time" in line)
    fields = [float(value) for value in line.rsplit(": ", 1)[1].split(":")]
    return sum(value * (60 ** power)
               for power, value in enumerate(reversed(fields)))


def measure(path: Path, key: str) -> float:
    for line in path.read_text().splitlines():
        if not line.startswith("# Measure "):
            continue
        fields = [part.strip() for part in line[len("# Measure "):].split(",")]
        if fields[0] + "." + fields[1] == key:
            return float(fields[3])
    raise KeyError((path, key))


def roi_field(rows: dict, field: str) -> dict:
    pairs = [(name, row[field]) for name, row in rows.items() if field in row]
    reference = np.array([item[1]["reference"] for item in pairs], dtype=float)
    candidate = np.array([item[1]["candidate"] for item in pairs], dtype=float)
    error = candidate - reference
    relative = np.divide(error, reference, out=np.full_like(error, np.nan), where=reference != 0)
    valid_relative = np.abs(relative[np.isfinite(relative)])
    ranked = np.argsort(np.abs(error))[::-1][:5]
    return {
        "regions": len(pairs),
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(candidate.mean()),
        "mean_signed_error": float(error.mean()),
        "mae": float(np.abs(error).mean()),
        "median_abs_error": float(np.median(np.abs(error))),
        "max_abs_error": float(np.abs(error).max()),
        "median_abs_relative_pct": float(np.median(valid_relative) * 100),
        "max_abs_relative_pct": float(valid_relative.max() * 100),
        "largest_absolute_errors": [
            {"region": pairs[i][0], "reference": float(reference[i]),
             "candidate": float(candidate[i]), "signed_error": float(error[i]),
             "signed_relative_pct": float(relative[i] * 100)}
            for i in ranked
        ],
    }


def geometry_pair(reference: Path, candidate: Path) -> dict:
    a, fa = fs.read_geometry(str(reference))
    b, fb = fs.read_geometry(str(candidate))
    out = {
        "reference_vertices": len(a), "candidate_vertices": len(b),
        "reference_faces": len(fa), "candidate_faces": len(fb),
        "ordered_faces_equal": bool(np.array_equal(fa, fb)),
        "pointwise_vertex_correspondence": bool(len(a) == len(b) and np.array_equal(fa, fb)),
    }
    if out["pointwise_vertex_correspondence"]:
        out["paired_displacement_mm"] = distribution(np.linalg.norm(a - b, axis=1))
        out["coordinates_exact"] = bool(np.array_equal(a, b))
    else:
        # Geometric coverage only. A nearest point is generally a different vertex.
        out["nearest_vertex_distance_mm_not_pointwise"] = {
            "reference_to_candidate": distribution(cKDTree(b).query(a, workers=4)[0]),
            "candidate_to_reference": distribution(cKDTree(a).query(b, workers=4)[0]),
        }
    return out


def main() -> None:
    base = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = base / "reconall_benchmark_pair_ac_20260924/official_subjects/a_official"
    case_root = base / "reconall_wm_agent_20260927/v3_e2e"
    candidate = case_root / "subjects/sub01"
    strict_path = case_root / "strict_138.json"
    spatial_path = out_dir / "comparison.json"
    strict = json.loads(strict_path.read_text())
    spatial = json.loads(spatial_path.read_text())
    run_path = candidate / "fnit-native-free-run.json"
    run = json.loads(run_path.read_text())
    original = json.loads((base / "reconall_benchmark_pair_ac_20260924/summary.json").read_text())
    official = next(row for row in original["results"] if row["label"] == "A_official")
    t1 = base.parent / "examples/data/sub-01_T1w.nii.gz"

    surfaces = {f"surf/{h}.{name}" for h in ("lh", "rh")
                for name in ("orig", "smoothwm", "inflated", "white", "white.preaparc",
                             "pial", "pial.T1", "sphere", "sphere.reg")}
    categories = {}
    for key, row in strict["files"].items():
        kind = ("mri" if key.startswith("mri/") else
                "surface" if key in surfaces else
                "vertex_map" if key.startswith("surf/") else
                "annotation" if key.startswith("label/") else "stats")
        categories.setdefault(kind, []).append((key, row))
    class_summary = {
        kind: {
            "checked": len(rows), "passed": sum(row["pass"] for _, row in rows),
            "missing_candidate": sum(row.get("missing_candidate", False) for _, row in rows),
            "failed_present": sum(not row["pass"] and not row.get("missing_candidate", False)
                                  for _, row in rows),
            "passing_files": [key for key, row in rows if row["pass"]],
            "missing_files": [key for key, row in rows if row.get("missing_candidate", False)],
        }
        for kind, rows in categories.items()
    }

    roi = {}
    for hemi in ("lh", "rh"):
        name = f"stats/{hemi}.aparc.stats"
        check = spatial["checks"][name]
        roi[hemi] = {
            "aparc_fields": {field: roi_field(check["rows"], field)
                             for field in ("ThickAvg", "SurfArea", "GrayVol", "MeanCurv")},
            "global": {},
        }
        for key in ("Cortex.MeanThickness", "Cortex.WhiteSurfArea", "Cortex.CortexVol",
                    "Cortex.NumVert"):
            ref = measure(reference / name, key)
            got = measure(candidate / name, key)
            roi[hemi]["global"][key] = {
                "reference": ref, "candidate": got, "signed_error": got - ref,
                "signed_relative_pct": (got - ref) / ref * 100 if ref else None,
            }
    roi["aseg_volume"] = roi_field(spatial["checks"]["stats/aseg.stats"]["rows"], "Volume_mm3")

    geometry = {}
    for hemi in ("lh", "rh"):
        geometry[hemi] = {
            name: geometry_pair(reference / f"surf/{hemi}.{name}",
                                candidate / f"surf/{hemi}.{name}")
            for name in ("orig.nofix", "smoothwm.nofix", "inflated.nofix",
                         "qsphere.nofix", "orig", "white", "pial", "sphere.reg")
        }

    maps = {}
    for hemi in ("lh", "rh"):
        maps[hemi] = {}
        for name in ("thickness", "area", "area.pial", "volume", "curv", "curv.pial"):
            left = fs.read_morph_data(str(reference / f"surf/{hemi}.{name}"))
            right = fs.read_morph_data(str(candidate / f"surf/{hemi}.{name}"))
            maps[hemi][name] = {
                "pointwise_comparison_allowed": False,
                "reason": "different vertex counts and ordered topology",
                "reference": distribution(left), "candidate": distribution(right),
            }

    volume_failures = {}
    for key, row in categories["mri"]:
        if not row["pass"] and not row.get("missing_candidate"):
            volume_failures[key] = {
                "dtype": row.get("dtype"),
                "outlier_voxels": row.get("voxels", {}).get("outliers"),
                "max_abs": row.get("voxels", {}).get("max_abs"),
                "first_outlier": row.get("voxels", {}).get("first_outlier"),
                "header_exact": row.get("header_exact"),
            }

    report = {
        "case": "real sub-01 T1, same input as archived FreeSurfer 8.2",
        "t1_sha256": digest(t1),
        "reference": str(reference), "candidate": str(candidate),
        "candidate_status": run["status"],
        "candidate_process_exit_code": int((case_root / "exit_code").read_text()),
        "candidate_reported_stage_total_seconds": run["total_seconds"],
        "candidate_process_wall_seconds": process_wall_seconds(case_root / "e2e.time"),
        "official_process_wall_seconds": process_wall_seconds(
            base / "reconall_benchmark_pair_ac_20260924/A_official.time.txt"),
        "official_launcher_wall_seconds": official["elapsed_seconds"],
        "timing_limit": "separate shared-node runs; outputs differ, so ratio is not equivalent-reconstruction acceleration",
        "strict": {
            "checked": strict["checked"], "passed": strict["passed"],
            "missing_candidate_total": sum(x["missing_candidate"] for x in class_summary.values()),
            "failed_present_total": sum(x["failed_present"] for x in class_summary.values()),
            "by_class": class_summary,
            "strict_sha256": digest(strict_path),
            "comparator_sha256": strict["comparator_sha256"],
        },
        "spatial_comparator_sha256": digest(spatial_path),
        "spatial_status_counts": dict(Counter(v["status"] for v in spatial["checks"].values())),
        "volume_failures": volume_failures,
        "first_surface_divergence": geometry,
        "vertex_map_distributions_not_pointwise": maps,
        "roi": roi,
        "stages_seconds": {s["name"]: s["seconds"] for s in run["stages"]},
        "nested_surfaces": run["surfaces"],
        "source_commit": "c925c3c19dd0a8df8c5f9ab9c0438c8b02d966e8",
        "run_json_sha256": digest(run_path),
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(out_dir / "benchmark_summary.json")


if __name__ == "__main__":
    main()
