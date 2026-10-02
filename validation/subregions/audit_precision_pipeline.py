"""Audit saved public-data precision runs without refitting or external tools."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np


def identity(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    api = json.loads((args.run / "api_report.json").read_text())
    manifest = json.loads((args.source / "source_manifest.json").read_text())
    paths = {entry["path"]: entry for entry in manifest["files"]}
    source_match = all((args.source / entry["path"]).stat().st_size == entry["bytes"] and
                       identity(args.source / entry["path"])["sha256"] == entry["sha256"]
                       for entry in manifest["files"])
    runtime_match = {name: paths["src/fnit/" + name]["sha256"] == digest
                     for name, digest in report["source_sha256"].items()}
    input_hash_matches = {name: identity(Path(name))["sha256"] == digest
                          for name, digest in report["input_sha256"].items()}
    input_image = nib.load(report["input"])
    image = nib.load(api["files"]["labels"])
    labels = np.asarray(image.dataobj)
    geometry_match = image.shape == input_image.shape and np.allclose(image.affine, input_image.affine,
                                                                     atol=1e-5, rtol=0)
    voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
    volume_checks = {}
    for label, volume in api["volumes"].items():
        expected = np.count_nonzero(labels == int(label)) * voxel_volume
        volume_checks[label] = bool(np.isclose(volume["hard_volume_mm3"], expected, atol=1e-6, rtol=1e-6))
    soft = np.asarray([entry["soft_volume_mm3"] for entry in api["volumes"].values()])
    allowed = {0, *[int(label) for label in api["labels"]]}
    highres = {}
    for key, filename in api["files"].items():
        if key.startswith("highres/"):
            volume = nib.load(filename)
            values = np.asarray(volume.dataobj)
            highres[key] = {**identity(Path(filename)), "shape": list(volume.shape),
                            "affine": volume.affine.tolist(), "dtype": str(values.dtype),
                            "finite": bool(np.isfinite(values).all()),
                            "allowed_labels": bool(set(np.unique(values).tolist()) <= allowed)}
    status_path = args.root / (args.run.name + "_status.json")
    status = json.loads(status_path.read_text())
    assets = []
    for path in sorted((args.root / "atlases").rglob("*")):
        if path.is_file():
            assets.append(identity(path))
    from fnit.weights import MODEL_FILES, WEIGHT_FILES
    models = {}
    for name in MODEL_FILES["synthseg-plus"]:
        record = identity(args.root / "weights" / name)
        record["matches_release_manifest"] = (
            record["bytes"], record["sha256"]) == tuple(WEIGHT_FILES[name][1:])
        models[name] = record
    checks = {"source_manifest_all_size_sha256_match": bool(source_match),
              "reported_source_all_matches_manifest": all(runtime_match.values()),
              "input_hash_matches_report": all(input_hash_matches.values()),
              "native_geometry_matches_input": bool(geometry_match), "native_dtype_int32": labels.dtype == np.int32,
              "native_labels_in_table": set(np.unique(labels).tolist()) <= allowed,
              "all_110_volume_entries": len(api["volumes"]) == 110,
              "all_hard_volumes_match_native_counts": all(volume_checks.values()),
              "all_soft_volumes_finite_nonnegative": bool(np.isfinite(soft).all() and (soft >= 0).all()),
              "four_positive_final_Jacobians": len(api["fit_min_jacobians"]) == 4 and
                                             all(value > 0 for value in api["fit_min_jacobians"].values()),
              "four_valid_highres_images": len(highres) == 4 and
                                            all(item["finite"] and item["allowed_labels"] for item in highres.values()),
              "five_models_match_manifest": len(models) == 5 and
                                            all(item["matches_release_manifest"] for item in models.values()),
              "monitor_completed_within_memory_limit": status["state"] == "completed" and
                  status["exit_code"] == 0 and status["max_own_process_memory_mib"] <= status["own_process_limit_mib"]}
    result = {"driver": identity(Path(__file__)), "source": str(args.source), "source_files": len(paths),
              "reported_module_count": len(runtime_match), "checks": checks,
              "all_passed": all(checks.values()), "input": identity(Path(report["input"])),
              "all_input_hashes_match_report": input_hash_matches,
              "native": {**identity(Path(api["files"]["labels"])), "shape": list(image.shape),
                         "affine": image.affine.tolist(), "dtype": str(labels.dtype)},
              "highres": highres, "volume_checks": volume_checks, "models": models, "atlases": assets,
              "reported_source_matches_manifest": runtime_match, "fit_min_jacobians": api["fit_min_jacobians"],
              "compute_seconds": report["wall_seconds"], "api_seconds": report["api_total_seconds"],
              "process_wall_seconds": status["finished_unix"] - status["started_unix"],
              "own_gpu_memory_peak_mib": status["max_own_process_memory_mib"],
              "own_gpu_memory_limit_mib": status["own_process_limit_mib"]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run": str(args.run), "all_passed": result["all_passed"], "checks": checks}))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
