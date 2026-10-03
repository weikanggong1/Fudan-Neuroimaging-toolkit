"""补查完整参考的保存后 GIFTI/CIFTI 合同；额外诊断时钟不计入原整例墙钟。"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check(report_path: Path, *, report_override: dict | None = None) -> dict:
    started = time.perf_counter()
    own_path = Path(__file__)
    own_before = sha256(own_path)
    report_before = sha256(report_path)
    report = json.loads(report_path.read_text()) if report_override is None else report_override
    if (report.get("status") != "complete" or report.get("command_exit_code") != 0
            or report.get("source_unchanged_during_run") is not True
            or report.get("input_unchanged_during_run") is not True
            or report.get("QC", {}).get("passed") is not True):
        raise ValueError("reference final completion/input/source/QC contract is not satisfied")
    case = report_path.parent
    frames, tr = report["frames"], report["repetition_time"]
    if frames != 180:
        raise ValueError("reference did not preserve the entire acquired run")
    result = {
        "case_id": report["subject"], "attempt": case.name,
        "status": "passed", "frames": frames, "repetition_time": tr,
        "scope": "Additional saved GIFTI/CIFTI/header checks after original whole clock; primary complete NIfTI all-finite checks remain bound to the unchanged original report.",
        "original_report_sha256": report_before, "validator_sha256": own_before,
        "normalized_report_override_used": report_override is not None,
        "validator_software_versions": {"python": sys.version.split()[0], "nibabel": nib.__version__, "numpy": np.__version__},
        "native_bold": {}, "msmsulc_spheres": {}, "errors": [],
    }
    checked = {}
    for item in report["QC"]["outputs"]:
        if item["kind"] not in ("fsnative_BOLD", "new_MSMSulc_sphere", "fsLR91k_CIFTI"):
            continue
        path = (case / item["relative_path"]).resolve()
        if not path.is_relative_to(case.resolve()):
            raise ValueError("saved output path escapes the current fresh attempt")
        initial = sha256(path)
        if initial != item["sha256"]:
            raise ValueError("saved output differs from original final QC binding")
        checked[path] = initial
        image = nib.load(str(path))
        if item["kind"] == "fsLR91k_CIFTI":
            series, brain = image.header.get_axis(0), image.header.get_axis(1)
            if (not isinstance(series, nib.cifti2.SeriesAxis)
                    or not isinstance(brain, nib.cifti2.BrainModelAxis)
                    or image.shape != (frames, 91282) or series.unit != "SECOND"
                    or not np.isclose(series.step, tr)
                    or not np.isfinite(np.asanyarray(image.dataobj)).all()):
                raise ValueError("CIFTI time/brain axes or saved values are invalid")
            if brain.nvertices != {"CIFTI_STRUCTURE_CORTEX_LEFT": 32492,
                                   "CIFTI_STRUCTURE_CORTEX_RIGHT": 32492}:
                raise ValueError("CIFTI cortical vertex domains differ from fsLR32k")
            volume = brain.volume_mask
            if (brain.volume_shape is None or brain.affine is None
                    or not np.isfinite(brain.affine).all()
                    or np.any(brain.voxel[volume] < 0)
                    or np.any(brain.voxel[volume] >= np.asarray(brain.volume_shape))):
                raise ValueError("CIFTI subcortical voxel domain is invalid")
            model_counts = {str(name): int(np.count_nonzero(brain.name == name))
                            for name in np.unique(brain.name)}
            result["dtseries"] = {
                "sha256": initial, "shape": list(image.shape), "series_start": float(series.start),
                "series_unit": series.unit, "series_step": float(series.step),
                "brain_model_counts": model_counts, "nvertices": {name: int(count) for name, count in brain.nvertices.items()},
                "volume_shape": [int(size) for size in brain.volume_shape], "volume_affine": brain.affine.tolist(),
                "all_finite": True,
            }
            continue
        match = re.search(r"hemi-([LR])_", path.name)
        if match is None:
            raise ValueError("saved GIFTI has no explicit hemisphere")
        hemi = match.group(1)
        if item["kind"] == "fsnative_BOLD":
            if len(image.darrays) != frames:
                raise ValueError("native GIFTI does not contain the complete run")
            shapes = [array.data.shape for array in image.darrays]
            if (not shapes or len(shapes[0]) != 1 or shapes[0][0] == 0
                    or any(shape != shapes[0] for shape in shapes)
                    or any(not np.isfinite(array.data).all() for array in image.darrays)):
                raise ValueError("native GIFTI has inconsistent or nonfinite vertex arrays")
            result["native_bold"][hemi] = {"sha256": initial, "shape": [frames, shapes[0][0]], "all_finite": True}
        else:
            points = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
            triangles = image.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
            if len(points) != 1 or len(triangles) != 1:
                raise ValueError("MSMSulc sphere point/triangle arrays are not unique")
            vertices, faces = points[0].data, triangles[0].data
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                    or not np.isfinite(vertices).all() or faces.ndim != 2
                    or faces.shape[1] != 3 or not len(faces)
                    or faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError("saved MSMSulc sphere geometry is invalid")
            result["msmsulc_spheres"][hemi] = {"sha256": initial, "vertices": len(vertices), "faces": len(faces), "all_finite": True}
    if set(result["native_bold"]) != {"L", "R"} or set(result["msmsulc_spheres"]) != {"L", "R"} or "dtseries" not in result:
        raise ValueError("complete output set lacks a hemisphere or CIFTI")
    for hemi in ("L", "R"):
        if result["native_bold"][hemi]["shape"][1] != result["msmsulc_spheres"][hemi]["vertices"]:
            raise ValueError("native BOLD and registration sphere vertex domains differ")
    result["saved_file_guards_equal"] = all(sha256(path) == digest for path, digest in checked.items())
    result["source_guards_equal"] = sha256(own_path) == own_before
    result["original_report_guard_equal"] = sha256(report_path) == report_before
    if not all(result[key] for key in ("saved_file_guards_equal", "source_guards_equal", "original_report_guard_equal")):
        raise ValueError("saved output, completed report or validator changed during checks")
    result["additional_diagnostic_wall_seconds"] = time.perf_counter() - started
    result["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["CON01", "CON03", "CON04", "CON05", "CON06", "CON07", "CON08", "CON09", "CON10", "CON11"])
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--report-name", default="report.public.json")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = list(args.subjects)
    state = {"status": "waiting", "cases": {}, "validator_sha256": sha256(Path(__file__)), "production_timing_includes_this_diagnostic": False}
    while pending:
        for subject in list(pending):
            target = args.output_root / (subject + ".public.json")
            if target.exists():
                state["cases"][subject] = json.loads(target.read_text())
                pending.remove(subject)
                continue
            reports = sorted((args.reference_root / "cases" / ("sub-" + subject)).glob("attempt-*/" + args.report_name))
            complete = [p for p in reports if json.loads(p.read_text()).get("status") == "complete"]
            if not complete:
                continue
            try:
                if len(complete) != 1:
                    raise ValueError("multiple completed fresh attempts require explicit selection")
                row = check(complete[0])
            except Exception as error:
                (args.output_root / (subject + ".private.txt")).write_text(traceback.format_exc())
                row = {"case_id": subject, "status": "failed", "error_type": type(error).__name__, "detail": "Saved reference contract failed; exact private diagnostic retained."}
            save(target, row)
            state["cases"][subject] = row
            pending.remove(subject)
        state["pending_cases"] = list(pending)
        state["status"] = ("waiting" if pending else
                           "passed" if all(row["status"] == "passed" for row in state["cases"].values()) else "failed")
        save(args.output_root / "cohort.public.json", state)
        if pending and args.watch:
            time.sleep(args.poll_seconds)
        else:
            break


if __name__ == "__main__":
    main()
