"""Compare an independent Python white-preaparc step to pinned FreeSurfer RAM probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import nibabel.freesurfer as fs
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])
STAGES = (
    ("initial", "clear", slice(0, 3)),
    ("intensity", "intensity", slice(6, 9)),
    ("averaged", "after_signed_average", slice(6, 9)),
    ("pre_normal_spring", "after_self_repulsion", slice(6, 9)),
    ("normal_spring", "normal_spring", slice(6, 9)),
    ("curvature", "curvature", slice(6, 9)),
    ("tangential_spring", "tangential_spring", slice(6, 9)),
    ("after_collision", "after_collision", slice(0, 3)),
)


def _compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"shape differs: {actual.shape} versus {expected.shape}")
    distance = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    result = {
        "exact_elements": int(np.count_nonzero(actual == expected)),
        "elements": int(actual.size),
        "max_absolute": float(distance.max(initial=0)),
        "p99_absolute": float(np.percentile(distance, 99)),
        "vertices_with_any_difference": int(np.count_nonzero(np.any(actual != expected, axis=1))),
    }
    if distance.shape[1] == 3:
        displacement = np.linalg.norm(distance, axis=1)
        result.update(
            max_euclidean_mm=float(displacement.max(initial=0)),
            p99_euclidean_mm=float(np.percentile(displacement, 99)),
            vertices_above_0_0001_mm=int(np.count_nonzero(displacement > 0.0001)),
            vertices_above_0_1_mm=int(np.count_nonzero(displacement > 0.1)),
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--candidate-step", type=Path, required=True)
    parser.add_argument("--probe-prefix", type=Path, required=True)
    parser.add_argument("--native-log", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    candidate = np.load(args.diagnostics)
    result = {}
    for name, suffix, columns in STAGES:
        reference = np.fromfile(f"{args.probe_prefix}.{suffix}", dtype=STATE)
        result[name] = _compare(candidate[name], reference["floats"][:, columns])
        if name == "initial":
            result["ripped"] = {
                "exact_elements": int(np.count_nonzero(candidate["ripped"] == reference["flags"][:, 0])),
                "elements": int(len(reference)),
            }
    current, faces = fs.read_geometry(str(args.candidate_step))
    native = np.fromfile(f"{args.probe_prefix}.after_collision", dtype=STATE)
    result["step1_coordinates"] = _compare(current.astype(np.float32), native["floats"][:, :3])
    original_faces = fs.read_geometry(str(args.subject / f"surf/{args.hemi}.orig"))[1]
    result["step1_faces"] = {
        "exact_elements": int(np.count_nonzero(faces == original_faces)),
        "elements": int(original_faces.size),
    }
    inputs = [args.subject / f"surf/{args.hemi}.orig",
              args.subject / f"surf/autodet.gw.stats.{args.hemi}.dat"]
    inputs += [args.subject / f"mri/{name}.mgz" for name in
               ("brain.finalsurfs", "wm", "aseg.presurf")]
    objectives = None
    if args.native_log is not None:
        pattern = re.compile(r"PY_OBJ_REF (initial|step1) rms=([^ ]+) sse=([^ ]+)")
        native = {}
        for line in args.native_log.read_text().splitlines():
            match = pattern.search(line)
            if match and match.group(1) not in native:
                native[match.group(1)] = (float(match.group(2)), float(match.group(3)))
        if set(native) != {"initial", "step1"}:
            raise ValueError("native log lacks first-pass PY_OBJ_REF initial/step1")
        objectives = {}
        for label, stem in (("initial", "initial"), ("step1", "step")):
            rms, sse = native[label]
            objectives[label] = {
                "native_rms": rms, "python_rms": float(candidate[f"{stem}_rms"]),
                "rms_absolute": abs(rms - float(candidate[f"{stem}_rms"])),
                "native_sse": sse, "python_sse": float(candidate[f"{stem}_sse"]),
                "sse_absolute": abs(sse - float(candidate[f"{stem}_sse"])),
            }
    report = {
        "scope": "one white.preaparc optimizer step on frozen real T1 inputs",
        "reference": "instrumented pinned FreeSurfer 8.2 source, not installed binary",
        "subject_inputs": {str(path.relative_to(args.subject)): {
            "resolved_path": str(path.resolve()), "sha256": _sha256(path),
        } for path in inputs},
        "comparisons": result,
        "objective": objectives,
        "first_difference": next((name for name, values in result.items()
                                  if values["exact_elements"] != values["elements"]), None),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and report["first_difference"] is not None:
        raise SystemExit("first white placement step differs from pinned source")


if __name__ == "__main__":
    main()
