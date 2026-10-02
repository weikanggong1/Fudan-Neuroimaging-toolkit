"""Render saved real whole-case outputs against fresh official labels on CPU.

Run after a successful final production all r1 in the explicitly selected queue. Only derived
PNG and JSON are public artifacts. Images stay on the original server.
No model inference or GPU fitting is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
from time import monotonic


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--mode", choices=("stage", "raw"), required=True)
    parser.add_argument("--queue", type=Path, required=True, help="actual final production six-run queue")
    parser.add_argument("--plot-driver", type=Path, required=True,
                        help="external plotting tool; never overwrite the frozen source")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output-prefix", default="final_brain_figures")
    parser.add_argument("--include-brainstem", action="store_true")
    args = parser.parse_args()
    if Path(args.output_prefix).name != args.output_prefix or args.output_prefix in (".", ".."):
        raise ValueError("output-prefix must be one filename component")
    started = monotonic()
    root = args.root.resolve()
    evidence = root / "reproducibility_20261002"
    queue = json.loads(args.queue.read_text())
    selected_runs = [entry for entry in queue["runs"] if entry["mode"] == args.mode
                     and entry["repeat"] == 1 and entry["structure"] == "all"
                     and entry.get("exit_code") == 0]
    if len(selected_runs) != 1:
        raise ValueError("figure requires one completed full real-image r1, no failed attempt")
    candidate_dir = Path(selected_runs[0]["output"])
    source = Path(queue["source"])
    actual_manifest = identity(source / "source_manifest.json")
    if args.source_manifest.resolve() != (source / "source_manifest.json").resolve() or actual_manifest["sha256"] != args.source_manifest_sha256 or queue["source_manifest"] != actual_manifest:
        raise ValueError("explicit final source manifest must match the queue and expected SHA")
    source_manifest = json.loads(args.source_manifest.read_text())
    for entry in source_manifest["files"]:
        actual = identity(source / entry["path"])
        if actual["bytes"] != entry["bytes"] or actual["sha256"] != entry["sha256"]:
            raise ValueError("frozen source changed: " + entry["path"])
    runtime = {entry["path"].removeprefix("src/fnit/"): entry["sha256"]
               for entry in source_manifest["files"]
               if entry["path"].startswith("src/fnit/") and entry["path"].endswith(".py")}
    candidate_report = json.loads((candidate_dir / "report.json").read_text())
    if candidate_report["source_sha256"] != runtime:
        raise ValueError("candidate report does not match its frozen runtime source")
    for path, expected in candidate_report["input_sha256"].items():
        if identity(Path(path))["sha256"] != expected:
            raise ValueError("candidate input no longer matches recorded input")
    baseline_dir = root / f"segment4_precision_export_final_full_{args.mode}_20261002"
    official = evidence / "official_r1"
    t1 = (root.parent / "reconall_reference_gpucw1/fs_sub01/mri/norm.mgz"
          if args.mode == "stage" else root.parent.parent / "examples/data/sub-01_T1w.nii.gz")
    candidate = candidate_dir / "subregions_native.nii.gz"
    baseline = baseline_dir / "subregions_native.nii.gz"
    folder = evidence / args.output_prefix / args.mode
    if (folder / "figures_manifest.json").exists():
        raise ValueError("preserve completed figure collection; choose a new output-prefix")
    folder.mkdir(parents=True, exist_ok=True)
    cases = [
        {"name": "thalamus_low_nuclei", "reference": official / "thalamus/ThalamicNuclei.FSvoxelSpace.mgz",
         "offset": 0, "labels": [8111, 8116, 8118, 8127, 8211],
         "names": ["L L-Sg", "L MV(Re)", "L Pf", "L VAmc", "R L-Sg"], "padding_mm": 2},
        {"name": "hippocampus_low_regions", "reference": official / "hippo-amygdala/rh.hippoAmygLabels.FSvoxelSpace.mgz",
         "offset": 10000, "labels": [10203, 10240, 10243, 10244],
         "names": ["Parasubiculum", "CA3-body", "DG-head", "DG-body"], "padding_mm": 2},
        {"name": "amygdala_AAA_Medial", "reference": official / "hippo-amygdala/rh.hippoAmygLabels.FSvoxelSpace.mgz",
         "offset": 10000, "labels": [17006, 17010],
         "names": ["Medial", "AAA"], "padding_mm": 5},
    ]
    if args.include_brainstem:
        cases.append({"name": "brainstem_low_regions",
                      "reference": official / "brainstem/brainstemSsLabels.FSvoxelSpace.mgz",
                      "offset": 0, "labels": [178, 175, 173],
                      "names": ["SCP", "Medulla", "Midbrain"], "padding_mm": 2})
    baseline_previous = json.loads((evidence / "brain_figures" / args.mode / "thalamus_low_nuclei.json").read_text())
    for key, path in (("t1", t1), ("baseline", baseline)):
        expected = baseline_previous["inputs"][key]
        actual = identity(path)
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError("audited pre-change full input changed: " + key)
    if identity(baseline_dir / "report.json") != baseline_previous["baseline_full_report"]:
        raise ValueError("audited pre-change full report changed")
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               NUMEXPR_NUM_THREADS="4", MPLBACKEND="Agg")
    records = []
    for case in cases:
        output = folder / (case["name"] + ".png")
        fixed_reference = queue["fixed_inputs_references_assets"][str(case["reference"])]
        if identity(case["reference"]) != fixed_reference:
            raise ValueError("fresh official reference identity changed")
        command = [args.python, str(args.plot_driver), "--t1", str(t1),
                   "--reference", str(case["reference"]), "--reference-offset", str(case["offset"]),
                   "--baseline", str(baseline), "--candidate", str(candidate),
                   "--baseline-name", "FNIT 4178a48", "--candidate-name", "FNIT final full run",
                   "--output", str(output), "--labels", *map(str, case["labels"]),
                   "--names", *case["names"], "--slice-padding-mm", str(case["padding_mm"])]
        subprocess.run(command, env=env, check=True)
        metadata_path = output.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        if len(metadata["slices_display_indices"]) != 6 or not metadata["slice_indices_unique"]:
            raise ValueError("predeclared official ROI padding did not yield six distinct axial planes")
        if output.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("plot output is not PNG")
        dimensions = list(struct.unpack(">II", output.read_bytes()[16:24]))
        import nibabel as nib
        from nibabel.processing import resample_from_to
        import numpy as np
        grid = (tuple(metadata["display_shape"]), np.asarray(metadata["display_affine"]))
        display_labels = []
        for index, path in enumerate((case["reference"], baseline, candidate)):
            array = np.asarray(resample_from_to(nib.load(path), grid, order=0).dataobj, dtype=np.int32)
            if index == 0 and case["offset"]:
                array = np.where(array != 0, array + case["offset"], 0)
            display_labels.append(np.isin(array, case["labels"]))
        points = np.argwhere(np.logical_or.reduce(display_labels))
        lower = np.maximum(points.min(0) - 6, 0)
        upper = np.minimum(points.max(0) + 7, metadata["display_shape"])
        metadata.update({"pipeline_mode": args.mode,
                         "figure_dimensions_pixels": dimensions,
                         "crop_display_indices_xy": {"lower_inclusive": lower[:2].tolist(), "upper_exclusive": upper[:2].tolist()},
                         "crop_scope": "same union-of-selected-labels crop for official, before and final, on one RAS grid",
                         "verified_source_files": len(source_manifest["files"]),
                         "pipeline_queue": identity(args.queue),
                         "successful_full_repeat": 1,
                         "baseline_behavior_commit": "4178a48",
                         "reference_scope": "fresh official FreeSurfer8.2 repeat1 with audited input/source identity",
                         "pipeline_source_manifest": identity(source / "source_manifest.json"),
                         "candidate_full_report": identity(candidate_dir / "report.json"),
                         "candidate_context_identity": identity(candidate_dir / "context_identity.json"),
                         "baseline_full_report": identity(baseline_dir / "report.json"),
                         "orchestrator": identity(Path(__file__)),
                         "scientific_scope": "CPU display of saved complete real-image runs; no fitting or Dice from display interpolation"})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        records.extend([{**identity(output), "figure_dimensions_pixels": dimensions}, identity(metadata_path)])
    manifest = {"scope": "Derived PNG and JSON metadata only; no imaging volumes, models, atlases or licenses.",
                "mode": args.mode, "candidate": str(candidate_dir),
                "driver": identity(args.plot_driver), "orchestrator": identity(Path(__file__)),
                "input_source_identity_verified": True,
                "verified_runtime_python_files": len(runtime),
                "source_manifest": actual_manifest,
                "output_prefix": args.output_prefix,
                "CPU_render_seconds": monotonic() - started,
                "files": [{**entry, "path": str(Path(entry["path"]).relative_to(evidence))}
                          for entry in records]}
    (folder / "figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"mode": args.mode, "figures": len(cases), "derived_files": len(records),
                      "output_dir": str(folder)}))


if __name__ == "__main__":
    main()

