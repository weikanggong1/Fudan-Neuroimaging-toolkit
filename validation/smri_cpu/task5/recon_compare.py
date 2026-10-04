"""Score a finished CPU subject using the existing full-output comparators.

The reference is independent. Vertex maps are never paired before ordered
topology is checked. Final white/pial distances use the full triangle mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate_state(candidate, runner_record=None):
    """The process receipt takes precedence over an unfinished pipeline report."""
    if runner_record is not None:
        if not runner_record.exists():
            return None, None
        receipt = json.loads(runner_record.read_text())
        if receipt.get("status") in {"failed", "timeout", "cancelled"}:
            return None, {"status": "candidate_failed", "runner_status": receipt["status"],
                          "returncode": receipt.get("returncode"),
                          "reason": "timed candidate process did not complete"}
        if receipt.get("status") != "complete":
            return None, None
    report = candidate / "fnit-native-free-run.json"
    if not report.exists():
        return None, None
    run = json.loads(report.read_text())
    if run.get("status") == "failed":
        return None, {"status": "candidate_failed", "failed_stage": run.get("failed_stage")}
    return (run, None) if run.get("status") == "complete" else (None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, required=True)
    parser.add_argument("--geometry-helper", type=Path, required=True)
    parser.add_argument("--label-table", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=21600)
    parser.add_argument("--runner-record", type=Path,
                        help="timed process receipt; rejects stale running/complete pipeline metadata")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    while True:
        run, failure = candidate_state(args.candidate, args.runner_record)
        if failure is not None:
            write(args.output / "status.json", failure)
            return
        if run is not None:
            break
        if time.monotonic() - started > args.wait_seconds:
            raise TimeoutError("candidate did not finish before comparison deadline")
        time.sleep(10)
    if run.get("threads") != 8 or run.get("device") != "cpu":
        raise ValueError("expected the declared CPU8 subject")
    if run.get("output_validation", {}).get("present") != 138:
        raise ValueError("candidate did not produce all 138 expected outputs")
    # This is posthoc computation, separate from both timed reconstructions.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import numba
    torch.set_num_threads(4)
    numba.set_num_threads(4)
    strict = load(args.scripts_dir / "compare_complete_subject.py")
    surface = load(args.scripts_dir / "compare_surface_chain.py")
    geometry = load(args.geometry_helper)
    result = strict.compare(args.reference, args.candidate)
    if result["checked"] != 138:
        raise ValueError("frozen strict item count changed")
    result["interpretation"] = "file diagnostics; indexed maps require separate topology correspondence"
    write(args.output / "strict.private.json", result)
    gate = geometry.geometry(args.reference, args.candidate, surface)
    write(args.output / "geometry.private.json", gate)
    write(args.output / "vertices.private.json",
          geometry.local_differences(args.reference, args.candidate, gate))
    commands = []
    for name, filename in [("compare_region_stats", "regions.private.json"),
                           ("compare_parcellation_dice", "dice.private.json")]:
        command = [sys.executable, str(args.scripts_dir / (name + ".py")),
                   "--reference", str(args.reference), "--candidate", str(args.candidate),
                   "--code-commit", args.source_label, "--output", str(args.output / filename)]
        if name == "compare_parcellation_dice":
            command += ["--label-table", str(args.label_table)]
        tick = time.monotonic()
        subprocess.run(command, check=True)
        commands.append({"comparator": name, "comparison_seconds": time.monotonic() - tick})
    finals = [gate["surfaces"][f"{hemi}.{name}"] for hemi in ("lh", "rh")
              for name in ("white", "pial")]
    if all(row.get("surface_ras_mm_comparable") and row["status"] == "available" for row in finals):
        surface.STAGES = ("white", "pial")
        distances = surface.compare(args.reference, args.candidate)
        distances["scope"] = "all vertices to final full triangle meshes; not continuous Hausdorff"
        write(args.output / "surfaces.private.json", distances)
    else:
        write(args.output / "surfaces.private.json", {"status": "not_assessed_space_or_invalid_mesh"})
    for kind, subject in [("fnit", args.candidate), ("official", args.reference)]:
        subprocess.run([sys.executable, str(args.scripts_dir / "benchmark_surface_quality_extended.py"),
                        "--subject", str(subject), "--output", str(args.output / ("quality_" + kind)),
                        "--code-version", args.source_label if kind == "fnit" else "FreeSurfer8.2.0-1",
                        "--source-kind", kind, "--threads", "4"], check=True)
    write(args.output / "status.json", {
        "status": "scored", "strict_passed": result["passed"], "strict_checked": result["checked"],
        "overall_numerical_equivalence": "not_assessed", "posthoc_commands": commands,
        "source_label": args.source_label,
        "comparator_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [args.geometry_helper, *args.scripts_dir.glob("*.py")]}})


if __name__ == "__main__":
    main()
