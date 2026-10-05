"""Isolated actual MATLAB reference launcher; no candidate FNIT imports."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def matlab_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--matlab", type=Path, required=True)
    parser.add_argument("--jvm", action="store_true", help="Enable original MATLAB graphics for the separate full-node/spectra validation")
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text())
    source_root = Path(binding["reference_root"])
    source_manifest = json.loads((source_root / "source_manifest.private.json").read_text())
    for entry in source_manifest["files"]:
        data = (source_root / entry["path"]).read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("Original HCP reference source changed")
    expression = ("try, addpath(" + matlab_string(Path(__file__).resolve().parent) + "); "
                  "run_hcp_reference(" + matlab_string(args.binding.resolve()) + ", "
                  + matlab_string(args.output_dir.resolve()) + ", " + str(args.threads) + "); "
                  "exit(0); catch exception, disp(getReport(exception, 'extended')); exit(1); end;")
    command = [str(args.matlab)]
    if not args.jvm:
        command.append("-nojvm")
    command += ["-nodisplay", "-nosplash", "-nodesktop"]
    if args.threads == 1:
        command.append("-singleCompThread")
    command += ["-r", expression]
    started = time.perf_counter()
    completed = subprocess.run(command)
    wall = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"Actual original MATLAB reference exited {completed.returncode}")
    report_path = args.output_dir / "reference_report.public.json"
    report = json.loads(report_path.read_text())
    report["matlab_process_wall_seconds"] = wall
    report["process_outside_wrapper_seconds"] = wall - report["wrapper_seconds"]
    report["outside_wrapper_scope"] = "MATLAB startup/shutdown and invocation overhead; separate from original function"
    report["jvm_requested"] = args.jvm
    report["original_source_manifest_sha256"] = hashlib.sha256(
        (source_root / "source_manifest.private.json").read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
