"""Run independent MRtrix CPU fixed-TCK reference, with hashes and wall times."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import os


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mrtrix-bin", required=True, type=Path)
    p.add_argument("--tracks", required=True, type=Path)
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--wait-seconds", type=int, default=0)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {name: args.input_dir / f"{name}_float64_geometry.nii" for name in ("wm_fod", "five_tissue", "fa")}
    started_wait = time.monotonic()
    while not all(path.is_file() for path in (args.tracks, *files.values(), args.input_dir / "reference_input_conversion.json")):
        if time.monotonic() - started_wait > args.wait_seconds:
            raise RuntimeError("real fixed TCK/reference input conversion not ready")
        (args.output_dir / "wait_status.json").write_text(json.dumps({"state": "waiting_real_checkpoints", "pid": os.getpid(), "elapsed_s": time.monotonic() - started_wait}) + "\n")
        time.sleep(30)
    report = dict(scope="official same fixed-TCK reference; CPU 8 threads; not raw full pipeline",
                  inputs={str(path): sha(path) for path in (args.tracks, *files.values())}, stages={})
    (args.output_dir / "reference_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, path in files.items():
        subprocess.run([str(args.mrtrix_bin / "mrinfo"), str(path),
                        "-config", "RealignTransform", "false", "-json_all",
                        str(args.output_dir / f"{name}_mrinfo.json")], check=True)
    commands = {
        "sift2": [str(args.mrtrix_bin / "tcksift2"), str(args.tracks), str(files["wm_fod"]),
                  str(args.output_dir / "sift2_weights.txt"), "-act", str(files["five_tissue"]),
                  "-nthreads", "8", "-csv", str(args.output_dir / "sift2_stats.csv")],
        "fa": [str(args.mrtrix_bin / "tcksample"), str(args.tracks), str(files["fa"]),
               str(args.output_dir / "mean_fa.txt"), "-precise", "-stat_tck", "mean", "-nthreads", "8"],
    }
    for name, command in commands.items():
        version = subprocess.check_output([command[0], "-version"], stderr=subprocess.STDOUT, text=True)
        started = time.perf_counter()
        with open(args.output_dir / f"{name}.log", "w") as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report["stages"][name] = dict(command=command, program_sha256=sha(command[0]), version=version,
                                     wall_s=time.perf_counter()-started, returncode=run.returncode)
        (args.output_dir / "reference_report.json").write_text(json.dumps(report, indent=2) + "\n")
        if run.returncode:
            raise RuntimeError(f"{name} official reference failed; see preserved log")
    report["completed"] = True
    (args.output_dir / "reference_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
