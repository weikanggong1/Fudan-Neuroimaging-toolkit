"""Validation only: independent official runs with immutable shared inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def identity(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freesurfer-home", type=Path, required=True)
    parser.add_argument("--fs-license", type=Path,
                        help="Existing server registration file; never copied into evidence")
    parser.add_argument("--repeat", type=int, required=True, choices=(1, 2, 3))
    args = parser.parse_args()
    subject_root = args.root.parent / "reconall_reference_gpucw1"
    out = args.root / "reproducibility_20261002" / f"official_r{args.repeat}"
    out.mkdir(parents=True, exist_ok=False)
    status_path = out / "queue.json"
    package = args.freesurfer_home / "python/lib/python3.8/site-packages/samseg"
    source = [identity(p) for p in sorted((package / "subregions").rglob("*.py"))]
    source += [identity(package / "cli/segment_subregions.py")]
    input_records = [identity(subject_root / "fs_sub01/mri" / name)
                     for name in ("norm.mgz", "aseg.mgz", "wmparc.mgz")]
    env = dict(os.environ, FREESURFER_HOME=str(args.freesurfer_home),
               SUBJECTS_DIR=str(subject_root), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
    env["PATH"] = str(args.freesurfer_home / "bin") + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONPATH", None)
    if args.fs_license is not None:
        if not args.fs_license.is_file():
            raise FileNotFoundError("Existing FreeSurfer registration file is required")
        env["FS_LICENSE"] = str(args.fs_license)
    status = {"state": "running", "repeat": args.repeat, "threads": 4,
              "hostname": os.uname().nodename, "started_unix": time.time(),
              "inputs": input_records, "official_python_source": source, "runs": []}

    def save():
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2) + "\n")
        tmp.replace(status_path)

    save()
    for structure in ("brainstem", "thalamus", "hippo-amygdala"):
        command = [str(args.freesurfer_home / "bin/segment_subregions"), structure,
                   "--cross", "fs_sub01", "--sd", str(subject_root), "--threads", "4",
                   "--out-dir", str(out / structure), "--temp-dir", str(out / ("temp_" + structure))]
        record = {"structure": structure, "command": command, "started_unix": time.time()}
        status["runs"].append(record)
        save()
        with (out / (structure + ".log")).open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        record.update(exit_code=result.returncode, finished_unix=time.time())
        record["wall_seconds"] = record["finished_unix"] - record["started_unix"]
        save()
        if result.returncode:
            status["state"] = "failed"
            save()
            raise SystemExit(result.returncode)
    status.update(state="completed", finished_unix=time.time())
    save()


if __name__ == "__main__":
    main()
