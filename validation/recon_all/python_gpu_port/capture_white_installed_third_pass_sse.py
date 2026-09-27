"""Capture installed FreeSurfer third-pass SSE terms from the same real T1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    for section in ("mri", "surf"):
        (args.out / section).mkdir(parents=True, exist_ok=True)
    for name in ("brain.finalsurfs.mgz", "wm.mgz", "aseg.presurf.mgz"):
        link = args.out / "mri" / name
        if not link.exists():
            link.symlink_to(args.subject / "mri" / name)
    for name in ("lh.orig", "autodet.gw.stats.lh.dat"):
        shutil.copy2(args.subject / "surf" / name, args.out / "surf" / name)
    script = args.out / "capture.gdb"
    script.write_text("""set pagination off
set confirm off
set $returns=0
break *0x4176b0
commands 1
 silent
 set $returns=$returns+1
 if $returns==3
  printf "THIRD_PASS_SSE_COMPLETE\\n"
  quit
 end
 continue
end
run
""")
    command = [
        str(args.binary), "--adgws-in", "../surf/autodet.gw.stats.lh.dat",
        "--wm", "wm.mgz", "--threads", "4", "--invol", "brain.finalsurfs.mgz",
        "--lh", "--i", "../surf/lh.orig", "--o", "../surf/lh.white.unused",
        "--white", "--seg", "aseg.presurf.mgz", "--restore-255", "--nsmooth", "5",
        "--rip-bg-no-annot", "--rip-bg", "--rip-bg-lof", "--restore-255",
        "--outvol", "mrisps.wpa.mgz",
    ]
    env = os.environ.copy()
    env.update(FREESURFER_HOME=str(args.binary.parent.parent),
               SUBJECTS_DIR=str(args.subject.parent), FS_LICENSE=str(args.license),
               OMP_NUM_THREADS="4", FREESURFER_logSSE="1")
    started = time.perf_counter()
    with (args.out / "capture.log").open("w") as log:
        status = subprocess.run(["gdb", "--batch", "-x", str(script), "--args", *command],
                                cwd=args.out / "mri", env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    content = (args.out / "capture.log").read_text(errors="replace")
    current: dict[str, float] = {}
    accepted = []
    for line in content.splitlines():
        for key, pattern in (
            ("repulsion_weighted", r"^new sse_repulse : ([0-9.]+)"),
            ("spring_weighted", r"^new sse_tspring : ([0-9.]+)"),
            ("intensity_weighted", r"^new sse_val : ([0-9.]+)"),
            ("total", r"^new sum = ([0-9.]+)"),
        ):
            match = re.match(pattern, line)
            if match:
                current[key] = float(match.group(1))
        step = re.match(r"^(\d{3}): dt: ([0-9.]+), sse=([0-9.]+), rms=([0-9.]+)", line)
        if step and int(step.group(1)) >= 27:
            accepted.append({"step": int(step.group(1)), "dt": float(step.group(2)),
                             "printed_sse": float(step.group(3)),
                             "printed_rms": float(step.group(4)), **current})
    report = {
        "reference": "installed FreeSurfer 8.2; third-pass per-step high-precision SSE only",
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "input_sha256": {str(args.subject / section / name): hashlib.sha256(
            (args.subject / section / name).read_bytes()).hexdigest()
            for section, names in (("mri", ("brain.finalsurfs.mgz", "wm.mgz", "aseg.presurf.mgz")),
                                   ("surf", ("lh.orig", "autodet.gw.stats.lh.dat")))
            for name in names},
        "command": command,
        "gdb_status": status.returncode,
        "third_pass_complete": "THIRD_PASS_SSE_COMPLETE" in content,
        "accepted": accepted,
        "log_sha256": hashlib.sha256((args.out / "capture.log").read_bytes()).hexdigest(),
        "seconds": time.perf_counter() - started,
    }
    (args.out / "capture_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gdb_status": report["gdb_status"],
                      "third_pass_complete": report["third_pass_complete"],
                      "steps": [row["step"] for row in accepted],
                      "seconds": report["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
