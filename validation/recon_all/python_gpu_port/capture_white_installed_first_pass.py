"""Capture the installed FreeSurfer 8.2 white optimizer's first-pass RAM mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    for section in ("mri", "surf"):
        (args.out / section).mkdir(parents=True, exist_ok=True)
    for name in ("brain.finalsurfs.mgz", "wm.mgz", "aseg.presurf.mgz"):
        link = args.out / "mri" / name
        if not link.exists():
            link.symlink_to(args.subject / "mri" / name)
    for name in (f"{args.hemisphere}.orig", f"autodet.gw.stats.{args.hemisphere}.dat"):
        destination = args.out / "surf" / name
        if not destination.exists():
            shutil.copy2(args.subject / "surf" / name, destination)
    command = [
        str(args.binary), "--adgws-in", f"../surf/autodet.gw.stats.{args.hemisphere}.dat",
        "--wm", "wm.mgz", "--threads", "4", "--invol", "brain.finalsurfs.mgz",
        f"--{args.hemisphere}", "--i", f"../surf/{args.hemisphere}.orig",
        "--o", f"../surf/{args.hemisphere}.white.ram", "--white",
        "--seg", "aseg.presurf.mgz", "--restore-255", "--nsmooth", "5",
        "--rip-bg-no-annot", "--rip-bg", "--rip-bg-lof", "--restore-255",
        "--outvol", "mrisps.wpa.mgz",
    ]
    script = args.out / "capture.gdb"
    script.write_text("""set pagination off
break *0x416970
commands 1
 silent
 set {int}($rbx+0x514)=FIRST_NITER
 printf "INSTALLED_RAM_NITER=%d\\n", {int}($rbx+0x514)
 continue
end
break *0x4176b0
commands 2
 silent
 printf "INSTALLED_RAM_AFTER_FIRST_OUTER\\n"
 set $rip=0x417776
 continue
end
run
""".replace("FIRST_NITER", str(args.iterations)))
    env = os.environ.copy()
    env.update(FREESURFER_HOME=str(args.binary.parent.parent),
               SUBJECTS_DIR=str(args.subject.parent), FS_LICENSE=str(args.license),
               OMP_NUM_THREADS="4")
    with (args.out / "capture.log").open("w") as log:
        status = subprocess.run(
            ["gdb", "--batch", "-x", str(script), "--args", *command],
            cwd=args.out / "mri", env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    content = (args.out / "capture.log").read_text(errors="replace")
    output = args.out / "surf" / f"{args.hemisphere}.white.ram"
    report = {
        "reference": "installed FreeSurfer 8.2, isolated first-pass RAM capture",
        "binary": str(args.binary),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "input_subject": str(args.subject), "gdb_status": status.returncode,
        "niterations_patch_seen": f"INSTALLED_RAM_NITER={args.iterations}" in content,
        "first_pass_stop_seen": "INSTALLED_RAM_AFTER_FIRST_OUTER" in content,
        "output": str(output), "output_exists": output.is_file(),
        "accepted_lines": [line for line in content.splitlines()
                           if line[:3].isdigit() and line[3:8] == ": dt:"],
        "command": command,
    }
    (args.out / "capture_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "gdb_status", "niterations_patch_seen", "first_pass_stop_seen",
        "output_exists", "binary_sha256")}, indent=2))


if __name__ == "__main__":
    main()
