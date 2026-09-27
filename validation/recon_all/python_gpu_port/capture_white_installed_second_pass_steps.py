"""Capture each accepted step in the installed white placement second pass."""

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
    for section in ("mri", "surf"):
        (args.out / section).mkdir(parents=True, exist_ok=True)
    for name in ("brain.finalsurfs.mgz", "wm.mgz", "aseg.presurf.mgz"):
        link = args.out / "mri" / name
        if not link.exists():
            link.symlink_to(args.subject / "mri" / name)
    for name in ("lh.orig", "autodet.gw.stats.lh.dat"):
        shutil.copy2(args.subject / "surf" / name, args.out / "surf" / name)
    script = args.out / "capture.gdb"
    script.write_text(f"""set pagination off
set confirm off
set $outer=0
set $returns=0
break *0x4176ab
commands 1
 silent
 set $outer=$outer+1
 if $outer==2
  set $surface=*(char**)$r13
  printf "SECOND_PASS_STARTED_NVERTICES=%d\\n",*(int*)($surface+4)
 end
 continue
end
break *0x4c4d7a
commands 2
 silent
 if $outer==2 && $rdx>=18
  set $verts=*(char**)($surface+40)
  eval "dump binary memory {args.out}/step%03d.raw $verts ($verts+464*106622)", $rdx
  printf "SECOND_PASS_STEP_SAVED=%d\\n",$rdx
 end
 continue
end
break *0x4176b0
commands 3
 silent
 set $returns=$returns+1
 if $returns==2
  printf "SECOND_PASS_COMPLETE\\n"
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
               OMP_NUM_THREADS="4")
    start = time.perf_counter()
    with (args.out / "capture.log").open("w") as log:
        status = subprocess.run(["gdb", "--batch", "-x", str(script), "--args", *command],
                                cwd=args.out / "mri", env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    content = (args.out / "capture.log").read_text(errors="replace")
    files = sorted(args.out.glob("step*.raw"))
    report = {
        "reference": "installed FreeSurfer 8.2, second-pass accepted-step RAM only",
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "subject": str(args.subject),
        "input_sha256": {str(args.subject / section / name): hashlib.sha256(
            (args.subject / section / name).read_bytes()).hexdigest()
            for section, names in (("mri", ("brain.finalsurfs.mgz", "wm.mgz", "aseg.presurf.mgz")),
                                   ("surf", ("lh.orig", "autodet.gw.stats.lh.dat")))
            for name in names},
        "command": command, "gdb_status": status.returncode,
        "second_pass_started": "SECOND_PASS_STARTED_NVERTICES=106622" in content,
        "second_pass_complete": "SECOND_PASS_COMPLETE" in content,
        "saved_step_messages": [int(x) for x in re.findall(r"SECOND_PASS_STEP_SAVED=(\d+)", content)],
        "files": {path.name: {"bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in files},
        "official_step_lines": [line for line in content.splitlines()
                                if re.match(r"0(?:1[89]|2[0-9]): dt:", line)],
        "seconds": time.perf_counter() - start,
    }
    (args.out / "capture_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("gdb_status", "second_pass_started", "second_pass_complete",
                       "saved_step_messages", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
