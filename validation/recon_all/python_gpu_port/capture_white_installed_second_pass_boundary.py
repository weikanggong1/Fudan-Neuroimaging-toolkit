"""Capture the installed FreeSurfer white optimizer immediately before pass two."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--after-first-step", action="store_true")
    args = parser.parse_args()
    for section in ("mri", "surf"):
        (args.out / section).mkdir(parents=True, exist_ok=True)
    for name in ("brain.finalsurfs.mgz", "wm.mgz", "aseg.presurf.mgz"):
        link = args.out / "mri" / name
        if not link.exists():
            link.symlink_to(args.subject / "mri" / name)
    for name in ("lh.orig", "autodet.gw.stats.lh.dat"):
        shutil.copy2(args.subject / "surf" / name, args.out / "surf" / name)
    raw = args.out / "vertices.raw"
    script = args.out / "capture.gdb"
    if args.after_first_step:
        script_text = f"""set pagination off
set confirm off
set $calls=0
set $returns=0
break *0x4176ab
commands 1
 silent
 set $calls=$calls+1
 if $calls==2
  set {{int}}($rbx+0x514)=1
  printf "SECOND_STEP_NITER=%d\\n",{{int}}($rbx+0x514)
 end
 continue
end
break *0x4176b0
commands 2
 silent
 set $returns=$returns+1
 if $returns==2
  set $mris=*(char**)$r13
  printf "SECOND_STEP_NVERTICES=%d\\n",*(int*)($mris+4)
  set $verts=*(char**)($mris+40)
  dump binary memory {raw} $verts ($verts+464*106622)
  printf "SECOND_STEP_DUMPED\\n"
  quit
 end
 continue
end
run
"""
    else:
        script_text = f"""set pagination off
set confirm off
set $calls=0
break *0x4176ab
commands 1
 silent
 set $calls=$calls+1
 if $calls==2
  set $mris=*(char**)$r13
  printf "SECOND_PASS_BOUNDARY_NVERTICES=%d\\n",*(int*)($mris+4)
  set $verts=*(char**)($mris+40)
  dump binary memory {raw} $verts ($verts+464*106622)
  printf "SECOND_PASS_BOUNDARY_DUMPED\\n"
  quit
 end
 continue
end
run
"""
    script.write_text(script_text)
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
    report = {
        "reference": "installed FreeSurfer 8.2; diagnostic RAM after pass two first step"
                     if args.after_first_step else "installed FreeSurfer 8.2; diagnostic RAM before outer pass two",
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "input_sha256": {str(args.subject / section / name): hashlib.sha256(
            (args.subject / section / name).read_bytes()).hexdigest()
            for section, names in (("mri", ("brain.finalsurfs.mgz", "wm.mgz", "aseg.presurf.mgz")),
                                   ("surf", ("lh.orig", "autodet.gw.stats.lh.dat")))
            for name in names},
        "command": command, "gdb_status": status.returncode,
        "capture_kind": "after_first_step" if args.after_first_step else "before_pass_two",
        "boundary_seen": ("SECOND_STEP_DUMPED" if args.after_first_step
                          else "SECOND_PASS_BOUNDARY_DUMPED") in content,
        "one_step_patch_seen": "SECOND_STEP_NITER=1" in content if args.after_first_step else None,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest() if raw.is_file() else None,
        "raw_size": raw.stat().st_size if raw.is_file() else None,
        "seconds": time.perf_counter() - start,
    }
    (args.out / "capture_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
