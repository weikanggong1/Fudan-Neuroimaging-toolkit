"""Capture installed FreeSurfer white vertices just before the third outer pass."""

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
    raw = args.out / "vertices.raw"
    script = args.out / "capture.gdb"
    script.write_text(f"""set pagination off
set confirm off
set $calls=0
break *0x4176ab
commands 1
 silent
 set $calls=$calls+1
 if $calls==3
  set $mris=*(char**)$r13
  printf "THIRD_PASS_BOUNDARY_NVERTICES=%d\\n",*(int*)($mris+4)
  set $verts=*(char**)($mris+40)
  dump binary memory {raw} $verts ($verts+464*106622)
  printf "THIRD_PASS_BOUNDARY_DUMPED\\n"
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
    report = {
        "reference": "installed FreeSurfer 8.2; diagnostic RAM before outer pass three",
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "input_sha256": {str(args.subject / section / name): hashlib.sha256(
            (args.subject / section / name).read_bytes()).hexdigest()
            for section, names in (("mri", ("brain.finalsurfs.mgz", "wm.mgz", "aseg.presurf.mgz")),
                                   ("surf", ("lh.orig", "autodet.gw.stats.lh.dat")))
            for name in names},
        "command": command, "gdb_status": status.returncode,
        "boundary_seen": "THIRD_PASS_BOUNDARY_DUMPED" in content,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest() if raw.is_file() else None,
        "raw_size": raw.stat().st_size if raw.is_file() else None,
        "seconds": time.perf_counter() - start,
    }
    (args.out / "capture_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("gdb_status", "boundary_seen", "raw_size", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
