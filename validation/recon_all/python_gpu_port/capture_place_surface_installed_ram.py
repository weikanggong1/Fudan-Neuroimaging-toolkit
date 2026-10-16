"""Capture a fixed installed FreeSurfer pial optimizer pass from process RAM."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--first-pass-iterations", type=int, required=True)
    parser.add_argument("--debug-vertex", type=int)
    args = parser.parse_args()
    line = next(
        line for line in args.installed_log.read_text().splitlines()
        if line.startswith("/") and "mris_place_surface" in line
    )
    command = shlex.split(line)
    surf = args.out / "surf"
    surf.mkdir(parents=True, exist_ok=True)
    source_input = Path(command[command.index("--i") + 1])
    copied_input = surf / source_input.name
    shutil.copy2(source_input, copied_input)
    command[command.index("--i") + 1] = str(copied_input)
    hemisphere = "rh" if "--rh" in command else "lh"
    command[command.index("--o") + 1] = str(surf / f"{hemisphere}.pial.ram")
    command[command.index("--target") + 1] = str(surf / f"{hemisphere}.pial.target")
    command.append("--no-pin-medial-wall")
    if args.debug_vertex is not None:
        command.extend(("--debug-vertex", str(args.debug_vertex)))
    script = args.out / "installed_ram.gdb"
    script.write_text("""set pagination off
break *0x416970
commands 1
 silent
 set {int}($rbx+0x514)=FIRST_NITER
 printf \"INSTALLED_RAM_NITER=%d\\n\", {int}($rbx+0x514)
 continue
end
break *0x4176b0
commands 2
 silent
 printf \"INSTALLED_RAM_AFTER_FIRST_OUTER\\n\"
 set $rip=0x417776
 continue
end
run
""".replace("FIRST_NITER", str(args.first_pass_iterations)))
    env = os.environ.copy()
    env.update(
        FREESURFER_HOME="/opt/reference/Freesurfer/8.2.0-1",
        SUBJECTS_DIR="/opt/reference/Freesurfer/8.2.0-1/subjects",
        FS_LICENSE=str(args.license),
    )
    with (args.out / "installed_ram.log").open("w") as log:
        subprocess.run(
            ["gdb", "--batch", "-x", str(script), "--args", *command],
            env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    log = (args.out / "installed_ram.log").read_text()
    report = {
        "mode": "installed FreeSurfer binary RAM checkpoint; stop immediately after first outer optimizer, skip remaining outer passes, final intersection cleanup, and medial wall pin",
        "installed_command_argv": command,
        "first_pass_iteration_limit": args.first_pass_iterations,
        "debug_vertex": args.debug_vertex,
        "gdb_breakpoint_addresses": ["0x416970", "0x4176b0"],
        "gdb_version": subprocess.check_output(["gdb", "--version"], text=True).splitlines()[0],
        "recorded_iteration_limit": next((line for line in log.splitlines() if line.startswith("INSTALLED_RAM_NITER=")), ""),
        "first_outer_end_captured": "INSTALLED_RAM_AFTER_FIRST_OUTER" in log,
        "initial_line": next((line for line in log.splitlines() if line.startswith("000: dt:")), ""),
        "accepted_lines": [line for line in log.splitlines() if line[:3].isdigit() and line[3:8] == ": dt:" and not line.startswith("000:")],
        "debug_gradient_lines": [line for line in log.splitlines()
                                 if args.debug_vertex is not None and line.startswith(f"vno={args.debug_vertex} ")],
        "mesh": str(surf / f"{hemisphere}.pial.ram"),
    }
    (args.out / "installed_ram_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("recorded_iteration_limit", "first_outer_end_captured", "initial_line", "accepted_lines")}, indent=2))


if __name__ == "__main__":
    main()
