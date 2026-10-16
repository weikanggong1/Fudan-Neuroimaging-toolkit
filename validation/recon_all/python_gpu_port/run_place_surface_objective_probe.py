"""Replay the fixed one-iteration LH pial command with a precision diagnostic."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-log", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--gradient-prefix", type=Path)
    parser.add_argument("--outer-prefix", type=Path)
    parser.add_argument("--original-parameters", action="store_true")
    args = parser.parse_args()
    line = next(
        line for line in args.previous_log.read_text().splitlines()
        if line.startswith("/") and "mris_place_surface_first_iteration" in line
    )
    command = shlex.split(line)
    command[0] = str(args.binary)
    if args.original_parameters:
        for option in ("--n_averages", "--subiters"):
            if option in command:
                offset = command.index(option)
                del command[offset:offset + 2]
    surf = args.out / "surf"
    surf.mkdir(parents=True, exist_ok=True)
    input_slot = command.index("--i") + 1
    copied_input = surf / Path(command[input_slot]).name
    shutil.copy2(command[input_slot], copied_input)
    command[input_slot] = str(copied_input)
    command[command.index("--o") + 1] = str(surf / "lh.pial.objective")
    command[command.index("--target") + 1] = str(surf / "lh.pial.target")
    env = os.environ.copy()
    env["FREESURFER_HOME"] = "/opt/reference/Freesurfer/8.2.0-1"
    env["SUBJECTS_DIR"] = "/opt/reference/Freesurfer/8.2.0-1/subjects"
    env["FS_LICENSE"] = str(args.license)
    if args.gradient_prefix is not None:
        env["PLACE_GRAD_PREFIX"] = str(args.gradient_prefix)
    if args.outer_prefix is not None:
        env["PLACE_OUTER_PREFIX"] = str(args.outer_prefix)
    with (args.out / "objective_probe.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for line in (args.out / "objective_probe.log").read_text().splitlines():
        if "PY_OBJ_REF" in line:
            print(line)


if __name__ == "__main__":
    main()
