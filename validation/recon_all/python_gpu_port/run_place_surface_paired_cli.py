"""Run a frozen pial command with a selected native binary and fresh output paths."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-log", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    args = parser.parse_args()
    line = next(
        line for line in args.reference_log.read_text().splitlines()
        if line.startswith("/") and "mris_place_surface" in line
    )
    command = shlex.split(line)
    command[0] = str(args.binary)
    surf = args.out / "surf"
    surf.mkdir(parents=True, exist_ok=True)
    original = Path(command[command.index("--i") + 1])
    copied = surf / original.name
    shutil.copy2(original, copied)
    command[command.index("--i") + 1] = str(copied)
    command[command.index("--o") + 1] = str(surf / "lh.pial.cli")
    command[command.index("--target") + 1] = str(surf / "lh.pial.target")
    env = os.environ.copy()
    env.update(
        FREESURFER_HOME="/opt/reference/Freesurfer/8.2.0-1",
        SUBJECTS_DIR="/opt/reference/Freesurfer/8.2.0-1/subjects",
        FS_LICENSE=str(args.license),
    )
    with (args.out / "normal_cli.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(args.out / "normal_cli.log")


if __name__ == "__main__":
    main()
