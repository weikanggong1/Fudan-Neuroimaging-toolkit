"""Observe GPU load during a real profile; never change any device policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True,
                        help="new private CSV of GPU utilization and memory")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply the unchanged real profile command after --")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    # A separate observer is in the runner's process group. A job timeout
    # therefore also stops this observer. It does not contact other processes.
    with args.csv.open("x") as output:
        observer = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits", "-lms", "200",
        ], stdout=output, stderr=subprocess.DEVNULL)
        try:
            result = subprocess.run(command, check=False)
        finally:
            observer.terminate()
            observer.wait(timeout=10)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
