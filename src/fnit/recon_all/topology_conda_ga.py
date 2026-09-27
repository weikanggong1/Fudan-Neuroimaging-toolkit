"""Conda C++ topology GA with an exact Python sphere-centering preflight.

The pinned FreeSurfer 8.2 source build requires build_recon_all_topology_fnit.py's three
source fixes. This stage stops at orig.premesh; the caller runs remeshing next.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .topology_preflight_python import write_centered_topology_sphere


def run_topology_ga_conda(subject: str | Path, hemisphere: str,
                          binary: str | Path, assets: str | Path) -> dict:
    """Run one patched topology GA and write surf/{hemi}.orig.premesh.

    Inputs: subject surf orig/inflated/qsphere.nofix and mri brain/wm.mgz,
    patched mris_fix_topology_fnit, and FreeSurfer data-only asset root.
    Outputs: a centered temporary sphere, orig.premesh, scripts log, and a
    dictionary of their paths plus preflight/native elapsed seconds.
    """
    subject, binary, assets = Path(subject).resolve(), Path(binary).resolve(), Path(assets).resolve()
    if hemisphere not in ("lh", "rh"):
        raise ValueError("hemisphere must be lh or rh")
    if binary.name != "mris_fix_topology_fnit" or not binary.is_file():
        raise ValueError("use the validated mris_fix_topology_fnit build")
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    surf, scripts, mri = (subject / name for name in ("surf", "scripts", "mri"))
    for source in (surf / f"{hemisphere}.orig.nofix",
                   surf / f"{hemisphere}.inflated.nofix",
                   surf / f"{hemisphere}.qsphere.nofix",
                   mri / "brain.mgz", mri / "wm.mgz"):
        if not source.is_file():
            raise FileNotFoundError(source)
    scripts.mkdir(exist_ok=True)
    centered = surf / f"{hemisphere}.topology-centered.sphere"
    preflight = write_centered_topology_sphere(
        surf / f"{hemisphere}.qsphere.nofix", centered)
    log = scripts / f"{hemisphere}.topology-ga-fnit.log"
    output = surf / f"{hemisphere}.orig.premesh"
    env = dict(os.environ, SUBJECTS_DIR=str(subject.parent),
               FREESURFER_HOME=str(assets), FNIT_CENTERED_COORDS=str(centered),
               OMP_NUM_THREADS="1")
    command = [str(binary), "-threads", "1", "-mgz", "-sphere",
               "qsphere.nofix", "-inflated", "inflated.nofix", "-orig",
               "orig.nofix", "-out", "orig.premesh", "-ga", "-seed",
               "1234", "-threads", "1", subject.name, hemisphere]
    started = time.perf_counter()
    with log.open("w") as stream:
        subprocess.run(command, cwd=scripts, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    seconds = time.perf_counter() - started
    if not output.is_file() or "FNIT_CENTERED_SUBSTITUTION_USED" not in log.read_text():
        raise RuntimeError("patched topology GA did not produce its expected output")
    return {"hemisphere": hemisphere, "preflight": preflight,
            "output": str(output), "log": str(log), "command": command,
            "native_seconds": seconds}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=Path)
    parser.add_argument("hemisphere", choices=("lh", "rh"))
    parser.add_argument("binary", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = run_topology_ga_conda(
        args.subject, args.hemisphere, args.binary, args.assets)
    content = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
