"""Run the FreeSurfer 8.2 white.preaparc placement using Conda C++."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time

from .autodet_gwstats_python import write_autodet_stats


def run_white_preaparc(subject_dir: str | Path, hemi: str,
                       binary: str | Path, assets_dir: str | Path,
                       *, threads: int = 4) -> dict:
    """Place one hemisphere's pre-aparc white surface from actual MRI/mesh inputs.

    Requires brain.finalsurfs, wm, aseg.presurf, orig.premesh and orig.
    Writes surf/H.white.preaparc, surf/autodet.gw.stats.H.dat and the
    mri/mrisps.wpa.mgz placement diagnostic. Returns paths and stage times.
    """
    if hemi not in ("lh", "rh"):
        raise ValueError("hemi must be lh or rh")
    subject = Path(subject_dir).resolve()
    binary = Path(binary).resolve()
    assets = Path(assets_dir).resolve()
    mri, surf = subject / "mri", subject / "surf"
    brain, wm, aseg = (mri / f"{name}.mgz"
                       for name in ("brain.finalsurfs", "wm", "aseg.presurf"))
    premesh, orig = (surf / f"{hemi}.{name}" for name in ("orig.premesh", "orig"))
    for path in (binary, brain, wm, aseg, premesh, orig):
        if not path.is_file():
            raise FileNotFoundError(path)
    stats = surf / f"autodet.gw.stats.{hemi}.dat"
    output, outvol = surf / f"{hemi}.white.preaparc", mri / "mrisps.wpa.mgz"
    tick = time.perf_counter()
    write_autodet_stats(brain, wm, premesh, stats, hemi)
    stats_seconds = time.perf_counter() - tick
    command = [str(binary), "--adgws-in", str(stats), "--wm", str(wm),
               "--threads", str(threads), "--invol", str(brain),
               f"--{hemi}", "--i", str(orig), "--o", str(output),
               "--white", "--seg", str(aseg), "--restore-255",
               "--nsmooth", "5", "--rip-bg-no-annot", "--rip-bg",
               "--rip-bg-lof", "--restore-255", "--outvol", str(outvol)]
    env = dict(os.environ, FREESURFER_HOME=str(assets),
               SUBJECTS_DIR=str(subject.parent))
    tick = time.perf_counter()
    subprocess.run(command, cwd=mri, env=env, check=True)
    place_seconds = time.perf_counter() - tick
    for path in (stats, output, outvol):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {"output": str(output), "stats": str(stats), "outvol": str(outvol),
            "stats_seconds": stats_seconds, "place_seconds": place_seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("hemi", choices=("lh", "rh"))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    print(run_white_preaparc(args.subject_dir, args.hemi,
                             args.binary, args.assets_dir, threads=args.threads))


if __name__ == "__main__":
    main()
