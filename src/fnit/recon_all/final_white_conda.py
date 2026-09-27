"""Place final white geometry with the Conda-built FreeSurfer 8.2 executable."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time


def run_final_white(subject_dir: str | Path, hemi: str,
                    binary: str | Path, assets_dir: str | Path,
                    *, threads: int = 4) -> dict:
    """Run the logged final white command on one hemisphere.

    Requires brain.finalsurfs, wm, aseg.presurf, white.preaparc,
    cortex.label, aparc.annot and autodet.gw.stats.H.dat. Writes
    surf/H.white and mri/mrisps.white.mgz. Returns their paths and
    placement wall time.
    """
    if hemi not in ("lh", "rh"):
        raise ValueError("hemi must be lh or rh")
    if threads < 1:
        raise ValueError("threads must be positive")
    subject = Path(subject_dir).resolve()
    binary = Path(binary).resolve()
    assets = Path(assets_dir).resolve()
    mri, surf, label = (subject / name for name in ("mri", "surf", "label"))
    preaparc = surf / f"{hemi}.white.preaparc"
    stats = surf / f"autodet.gw.stats.{hemi}.dat"
    cortex = label / f"{hemi}.cortex.label"
    aparc = label / f"{hemi}.aparc.annot"
    brain, wm, aseg = (mri / f"{name}.mgz"
                       for name in ("brain.finalsurfs", "wm", "aseg.presurf"))
    for path in (binary, preaparc, stats, cortex, aparc, brain, wm, aseg):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not os.access(binary, os.X_OK):
        raise PermissionError(f"binary is not executable: {binary}")
    if not assets.is_dir():
        raise NotADirectoryError(assets)
    output, outvol = surf / f"{hemi}.white", mri / "mrisps.white.mgz"
    command = [
        str(binary), "--adgws-in", str(stats), "--seg", str(aseg),
        "--threads", str(threads), "--wm", str(wm), "--invol", str(brain),
        f"--{hemi}", "--i", str(preaparc), "--o", str(output),
        "--white", "--nsmooth", "0", "--rip-label", str(cortex),
        "--rip-bg", "--rip-surf", str(preaparc), "--aparc", str(aparc),
        "--restore-255", "--restore-255", "--outvol", str(outvol),
        "--rip-bg-lof",
    ]
    env = dict(os.environ, FREESURFER_HOME=str(assets),
               SUBJECTS_DIR=str(subject.parent))
    tick = time.perf_counter()
    subprocess.run(command, cwd=mri, env=env, check=True)
    seconds = time.perf_counter() - tick
    for path in (output, outvol):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"mris_place_surface produced no output: {path}")
    return {"output": str(output), "outvol": str(outvol), "seconds": seconds}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("hemi", choices=("lh", "rh"))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    print(run_final_white(args.subject_dir, args.hemi, args.binary,
                          args.assets_dir, threads=args.threads))


if __name__ == "__main__":
    main()
