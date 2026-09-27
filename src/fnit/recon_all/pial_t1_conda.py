"""Place pial.T1 with the Conda-built FreeSurfer 8.2 executable."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time


def run_pial_t1(subject_dir: str | Path, hemi: str,
                binary: str | Path, assets_dir: str | Path,
                *, threads: int = 4) -> dict:
    """Run the logged pial command; write surf/H.pial.T1 and return its path/time.

    Requires the final white, brain.finalsurfs, wm, aseg.presurf,
    autodet thresholds, cortex and cortex+hipamyg labels, and aparc annotation.
    """
    if hemi not in ("lh", "rh"):
        raise ValueError("hemi must be lh or rh")
    if threads < 1:
        raise ValueError("threads must be positive")
    subject = Path(subject_dir).resolve()
    binary = Path(binary).resolve()
    assets = Path(assets_dir).resolve()
    mri, surf, label = (subject / name for name in ("mri", "surf", "label"))
    white = surf / f"{hemi}.white"
    stats = surf / f"autodet.gw.stats.{hemi}.dat"
    cortex = label / f"{hemi}.cortex.label"
    hipamyg = label / f"{hemi}.cortex+hipamyg.label"
    aparc = label / f"{hemi}.aparc.annot"
    brain, wm, aseg = (mri / f"{name}.mgz"
                       for name in ("brain.finalsurfs", "wm", "aseg.presurf"))
    for path in (binary, white, stats, cortex, hipamyg, aparc, brain, wm, aseg):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not os.access(binary, os.X_OK):
        raise PermissionError(f"binary is not executable: {binary}")
    if not assets.is_dir():
        raise NotADirectoryError(assets)
    output = surf / f"{hemi}.pial.T1"
    command = [
        str(binary), "--adgws-in", str(stats), "--seg", str(aseg),
        "--threads", str(threads), "--wm", str(wm), "--invol", str(brain),
        f"--{hemi}", "--i", str(white), "--o", str(output),
        "--pial", "--nsmooth", "0", "--rip-label", str(hipamyg),
        "--pin-medial-wall", str(cortex), "--aparc", str(aparc),
        "--repulse-surf", str(white), "--white-surf", str(white),
        "--restore-255",
    ]
    env = dict(os.environ, FREESURFER_HOME=str(assets),
               SUBJECTS_DIR=str(subject.parent))
    tick = time.perf_counter()
    subprocess.run(command, cwd=mri, env=env, check=True)
    seconds = time.perf_counter() - tick
    if not output.is_file() or output.stat().st_size == 0:
        raise FileNotFoundError(f"mris_place_surface produced no output: {output}")
    return {"output": str(output), "seconds": seconds}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("hemi", choices=("lh", "rh"))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    print(run_pial_t1(args.subject_dir, args.hemi, args.binary,
                      args.assets_dir, threads=args.threads))


if __name__ == "__main__":
    main()
