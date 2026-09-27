"""Connect one T1 to FreeSurfer's initial brainmask in Python."""

from __future__ import annotations

from pathlib import Path
import time

from .input_n4_chain import run_input_n4_chain
from .mri_mask_gpu import mask_volume
from .normalization import normalize_t1


def run_input_brainmask_chain(t1: str | Path, subject_dir: str | Path,
                              weights_dir: str | Path, assets_dir: str | Path,
                              *, n4_binary: str | Path, device: str = "cpu",
                              threads: int = 4) -> dict:
    """Produce orig, synthstrip, Talairach, nu, T1 and initial brainmask."""
    result = run_input_n4_chain(t1, subject_dir, weights_dir, assets_dir,
                                n4_binary=n4_binary, device=device, threads=threads)
    mri = Path(subject_dir) / "mri"
    t1_out = mri / "T1.mgz"
    started = time.perf_counter()
    normalized = normalize_t1(result["nu"], result["talairach_xfm"],
                              t1_out, device=device)
    normalize_seconds = time.perf_counter() - started
    brainmask = mri / "brainmask.mgz"
    started = time.perf_counter()
    mask_volume(t1_out, result["synthstrip"], brainmask, device=device)
    mask_seconds = time.perf_counter() - started
    return {**result, "T1": str(t1_out), "brainmask": str(brainmask),
            "normalize_first_seconds": normalize_seconds,
            "normalize_first_report": normalized,
            "brainmask_seconds": mask_seconds}
