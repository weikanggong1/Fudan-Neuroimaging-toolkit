"""Connect one T1 to FreeSurfer's atlas-normalized norm.mgz in Python."""

from __future__ import annotations

from pathlib import Path
import time

from .ca_normalize_python import run_ca_normalize
from .input_brainmask_chain import run_input_brainmask_chain
from .mri_em_register_python import register_t1


def run_input_ca_normalize_chain(t1: str | Path, subject_dir: str | Path,
                                 weights_dir: str | Path, assets_dir: str | Path,
                                 *, n4_binary: str | Path, device: str = "cpu",
                                 threads: int = 4) -> dict:
    """Produce nu, initial brainmask, Talairach LTA, norm and control points."""
    atlas = Path(assets_dir) / "average/RB_all_2020-01-02.gca"
    if not atlas.is_file():
        raise FileNotFoundError(atlas)
    result = run_input_brainmask_chain(t1, subject_dir, weights_dir, assets_dir,
                                       n4_binary=n4_binary, device=device, threads=threads)
    mri = Path(subject_dir) / "mri"
    lta = mri / "transforms/talairach.lta"
    started = time.perf_counter()
    registration = register_t1(result["nu"], atlas, result["brainmask"], lta)
    registration_seconds = time.perf_counter() - started
    norm, controls = mri / "norm.mgz", mri / "ctrl_pts.mgz"
    started = time.perf_counter()
    normalization = run_ca_normalize(result["nu"], result["brainmask"],
                                     atlas, lta, norm, controls)
    ca_seconds = time.perf_counter() - started
    return {**result, "talairach_lta": str(lta), "norm": str(norm),
            "ctrl_pts": str(controls), "em_registration_report": registration,
            "em_registration_seconds": registration_seconds,
            "ca_normalize_report": normalization,
            "ca_normalize_seconds": ca_seconds}
