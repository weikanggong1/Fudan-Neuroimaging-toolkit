"""Connect the original T1 to FreeSurfer's N4-corrected nu.mgz in Python."""

from __future__ import annotations

from pathlib import Path
import time

from .input_talairach_chain import run_input_talairach_chain
from .n4_itk import correct_volume
from .n4_wrapper import make_nu


def run_input_n4_chain(t1: str | Path, subject_dir: str | Path,
                       weights_dir: str | Path, assets_dir: str | Path,
                       *, n4_binary: str | Path, device: str = "cpu",
                       threads: int = 4) -> dict:
    """Produce orig, synthstrip, talairach.xfm and nu.mgz from one T1.

    The neural calls use ``device``. The Conda C++ N4 binary runs on CPU. The subject
    directory must be empty, as required by ``run_input_talairach_chain``.
    """
    result = run_input_talairach_chain(
        t1, subject_dir, weights_dir, assets_dir,
        device=device, threads=threads)
    mri = Path(subject_dir) / "mri"
    scratch = mri / "tmp"
    scratch.mkdir(exist_ok=True)
    nu0 = scratch / "nu0.mgz"
    nu = mri / "nu.mgz"
    started = time.perf_counter()
    correct_volume(mri / "orig.mgz", nu0, binary=n4_binary)
    n4_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scale, histogram_bins = make_nu(
        mri / "orig.mgz", nu0, result["talairach_xfm"], nu)
    wrapper_seconds = time.perf_counter() - started
    return {**result, "nu0": str(nu0), "nu": str(nu),
            "n4_seconds": n4_seconds, "n4_wrapper_seconds": wrapper_seconds,
            "n4_global_mean_scale": scale, "n4_histogram_bins": histogram_bins}
