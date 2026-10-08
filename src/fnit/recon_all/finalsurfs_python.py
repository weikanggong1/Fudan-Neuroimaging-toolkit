"""Reproduce recon-all's five post-segmentation brain.finalsurfs edits."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from .mri_mask_gpu import mask_volume
from .wm_edits_python import fix_ento_wm


def _fix_ento_wm_for_device(input_file: Path, label_file: Path,
                            output_file: Path, *, level: int,
                            left_value: int, right_value: int,
                            device: str) -> int:
    """Apply the validated edit kernel on the requested device.

    The CUDA implementation has the same file contract and scan-order tie
    rule as :func:`wm_edits_python.fix_ento_wm`. Keep the CPU implementation
    as the explicit fallback for CPU runs; do not silently move a CUDA run
    back to CPU because that hides a device-selection bug in recon-all.
    """
    if str(device).startswith("cuda"):
        from .wm_edits_gpu import fix_ento_wm_gpu
        return fix_ento_wm_gpu(
            input_file, label_file, output_file, level=level,
            left_value=left_value, right_value=right_value, device=device)
    return fix_ento_wm(
        input_file, label_file, output_file, level=level,
        left_value=left_value, right_value=right_value)


def run_finalsurfs(subject_dir: str | Path, *, device: str = "cpu") -> Path:
    """Create ``mri/brain.finalsurfs.mgz`` from completed MRI segmentations.

    Requires ``brain``, ``brainmask``, ``mca-dura``, ``vsinus``, ``entowm``
    and ``aseg.presurf`` in the subject's ``mri`` directory. Also writes the
    pre-manual-edit checkpoint ``brain.finalsurfs.manedit.mgz``. Returns the final
    MGZ path. MRI images must share the conformed voxel grid.
    """
    mri = Path(subject_dir) / "mri"
    required = ("brain", "brainmask", "mca-dura", "vsinus", "entowm",
                "aseg.presurf")
    missing = [name for name in required if not (mri / f"{name}.mgz").is_file()]
    if missing:
        raise FileNotFoundError(f"missing finalsurfs inputs: {', '.join(missing)}")
    output = mri / "brain.finalsurfs.mgz"
    mask_volume(mri / "brain.mgz", mri / "brainmask.mgz", output,
                threshold=5, device=device)
    for name in ("mca-dura", "vsinus"):
        mask_volume(output, mri / f"{name}.mgz", output, invert=True,
                    outside_value=1, device=device)
    _fix_ento_wm_for_device(output, mri / "entowm.mgz", output, level=2,
                             left_value=255, right_value=255, device=device)
    # ACJ is a separate semantic operation. The CUDA path computes the
    # 7030/7031 junction labels before applying the level-3 edit.
    if str(device).startswith("cuda"):
        from .wm_edits_gpu import fix_ento_wm_gpu
        fix_ento_wm_gpu(output, mri / "aseg.presurf.mgz", output, level=3,
                        left_value=255, right_value=255, device=device, acj=True)
    else:
        fix_ento_wm(output, mri / "aseg.presurf.mgz", output, level=3,
                    left_value=255, right_value=255, acj=True)
    shutil.copyfile(output, mri / "brain.finalsurfs.manedit.mgz")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(run_finalsurfs(args.subject_dir, device=args.device))


if __name__ == "__main__":
    main()
