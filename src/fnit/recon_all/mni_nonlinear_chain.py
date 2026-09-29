"""Generate recon-all's MNI152 nonlinear warp and registration check image."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit._transforms import AffineTransform
from fnit.synthmorph import SynthMorph

from .mni_aux_chain import TEMPLATE_DIR, write_mni_voxel_lta


def run_mni_nonlinear_chain(subject_dir: str | Path, weights_dir: str | Path,
                            assets_dir: str | Path, *, warp_convert: str | Path,
                            ca_register: str | Path, mri_convert: str | Path,
                            device: str = "cpu", threads: int = 4) -> dict:
    """Register the cropped T1, convert the warp, invert it and check resampling."""
    subject = Path(subject_dir)
    mri = subject / "mri"
    transforms = mri / "transforms/synthmorph.1.0mm.1.0mm"
    temporary = transforms / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    target_dir = Path(assets_dir) / TEMPLATE_DIR
    cropped_target = target_dir / "mni152.1.0mm.cropped.nii.gz"
    full_target = target_dir / "mni152.1.0mm.nii.gz"
    crop = transforms / "invol.crop.nii.gz"
    original = mri / "orig.mgz"
    affine = transforms / "aff.lta"
    weight = Path(weights_dir) / "synthmorph.deform.3.h5"
    for path in (cropped_target, full_target, crop, original, affine, weight):
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(threads)
    result = SynthMorph(weights=weights_dir, device=device, model="deform", extent=256)(
        crop, cropped_target, init=affine)
    ras_warp = temporary / "deform.mgz"
    result.transform.save(ras_warp)

    crop_image, original_image = nib.load(str(crop)), nib.load(str(original))
    write_mni_voxel_lta(temporary / "reg.crop-to-invol.lta",
                        np.linalg.inv(original_image.affine) @ crop_image.affine,
                        crop, original)
    AffineTransform(np.eye(4), source=nib.load(str(cropped_target)),
                    target=nib.load(str(full_target)), space="world").save(
                        temporary / "reg.crop-to-full.lta")
    forward = transforms / "warp.to.mni152.1.0mm.1.0mm.nii.gz"
    inverse = transforms / "warp.to.mni152.1.0mm.1.0mm.inv.nii.gz"
    check = transforms / "test.nii.gz"
    environment = dict(os.environ, FREESURFER_HOME=str(Path(assets_dir)))
    subprocess.run([str(warp_convert), "--inras", str(ras_warp),
                    "--insrcgeom", str(crop), "--outfswarp", str(forward),
                    "--vg-thresh", "1e-4", "--lta1-inv",
                    str(temporary / "reg.crop-to-invol.lta"), "--lta2",
                    str(temporary / "reg.crop-to-full.lta")],
                   env=environment, check=True)
    subprocess.run([str(ca_register), "-invert-and-save", str(forward),
                    str(inverse)], env=environment, check=True)
    subprocess.run([str(mri_convert), "-rt", "nearest", str(original),
                    "-at", str(forward), str(check)], env=environment, check=True)
    if nib.load(str(forward)).shape != (*nib.load(str(full_target)).shape, 1, 3) \
            or nib.load(str(inverse)).shape != (*original_image.shape, 1, 3) \
            or nib.load(str(check)).shape != nib.load(str(full_target)).shape:
        raise ValueError("MNI152 nonlinear outputs have unexpected grids")
    return {"forward": str(forward), "inverse": str(inverse), "check": str(check),
            "model": "pytorch-synthmorph-deform", "device": device}
