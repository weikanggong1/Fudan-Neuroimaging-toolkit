"""UK Biobank single-subject EDDY input preparation."""

from __future__ import annotations
from pathlib import Path
from numbers import Integral
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_fill_holes, binary_erosion, label
from .._dmri import image_like, load_bvals
from ..topup.ukb import _best_b0, _load_b0_candidates
from . import TorchEDDY


def _brain_mask(image):
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data.mean(3)
    positive = data[data > 0]
    if positive.size == 0:
        raise ValueError("cannot estimate a brain mask from an empty image")
    threshold = 0.18 * np.percentile(positive, 98)
    mask = binary_fill_holes(binary_closing(data > threshold, iterations=2))
    labels, count = label(mask)
    if count:
        sizes = np.bincount(labels.reshape(-1))
        sizes[0] = 0
        mask = labels == sizes.argmax()
    mask = binary_erosion(mask, iterations=1)
    return mask


def prepare_ukb_eddy(raw_dir, topup_dir, output_dir, *, device=None, overwrite=False,
                     ref_scan_no=None):
    """Prepare EDDY inputs; optionally reuse TOPUP's selected AP b0 index.

    ``ref_scan_no`` is the zero-based AP volume index returned as ``ap_index``
    by ``prepare_ukb_topup``. Omitting it retains pairwise b0 selection.
    """
    raw_dir = Path(raw_dir)
    topup_dir = Path(topup_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [raw_dir / f"AP.{suffix}" for suffix in ("nii.gz", "bval", "bvec")] + [
        topup_dir / "acqparams.txt",
        topup_dir / "fieldmap_out_fieldcoef.nii.gz",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    bvals = load_bvals(raw_dir / "AP.bval")
    if ref_scan_no is not None:
        if (isinstance(ref_scan_no, bool) or not isinstance(ref_scan_no, Integral)
                or not 0 <= ref_scan_no < bvals.size
                or not bvals[ref_scan_no] < 100):
            raise ValueError("ref_scan_no must index an AP b<100 volume")
        ref_scan_no = int(ref_scan_no)
    index_path = output_dir / "eddy_index.txt"
    mask_path = output_dir / "nodif_brain_mask.nii.gz"
    existing = [p for p in (index_path, mask_path) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output exists: {existing[0]}; pass overwrite=True")
    np.savetxt(index_path, np.ones((1, bvals.size), dtype=int), fmt="%d")
    corrected_path = topup_dir / "fieldmap_iout.nii.gz"
    if corrected_path.exists():
        corrected = nib.load(str(corrected_path))
    else:
        corrected = nib.load(str(raw_dir / "AP.nii.gz"))
    mask = _brain_mask(corrected)
    nib.save(image_like(mask.astype(np.float32), corrected), str(mask_path))
    if ref_scan_no is None:
        ap_image, candidates, indices = _load_b0_candidates(raw_dir, "AP")
        voxel_sizes = tuple(float(v) for v in ap_image.header.get_zooms()[:3])
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        best, _ = _best_b0(candidates, voxel_sizes, selected_device)
        ref_scan_no = int(indices[best])
    return {
        "imain": raw_dir / "AP.nii.gz",
        "mask": mask_path,
        "acqp": topup_dir / "acqparams.txt",
        "index": index_path,
        "bvecs": raw_dir / "AP.bvec",
        "bvals": raw_dir / "AP.bval",
        "topup": topup_dir / "fieldmap_out",
        "ref_scan_no": ref_scan_no,
    }


def run_ukb_eddy(raw_dir, topup_dir, output_dir, *, device=None, overwrite=False):
    inputs = prepare_ukb_eddy(
        raw_dir, topup_dir, output_dir, device=device, overwrite=overwrite
    )
    result = TorchEDDY(device=device).run(
        **inputs, out=Path(output_dir) / "data", overwrite=overwrite
    )
    return result, inputs


__all__ = ["prepare_ukb_eddy", "run_ukb_eddy"]
