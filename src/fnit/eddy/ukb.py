"""UK Biobank single-subject EDDY input preparation."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from numbers import Integral
import time
import nibabel as nib
import numpy as np
import torch
from .._dmri import configure_device, image_like, load_bvals
from ..synthstrip import SynthStrip
from ..topup.ukb import _best_b0, _load_b0_candidates
from ..weights import WEIGHT_FILES, resolve_weights, verify_file
from . import TorchEDDY


def _get_synthstrip(weights, device, cache=None):
    """Resolve and verify the standard checkpoint; cache only within a caller."""
    device = configure_device(device)
    filename = "synthstrip.1.pt"
    path = Path(resolve_weights(filename, explicit=weights))
    status = path.stat()
    key = (str(path.resolve()), str(device), status.st_size, status.st_mtime_ns)
    if cache is not None and cache.get("key") == key:
        return cache["model"], dict(cache["weights"]), True
    _, expected_size, expected_sha256 = WEIGHT_FILES[filename]
    if not verify_file(path, expected_size, expected_sha256):
        raise ValueError("SynthStrip checkpoint does not match the registered "
                         "synthstrip.1.pt size and SHA-256")
    model = SynthStrip(weights=path, device=device, no_csf=False)
    metadata = {
        "filename": filename,
        "size_bytes": expected_size,
        "sha256": expected_sha256,
        "verified": True,
        "resolution": "explicit" if weights is not None else "fnit_local_resolver",
    }
    if cache is not None:
        cache.clear()
        cache.update(key=key, model=model, weights=metadata)
    return model, metadata, False


def _prepare_brain_mask(image, output_dir, *, device=None, synthstrip_weights=None,
                        _synthstrip_cache=None, source, input_volumes, started=None):
    """Infer a b0 mask on the input grid and save anonymous provenance/QC."""
    device = configure_device(device)
    started = time.perf_counter() if started is None else started
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    values = np.asarray(image.dataobj, dtype=np.float32)
    if (values.ndim != 3 or not np.isfinite(values).all()
            or not np.any(values > 0)):
        raise ValueError("SynthStrip requires a finite, nonempty 3D b0 mean image")
    image = image_like(values, image)
    stage_started = time.perf_counter()
    model, weights, reused = _get_synthstrip(
        synthstrip_weights, device, _synthstrip_cache
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    stripped = model(image, border=1, fill=None)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - stage_started
    mask = np.asarray(stripped.mask.dataobj)
    if (mask.shape != values.shape
            or not np.allclose(stripped.mask.affine, image.affine, atol=1e-5, rtol=0)):
        raise ValueError("SynthStrip mask must retain the b0 input grid")
    if not np.isin(mask, (0, 1)).all() or not np.any(mask):
        raise ValueError("SynthStrip produced an empty or nonbinary brain mask")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_started = time.perf_counter()
    nib.save(image_like(mask.astype(np.float32), image),
             str(output_dir / "nodif_brain_mask.nii.gz"))
    save_seconds = time.perf_counter() - stage_started
    qc = {
        "algorithm": "fnit.synthstrip.SynthStrip",
        "input": {"source": source, "volumes_averaged": int(input_volumes),
                  "dtype": "float32", "external_intensity_normalization": False,
                  "mean_sha256": hashlib.sha256(values.tobytes()).hexdigest()},
        "weights": weights,
        "model_reused": reused,
        "parameters": {"border_mm": 1, "no_csf": False, "fill": None},
        "geometry": {"shape": list(values.shape), "affine": image.affine.tolist(),
                     "voxel_sizes_mm": list(map(float, image.header.get_zooms()[:3])),
                     "mask_matches_input_grid": True},
        "brain_voxels": int(np.count_nonzero(mask)),
        "device": str(device),
        "dtype": "float32",
        "tf32": device.type == "cuda",
        "timings_seconds": {"weights_resolve_verify_load": setup_seconds,
                            "synthstrip": inference_seconds,
                            "mask_save": save_seconds},
        "elapsed_seconds": time.perf_counter() - started,
        "timing_scope": "b0 input read and float32 mean, checkpoint resolution/"
                        "verification/load, SynthStrip preprocessing/inference/"
                        "native-grid mask, mask NIfTI save; excludes QC JSON write",
        "cuda_peak_allocated_gb": (torch.cuda.max_memory_allocated(device) / 1e9
                                   if device.type == "cuda" else None),
        "cuda_peak_reserved_gb": (torch.cuda.max_memory_reserved(device) / 1e9
                                  if device.type == "cuda" else None),
        "memory_scope": "process PyTorch allocator peak reset at mask preparation; "
                        "includes live tensors/model and existing allocator reserve; "
                        "excludes CUDA context and other processes",
    }
    (output_dir / "nodif_brain_mask_report.json").write_text(
        json.dumps(qc, indent=2) + "\n", encoding="utf-8"
    )
    return qc


def prepare_ukb_eddy(raw_dir, topup_dir, output_dir, *, device=None, overwrite=False,
                     ref_scan_no=None, synthstrip_weights=None, _synthstrip_cache=None):
    """Prepare EDDY inputs with a verified PyTorch SynthStrip b0 brain mask.

    ``ref_scan_no`` is the zero-based AP volume index returned as ``ap_index``
    by ``prepare_ukb_topup``. Omitting it retains pairwise b0 selection.
    ``synthstrip_weights`` is the standard checkpoint file/directory, or None
    to use FNIT's local resolver. No weights are downloaded during inference.
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
    existing = [p for p in (index_path, mask_path,
                            output_dir / "nodif_brain_mask_report.json") if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output exists: {existing[0]}; pass overwrite=True")
    np.savetxt(index_path, np.ones((1, bvals.size), dtype=int), fmt="%d")
    started = time.perf_counter()
    ap_image = nib.load(str(raw_dir / "AP.nii.gz"))
    if len(ap_image.shape) != 4 or ap_image.shape[3] != bvals.size:
        raise ValueError("AP image and bvals have inconsistent volume counts")
    corrected_path = topup_dir / "fieldmap_iout.nii.gz"
    if corrected_path.exists():
        corrected = nib.load(str(corrected_path))
        data = np.asarray(corrected.dataobj, dtype=np.float32)
        if data.ndim != 4 or data.shape[3] == 0:
            raise ValueError("TOPUP fieldmap_iout must contain corrected b0 volumes")
        mean = data.mean(3, dtype=np.float32)
        source = "topup_corrected_b0_mean"
        input_volumes = data.shape[3]
    else:
        corrected = ap_image
        indices = np.flatnonzero(bvals < 100)
        if indices.size == 0:
            raise ValueError("AP acquisition contains no b<100 volume")
        # Decode only b0 volumes, rather than averaging the diffusion-weighted data.
        mean = np.zeros(ap_image.shape[:3], dtype=np.float32)
        for index in indices:
            mean += np.asarray(ap_image.dataobj[..., index], dtype=np.float32)
        mean /= indices.size
        source = "raw_ap_b_lt_100_mean"
        input_volumes = indices.size
    if (corrected.shape[:3] != ap_image.shape[:3]
            or not np.allclose(corrected.affine, ap_image.affine, atol=1e-5, rtol=0)):
        raise ValueError("TOPUP b0 grid must match AP DWI for the EDDY mask")
    _prepare_brain_mask(
        image_like(mean, corrected), output_dir, device=device,
        synthstrip_weights=synthstrip_weights, _synthstrip_cache=_synthstrip_cache,
        source=source, input_volumes=input_volumes, started=started,
    )
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


def run_ukb_eddy(raw_dir, topup_dir, output_dir, *, device=None, overwrite=False,
                 synthstrip_weights=None):
    inputs = prepare_ukb_eddy(
        raw_dir, topup_dir, output_dir, device=device, overwrite=overwrite,
        synthstrip_weights=synthstrip_weights,
    )
    result = TorchEDDY(device=device).run(
        **inputs, out=Path(output_dir) / "data", overwrite=overwrite
    )
    return result, inputs


__all__ = ["prepare_ukb_eddy", "run_ukb_eddy"]
