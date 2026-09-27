"""Real-data MRtrix SIFT2 precise track-to-fixel mapping benchmark.

Example (official FMLS LUT):
  python tools/benchmark_connectome_sift2_mapping.py --tracks tracks_10000.tck \
    --fixel-dir official_fixels_nifti1 --reference-tdi official_tdi_fixels/tdi.nii \
    --processing-mask proc_mask.nii.gz --mu official_mu.txt \
    --directions mrtrix_sift2_1281.npz --lookup fmls_lut.npz \
    --output sift2_mapping_fmls.public.json
Without --lookup, an official-peak nearest-direction LUT is used only as an
explicit approximation diagnostic; that result cannot establish parity.
Original command: tcksift2 tracks_10000.tck wm_fod_norm.mif weights.txt \
    -act 5tt_dwi.mif -nthreads 8 -debug -force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from fnit.connectome.sift2_mapping import map_streamlines_to_fixels


def _sha256(path: Path) -> str:
    """Return SHA-256 hex digest of one benchmark input file."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _peak_proxy_lut(
    voxel_ids: np.ndarray, count: np.ndarray, offset: np.ndarray,
    peaks: np.ndarray, sphere: np.ndarray, device: torch.device,
) -> torch.Tensor:
    """Approximate [V,1281] LUT by nearest official fixel peak per voxel.

    Inputs use NIfTI C-order voxel IDs and MRtrix global fixel order; output is
    uint8 on ``device``. This lacks FMLS lobe-boundary/dilation decisions and
    must only be used as a diagnostic comparison with ``tcksift2 -debug``.
    """
    table = torch.empty((len(voxel_ids), 1281), device=device, dtype=torch.uint8)
    dirs = torch.as_tensor(sphere, device=device, dtype=torch.float32)
    for start in range(0, len(voxel_ids), 512):
        stop = min(start + 512, len(voxel_ids))
        selected = voxel_ids[start:stop]
        n = count[selected]
        off = offset[selected]
        slots = off[:, None] + np.arange(int(n.max()))[None, :]
        vectors = torch.as_tensor(peaks[np.clip(slots, 0, len(peaks) - 1)], device=device, dtype=torch.float32)
        score = torch.einsum("bfc,dc->bfd", vectors, dirs).abs()
        bad = torch.as_tensor(np.arange(slots.shape[1])[None, :] >= n[:, None], device=device)
        score.masked_fill_(bad[:, :, None], -1)
        table[start:stop] = score.argmax(dim=1).to(torch.uint8)
    return table


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Report fixelwise Pearson r and absolute errors for aligned float arrays."""
    union = (reference != 0) | (candidate != 0)
    return {
        "pearson_all": float(np.corrcoef(reference, candidate)[0, 1]),
        "pearson_union_nonzero": float(np.corrcoef(reference[union], candidate[union])[0, 1]),
        "mae_all": float(np.mean(np.abs(reference - candidate))),
        "mae_union_nonzero": float(np.mean(np.abs(reference[union] - candidate[union]))),
        "relative_mae_reference_nonzero": float(np.mean(np.abs(reference[reference != 0] - candidate[reference != 0])) / np.mean(reference[reference != 0])),
        "sum_reference": float(reference.sum()),
        "sum_candidate": float(candidate.sum()),
        "nonzero_reference": int(np.count_nonzero(reference)),
        "nonzero_candidate": int(np.count_nonzero(candidate)),
    }


def _figure(path: Path, reference: np.ndarray, candidate: np.ndarray, voxel_fixel: np.ndarray,
            volume_shape: tuple[int, int, int]) -> None:
    """Save two real brain TDI slices and fixel density scatter as a PNG."""
    ref_vol = np.bincount(voxel_fixel, weights=reference, minlength=np.prod(volume_shape)).reshape(volume_shape)
    cand_vol = np.bincount(voxel_fixel, weights=candidate, minlength=np.prod(volume_shape)).reshape(volume_shape)
    z = int(np.argmax(ref_vol.sum(axis=(0, 1))))
    cap = np.quantile(ref_vol[:, :, z][ref_vol[:, :, z] > 0], .995)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, vol, label in zip(axes[:2], (ref_vol, cand_vol), ("MRtrix", "PyTorch")):
        ax.imshow(np.rot90(vol[:, :, z]), cmap="magma", vmin=0, vmax=cap)
        ax.set_title(f"{label} fixel TDI sum, z={z}")
        ax.axis("off")
    nonzero = np.flatnonzero((reference > 0) | (candidate > 0))
    rng = np.random.default_rng(0)
    selected = rng.choice(nonzero, size=min(10000, len(nonzero)), replace=False)
    axes[2].hexbin(reference[selected], candidate[selected], gridsize=65, bins="log", mincnt=1, cmap="viridis")
    ceiling = max(np.quantile(reference[selected], .99), np.quantile(candidate[selected], .99))
    axes[2].plot([0, ceiling], [0, ceiling], "r--", linewidth=1)
    axes[2].set(xlabel="MRtrix TDI", ylabel="PyTorch TDI", title="Same-input fixel density")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    """Load official real data, run mapping, and write JSON plus PNG evidence."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--fixel-dir", type=Path, required=True)
    p.add_argument("--reference-tdi", type=Path, required=True)
    p.add_argument("--processing-mask", type=Path, required=True)
    p.add_argument("--mu", type=Path, required=True)
    p.add_argument("--directions", type=Path, required=True)
    p.add_argument("--lookup", type=Path)
    p.add_argument("--records-output", type=Path, help="save official-order sparse records for downstream optimizer")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    index_path = args.fixel_dir / "index.nii.gz"
    peaks_path = args.fixel_dir / "directions.nii.gz"
    target_path = args.fixel_dir / "target.nii.gz"
    index_img = nib.load(index_path)
    index = np.asarray(index_img.dataobj, dtype=np.int64)
    shape = index.shape[:3]
    count = index[..., 0].ravel()
    offset = index[..., 1].ravel()
    voxels = np.flatnonzero(count).astype(np.int64)
    # MRtrix sparse fixel-vector images have reversed first-axis NIfTI storage.
    peaks = np.asarray(nib.load(peaks_path).dataobj, dtype=np.float32).reshape(-1, 3)[::-1].copy()
    target = np.asarray(nib.load(target_path).dataobj, dtype=np.float32).ravel()[::-1].copy()
    reference = np.asarray(nib.load(args.reference_tdi).dataobj, dtype=np.float32).ravel()[::-1].copy()
    mask = np.asarray(nib.load(args.processing_mask).dataobj, dtype=np.float32).ravel()
    sphere = np.load(args.directions)["directions"].astype(np.float64)
    if len(reference) != len(peaks) or len(reference) != len(target):
        raise ValueError("official fixel arrays have mismatched lengths")
    fixed_tracks = nib.streamlines.load(args.tracks)
    paths = [torch.as_tensor(np.asarray(path, dtype=np.float32).copy(), device=device) for path in fixed_tracks.streamlines]
    step_size = float(fixed_tracks.header["step_size"])
    if args.lookup:
        lookup = np.load(args.lookup)
        voxel_ids = lookup["voxel_ids"].astype(np.int64)
        first = lookup["first_fixel_index"].astype(np.int64)
        n = lookup["count"].astype(np.uint8)
        lut = torch.as_tensor(lookup["lookup_table"], device=device, dtype=torch.uint8)
        method = "PyTorch FMLS LUT; official voxel counts and target integrals matched"
        if not np.array_equal(voxel_ids, voxels) or not np.array_equal(n, count[voxels]):
            raise ValueError("FMLS LUT does not match official voxel occupancy and fixel count")
    else:
        voxel_ids, first, n = voxels, offset[voxels] + 1, count[voxels].astype(np.uint8)
        lut = _peak_proxy_lut(voxel_ids, count, offset, peaks, sphere, device)
        method = "official peak nearest-direction proxy; no FMLS parity claim"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = map_streamlines_to_fixels(
        paths, torch.as_tensor(index_img.affine, device=device, dtype=torch.float32), shape,
        torch.as_tensor(voxel_ids, device=device), torch.as_tensor(first, device=device),
        torch.as_tensor(n, device=device), lut, torch.as_tensor(sphere, device=device),
        step_size_mm=step_size, n_fixels=len(reference) + 1,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    internal_to_official = np.zeros(len(reference) + 1, dtype=np.int64)
    local = np.arange(int(count.max()))
    valid_local = local[None, :] < count[voxels, None]
    internal_slots = (first[:, None] + local[None, :])[valid_local]
    official_slots = (offset[voxels, None] + local[None, :] + 1)[valid_local]
    internal_to_official[internal_slots] = official_slots
    if len(np.unique(official_slots)) != len(reference) or len(np.unique(internal_slots)) != len(reference):
        raise ValueError("FMLS-to-official fixel permutation is incomplete")
    raw_internal = result.tdi_mm.cpu().numpy().astype(np.float64)
    raw = np.zeros(len(reference), dtype=np.float64)
    raw[official_slots - 1] = raw_internal[internal_slots]
    track_ids = result.track_index.cpu().numpy().astype(np.int64)
    fixel_ids = internal_to_official[result.fixel_index.cpu().numpy().astype(np.int64)]
    pair_ids = track_ids * (len(reference) + 1) + fixel_ids
    pair_unique, pair_counts = np.unique(pair_ids, return_counts=True)
    mu = float(args.mu.read_text().strip())
    candidate = mu * raw
    pm = np.zeros(len(reference), dtype=np.float64)
    voxel_fixel = np.zeros(len(reference), dtype=np.int64)
    slots = offset[voxels, None] + np.arange(int(count.max()))[None, :]
    valid = np.arange(slots.shape[1])[None, :] < count[voxels, None]
    pm[slots[valid]] = np.broadcast_to(mask[voxels, None], slots.shape)[valid]
    voxel_fixel[slots[valid]] = np.broadcast_to(voxels[:, None], slots.shape)[valid]
    if args.records_output:
        args.records_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.records_output, track_index=track_ids, fixel_index=fixel_ids,
                            length_mm=result.length_mm.cpu().numpy(), target=np.concatenate(([0.], target)),
                            processing_mask=np.concatenate(([0.], pm)), tdi_mm=np.concatenate(([0.], raw)))
    mu_candidate = float(np.dot(pm, target) / np.dot(pm, raw))
    fig_path = args.output.with_suffix(".png")
    _figure(fig_path, reference, candidate, voxel_fixel, shape)
    inputs = [args.tracks, index_path, peaks_path, target_path, args.reference_tdi,
              args.processing_mask, args.mu, args.directions]
    if args.lookup:
        inputs.append(args.lookup)
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm; fixed real tracks/FOD/fixels from MRtrix3 3.0.3-103-g026e850d",
        "mapping_method": method,
        "original_command": "tcksift2 tracks_10000.tck wm_fod_norm.mif weights.txt -act 5tt_dwi.mif -nthreads 8 -debug -force",
        "comparison": "same TCK, FOD voxel affine, official fixel count/peaks and before_tdi_fixel; FMLS global fixels permuted by matching voxel and local lobe; reference includes official mu",
        "input_sha256": {path.name: _sha256(path) for path in inputs},
        "n_tracks": len(paths), "n_fixels": len(reference), "n_contributions": len(result.track_index),
        "n_unique_track_fixel_pairs": len(pair_unique),
        "track_fixel_pairs_with_overflow_records": int((pair_counts > 1).sum()),
        "fixel_permutation_nonidentity": int(np.count_nonzero(internal_to_official[1:] != np.arange(1, len(reference) + 1))),
        "records_output": args.records_output.name if args.records_output else None,
        "step_size_mm": step_size, "upsample_ratio": int(np.ceil(step_size / (min(np.linalg.norm(index_img.affine[:3, :3], axis=0)) * .1))),
        "device": str(device), "dtype": "float32 coordinates, float64 lengths and direction lookup; TF32 enabled on CUDA; no float16",
        "mapping_seconds": seconds,
        "peak_cuda_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
        "official_tcksift2_total_seconds": 24.64,
        "timing_note": "MRtrix 24.64 s comes from corrected_mrtrix_fs5tt_act_adapted/stage_times.tsv and includes FMLS, mapping and optimization. PyTorch time starts after input loading and covers mapping only; no direct stage speed ratio.",
        "official_mu": mu, "candidate_mu_from_same_target_and_mask": mu_candidate,
        "tdi_scaled_by_official_mu": _metrics(reference.astype(np.float64), candidate),
        "figure": fig_path.name,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
