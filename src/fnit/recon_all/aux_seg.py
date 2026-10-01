"""PyTorch replacements for the recon-all MCA/dura and venous-sinus wrappers.

These commands need the existing SynthMorph affine LTA and FreeSurfer model
assets, but do not invoke FreeSurfer executables or TensorFlow.
"""

from __future__ import annotations

import argparse
import os
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from .sclimbic import _etiv_from_lta, LimbicUNet, segment_sclimbic_image


MCA_MODEL = "mca-dura.both-lh.nstd21.fhs.h5"
VSINUS_MODEL = "vsinus.no-sp.m.all.nstd10-070.h5"
VSINUS_ROWS = (
    (0, "Unknown"),
    (6111, "Left-Transverse-Sinus"),
    (6112, "Right-Transverse-Sinus"),
    (6115, "Straight-Sinus"),
    (6116, "Superior-Sinus-P"),
    (6117, "Superior-Sinus-D"),
)


def _assets_root(path: str | Path) -> Path:
    root = Path(path)
    return root.parent if root.name == "models" else root


def _models_root(root: Path) -> Path:
    return Path(os.environ.get("FS_TORCH_MODEL_DIR") or root / "models")


def _lta_matrix(path: str | Path) -> np.ndarray:
    lines = Path(path).read_text().splitlines()
    kind = next(line.split("#")[0].split("=")[1].strip()
                for line in lines if line.startswith("type"))
    if kind != "0":
        raise ValueError("SynthMorph prior registration must be a voxel-to-voxel LTA")
    first = next(i + 1 for i, line in enumerate(lines) if line.strip() == "1 4 4")
    stored = np.asarray([[float(v) for v in row.split()]
                         for row in lines[first:first + 4]], dtype=np.float64)
    # The stored matrix maps prior voxels to native voxels; pull sampling reverses it.
    return np.linalg.inv(stored)


def _resample_prior(prior: nib.spatialimages.SpatialImage,
                    native: nib.spatialimages.SpatialImage,
                    lta: np.ndarray, device: str) -> torch.Tensor:
    """Map an MNI prior to native voxels using the inverted voxel LTA."""
    matrix = torch.as_tensor(lta, dtype=torch.float64, device=device)
    source = torch.as_tensor(np.asanyarray(prior.dataobj).astype("float32"),
                             device=device)[None, None]
    result = torch.empty(native.shape[:3], dtype=torch.float32, device=device)
    nx, ny, nz = native.shape[:3]
    for first in range(0, nx, 24):
        coords = torch.stack(torch.meshgrid(
            torch.arange(first, min(first + 24, nx), device=device),
            torch.arange(ny, device=device),
            torch.arange(nz, device=device), indexing="ij"), -1)
        pos = coords.to(torch.float64) @ matrix[:3, :3].T + matrix[:3, 3]
        grid = torch.stack([2 * pos[..., axis] / (prior.shape[axis] - 1) - 1
                            for axis in (2, 1, 0)], -1).float()[None]
        result[first:first + len(coords)] = F.grid_sample(
            source, grid, mode="bilinear", padding_mode="zeros",
            align_corners=True)[0, 0]
    return result


def _round(x: torch.Tensor) -> torch.Tensor:
    """C++ round(), including half-integers away from zero."""
    return x.sign() * torch.floor(x.abs() + 0.5)


def _crop_start(prior: torch.Tensor, fov: int, kind: str) -> np.ndarray:
    hits = (prior >= 0.001).nonzero()
    if not hits.numel():
        raise ValueError("Registered prior contains no voxels above 0.001")
    if kind == "mca":
        centre = hits.double().mean(0)
        start = _round(centre - fov / 2).clamp(min=0)
    else:
        low = hits.min(0).values.double()
        high = hits.max(0).values.double()
        centre = low + (high - low) / 2
        start = _round(centre - np.floor((fov - 1) / 2))
    return start.cpu().numpy().astype(int)


def _extract(volume: np.ndarray, start: np.ndarray, fov: int) -> np.ndarray:
    output = np.zeros((fov,) * 3, dtype=volume.dtype)
    low = np.maximum(start, 0)
    high = np.minimum(start + fov, volume.shape)
    if np.any(high <= low):
        return output
    target = tuple(slice(int(a - s), int(b - s)) for a, b, s in zip(low, high, start))
    source = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    output[target] = volume[source]
    return output


def _paste(output: np.ndarray, crop: np.ndarray, start: np.ndarray) -> None:
    low = np.maximum(start, 0)
    high = np.minimum(start + crop.shape, output.shape)
    if np.any(high <= low):
        return
    target = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    source = tuple(slice(int(a - s), int(b - s)) for a, b, s in zip(low, high, start))
    output[target] += crop[source]


def _infer_crop(crop: np.ndarray, native: nib.spatialimages.SpatialImage,
                start: np.ndarray, model: Path, rows: tuple[tuple[int, str], ...],
                fov: int, device: str, *, model_cache: dict | None = None,
                precision_report: list | None = None) -> np.ndarray:
    """复用同次调用的权重/设备模型，在内存MGH上执行原裁剪推理。

    crop为float32三维强度，start为原网格起点（体素），native提供affine。
    rows、fov和device与sclimbic相同；局部model_cache以权重SHA和设备为键，
    仅跨MCA双侧复用，不持久驻留。返回crop原网格int32标签。
    不创建临时MGZ或ctab；权重、几何及推理错误向上传递。
    """
    affine = native.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ start
    source = nib.MGHImage(np.ascontiguousarray(crop), affine)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    key = (digest, str(torch.device(device)), True)
    loaded = None if model_cache is None else model_cache.get(key)
    if loaded is None:
        loaded = LimbicUNet.from_h5(model).to(device).eval()
        if model_cache is not None:
            model_cache[key] = loaded
    result = segment_sclimbic_image(
        source, model_path=model, rows=rows, fov=fov, device=device,
        model=loaded, precision_report=precision_report)
    return np.asarray(result.dataobj).astype(np.int32)


def _save_labels(path: str | Path, labels: np.ndarray,
                 native: nib.spatialimages.SpatialImage) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = native.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.MGHImage(np.ascontiguousarray(labels, dtype=np.float32),
                          None, header=header), str(output))
    return output


def mri_mcadura_seg(input_path: str | Path, output_path: str | Path,
                    synthmorphdir: str | Path, assets: str | Path, *,
                    device: str = "cpu", weights_dir: str | Path | None = None,
                    precision_report: list | None = None) -> Path:
    """Segment bilateral MCA-associated dura on an existing 1 mm recon-all MRI."""
    root = _assets_root(assets)
    native = nib.load(str(input_path))
    image = np.asarray(native.dataobj).astype(np.float32)
    lta = _lta_matrix(Path(synthmorphdir) / "reg.targ_to_invol.lta")
    output = np.zeros(native.shape[:3], dtype=np.int32)
    model_cache = {}
    for hemi, label in (("lh", 6101), ("rh", 6102)):
        prior = nib.load(str(root / "average" /
                             f"mca-dura.prior.warp.mni152.1.0mm.{hemi}.nii.gz"))
        start = _crop_start(_resample_prior(prior, native, lta, device), 80, "mca")
        crop = _extract(image, start, 80)
        if hemi == "rh":
            crop = crop[::-1].copy()
        models = Path(weights_dir) if weights_dir is not None else _models_root(root)
        seg = _infer_crop(crop, native, start, models / MCA_MODEL,
                          ((0, "Unknown"), (6101, "Left-Dura-MCA")), 72, device,
                          model_cache=model_cache, precision_report=precision_report)
        if hemi == "rh":
            seg = seg[::-1].copy()
            seg[seg == 6101] = 6102
        _paste(output, seg, start)
    return _save_labels(output_path, output, native)


def _write_vsinus_stats(path: str | Path, labels: np.ndarray, intensity: np.ndarray,
                        affine: np.ndarray, etiv: float | None) -> None:
    volume = abs(np.linalg.det(affine[:3, :3]))
    present = [(label, name) for label, name in VSINUS_ROWS[1:] if np.any(labels == label)]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        file.write("# Venous sinus segmentation statistics\n")
        if etiv is not None:
            file.write("# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total "
                       f"Intracranial Volume, {etiv:.6f}, mm^3\n")
        file.write(f"# VoxelVolume_mm3 {volume:g}\n")
        file.write(f"# NRows {len(present)}\n# NTableCols 10\n")
        file.write("# ColHeaders  Index SegId NVoxels Volume_mm3 StructName "
                   "Mean StdDev Min Max Range\n")
        for index, (label, name) in enumerate(present, 1):
            values = intensity[labels == label].astype(float)
            file.write(f"{index:3} {label:5} {len(values):9} {len(values)*volume:10.1f} "
                       f"{name:30} {values.mean():9.4f} "
                       f"{values.std(ddof=1) if len(values) > 1 else 0.0:9.4f} "
                       f"{values.min():9.4f} {values.max():9.4f} "
                       f"{np.ptp(values):9.4f}\n")


def mri_vsinus_seg(input_path: str | Path, output_path: str | Path,
                   synthmorphdir: str | Path, assets: str | Path, *,
                   ctxseg_path: str | Path | None = None,
                   stats_path: str | Path | None = None,
                   talairach_lta: str | Path | None = None,
                   device: str = "cpu",
                   weights_dir: str | Path | None = None,
                   precision_report: list | None = None) -> Path:
    """Segment venous sinuses and optionally suppress cortical overlap."""
    root = _assets_root(assets)
    native = nib.load(str(input_path))
    image = np.asarray(native.dataobj).astype(np.float32)
    lta = _lta_matrix(Path(synthmorphdir) / "reg.targ_to_invol.lta")
    prior = nib.load(str(root / "average" / "vsinus.no-sp.prior.mni152.1.0mm.mgz"))
    start = _crop_start(_resample_prior(prior, native, lta, device), 144, "vsinus")
    crop = _extract(image, start, 144)
    models = Path(weights_dir) if weights_dir is not None else _models_root(root)
    seg = _infer_crop(crop, native, start, models / VSINUS_MODEL,
                      VSINUS_ROWS, 144, device, precision_report=precision_report)
    output = np.zeros(native.shape[:3], dtype=np.int32)
    _paste(output, seg, start)
    if ctxseg_path is not None:
        cortex = nib.load(str(ctxseg_path))
        if cortex.shape[:3] != native.shape[:3] or not np.allclose(cortex.affine, native.affine):
            raise ValueError("Cortex segmentation must share the native MRI grid")
        output[np.isin(np.asarray(cortex.dataobj), (3, 42))] = 0
    result = _save_labels(output_path, output, native)
    if stats_path is not None:
        etiv = _etiv_from_lta(talairach_lta) if talairach_lta else None
        _write_vsinus_stats(stats_path, output, image, native.affine, etiv)
    return result


def _parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"mri_{command}_seg")
    parser.add_argument("--i")
    parser.add_argument("--o")
    parser.add_argument("--s")
    parser.add_argument("--sd", default=os.environ.get("SUBJECTS_DIR"))
    parser.add_argument("--synthmorphdir")
    parser.add_argument("--assets", default=os.environ.get("FNIT_ASSETS"))
    parser.add_argument("--weights", default=os.environ.get("FS_TORCH_MODEL_DIR"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    if command == "vsinus":
        parser.add_argument("--rca-synthseg", action="store_true")
        parser.add_argument("--ctxseg")
    return parser


def _args(command: str, argv: list[str] | None):
    parser = _parser(command)
    args = parser.parse_args(argv)
    if args.assets is None:
        parser.error("--assets or FNIT_ASSETS is required")
    if args.s and not args.sd:
        parser.error("--sd or SUBJECTS_DIR is required with --s")
    subject = Path(args.sd) / args.s if args.s else None
    if not args.i:
        if subject is None:
            parser.error("--i or --s is required")
        args.i = str(subject / "mri" / "nu.mgz")
    if not args.o:
        if subject is None:
            parser.error("--o or --s is required")
        name = "mca-dura.mgz" if command == "mcadura" else "vsinus.mgz"
        args.o = str(subject / "mri" / name)
    if not args.synthmorphdir:
        if subject is None:
            parser.error("--synthmorphdir is required without --s")
        args.synthmorphdir = str(subject / "mri" / "transforms" /
                                  "synthmorph.1.0mm.1.0mm")
    torch.set_num_threads(args.threads)
    args.device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    return args, subject


def main_mcadura(argv: list[str] | None = None) -> int:
    args, _ = _args("mcadura", argv)
    mri_mcadura_seg(args.i, args.o, args.synthmorphdir, args.assets,
                    device=args.device, weights_dir=args.weights)
    return 0


def main_vsinus(argv: list[str] | None = None) -> int:
    args, subject = _args("vsinus", argv)
    if args.rca_synthseg and args.ctxseg:
        _parser("vsinus").error("--rca-synthseg and --ctxseg are mutually exclusive")
    if args.rca_synthseg and subject is None:
        _parser("vsinus").error("--rca-synthseg requires --s")
    ctxseg = subject / "mri" / "synthseg.rca.mgz" if args.rca_synthseg else args.ctxseg
    mri_vsinus_seg(args.i, args.o, args.synthmorphdir, args.assets,
                   ctxseg_path=ctxseg,
                   stats_path=subject / "stats" / "vsinus.stats" if subject else None,
                   talairach_lta=subject / "mri" / "transforms" /
                   "talairach.xfm.lta" if subject else None,
                   device=args.device, weights_dir=args.weights)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aux_seg")
    parser.add_argument("command", choices=("mcadura", "vsinus"))
    args, remaining = parser.parse_known_args(argv)
    return (main_mcadura if args.command == "mcadura" else main_vsinus)(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
