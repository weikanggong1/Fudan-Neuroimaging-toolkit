"""MNI152, fsaverage and fsLR map conversion using RF-ANTs and HCP 2017 assets."""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import nibabel as nib
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


_FSAVG = {"3k": ("fsaverage4", 2562), "10k": ("fsaverage5", 10242),
          "41k": ("fsaverage6", 40962), "164k": ("fsaverage", 163842)}
_FSLR = {"32k": 32492, "59k": 59292, "164k": 163842}
_RF = "rf_ants"


def _device(name: str) -> torch.device:
    import torch

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def _density(space: str, density: str | None) -> int:
    choices = _FSAVG if space == "fsaverage" else _FSLR
    if density not in choices:
        raise ValueError(f"{space} density must be one of {tuple(choices)}")
    return choices[density][1] if space == "fsaverage" else choices[density]


def _gifti_values(path: str | Path, count: int) -> tuple[np.ndarray, bool]:
    image = nib.load(str(path))
    if not isinstance(image, nib.gifti.GiftiImage) or not image.darrays:
        raise ValueError(f"Expected a GIFTI metric or label: {path}")
    data = np.stack([np.asarray(frame.data) for frame in image.darrays], axis=1)
    if data.shape[0] != count:
        raise ValueError(f"{path}: expected {count} vertices, got {data.shape[0]}")
    is_label = all(frame.intent == nib.nifti1.intent_codes["NIFTI_INTENT_LABEL"]
                   for frame in image.darrays)
    return data, is_label


def _save_gifti(values: np.ndarray, path: Path, label: bool) -> Path:
    intent = "NIFTI_INTENT_LABEL" if label else "NIFTI_INTENT_SHAPE"
    dtype = np.int32 if label else np.float32
    image = nib.gifti.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(np.asarray(values[:, frame], dtype=dtype), intent=intent)
        for frame in range(values.shape[1])
    ])
    nib.save(image, str(path))
    return path


def _sphere_area(root: Path, space: str, density: str, hemi: str,
                 paired_space: str) -> tuple[Path, Path]:
    atlas = root / "hcp_2017" / "resample_fsaverage"
    if space == "fsaverage":
        prefix = _FSAVG[density][0]
        sphere = atlas / f"{prefix}_std_sphere.{hemi}.{density}_fsavg_{hemi}.surf.gii"
        area = atlas / f"{prefix}.{hemi}.midthickness_va_avg.{density}_fsavg_{hemi}.shape.gii"
    else:
        sphere = (atlas / f"fs_LR-deformed_to-fsaverage.{hemi}.sphere.{density}_fs_LR.surf.gii"
                  if paired_space == "fsaverage" else
                  root / "hcp_2017" / (f"{hemi}.sphere.{density}_fs_LR.surf.gii" if density != "164k"
                                         else f"fsaverage.{hemi}_LR.spherical_std.164k_fs_LR.surf.gii"))
        area = atlas / f"fs_LR.{hemi}.midthickness_va_avg.{density}_fs_LR.shape.gii"
    for path in (sphere, area):
        if not path.is_file():
            raise FileNotFoundError(path)
    return sphere, area


def _surface_resample(source: Path, destination: Path, root: Path,
                      source_space: str, source_density: str,
                      target_space: str, target_density: str,
                      hemi: str, label: bool, wb_command: str,
                      cpu_threads: int | None = None) -> Path:
    if source_space == target_space and source_density == target_density:
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        return destination
    wb = shutil.which(wb_command)
    if wb is None:
        raise FileNotFoundError(wb_command)
    src_sphere, src_area = _sphere_area(root, source_space, source_density, hemi, target_space)
    dst_sphere, dst_area = _sphere_area(root, target_space, target_density, hemi, source_space)
    command = "-label-resample" if label else "-metric-resample"
    environment = None
    if cpu_threads is not None:
        from ._hemisphere_parallel import workbench_environment
        environment = workbench_environment(cpu_threads)
    subprocess.run([wb, command, str(source), str(src_sphere), str(dst_sphere),
                    "ADAP_BARY_AREA", str(destination), "-area-metrics", str(src_area),
                    str(dst_area)], check=True, capture_output=True, text=True,
                   env=environment)
    return destination


def _sample_volume(image: nib.spatialimages.SpatialImage, ras: np.ndarray,
                   device: torch.device, nearest: bool) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    values = np.asarray(image.dataobj, dtype=np.float32)
    if values.ndim == 3:
        values = values[..., None]
    if values.ndim != 4:
        raise ValueError("MNI input must be a 3D or 4D NIfTI image")
    volume = torch.from_numpy(np.ascontiguousarray(values.transpose(3, 2, 1, 0)))[None].to(device)
    voxels = nib.affines.apply_affine(np.linalg.inv(image.affine), ras).astype(np.float32)
    output = np.empty((len(ras), values.shape[3]), np.float32)
    for start in range(0, len(ras), 65536):
        chunk = voxels[start:start + 65536]
        grid = torch.from_numpy(chunk).to(device)
        dims = np.asarray(image.shape[:3], np.float32)
        grid = (2 * grid / torch.as_tensor(dims - 1, device=device) - 1).reshape(1, -1, 1, 1, 3)
        sampled = F.grid_sample(volume, grid, mode="nearest" if nearest else "bilinear",
                                padding_mode="zeros", align_corners=True)
        output[start:start + len(chunk)] = sampled[0, :, :, 0, 0].T.cpu().numpy()
    return output


def _volume_to_fsaverage(source: Path, output: Path, root: Path,
                         density: str, device: torch.device, label: bool) -> tuple[Path, Path]:
    from scipy.io import loadmat

    count = _density("fsaverage", density)
    image = nib.load(str(source))
    result = []
    for hemi, side in (("L", "lh"), ("R", "rh")):
        path = root / _RF / f"{side}.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.mat"
        if not path.is_file():
            raise FileNotFoundError(path)
        ras = np.asarray(loadmat(path)["ras"].T[:count])
        values = _sample_volume(image, ras, device, label)
        result.append(_save_gifti(values, output / f"{hemi}.fsaverage.{density}.{'label' if label else 'func'}.gii", label))
    return tuple(result)


def _surface_to_volume(sources: tuple[Path, Path], output: Path, root: Path,
                       reference: Path, device: torch.device, label: bool) -> Path:
    from scipy.io import loadmat
    import torch
    import torch.nn.functional as F

    ref = nib.load(str(reference))
    if len(ref.shape) < 3 or len(ref.shape) > 4:
        raise ValueError("MNI reference must be a NIfTI image with a 3D grid")
    mapping = root / _RF / "allSub_fsaverage_to_FSL_MNI152_FS4.5.0_RF_ANTs_avgMapping.vertex.mat"
    mask_path = root / _RF / "FSL_MNI152_FS4.5.0_cortex_estimate.nii.gz"
    for path in (mapping, mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference_mask = nib.load(str(mask_path))
    if reference_mask.shape[:3] != (256, 256, 256):
        raise ValueError("RF-ANTs cortex mask must have a 256-cubed grid")
    maps = loadmat(mapping)
    if device.type == "cpu":
        return _surface_to_volume_cpu(sources, output, ref, reference_mask, maps, label)
    vertex_maps = []
    values = []
    nframes = None
    for hemi, path in (("lh", sources[0]), ("rh", sources[1])):
        data, _ = _gifti_values(path, 163842)
        if nframes is not None and data.shape[1] != nframes:
            raise ValueError("Left and right GIFTI files have different frame counts")
        nframes = data.shape[1]
        values.append(torch.from_numpy(np.ascontiguousarray(data.astype(np.float32))).to(device))
        # CBIG/MRIread stores the first two axes in MATLAB row-column order.
        vertices = np.ascontiguousarray(maps[f"{hemi}_vertex"].transpose(2, 0, 1), dtype=np.float32)
        vertex_maps.append(torch.from_numpy(vertices)[None, None].to(device))
    mask = np.asarray(reference_mask.dataobj, dtype=np.float32)
    if mask.ndim == 4:
        mask = mask[..., 0]
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 1, 0)))[None, None].to(device)
    shape = ref.shape[:3]
    result = np.zeros((*shape, nframes), np.float32)
    transform = np.linalg.inv(reference_mask.affine) @ ref.affine
    mode = "nearest"  # CBIG's published vertex map is nearest-vertex RF-ANTs.
    for z in range(shape[2]):
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        ijk = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))
        mapped = nib.affines.apply_affine(transform, ijk).astype(np.float32)
        grid = torch.from_numpy(2 * mapped / 255 - 1).to(device).reshape(1, shape[0], shape[1], 1, 3)
        # grid_sample uses output dimensions D,H,W; flattening preserves x,y order.
        valid = F.grid_sample(mask_tensor, grid, mode=mode, align_corners=True)[0, 0].reshape(-1) > 0.5
        layer = torch.zeros((x.size, nframes), device=device, dtype=torch.float32)
        for vertex_map, data in zip(vertex_maps, values):
            indices = F.grid_sample(vertex_map, grid, mode=mode, align_corners=True)[0, 0].reshape(-1).long()
            valid_hemi = valid & (indices > 0)
            layer[valid_hemi] += data[indices[valid_hemi] - 1]
        result[:, :, z, :] = layer.cpu().numpy().reshape(shape[0], shape[1], nframes)
    if label:
        result = np.rint(result).astype(np.int32)
    if nframes == 1:
        result = result[..., 0]
    _save_volume(result, ref, output, label)
    return output


def _save_volume(result: np.ndarray, reference: nib.spatialimages.SpatialImage,
                 output: Path, label: bool) -> None:
    # A cortex mask or atlas reference often has an integer storage dtype.
    # Preserve its geometry, while avoiding quantization of continuous maps.
    header = reference.header.copy()
    header.set_data_dtype(np.int32 if label else np.float32)
    nib.save(nib.Nifti1Image(result, reference.affine, header), str(output))


def _surface_to_volume_cpu(sources: tuple[Path, Path], output: Path,
                           reference: nib.spatialimages.SpatialImage,
                           reference_mask: nib.spatialimages.SpatialImage,
                           maps: dict, label: bool) -> Path:
    from ._space_conversion_cpu import project_surface_layer

    values = [_gifti_values(path, 163842)[0] for path in sources]
    if values[0].shape[1] != values[1].shape[1]:
        raise ValueError("Left and right GIFTI files have different frame counts")
    values = [np.ascontiguousarray(data, dtype=np.float32) for data in values]
    vertices = [np.ascontiguousarray(maps[f"{hemi}_vertex"], dtype=np.float32)
                for hemi in ("lh", "rh")]
    for vertex_map in vertices:
        if vertex_map.shape != (256, 256, 256):
            raise ValueError("RF-ANTs vertex maps must have a 256-cubed grid")
        if (not np.isfinite(vertex_map).all() or vertex_map.min() < 0
                or vertex_map.max() > 163842):
            raise ValueError("RF-ANTs vertex map contains invalid vertex indices")
    mask = np.asarray(reference_mask.dataobj, dtype=np.float32)
    if mask.ndim == 4:
        mask = mask[..., 0]
    mask = np.ascontiguousarray(mask)
    shape = reference.shape[:3]
    nframes = values[0].shape[1]
    result = np.zeros((*shape, nframes), np.float32)
    transform = np.linalg.inv(reference_mask.affine) @ reference.affine
    # The x/y plane and affine multiplication order match the Torch path.
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    ijk = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    with _cpu_thread_budget():
        for z in range(shape[2]):
            ijk[:, 2] = z
            mapped = nib.affines.apply_affine(transform, ijk).astype(np.float32)
            grid = np.ascontiguousarray(2 * mapped / 255 - 1, dtype=np.float32)
            layer = project_surface_layer(grid, mask, vertices[0], vertices[1],
                                          values[0], values[1])
            result[:, :, z, :] = layer.reshape(shape[0], shape[1], nframes)
    if label:
        result = np.rint(result).astype(np.int32)
    if nframes == 1:
        result = result[..., 0]
    _save_volume(result, reference, output, label)
    return output


@contextmanager
def _cpu_thread_budget():
    """Respect an existing Numba mask and restore it even on failed calls."""
    from numba import get_num_threads, set_num_threads
    import torch

    previous = get_num_threads()
    budget = min(previous, torch.get_num_threads())
    try:
        if budget != previous:
            set_num_threads(budget)
        yield
    finally:
        if budget != previous:
            set_num_threads(previous)


def convert_space(
    source: str | Path | tuple[str | Path, str | Path],
    source_space: str,
    target_space: str,
    output_dir: str | Path,
    assets_dir: str | Path,
    *,
    source_density: str | None = None,
    target_density: str | None = None,
    reference: str | Path | None = None,
    device: str = "cpu",
    label: bool = False,
    wb_command: str = "wb_command",
) -> Path | tuple[Path, Path]:
    """Convert a cortical scalar or label map among MNI152, fsaverage and fsLR.

    MNI input is a 3D/4D NIfTI. Surface input is an (L, R) GIFTI pair.
    Surface output is an (L, R) pair; MNI output is a cortical NIfTI on
    ``reference``'s grid. ``assets_dir`` holds CBIG RF-ANTs and HCP 2017 assets.
    """
    spaces = {"MNI152", "fsaverage", "fsLR"}
    if source_space not in spaces or target_space not in spaces:
        raise ValueError(f"Spaces must be in {spaces}")
    if source_space != "MNI152":
        _density(source_space, source_density)
    if target_space != "MNI152":
        _density(target_space, target_density)
    # A CPU-only surface route delegates its computation to Workbench.
    # Leave its CLI independent of volume/GPU library startup. Other device
    # values still follow the existing Torch validation and TF32 setup.
    surface_only_cpu = (device == "cpu" and source_space != "MNI152"
                        and target_space != "MNI152")
    work_device = None if surface_only_cpu else _device(device)
    root = Path(assets_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if target_space == "MNI152" and reference is None:
        reference = root / _RF / "FSL_MNI152_FS4.5.0_cortex_estimate.nii.gz"
    with tempfile.TemporaryDirectory(prefix=".space-", dir=output) as temporary:
        scratch = Path(temporary)
        if source_space == "MNI152":
            if isinstance(source, (tuple, list)):
                raise ValueError("MNI152 source must be a NIfTI file")
            if target_space == "MNI152":
                raise ValueError("MNI152-to-MNI152 grid resampling is outside this function")
            source_pair = _volume_to_fsaverage(Path(source), scratch, root, "164k", work_device, label)
            source_space, source_density = "fsaverage", "164k"
        else:
            if not isinstance(source, (tuple, list)) or len(source) != 2:
                raise ValueError("Surface source must be a (left, right) GIFTI pair")
            source_pair = (Path(source[0]), Path(source[1]))
            frame_counts = [_gifti_values(path, _density(source_space, source_density))[0].shape[1]
                            for path in source_pair]
            if frame_counts[0] != frame_counts[1]:
                raise ValueError("Left and right GIFTI files have different frame counts")
        if target_space == "MNI152":
            if source_space != "fsaverage" or source_density != "164k":
                source_pair = tuple(_surface_resample(
                    path, scratch / f"{hemi}.fsaverage.164k.gii", root,
                    source_space, source_density, "fsaverage", "164k", hemi, label, wb_command)
                    for hemi, path in zip(("L", "R"), source_pair))
            return _surface_to_volume(source_pair, output / "space-MNI152_cortex.nii.gz", root,
                                      Path(reference), work_device, label)
        configured_threads = os.environ.get("OMP_NUM_THREADS") if surface_only_cpu else None
        if configured_threads is not None:
            from ._hemisphere_parallel import map_hemispheres, resolve_cpu_threads
            budget = resolve_cpu_threads(int(configured_threads))
            loaded_torch = sys.modules.get("torch")
            if loaded_torch is not None:
                budget = min(budget, loaded_torch.get_num_threads())
            sources_by_hemi = dict(zip(("L", "R"), source_pair))

            def resample_hemisphere(hemi, threads):
                destination = output / f"{hemi}.{target_space}.{target_density}.{'label' if label else 'func'}.gii"
                return _surface_resample(sources_by_hemi[hemi], destination, root,
                                         source_space, source_density, target_space,
                                         target_density, hemi, label, wb_command,
                                         cpu_threads=threads)

            return map_hemispheres(resample_hemisphere, cpu_threads=budget)
        result = []
        for hemi, path in zip(("L", "R"), source_pair):
            destination = output / f"{hemi}.{target_space}.{target_density}.{'label' if label else 'func'}.gii"
            result.append(_surface_resample(path, destination, root, source_space,
                                            source_density, target_space, target_density,
                                            hemi, label, wb_command))
        return tuple(result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-space", required=True, choices=("MNI152", "fsaverage", "fsLR"))
    parser.add_argument("--target-space", required=True, choices=("MNI152", "fsaverage", "fsLR"))
    parser.add_argument("--volume", type=Path, help="MNI152 NIfTI input")
    parser.add_argument("--left", type=Path, help="Left GIFTI input")
    parser.add_argument("--right", type=Path, help="Right GIFTI input")
    parser.add_argument("--source-density", choices=("3k", "10k", "32k", "41k", "59k", "164k"))
    parser.add_argument("--target-density", choices=("3k", "10k", "32k", "41k", "59k", "164k"))
    parser.add_argument("--reference", type=Path, help="MNI152 output reference NIfTI")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--label", action="store_true")
    args = parser.parse_args(argv)
    source = args.volume if args.source_space == "MNI152" else (args.left, args.right)
    result = convert_space(source=source, source_space=args.source_space,
                           target_space=args.target_space, output_dir=args.output_dir,
                           assets_dir=args.assets_dir, source_density=args.source_density,
                           target_density=args.target_density, reference=args.reference,
                           device=args.device, label=args.label)
    for path in result if isinstance(result, tuple) else (result,):
        print(path)


if __name__ == "__main__":
    main()
