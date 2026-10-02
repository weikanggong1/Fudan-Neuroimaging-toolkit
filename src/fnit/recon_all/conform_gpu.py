"""PyTorch translation of recon-all's ``mri_convert --conform`` stage.

This implements the default 1 mm, coronal, trilinear path used for one T1
volume. It does not implement mri_convert's other flags or the subsequent
``mri_add_xform_to_header`` recon-all command.
"""

from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path
import struct

import nibabel as nib
import numpy as np
import torch


def _uchar_from_float(volume: torch.Tensor) -> torch.Tensor:
    """Apply MRIchangeType's 1000-bin, 0--99.9% intensity scaling."""
    values = volume.to(torch.float32)
    minimum = values.min().item()
    maximum = values.max().item()
    if maximum == minimum:
        return values.clamp(0, 255).add(0.5).floor().to(torch.uint8)
    bin_size = np.float32((maximum - minimum) / 1000)
    bins = ((values - minimum) / float(bin_size)).to(torch.int64).clamp_(0, 999)
    counts = torch.bincount(bins.reshape(-1), minlength=1000).cpu().tolist()
    nonzero = torch.count_nonzero(values).item()
    target = int((1.0 - float(np.float32(0.999))) * nonzero)
    high_bin = 999
    passed = 0
    while passed < target and high_bin > 0:
        passed += counts[high_bin]
        high_bin -= 1
    clipped_max = np.float32(high_bin * bin_size + np.float32(minimum))
    scale = np.float32(255.0 / (clipped_max - np.float32(minimum))) if clipped_max != minimum else np.float32(1.0)
    return ((values - minimum) * float(scale)).clamp_(0, 255).add_(0.5).floor_().to(torch.uint8)


def _coronal_affine(source: nib.spatialimages.SpatialImage, width: int) -> np.ndarray:
    directions = np.array([[-1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float32)
    center = np.asarray(source.header['Pxyz_c'], dtype=np.float32)
    affine = np.eye(4, dtype=np.float32)
    affine[:3, :3] = directions
    affine[:3, 3] = center - directions @ np.full(3, width / 2, dtype=np.float32)
    return affine


def _source_affine(source: nib.spatialimages.SpatialImage) -> np.ndarray:
    """Reproduce MRIxfmCRS2XYZ's float matrix with double dot accumulation."""
    header = source.header
    matrix = np.eye(4, dtype=np.float32)
    directions = np.asarray(header['Mdc'], dtype=np.float32).T
    sizes = np.asarray(header['delta'], dtype=np.float32)
    matrix[:3, :3] = (directions.astype(np.float64) * sizes.astype(np.float64)).astype(np.float32)
    midpoint = np.asarray(source.shape, dtype=np.float32) / 2
    for row in range(3):
        offset = np.float32(sum(float(matrix[row, col]) * float(midpoint[col]) for col in range(3)))
        matrix[row, 3] = np.float32(header['Pxyz_c'][row]) - offset
    return matrix


def _multiply32(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 4), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            for inner in range(4):
                result[row, col] = np.float32(result[row, col] + np.float32(left[row, inner] * right[inner, col]))
    return result


def _det3(m: np.ndarray) -> np.float32:
    """按 VNL 余子式的 float32 求值顺序计算三阶行列式。"""
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    return a * e * i - a * f * h - d * b * i + d * c * h + g * b * f - g * c * e


def _inverse32(m: np.ndarray) -> np.ndarray:
    """The single-precision cofactor inverse used by FreeSurfer's VNL path."""
    determinant = _det3(m[:3, :3])
    reciprocal = np.float32(1) / determinant
    inverse = np.empty((4, 4), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            minor = np.delete(np.delete(m, col, axis=0), row, axis=1)
            cofactor = _det3(minor)
            inverse[row, col] = (-cofactor if (row + col) % 2 else cofactor) * reciprocal
    return inverse


def _resampled_header(source: nib.spatialimages.SpatialImage,
                      transform: np.ndarray, width: int) -> nib.MGHHeader:
    """Mirror MRIresampleFill's post-resampling Mdc and center calculation."""
    source_matrix = _source_affine(source)
    direction = _multiply32(source_matrix, transform)
    center_voxel = np.array([width / 2, width / 2, width / 2, 1], dtype=np.float32)
    source_voxel = np.zeros(4, dtype=np.float32)
    center = np.zeros(4, dtype=np.float32)
    for row in range(4):
        for col in range(4):
            source_voxel[row] = np.float32(source_voxel[row] + np.float32(transform[row, col] * center_voxel[col]))
    for row in range(4):
        for col in range(4):
            center[row] = np.float32(center[row] + np.float32(source_matrix[row, col] * source_voxel[col]))
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    header['delta'] = np.ones(3, dtype=np.float32)
    header['Mdc'] = direction[:3, :3].T
    header['Pxyz_c'] = center[:3]
    return header


def _sample_trilinear(volume: torch.Tensor, ijk: torch.Tensor) -> torch.Tensor:
    """Use FreeSurfer's eight-neighbor order and zero beyond each image edge."""
    base = torch.floor(ijk).to(torch.int64)
    fraction = ijk.to(torch.float32) - base
    output = torch.zeros(ijk.shape[:-1], dtype=torch.float64, device=volume.device)
    fraction64 = fraction.to(torch.float64)
    inverse64 = 1.0 - fraction64
    nx, ny, nz = volume.shape
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                i, j, k = base[..., 0] + dx, base[..., 1] + dy, base[..., 2] + dz
                valid = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
                voxel = volume[i.clamp(0, nx - 1), j.clamp(0, ny - 1), k.clamp(0, nz - 1)].to(torch.float32)
                if dx == dy == dz == 1:
                    weight = fraction[..., 0] * fraction[..., 1]
                    weight = weight * fraction[..., 2]
                    term = weight * torch.where(valid, voxel, 0)
                else:
                    weight = (fraction64[..., 0] if dx else inverse64[..., 0])
                    weight = weight * (fraction64[..., 1] if dy else inverse64[..., 1])
                    weight = weight * (fraction64[..., 2] if dz else inverse64[..., 2])
                    term = weight * torch.where(valid, voxel, 0)
                output = output + term
    return output.to(torch.float32).clamp_(0, 255).add_(0.5).floor_().to(torch.uint8)


def conform_volume(input_file: str | Path, output_file: str | Path, *, device: str = 'cuda:0',
                   slab: int = 8) -> None:
    """Conform one FreeSurfer MGH/MGZ T1 volume, preserving pulse parameters."""
    source = nib.load(str(input_file))
    if not isinstance(source, nib.MGHImage) or len(source.shape) != 3:
        raise ValueError('expected a 3D MGH/MGZ input from recon-all rawavg')
    if source.get_data_dtype().newbyteorder('=') not in (np.dtype('float32'), np.dtype('uint8')):
        raise ValueError('this recon-all path expects float32 or uint8 source voxels')
    if slab < 1:
        raise ValueError('slab must be positive')
    width = max(256, math.ceil(max(np.array(source.shape) * np.asarray(source.header['delta']))))
    if width > 256 and (width - 256) / 256 < 0.1:
        width = 256
    affine = _coronal_affine(source, width)
    transform = _multiply32(_inverse32(_source_affine(source)), affine)
    matrix = torch.as_tensor(transform, dtype=torch.float32, device=device)
    native_dtype = source.get_data_dtype().newbyteorder('=')
    data = torch.as_tensor(np.asarray(source.dataobj).astype(native_dtype, copy=True), device=device)
    if data.dtype != torch.uint8:
        data = _uchar_from_float(data)
    output = np.empty((width, width, width), dtype=np.uint8)
    x = torch.arange(width, device=device, dtype=torch.float32)
    y = torch.arange(width, device=device, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, width, slab):
            z = torch.arange(start, min(start + slab, width), device=device, dtype=torch.float32)
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            ijk = torch.stack(tuple(
                ((matrix[row, 0] * xx + matrix[row, 1] * yy)
                 + matrix[row, 2] * zz) + matrix[row, 3]
                for row in range(3)), -1)
            values = _sample_trilinear(data, ijk)
            output[:, :, start:start + len(z)] = values.cpu().numpy()
    header = _resampled_header(source, transform, width)
    target = nib.MGHImage(output, None, header)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    nib.save(target, str(output_file))


def add_xform_to_header(input_file: str | Path, output_file: str | Path,
                        xform_file: str | Path) -> None:
    """Write the TAG_MGH_XFORM name used by recon-all's ``-c`` command."""
    source = Path(input_file)
    target = Path(output_file)
    if source.suffix not in ('.mgh', '.mgz') or target.suffix not in ('.mgh', '.mgz'):
        raise ValueError('expected MGH/MGZ paths')
    image = nib.load(str(source))
    raw = gzip.decompress(source.read_bytes()) if source.suffix == '.mgz' else source.read_bytes()
    end = int(image.header.get_data_offset()) + math.prod(image.shape) * image.get_data_dtype().itemsize
    if len(raw) - end != 20:
        raise ValueError('expected a conformed image with the standard 20-byte MGH footer')
    name = str(xform_file).encode() + b'\0'
    raw += struct.pack('>iq', 31, len(name)) + name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gzip.compress(raw, mtime=0) if target.suffix == '.mgz' else raw)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--slab', type=int, default=8)
    parser.add_argument('--xform', help='write recon-all TAG_MGH_XFORM after conforming')
    args = parser.parse_args(argv)
    conform_volume(args.input_file, args.output_file, device=args.device, slab=args.slab)
    if args.xform:
        add_xform_to_header(args.output_file, args.output_file, args.xform)


if __name__ == '__main__':
    main()
