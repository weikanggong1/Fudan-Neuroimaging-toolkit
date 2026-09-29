"""Sample the fixed recon-all white/gray contrast points with PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import torch


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """FreeSurfer's corner-normal sum for a white surface."""
    xyz = np.asarray(vertices, dtype=np.float32)
    contributions = np.empty((len(faces), 3, 3), dtype=np.float32)
    for corner in range(3):
        previous = xyz[faces[:, (corner - 1) % 3]]
        current = xyz[faces[:, corner]]
        following = xyz[faces[:, (corner + 1) % 3]]
        edge0 = current - previous
        edge1 = following - current
        edge0 /= np.linalg.norm(edge0, axis=1)[:, None]
        edge1 /= np.linalg.norm(edge1, axis=1)[:, None]
        face_normal = np.cross(edge0, edge1)
        contributions[:, corner] = face_normal / np.linalg.norm(face_normal, axis=1)[:, None]
    normals = np.zeros_like(xyz)
    np.add.at(normals, faces.ravel(), contributions.reshape(-1, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals


def _det3(matrix: np.ndarray) -> np.float32:
    a, b, c = matrix
    return (a[0] * b[1] * c[2] - a[0] * c[1] * b[2] -
            b[0] * a[1] * c[2] + b[0] * c[1] * a[2] +
            c[0] * a[1] * b[2] - c[0] * b[1] * a[2])


def _inverse_affine32(matrix: np.ndarray) -> np.ndarray:
    """Match the float32 cofactor inverse used by FreeSurfer for 4x4 affines."""
    reciprocal = np.float32(1) / _det3(matrix[:3, :3])
    result = np.empty((4, 4), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            minor = matrix[[i for i in range(4) if i != col]][
                :, [i for i in range(4) if i != row]]
            cofactor = _det3(minor)
            result[row, col] = (np.float32(-cofactor if (row + col) & 1 else cofactor)
                                * reciprocal)
    return result


def _multiply32(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Accumulate each MATRIX_REAL product in FreeSurfer's scalar order."""
    result = np.empty((4, 4), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            value = np.float32(0)
            for inner in range(4):
                value = np.float32(value + np.float32(left[row, inner] *
                                                      right[inner, col]))
            result[row, col] = value
    return result


def _native_vox2ras(image: nib.MGHImage) -> np.ndarray:
    """Build MRIxfmCRS2XYZ from MGH geometry, preserving its float rounding."""
    header = image.header
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = (np.asarray(header["Mdc"], dtype=np.float32).T *
                      np.asarray(header["delta"], dtype=np.float32))
    center = np.asarray(image.shape[:3], dtype=np.float32) / 2
    offset = np.asarray([sum(float(matrix[row, col]) * float(center[col])
                             for col in range(3)) for row in range(3)], dtype=np.float32)
    matrix[:3, 3] = np.asarray(header["Pxyz_c"], dtype=np.float32) - offset
    return matrix


def _surface_to_source(moving: nib.MGHImage, reference: nib.MGHImage) -> np.ndarray:
    moving_tkr = moving.header.get_vox2ras_tkr().astype(np.float32)
    reference_tkr = reference.header.get_vox2ras_tkr().astype(np.float32)
    registration = _multiply32(moving_tkr, _inverse_affine32(_native_vox2ras(moving)))
    registration = _multiply32(registration, _native_vox2ras(reference))
    registration = _multiply32(registration, _inverse_affine32(reference_tkr))
    return _multiply32(_inverse_affine32(moving_tkr), registration)


def _transform_points32(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    result = np.zeros((len(points), 3), dtype=np.float32)
    for row in range(3):
        for col in range(3):
            result[:, row] = np.float32(result[:, row] +
                                       np.float32(points[:, col] * matrix[row, col]))
        result[:, row] = np.float32(result[:, row] + matrix[row, 3])
    return result


@torch.inference_mode()
def _trilinear(volume: np.ndarray, coordinates: np.ndarray, device: str) -> np.ndarray:
    source = torch.as_tensor(volume, dtype=torch.float32, device=device)
    coord = torch.as_tensor(coordinates, dtype=torch.float64, device=device)
    rounded = torch.floor(coord + 0.5).to(torch.int64)
    shape = torch.tensor(volume.shape, dtype=torch.int64, device=device)
    inside = ((rounded >= 0) & (rounded < shape)).all(dim=1)
    coord = torch.minimum(torch.maximum(coord, torch.zeros_like(coord)),
                          (shape - 1).to(torch.float64))
    low = torch.floor(coord).to(torch.int64)
    high = torch.minimum(low + 1, shape - 1)
    lower = coord - low.to(torch.float64)
    upper = 1.0 - lower
    result = torch.zeros(len(coord), dtype=torch.float64, device=device)
    for x, y, z in ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)):
        indices = [high[:, dim] if bit else low[:, dim]
                   for dim, bit in enumerate((x, y, z))]
        weight = ((lower[:, 0] if x else upper[:, 0]) *
                  (lower[:, 1] if y else upper[:, 1]) *
                  (lower[:, 2] if z else upper[:, 2]))
        result += weight * source[indices[0], indices[1], indices[2]].to(torch.float64)
    result[~inside] = 0
    return result.to(torch.float32).cpu().numpy()


def sample_contrast(subject: str | Path, hemi: str, projection: str,
                    *, device: str = "cpu") -> np.ndarray:
    """Return `pctsurfcon`'s white (`wm`) or 30% thickness (`gm`) samples."""
    if hemi not in ("lh", "rh") or projection not in ("wm", "gm"):
        raise ValueError("Expected hemisphere lh/rh and projection wm/gm")
    subject = Path(subject)
    surface, faces = fsio.read_geometry(str(subject / "surf" / f"{hemi}.white"))
    surface = np.asarray(surface, dtype=np.float32)
    normals = _vertex_normals(surface, faces)
    if projection == "wm":
        points = surface - normals
    else:
        thickness = fsio.read_morph_data(str(subject / "surf" / f"{hemi}.thickness"))
        points = surface + np.float32(np.float32(0.3) * thickness)[:, None] * normals
    moving = nib.load(str(subject / "mri" / "rawavg.mgz"))
    reference = nib.load(str(subject / "mri" / "orig.mgz"))
    matrix = _surface_to_source(moving, reference)
    coordinates = _transform_points32(points, matrix)
    values = _trilinear(np.asarray(moving.dataobj, dtype=np.float32), coordinates, device)
    cortex = fsio.read_label(str(subject / "label" / f"{hemi}.cortex.label"))
    mask = np.zeros(len(values), dtype=bool)
    mask[cortex] = True
    values[~mask] = 0
    return values


def contrast_percentage(subject: str | Path, hemi: str, *, device: str = "cpu") -> np.ndarray:
    """Apply recon-all's ``mri_concat --paired-diff-norm --mul 100`` rule."""
    white = sample_contrast(subject, hemi, "wm", device=device).astype(np.float64)
    gray = sample_contrast(subject, hemi, "gm", device=device).astype(np.float64)
    result = np.zeros(len(white), dtype=np.float32)
    nonzero = (white + gray) != 0
    result[nonzero] = ((white[nonzero] - gray[nonzero]) /
                       ((white[nonzero] + gray[nonzero]) / 2)).astype(np.float32)
    return (result.astype(np.float64) * 100).astype(np.float32)


def write_contrast_percentage(subject: str | Path, hemi: str,
                              output: str | Path, *, device: str = "cpu") -> Path:
    """保存每顶点 white/gray 百分比图，保留固定 MGH 头。"""
    subject, output = Path(subject), Path(output)
    values = contrast_percentage(subject, hemi, device=device)
    header = nib.load(str(subject / "mri/rawavg.mgz")).header.copy()
    header.set_data_shape((len(values), 1, 1))
    header["fov"] = float(len(values))
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.MGHImage(values[:, None, None], None, header=header), str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=Path)
    parser.add_argument("hemi", choices=("lh", "rh"))
    parser.add_argument("projection", choices=("wm", "gm", "pct"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.projection == "pct":
        write_contrast_percentage(args.subject, args.hemi, args.output,
                                  device=args.device)
        return
    values = sample_contrast(args.subject, args.hemi, args.projection,
                             device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = nib.load(str(args.subject / "mri" / "rawavg.mgz")).header.copy()
    header.set_data_shape((len(values), 1, 1))
    nib.save(nib.MGHImage(values[:, None, None], None, header=header), str(args.output))


if __name__ == "__main__":
    main()
