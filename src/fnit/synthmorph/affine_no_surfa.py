"""MGH/NIfTI geometry and LTA output for affine-only SynthMorph inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


_LIA = np.array([[-1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])


@dataclass(frozen=True)
class _Matrix:
    matrix: np.ndarray


@dataclass(frozen=True)
class AffineGeometry:
    shape: tuple[int, int, int]
    matrix: np.ndarray
    voxsize: np.ndarray
    rotation: np.ndarray
    center: np.ndarray

    @property
    def vox2world(self) -> _Matrix:
        return _Matrix(self.matrix)


def _geometry(image: nib.spatialimages.SpatialImage) -> AffineGeometry:
    shape = tuple(image.shape)
    if len(shape) != 3:
        raise ValueError("affine SynthMorph requires a single-frame 3D image")
    if isinstance(image, nib.MGHImage):
        voxsize = np.asarray(image.header["delta"], dtype=np.float64)
        rotation = np.asarray(image.header["Mdc"], dtype=np.float64).T
        center = np.asarray(image.header["Pxyz_c"], dtype=np.float64)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation @ np.diag(voxsize)
        matrix[:3, 3] = center - (matrix @ np.append(np.asarray(shape) / 2, 1))[:3]
    else:
        matrix = np.asarray(image.affine, dtype=np.float64)
        center = (matrix @ np.append(np.asarray(shape) / 2, 1))[:3]
        q, r = np.linalg.qr(matrix[:3, :3])
        indices = np.diag_indices(3)
        voxsize = np.abs(r[indices])
        signs = np.eye(3)
        signs[indices] = r[indices] / voxsize
        rotation = q @ signs
        voxsize = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    return AffineGeometry(shape, matrix, voxsize, rotation, center)


def load_affine_image(path: str | Path) -> tuple[np.ndarray, AffineGeometry]:
    """Load real MRI intensities with Surfa-compatible centered image geometry."""
    image = nib.load(str(path))
    geometry = _geometry(image)
    data = np.asarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError("input contains NaN or infinity")
    return data, geometry


def network_space_affine(image: AffineGeometry, shape: tuple[int, int, int]
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Map the LIA, 1-mm, centered SynthMorph network grid to MRI voxels."""
    network = np.eye(4)
    network[:3, :3] = _LIA
    network[:3, 3] = image.center - (_LIA @ (np.asarray(shape) / 2))
    return np.linalg.inv(image.matrix) @ network, np.linalg.inv(network) @ image.matrix


def _volume_info(geometry: AffineGeometry) -> list[str]:
    lines = ["valid = 1", "filename = none",
             "volume = " + " ".join(str(value) for value in geometry.shape),
             "voxelsize = " + " ".join("%.15e" % value for value in geometry.voxsize)]
    for name, axis in zip(("xras", "yras", "zras"), geometry.rotation.T):
        lines.append(name + "   = " + " ".join("%.15e" % value for value in axis))
    lines.append("cras   = " + " ".join("%.15e" % value for value in geometry.center))
    return lines


@dataclass(frozen=True)
class AffineTransform:
    """World-space affine and source/target geometry, with FreeSurfer LTA save."""

    matrix: np.ndarray
    source: AffineGeometry
    target: AffineGeometry
    space: str = "world"

    def convert(self, *, space: str) -> AffineTransform:
        if space == self.space:
            return self
        if self.space == "world" and space == "voxel":
            result = np.linalg.inv(self.target.matrix) @ self.matrix @ self.source.matrix
        elif self.space == "voxel" and space == "world":
            result = self.target.matrix @ self.matrix @ np.linalg.inv(self.source.matrix)
        else:
            raise ValueError("affine space must be world or voxel")
        return AffineTransform(result, self.source, self.target, space)

    def save(self, path: str | Path) -> None:
        kind = "1 # LINEAR_RAS_TO_RAS" if self.space == "world" else "0 # LINEAR_VOX_TO_VOX"
        lines = [f"type      = {kind}", "nxforms   = 1",
                 "mean      = 0.0000 0.0000 0.0000", "sigma     = 1.0000", "1 4 4"]
        lines.extend(" ".join("%.15e" % value for value in row) for row in self.matrix)
        lines.extend(["src volume info", *_volume_info(self.source),
                      "dst volume info", *_volume_info(self.target)])
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n")
