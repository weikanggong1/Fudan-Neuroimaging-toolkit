"""SynthSeg label image I/O without a Surfa runtime dependency."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import gzip
from pathlib import Path
import struct

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class LabelEntry:
    name: str
    color: np.ndarray  # RGB in 0..255; opacity in 0..1.


@dataclass(frozen=True)
class _Affine:
    matrix: np.ndarray


@dataclass(frozen=True)
class _Geometry:
    vox2world: _Affine


def read_color_lut(path: str | Path) -> OrderedDict[int, LabelEntry]:
    """Read FreeSurferColorLUT.txt's ID, name, RGB and transparency columns."""
    labels: OrderedDict[int, LabelEntry] = OrderedDict()
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        color = np.asarray([int(value) for value in fields[2:6]], dtype=np.float64)
        color[3] = (255 - color[3]) / 255
        labels[int(fields[0])] = LabelEntry(fields[1], color)
    return labels


def _geometry(shape: tuple[int, int, int], affine: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the centered QR geometry stored by FreeSurfer's MGH writer."""
    center = (affine @ np.append(np.asarray(shape) / 2, 1))[:3]
    q, r = np.linalg.qr(affine[:3, :3])
    diagonal = np.diag_indices(3)
    voxsize = np.abs(r[diagonal])
    signs = np.eye(3)
    signs[diagonal] = r[diagonal] / voxsize
    rotation = q @ signs
    shear_matrix = (signs @ r) / voxsize[:, None]
    shear = shear_matrix[0, 1], shear_matrix[0, 2], shear_matrix[1, 2]
    if np.any(np.asarray(shear) > 1e-5):
        voxsize = np.linalg.norm(affine[:3, :3], axis=0)
        rotation = affine[:3, :3] / voxsize
    return voxsize, rotation, center


def _binary_lut(labels: OrderedDict[int, LabelEntry]) -> bytes:
    chunks = [struct.pack(">iiii", -2, max(labels) + 1, 0, len(labels))]
    for index, label in labels.items():
        name = (label.name + "\0").encode("utf-8")
        rgb = label.color[:3].astype(np.uint8).astype(np.int32)
        alpha = int(np.asarray(255 * (1 - label.color[3])).astype(int))
        chunks.extend((struct.pack(">ii", index, len(label.name) + 1), name,
                       struct.pack(">iiii", *rgb, alpha)))
    return b"".join(chunks)


def _tag(number: int, content: bytes) -> bytes:
    return struct.pack(">iq", number, len(content)) + content


def _nifti_extension(labels: OrderedDict[int, LabelEntry] | None) -> bytes:
    content = [b">\x00\x00\x01"]
    if labels:
        content.append(_tag(1, _binary_lut(labels)))
    content.append(_tag(7, struct.pack(">i", 1)))
    content.append(_tag(45, struct.pack(">ff", 0., 0.) + struct.pack(">d", 0.)
                        + struct.pack(">I", 0x7fc00000) + b"UNKNOWN"))
    content.append(_tag(-1, b"*"))
    return b"".join(content)


def _resample_nearest(data: np.ndarray, source_affine: np.ndarray,
                      target_shape: tuple[int, int, int],
                      target_affine: np.ndarray) -> np.ndarray:
    """Match Surfa's float32 voxel-coordinate rounding at nearest-neighbor ties."""
    transform = (np.linalg.inv(source_affine) @ target_affine).astype(np.float32)
    output = np.zeros(target_shape, dtype=data.dtype)
    y = np.arange(target_shape[1], dtype=np.float32)[None, :, None]
    z = np.arange(target_shape[2], dtype=np.float32)[None, None, :]
    for start in range(0, target_shape[0], 16):
        stop = min(start + 16, target_shape[0])
        x = np.arange(start, stop, dtype=np.float32)[:, None, None]
        coordinates = [np.floor((transform[axis, 0] * x + transform[axis, 1] * y
                                + transform[axis, 2] * z + transform[axis, 3]) + 0.5
                                ).astype(np.int32) for axis in range(3)]
        valid = np.ones((stop - start, *target_shape[1:]), dtype=bool)
        for axis, coordinate in enumerate(coordinates):
            valid &= (coordinate >= 0) & (coordinate < data.shape[axis])
        output[start:stop][valid] = data[tuple(coordinate[valid] for coordinate in coordinates)]
    return output


@dataclass
class SynthSegVolume:
    """Float32 label image with the ``data``, ``geom`` and ``save`` public API."""

    data: np.ndarray
    affine: np.ndarray
    labels: OrderedDict[int, LabelEntry] | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    @property
    def geom(self) -> _Geometry:
        return _Geometry(_Affine(self.affine))

    def resample_like(self, image: str | Path) -> SynthSegVolume:
        target = nib.load(str(image))
        data = _resample_nearest(self.data, self.affine, target.shape[:3], target.affine)
        return SynthSegVolume(data, target.affine.copy(), self.labels)

    def save(self, path: str | Path) -> None:
        """Write NIfTI or MGH/MGZ with the optional FreeSurfer color table."""
        path = Path(path)
        voxsize, rotation, center = _geometry(self.shape, self.affine)
        if path.suffix in {".mgz", ".mgh"}:
            image = nib.MGHImage(self.data, self.affine)
            header = image.header
            header["dof"] = 1
            header["delta"] = voxsize
            header["Mdc"] = rotation.T
            header["Pxyz_c"] = center
            header["fov"] = max(voxsize * self.shape)
            trailing = b""
            if self.labels:
                trailing = struct.pack(">i", 1) + _binary_lut(self.labels)
            trailing += _tag(41, b"UNKNOWN") + _tag(43, struct.pack(">f", 0.))
            opener = gzip.open if path.suffix == ".mgz" else open
            with opener(path, "wb") as stream:
                stream.write(image.to_bytes())
                stream.write(trailing)
            return
        if str(path).endswith((".nii", ".nii.gz")):
            image = nib.Nifti1Image(self.data, np.eye(4))
            image.header["pixdim"][:] = 1
            image.header["xyzt_units"] = 10  # millimeters and seconds
            image.set_sform(self.affine, 1)
            image.set_qform(self.affine, 1)
            image.header["pixdim"][1:4] = voxsize.astype(np.float32)
            image.header.extensions.append(nib.nifti1.Nifti1Extension(
                14, _nifti_extension(self.labels)))
            nib.save(image, str(path))
            return
        raise ValueError("SynthSeg label output must end with .nii, .nii.gz, .mgh or .mgz")
