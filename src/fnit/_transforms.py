"""Small affine and dense-warp containers used by FNIT registration tools."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from ._nib import FNITNifti1Image


class ImageGeometry:
    """Image shape and voxel-to-world matrix with Surfa-compatible accessors."""

    def __init__(self, shape, affine):
        shape = tuple(int(value) for value in tuple(shape)[:3])
        affine = np.asarray(affine, dtype=np.float64)
        if len(shape) != 3 or any(value < 1 for value in shape):
            raise ValueError("image geometry must have three positive dimensions")
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("image geometry affine must be finite and 4x4")
        if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-10:
            raise ValueError("image geometry affine must be invertible")
        self.shape = shape
        self.affine = np.array(affine, copy=True)

    @property
    def vox2world(self):
        return SimpleNamespace(matrix=self.affine)

    @property
    def world2vox(self):
        return SimpleNamespace(matrix=np.linalg.inv(self.affine))

    @property
    def voxsize(self):
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def center(self):
        # FreeSurfer VOL_GEOM uses dimension / 2, rather than the centre voxel.
        return nib.affines.apply_affine(self.affine, np.asarray(self.shape) / 2.0)

    def copy(self):
        return ImageGeometry(self.shape, self.affine)


def image_geometry(value) -> ImageGeometry:
    if isinstance(value, ImageGeometry):
        return value
    if isinstance(value, nib.spatialimages.SpatialImage):
        return ImageGeometry(value.shape[:3], value.affine)
    shape = getattr(value, "shape", None)
    vox2world = getattr(getattr(value, "vox2world", None), "matrix", None)
    if shape is not None and vox2world is not None:
        return ImageGeometry(shape, vox2world)
    raise TypeError("geometry must be a nibabel image or ImageGeometry")


def same_geometry(left, right, *, tolerance=1e-3) -> bool:
    left = image_geometry(left)
    right = image_geometry(right)
    return left.shape == right.shape and np.allclose(
        left.affine, right.affine, atol=tolerance, rtol=0
    )


class AffineTransform:
    """Geometry-tagged source-to-target affine in voxel or world coordinates."""

    def __init__(self, matrix, *, source, target, space="world"):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("affine matrix must be finite and 4x4")
        if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
            raise ValueError("affine matrix must be homogeneous")
        if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-10:
            raise ValueError("affine matrix must be invertible")
        if space not in ("voxel", "world"):
            raise ValueError("affine space must be voxel or world")
        self.matrix = np.array(matrix, copy=True)
        self.source = image_geometry(source)
        self.target = image_geometry(target)
        self.space = space

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.matrix, dtype=dtype)
        return np.array(array, copy=True) if copy else array

    def copy(self):
        return AffineTransform(
            self.matrix, source=self.source, target=self.target, space=self.space
        )

    def convert(self, *, space, source=None, target=None):
        source = self.source if source is None else image_geometry(source)
        target = self.target if target is None else image_geometry(target)
        if self.space == "world":
            world = np.array(self.matrix, copy=True)
        else:
            world = self.target.affine @ self.matrix @ np.linalg.inv(
                self.source.affine
            )
        if space == "world":
            matrix = world
        elif space == "voxel":
            matrix = np.linalg.inv(target.affine) @ world @ source.affine
        else:
            raise ValueError("affine space must be voxel or world")
        matrix[3] = (0, 0, 0, 1)
        return AffineTransform(matrix, source=source, target=target, space=space)

    def save(self, path) -> None:
        write_lta(self, path)


def _lta_geometry_lines(label, geometry):
    geometry = image_geometry(geometry)
    linear = geometry.affine[:3, :3]
    sizes = np.linalg.norm(linear, axis=0)
    rotation = linear / sizes
    center = geometry.center
    values = lambda row: " ".join(f"{float(value):.15g}" for value in row)
    return [
        f"{label} volume info",
        "valid = 1",
        "filename = unknown",
        f"volume = {' '.join(str(value) for value in geometry.shape)}",
        f"voxelsize = {values(sizes)}",
        f"xras = {values(rotation[:, 0])}",
        f"yras = {values(rotation[:, 1])}",
        f"zras = {values(rotation[:, 2])}",
        f"cras = {values(center)}",
    ]


def write_lta(transform, path) -> None:
    if not isinstance(transform, AffineTransform):
        raise TypeError("transform must be an AffineTransform")
    path = Path(path)
    if path.suffix.lower() != ".lta":
        raise ValueError("affine transforms must be saved with a .lta extension")
    kind = 1 if transform.space == "world" else 0
    label = "LINEAR_RAS_TO_RAS" if kind == 1 else "LINEAR_VOX_TO_VOX"
    lines = [
        f"type      = {kind} # {label}",
        "nxforms   = 1",
        "mean      = 0.0 0.0 0.0",
        "sigma     = 1.0",
        "1 4 4",
    ]
    lines.extend(
        " ".join(f"{float(value):.15g}" for value in row)
        for row in transform.matrix
    )
    lines.extend(_lta_geometry_lines("src", transform.source))
    lines.extend(_lta_geometry_lines("dst", transform.target))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _lta_value(lines, start, key):
    prefix = f"{key} ="
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith(prefix):
            return stripped.split("=", 1)[1].strip()
        if index > start and stripped.endswith("volume info"):
            break
    raise ValueError(f"LTA is missing {key}")


def _lta_geometry(lines, label):
    try:
        start = next(
            index for index, line in enumerate(lines)
            if line.strip() == f"{label} volume info"
        )
    except StopIteration as error:
        raise ValueError(f"LTA is missing {label} volume info") from error
    shape = tuple(int(value) for value in _lta_value(lines, start, "volume").split())
    sizes = np.fromstring(_lta_value(lines, start, "voxelsize"), sep=" ")
    rotation = np.column_stack(
        [
            np.fromstring(_lta_value(lines, start, key), sep=" ")
            for key in ("xras", "yras", "zras")
        ]
    )
    center = np.fromstring(_lta_value(lines, start, "cras"), sep=" ")
    if len(shape) != 3 or sizes.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError(f"invalid {label} LTA geometry")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation * sizes
    affine[:3, 3] = center - affine[:3, :3] @ (np.asarray(shape) / 2.0)
    return ImageGeometry(shape, affine)


def load_lta(path) -> AffineTransform:
    lines = Path(path).read_text().splitlines()
    type_lines = [line for line in lines if line.strip().startswith("type")]
    if not type_lines:
        raise ValueError("LTA is missing its transform type")
    kind = int(type_lines[0].split("=", 1)[1].split("#", 1)[0])
    if kind not in (0, 1):
        raise NotImplementedError(f"LTA transform type {kind} is not supported")
    try:
        start = next(
            index + 1
            for index, line in enumerate(lines)
            if line.split("#", 1)[0].strip() == "1 4 4"
        )
    except StopIteration as error:
        raise ValueError("LTA is missing its 4x4 matrix") from error
    matrix = np.array(
        [[float(value) for value in lines[start + row].split()] for row in range(4)],
        dtype=np.float64,
    )
    return AffineTransform(
        matrix,
        source=_lta_geometry(lines, "src"),
        target=_lta_geometry(lines, "dst"),
        space="world" if kind == 1 else "voxel",
    )


class DenseWarp(FNITNifti1Image):
    """Target-grid target-to-source displacement in world-RAS millimetres."""

    def __init__(self, data, *, source, target):
        array = np.asarray(data, dtype=np.float32)
        source_geometry = image_geometry(source)
        target_geometry = image_geometry(target)
        if array.shape != (*target_geometry.shape, 3):
            raise ValueError("dense warp must have target shape (X, Y, Z, 3)")
        if not np.isfinite(array).all():
            raise ValueError("dense warp must contain only finite values")
        if isinstance(target, nib.spatialimages.SpatialImage):
            header = nib.Nifti1Header.from_header(target.header)
        else:
            header = nib.Nifti1Header()
        header.set_data_dtype(np.float32)
        header.set_intent("vector")
        super().__init__(array, target_geometry.affine, header)
        self.set_qform(self.affine, int(header["qform_code"]) or 1)
        self.set_sform(self.affine, int(header["sform_code"]) or 1)
        self.source = source_geometry
        self.target = target_geometry
        self.format = "disp_ras"

    def convert(self, *, format="disp_ras", copy=True):
        if format not in ("disp_ras",):
            raise NotImplementedError("DenseWarp stores only disp_ras fields")
        if not copy:
            return self
        return DenseWarp(
            np.array(self.dataobj, copy=True), source=self.source, target=self.target
        )

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name.lower().endswith((".mgz", ".mgh")):
            image = nib.MGHImage(
                np.asarray(self.dataobj, dtype=np.float32), self.affine
            )
            nib.save(image, str(path))
        else:
            nib.save(self, str(path))


def load_dense_warp(path, *, source=None) -> DenseWarp:
    image = nib.load(str(path))
    return DenseWarp(
        np.asanyarray(image.dataobj),
        source=image if source is None else source,
        target=image,
    )


def voxel_displacement_to_ras(displacement, source, target):
    """Convert target-grid source-voxel displacement to RAS displacement."""
    displacement = np.asarray(displacement, dtype=np.float32)
    source = image_geometry(source)
    target = image_geometry(target)
    if displacement.shape != (*target.shape, 3):
        raise ValueError("voxel displacement must have target shape (X, Y, Z, 3)")
    source_linear = source.affine[:3, :3].astype(np.float32)
    target_linear = target.affine[:3, :3].astype(np.float32)
    output = np.einsum(
        "ab,...b->...a", source_linear, displacement, optimize=True
    ).astype(np.float32, copy=False)
    delta = source_linear - target_linear
    for axis, size in enumerate(target.shape):
        coordinate = np.arange(size, dtype=np.float32)
        shape = [1, 1, 1, 1]
        shape[axis] = size
        output += coordinate.reshape(shape) * delta[:, axis]
    output += source.affine[:3, 3].astype(np.float32)
    output -= target.affine[:3, 3].astype(np.float32)
    return output


def ras_displacement_to_voxel(displacement, source, target):
    """Convert a target-grid RAS displacement to source-voxel displacement."""
    displacement = np.asarray(displacement, dtype=np.float32)
    source = image_geometry(source)
    target = image_geometry(target)
    if displacement.shape != (*target.shape, 3):
        raise ValueError("RAS displacement must have target shape (X, Y, Z, 3)")
    inverse = np.linalg.inv(source.affine).astype(np.float32)
    target_affine = target.affine.astype(np.float32)
    output = np.einsum(
        "ab,...b->...a", inverse[:3, :3], displacement, optimize=True
    ).astype(np.float32, copy=False)
    linear = inverse[:3, :3] @ target_affine[:3, :3] - np.eye(3, dtype=np.float32)
    offset = inverse[:3, :3] @ target_affine[:3, 3] + inverse[:3, 3]
    for axis, size in enumerate(target.shape):
        coordinate = np.arange(size, dtype=np.float32)
        shape = [1, 1, 1, 1]
        shape[axis] = size
        output += coordinate.reshape(shape) * linear[:, axis]
    output += offset.astype(np.float32)
    return output


__all__ = [
    "AffineTransform",
    "DenseWarp",
    "ImageGeometry",
    "image_geometry",
    "load_dense_warp",
    "load_lta",
    "ras_displacement_to_voxel",
    "same_geometry",
    "voxel_displacement_to_ras",
    "write_lta",
]
