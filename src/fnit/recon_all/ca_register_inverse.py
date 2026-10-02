"""Coordinate conversion and splatting for the fixed inverse-GCAM call.

The input is a FreeSurfer NIfTI warp with displacement vectors in RAS space.
Gap filling and output writing live in sibling modules.
"""

from __future__ import annotations

import struct

import numpy as np
from numba import njit


def _freesurfer_vox2ras(fields: tuple[int | float, ...]) -> np.ndarray:
    if any(fields[19:22]):
        raise ValueError("the fixed warp must have zero geometry shear")
    matrix = np.eye(4, dtype=np.float32)
    directions = np.asarray(fields[7:16], dtype=np.float32).reshape(3, 3).T
    matrix[:3, :3] = np.float32(directions * np.asarray(fields[4:7], dtype=np.float32))
    center_voxel = np.asarray(fields[1:4], dtype=np.float64) / 2.0
    for axis in range(3):
        offset = np.float32(sum(float(matrix[axis, j]) * center_voxel[j] for j in range(3)))
        matrix[axis, 3] = np.float32(np.float32(fields[16 + axis]) - offset)
    return matrix


def read_warp_geometries(image: object) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """读取固定FS DISP_RAS warp的几何并验证编码、spacing=1及(X,Y,Z,1,3)。

    image为nibabel NIfTI；source/target为scanner RAS voxel-to-mm矩阵。
    不接受ABS_RAS/DISP_CRS、多个向量帧、非单位节点间距或损坏的FS扩展。
    返回source矩阵、target矩阵和source网格；没有合法扩展时抛ValueError。
    """
    if len(image.shape) != 5 or tuple(image.shape[3:]) != (1, 3):
        raise ValueError("FS warp must have complete shape (X,Y,Z,1,3)")
    if int(image.header["intent_code"]) != 1006:
        raise ValueError("FS warp must have displacement-vector NIfTI intent")
    for extension in image.header.extensions:
        if extension.get_code() != 14:
            continue
        payload = extension.get_content()
        if payload[:4] != b">\x00\x03\x01":
            continue
        cursor, tags = 4, {}
        while cursor + 12 <= len(payload):
            tag, length = struct.unpack_from(">iq", payload, cursor)
            end = cursor + 12 + length
            if length < 0 or end > len(payload):
                raise ValueError("invalid FreeSurfer warp tag length")
            if tag in tags:
                raise ValueError("duplicate FreeSurfer warp metadata tag")
            tags[tag] = (cursor + 12, length)
            cursor = end
            if tag == -1:
                break
        if 13 not in tags or tags[13][1] != 12:
            raise ValueError("missing FreeSurfer warp format metadata")
        encoding, spacing, exponent = struct.unpack_from(">iif", payload, tags[13][0])
        if encoding != 3 or spacing != 1 or not np.isfinite(exponent):
            raise ValueError("fixed FS warp requires DISP_RAS encoding and spacing=1")
        if 15 not in tags:
            raise ValueError("missing FreeSurfer warp geometry extension")
        start, length = tags[15]
        if length < 184:
            raise ValueError("truncated FreeSurfer warp geometry")
        source = struct.unpack_from(">4i18f", payload, start)
        source_name_length = struct.unpack_from(">i", payload, start + 88)[0]
        target_start = start + 92 + source_name_length
        if source_name_length < 0 or target_start + 92 > start + length:
            raise ValueError("invalid FreeSurfer source geometry length")
        target = struct.unpack_from(">4i18f", payload, target_start)
        target_name_length = struct.unpack_from(">i", payload, target_start + 88)[0]
        if target_name_length < 0 or target_start + 92 + target_name_length != start + length:
            raise ValueError("invalid FreeSurfer target geometry length")
        if (source[0] != 1 or target[0] != 1 or
                tuple(target[1:4]) != image.shape[:3] or min(source[1:4]) < 1):
            raise ValueError("invalid FreeSurfer warp geometry extension")
        source_matrix, target_matrix = _freesurfer_vox2ras(source), _freesurfer_vox2ras(target)
        if not np.isfinite(source_matrix).all() or not np.isfinite(target_matrix).all():
            raise ValueError("nonfinite FreeSurfer warp geometry")
        return source_matrix, target_matrix, tuple(source[1:4])
    raise ValueError("missing FreeSurfer warp geometry extension")


def _inverse_4x4_native(matrix: np.ndarray) -> np.ndarray:
    """Float32 cofactor inversion used by VNL for this fixed 4x4 geometry."""
    matrix = np.asarray(matrix, dtype=np.float32)
    cofactors = np.empty((4, 4), dtype=np.float32)
    for row in range(4):
        for column in range(4):
            a, b, c, d, e, f, g, h, i = np.delete(np.delete(matrix, row, axis=0), column, axis=1).flat
            minor = a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g
            cofactors[row, column] = minor if (row + column) % 2 == 0 else -minor
    determinant = (
        matrix[0, 0] * cofactors[0, 0]
        + matrix[0, 1] * cofactors[0, 1]
        + matrix[0, 2] * cofactors[0, 2]
        + matrix[0, 3] * cofactors[0, 3]
    )
    return cofactors.T * np.float32(np.float32(1.0) / determinant)


def warp_to_source_voxels(
    displacement_ras: np.ndarray,
    atlas_vox2ras: np.ndarray,
    source_vox2ras: np.ndarray,
) -> np.ndarray:
    """Convert atlas-grid RAS displacement vectors to source voxel coordinates."""
    displacement = np.asarray(displacement_ras, dtype=np.float32)
    if displacement.ndim != 4 or displacement.shape[-1] != 3:
        raise ValueError("expected displacement with shape (X, Y, Z, 3)")
    atlas = np.asarray(atlas_vox2ras, dtype=np.float32)
    inverse_source = _inverse_4x4_native(source_vox2ras)
    if atlas.shape != (4, 4) or inverse_source.shape != (4, 4):
        raise ValueError("expected two 4x4 voxel-to-RAS matrices")

    shape = displacement.shape[:3]
    coords = np.empty(displacement.shape, dtype=np.float32)
    x = np.arange(shape[0], dtype=np.float64)[:, None]
    y = np.arange(shape[1], dtype=np.float64)[None, :]
    for z in range(shape[2]):
        ras = np.empty((shape[0], shape[1], 3), dtype=np.float32)
        for axis in range(3):
            atlas_ras = np.float32(
                float(atlas[axis, 0]) * x
                + float(atlas[axis, 1]) * y
                + float(atlas[axis, 2]) * z
                + float(atlas[axis, 3])
            )
            ras[..., axis] = np.float32(
                atlas_ras.astype(np.float64) + displacement[:, :, z, axis].astype(np.float64)
            )
        for axis in range(3):
            value = np.zeros(shape[:2], dtype=np.float64)
            for column in range(3):
                value += float(inverse_source[axis, column]) * ras[..., column].astype(np.float64)
            value += float(inverse_source[axis, 3])
            coords[:, :, z, axis] = value.astype(np.float32)
    return coords


@njit(cache=True)
def _splat_counts(node_coordinates: np.ndarray, width: int, height: int, depth: int) -> np.ndarray:
    counts = np.zeros((width, height, depth), dtype=np.float32)
    for z in range(node_coordinates.shape[2]):
        for y in range(node_coordinates.shape[1]):
            for x in range(node_coordinates.shape[0]):
                # GCAMinvert clips only outside [0,size), then the native
                # MRIinterpolateIntoVolume rejects by MRIindexNotInVolume/rint.
                xf = max(float(node_coordinates[x, y, z, 0]), 0.0)
                yf = max(float(node_coordinates[x, y, z, 1]), 0.0)
                zf = max(float(node_coordinates[x, y, z, 2]), 0.0)
                if xf >= width: xf = width - 1.0
                if yf >= height: yf = height - 1.0
                if zf >= depth: zf = depth - 1.0
                if np.rint(xf) >= width or np.rint(yf) >= height or np.rint(zf) >= depth:
                    continue
                xm, ym, zm = int(xf), int(yf), int(zf)
                xp, yp, zp = min(xm + 1, width - 1), min(ym + 1, height - 1), min(zm + 1, depth - 1)
                xmd, ymd, zmd = xf - xm, yf - ym, zf - zm
                xpd, ypd, zpd = 1.0 - xmd, 1.0 - ymd, 1.0 - zmd
                counts[xm, ym, zm] += xpd * ypd * zpd
                counts[xm, ym, zp] += xpd * ypd * zmd
                counts[xm, yp, zm] += xpd * ymd * zpd
                counts[xm, yp, zp] += xpd * ymd * zmd
                counts[xp, ym, zm] += xmd * ypd * zpd
                counts[xp, ym, zp] += xmd * ypd * zmd
                counts[xp, yp, zm] += xmd * ymd * zpd
                counts[xp, yp, zp] += xmd * ymd * zmd
    return counts


def splat_inverse_counts(node_coordinates: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    """Sequential float32 trilinear accumulation from ``GCAMinvert``."""
    positions = np.asarray(node_coordinates, dtype=np.float32)
    if positions.ndim != 4 or positions.shape[-1] != 3 or len(image_shape) != 3:
        raise ValueError("expected node coordinates (X, Y, Z, 3) and image shape (3,)")
    return _splat_counts(positions, *map(int, image_shape))


@njit(cache=True)
def _splat_coordinate_sums(positions: np.ndarray, width: int, height: int, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums = np.zeros((3, width, height, depth), dtype=np.float32)
    for z in range(positions.shape[2]):
        for y in range(positions.shape[1]):
            for x in range(positions.shape[0]):
                # GCAMinvert clips only outside [0,size), then the native
                # MRIinterpolateIntoVolume rejects by MRIindexNotInVolume/rint.
                xf = max(float(positions[x, y, z, 0]), 0.0)
                yf = max(float(positions[x, y, z, 1]), 0.0)
                zf = max(float(positions[x, y, z, 2]), 0.0)
                if xf >= width: xf = width - 1.0
                if yf >= height: yf = height - 1.0
                if zf >= depth: zf = depth - 1.0
                if np.rint(xf) >= width or np.rint(yf) >= height or np.rint(zf) >= depth:
                    continue
                xm, ym, zm = int(xf), int(yf), int(zf)
                xp, yp, zp = min(xm + 1, width - 1), min(ym + 1, height - 1), min(zm + 1, depth - 1)
                xmd, ymd, zmd = xf - xm, yf - ym, zf - zm
                xpd, ypd, zpd = 1.0 - xmd, 1.0 - ymd, 1.0 - zmd
                for dx in range(2):
                    cx = xm if dx == 0 else xp
                    wx = xpd if dx == 0 else xmd
                    for dy in range(2):
                        cy = ym if dy == 0 else yp
                        wy = ypd if dy == 0 else ymd
                        for dz in range(2):
                            cz = zm if dz == 0 else zp
                            wz = zpd if dz == 0 else zmd
                            weight = wx * wy * wz
                            sums[0, cx, cy, cz] += weight * x
                            sums[1, cx, cy, cz] += weight * y
                            sums[2, cx, cy, cz] += weight * z
    return sums[0], sums[1], sums[2]


def splat_inverse_coordinate_sums(
    node_coordinates: np.ndarray, image_shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate atlas x/y/z coordinates before inverse-GCAM averaging."""
    positions = np.asarray(node_coordinates, dtype=np.float32)
    if positions.ndim != 4 or positions.shape[-1] != 3 or len(image_shape) != 3:
        raise ValueError("expected node coordinates (X, Y, Z, 3) and image shape (3,)")
    return _splat_coordinate_sums(positions, *map(int, image_shape))
