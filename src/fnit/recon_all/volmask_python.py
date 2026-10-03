"""Build FreeSurfer's bilateral white/cortical ribbon from four surfaces."""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
import struct

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _owns_projected_edge(dy: float, dz: float) -> bool:
    """One owner for a shared edge after normalizing projected winding."""
    return dz > 0 or (dz == 0 and dy < 0)


@njit(cache=True, inline="always")
def _projected_edge(first: np.ndarray, second: np.ndarray,
                    first_id: int, second_id: int, yy: float, zz: float) -> float:
    """Evaluate a shared edge in canonical vertex-ID order, then orient it."""
    forward = first_id < second_id
    low, high = (first, second) if forward else (second, first)
    value = ((low[1] - yy) * (high[2] - zz)
             - (low[2] - zz) * (high[1] - yy))
    return value if forward else -value


@njit(cache=True)
def _inside_mesh(vertices: np.ndarray, faces: np.ndarray,
                 shape: tuple[int, int, int]) -> tuple[np.ndarray, int, int]:
    # Keep saved surface coordinates; only the ray/triangle predicate and
    # intersection arithmetic use float64. Mixed float32 determinants and
    # float64 numerators can count both triangles next to a shared edge.
    vertices = vertices.astype(np.float64)
    width, height, depth = shape
    crossings = np.zeros((height, depth, 128), dtype=np.float64)
    counts = np.zeros((height, depth), dtype=np.int32)
    offset = 1e-5
    for face in faces:
        a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        denominator = (b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2])
        # Only a zero projection is parallel/degenerate. A small nonzero
        # projected triangle can still contain a sampled ray.
        if denominator == 0:
            continue
        sign = 1.0 if denominator > 0 else -1.0
        # Bounds must cover the actual sample (y + offset, z + offset).
        y0 = max(0, int(np.ceil(min(a[1], b[1], c[1]) - offset)))
        y1 = min(height - 1, int(np.floor(max(a[1], b[1], c[1]) - offset)))
        z0 = max(0, int(np.ceil(min(a[2], b[2], c[2]) - offset)))
        z1 = min(depth - 1, int(np.floor(max(a[2], b[2], c[2]) - offset)))
        owns_a = _owns_projected_edge(sign * (c[1] - b[1]), sign * (c[2] - b[2]))
        owns_b = _owns_projected_edge(sign * (a[1] - c[1]), sign * (a[2] - c[2]))
        owns_c = _owns_projected_edge(sign * (b[1] - a[1]), sign * (b[2] - a[2]))
        for y in range(y0, y1 + 1):
            yy = y + offset
            for z in range(z0, z1 + 1):
                zz = z + offset
                edge_a = _projected_edge(b, c, face[1], face[2], yy, zz)
                edge_b = _projected_edge(c, a, face[2], face[0], yy, zz)
                edge_c = _projected_edge(a, b, face[0], face[1], yy, zz)
                sa, sb, sc = sign * edge_a, sign * edge_b, sign * edge_c
                if ((sa > 0 or (sa == 0 and owns_a))
                        and (sb > 0 or (sb == 0 and owns_b))
                        and (sc > 0 or (sc == 0 and owns_c))):
                    count = counts[y, z]
                    if count < 128:
                        crossings[y, z, count] = (
                            edge_a * a[0] + edge_b * b[0] + edge_c * c[0]
                        ) / (edge_a + edge_b + edge_c)
                    counts[y, z] += 1
    inside = np.zeros(shape, dtype=np.uint8)
    odd = overflow = 0
    for y in range(height):
        for z in range(depth):
            count = counts[y, z]
            if count == 0:
                continue
            if count >= 128:
                overflow += 1
                continue
            if count % 2:
                odd += 1
            for i in range(1, count):
                value = crossings[y, z, i]
                j = i - 1
                while j >= 0 and crossings[y, z, j] > value:
                    crossings[y, z, j + 1] = crossings[y, z, j]
                    j -= 1
                crossings[y, z, j + 1] = value
            position = 0
            for x in range(width):
                while position < count and crossings[y, z, position] <= x:
                    position += 1
                if position % 2:
                    inside[x, y, z] = 1
    return inside, odd, overflow


def ribbon_arrays(
    shape: tuple[int, int, int], vox2ras_tkr: np.ndarray,
    surfaces: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return combined 0/2/3/41/42 mask and both binary cortical ribbons."""
    if len(shape) != 3 or np.shape(vox2ras_tkr) != (4, 4) or set(surfaces) != {"lh", "rh"}:
        raise ValueError("expected a 3D grid and left/right white/pial surfaces")
    inverse = np.linalg.inv(vox2ras_tkr)
    hemispheres = {}
    for hemi in ("lh", "rh"):
        if set(surfaces[hemi]) != {"white", "pial"}:
            raise ValueError("each hemisphere needs white and pial surfaces")
        masks = {}
        for name in ("white", "pial"):
            xyz, faces = surfaces[hemi][name]
            if xyz.ndim != 2 or xyz.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError("invalid ordered surface geometry")
            voxels = (xyz @ inverse[:3, :3].T + inverse[:3, 3]).astype(np.float32)
            mask, odd, overflow = _inside_mesh(voxels, faces.astype(np.int32), shape)
            if odd or overflow:
                raise ValueError(f"{hemi}.{name} has {odd} odd and {overflow} overflowing rays")
            masks[name] = mask.astype(bool)
        hemispheres[hemi] = masks
    left = np.where(hemispheres["lh"]["white"], 2,
                    np.where(hemispheres["lh"]["pial"], 3, 0)).astype(np.uint8)
    right = np.where(hemispheres["rh"]["white"], 41,
                     np.where(hemispheres["rh"]["pial"], 42, 0)).astype(np.uint8)
    combined = np.where(left != 0, left, right).astype(np.uint8)
    return combined, (left == 3).astype(np.uint8), (right == 42).astype(np.uint8)


def _color_lut_tags(color_lut_file: str | Path) -> bytes:
    """Serialize the fixed FreeSurfer 8.2 MGH color table footer."""
    color_lut_file = Path(color_lut_file)
    entries = {}
    for line in color_lut_file.read_text(encoding="utf-8").splitlines():
        columns = line.split()
        if len(columns) < 6:
            continue
        try:
            index = int(columns[0])
            colors = tuple(int(value) for value in columns[2:6])
        except ValueError:
            continue
        entries.setdefault(index, (columns[1], colors))
    if not entries:
        raise ValueError("color lookup table has no entries")
    pack_int = lambda value: struct.pack(">i", value)
    filename = os.fsencode(color_lut_file) + b"\0"
    tags = bytearray(pack_int(41) + struct.pack(">q", 7) + b"UNKNOWN")
    tags += pack_int(43) + struct.pack(">q", 4) + pack_int(0)
    tags += pack_int(1) + pack_int(-2) + pack_int(max(entries) + 1)
    tags += pack_int(len(filename)) + filename + pack_int(len(entries))
    for index, (name, colors) in sorted(entries.items()):
        encoded = name.encode("utf-8") + b"\0"
        tags += pack_int(index) + pack_int(len(encoded)) + encoded
        tags += b"".join(pack_int(value) for value in colors)
    return bytes(tags)


def write_ribbon(template_file: str | Path, surface_dir: str | Path,
                 output_dir: str | Path, color_lut_file: str | Path) -> None:
    template = nib.load(str(template_file))
    if not isinstance(template, nib.MGHImage) or len(template.shape) != 3:
        raise ValueError("template must be a 3D MGH/MGZ volume")
    surface_dir, output_dir = Path(surface_dir), Path(output_dir)
    surfaces = {hemi: {name: fsio.read_geometry(str(surface_dir / f"{hemi}.{name}"))
                       for name in ("white", "pial")}
                for hemi in ("lh", "rh")}
    arrays = ribbon_arrays(template.shape, template.header.get_vox2ras_tkr(), surfaces)
    footer = _color_lut_tags(color_lut_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in zip(("ribbon", "lh.ribbon", "rh.ribbon"), arrays):
        header = template.header.copy()
        header.set_data_dtype(np.uint8)
        path = output_dir / f"{name}.mgz"
        nib.save(nib.MGHImage(values, template.affine, header=header), str(path))
        path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()) + footer, mtime=0))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template_file", type=Path)
    parser.add_argument("surface_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("color_lut_file", type=Path)
    args = parser.parse_args(argv)
    write_ribbon(args.template_file, args.surface_dir, args.output_dir, args.color_lut_file)


if __name__ == "__main__":
    main()
