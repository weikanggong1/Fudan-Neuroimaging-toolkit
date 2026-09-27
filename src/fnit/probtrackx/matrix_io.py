"""Volume matrix indexing and sparse output compatible with FSL probtrackx2."""

from pathlib import Path

import numpy as np


def ordered_voxels(mask):
    """Return nonzero internal voxel coordinates in FSL's z, y, x order."""
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError("volume mask must be 3D")
    return np.argwhere(mask.transpose(2, 1, 0) != 0)[:, ::-1]


def write_dot(path, entries, shape):
    """Write FSL's 1-based sparse triplets and mandatory shape sentinel."""
    nrows, ncols = map(int, shape)
    if nrows < 0 or ncols < 0:
        raise ValueError("matrix dimensions must be nonnegative")
    lines = []
    for (row, col), value in sorted(entries.items(), key=lambda item: (item[0][1], item[0][0])):
        if not (0 <= row < nrows and 0 <= col < ncols):
            raise ValueError("matrix entry is outside its shape")
        if value:
            lines.append(f"{row + 1}  {col + 1}  {value:.8g}\n")
    lines.append(f"{nrows}  {ncols}  0\n")
    Path(path).write_text("".join(lines))


def write_volume_coords(path, masks, *, flip_x=False, with_roi=True):
    """Write FSL volume coordinate table; masks are in internal voxel space.

    `with_roi=True` writes x, y, z, zero-based ROI index, and one-based
    unique-volume location. `False` writes x, y, z for target2/target4.
    """
    masks = [masks] if isinstance(masks, np.ndarray) else list(masks)
    if not masks:
        raise ValueError("at least one volume mask is required")
    shape = np.asarray(masks[0]).shape
    if any(np.asarray(mask).shape != shape for mask in masks):
        raise ValueError("volume masks must have the same shape")
    seen = {}
    lines = []
    for roi, mask in enumerate(masks):
        for x, y, z in ordered_voxels(mask):
            key = (int(x), int(y), int(z))
            seen.setdefault(key, len(seen) + 1)
            x_out = shape[0] - 1 - x if flip_x else x
            values = (x_out, y, z, roi, seen[key]) if with_roi else (x_out, y, z)
            lines.append("  ".join(map(str, values)) + "  \n")
    Path(path).write_text("".join(lines))
