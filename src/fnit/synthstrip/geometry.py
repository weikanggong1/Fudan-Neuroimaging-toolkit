"""Small image geometry operations used by SynthStrip without Surfa."""

import os
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
from scipy import ndimage


class Volume:
    """Image data and RAS voxel affine with ``save`` compatible with StripResult."""

    def __init__(self, data, affine, voxsize=None):
        self.data = np.asarray(data)
        self.affine = np.asarray(affine, dtype=np.float64)
        if voxsize is None:
            _, r = np.linalg.qr(self.affine[:3, :3])
            voxsize = np.abs(np.diag(r))
        self.voxsize = np.asarray(voxsize, dtype=np.float64)
        self.geom = SimpleNamespace(
            vox2world=SimpleNamespace(matrix=self.affine), voxsize=self.voxsize,
            shape=np.asarray(self.data.shape[:3]),
        )

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def nframes(self):
        return 1 if self.data.ndim == 3 else self.data.shape[3]

    @property
    def framed_data(self):
        return self.data[..., None] if self.data.ndim == 3 else self.data

    def new(self, data, affine=None, voxsize=None):
        return Volume(data, self.affine if affine is None else affine,
                      self.voxsize if voxsize is None else voxsize)

    def copy(self):
        return self.new(self.data.copy())

    def min(self):
        return self.data.min()

    def max(self):
        return self.data.max()

    def percentile(self, q):
        return np.percentile(self.data, q)

    def save(self, path):
        path = Path(path)
        data = self.data.astype(np.uint8) if self.data.dtype == np.bool_ else self.data
        if path.suffix.lower() in ('.mgz', '.mgh'):
            data = data.astype(np.float32) if data.dtype not in (np.uint8, np.int16, np.int32, np.float32) else data
            image = nib.MGHImage(data, self.affine)
        else:
            image = nib.Nifti1Image(data, self.affine)
        nib.save(image, str(path))


def load_volume(image):
    if isinstance(image, Volume):
        return image
    if isinstance(image, (str, os.PathLike)):
        image = nib.load(str(image))
    elif not isinstance(image, nib.spatialimages.SpatialImage):
        return Volume(image.data, image.geom.vox2world.matrix, image.geom.voxsize)
    if isinstance(image, nib.MGHImage):
        header = image.header
        affine = np.eye(4)
        affine[:3, :3] = np.asarray(header['Mdc'], dtype=np.float64).T @ np.diag(
            np.asarray(header['delta'], dtype=np.float64))
        affine[:3, 3] = np.asarray(header['Pxyz_c'], dtype=np.float64) - (
            affine[:3, :3] @ (np.asarray(image.shape[:3], dtype=np.float64) / 2))
        return Volume(np.asanyarray(image.dataobj), affine, header['delta'])
    return Volume(np.asanyarray(image.dataobj), image.affine, image.header.get_zooms()[:3])


def conform_lia(image):
    original_shape = image.shape[:3]
    original_orientation = nib.orientations.io_orientation(image.affine)
    target_orientation = nib.orientations.axcodes2ornt(('L', 'I', 'A'))
    orientation = nib.orientations.ornt_transform(original_orientation, target_orientation)
    data = nib.orientations.apply_orientation(image.data, orientation)
    affine = image.affine @ nib.orientations.inv_ornt_aff(orientation, original_shape)
    voxsize = image.voxsize[orientation[:, 0].astype(int)]
    target_shape = np.ceil(voxsize * np.asarray(data.shape[:3])).astype(int)
    resized = not np.allclose(voxsize, 1.0, atol=1e-5, rtol=0)
    if resized:
        q, r = np.linalg.qr(affine[:3, :3])
        p = np.eye(3)
        p[np.diag_indices(3)] = np.diag(r) / np.abs(np.diag(r))
        rotation = q @ p
        center = (affine @ np.r_[np.asarray(data.shape[:3]) / 2, 1])[:3]
        new_affine = np.eye(4)
        new_affine[:3, :3] = rotation
        new_affine[:3, 3] = center - (new_affine @ np.r_[target_shape / 2, 1])[:3]
        matrix = np.linalg.inv(affine) @ new_affine
        data = ndimage.affine_transform(data, matrix[:3, :3], matrix[:3, 3],
                                        output_shape=tuple(target_shape), order=0,
                                        mode='constant', cval=0, prefilter=False)
        affine = new_affine
    return Volume(np.array(data, dtype=np.float32, copy=True), affine,
                  np.ones(3) if resized else voxsize)


def crop_bbox(image):
    points = np.nonzero(image.data > 0)
    if not points[0].size:
        return image.copy()
    start = np.array([axis.min() for axis in points])
    stop = np.array([axis.max() + 1 for axis in points])
    slices = tuple(slice(a, b) for a, b in zip(start, stop))
    affine = image.affine.copy()
    affine[:3, 3] = (image.affine @ np.r_[start, 1])[:3]
    return image.new(image.data[slices], affine=affine)


def reshape(image, shape):
    shape = np.asarray(shape[:3], dtype=int)
    delta = (shape - np.asarray(image.shape[:3])) / 2
    low, high = np.floor(delta).astype(int), np.ceil(delta).astype(int)
    padding = [(max(n, 0), max(m, 0)) for n, m in zip(low, high)]
    data = np.pad(image.data, padding, mode='constant')
    c_low = np.maximum(-high, 0)
    c_high = np.asarray(data.shape[:3]) - np.maximum(-low, 0)
    data = data[tuple(slice(a, b) for a, b in zip(c_low, c_high))]
    origin = np.maximum(-high, 0) - np.maximum(low, 0)
    affine = image.affine.copy()
    affine[:3, 3] = (image.affine @ np.r_[origin, 1])[:3]
    return image.new(data, affine=affine)


def resample_linear(image, target, fill=100):
    # The reference resampler casts the voxel transform and coordinates to float32.
    affine = (np.linalg.inv(image.affine) @ target.affine).astype(np.float32)
    result = np.empty(target.shape[:3], dtype=image.data.dtype)
    source_shape = np.asarray(image.shape[:3], dtype=np.float32)[:, None]
    for start in range(0, result.shape[0], 16):
        stop = min(start + 16, result.shape[0])
        grid = np.indices((stop - start, *result.shape[1:]),
                          dtype=np.float32).reshape(3, -1)
        grid[0] += start
        coordinates = np.empty_like(grid)
        for axis in range(3):
            coordinates[axis] = (
                (affine[axis, 0] * grid[0] + affine[axis, 1] * grid[1])
                + affine[axis, 2] * grid[2]
            ) + affine[axis, 3]
        inside = np.all((coordinates >= 0) & (coordinates < source_shape), axis=0)
        values = ndimage.map_coordinates(image.data, coordinates, order=1,
                                         mode='nearest', prefilter=False,
                                         output=image.data.dtype)
        values[~inside] = fill
        result[start:stop] = values.reshape(stop - start, *result.shape[1:])
    return target.new(result)


def largest_filled_component(mask):
    labels, _ = ndimage.label(mask)
    counts = np.bincount(labels.flat)[1:]
    if not counts.size:
        return np.zeros_like(mask, dtype=bool)
    largest = (-counts).argsort()[:1] + 1
    return ndimage.binary_fill_holes(np.isin(labels, largest))
