"""Small nibabel image helpers shared by public FNIT pipelines."""

from __future__ import annotations

import os
from types import SimpleNamespace

import nibabel as nib
import numpy as np


class FNITNifti1Image(nib.Nifti1Image):
    """A NIfTI image with the ``save`` convenience used by FNIT results."""

    def save(self, path) -> None:
        nib.save(self, str(path))

    @property
    def data(self):
        """Array view retained for compatibility with existing FNIT consumers."""
        return np.asanyarray(self.dataobj)

    @property
    def geom(self):
        """Read-only shape, affine and voxel-size view without surfa."""
        return SimpleNamespace(
            shape=self.shape[:3],
            vox2world=SimpleNamespace(matrix=self.affine),
            voxsize=np.linalg.norm(self.affine[:3, :3], axis=0),
        )

    def new(self, data):
        return new_image(data, self)

    def copy(self):
        return new_image(np.array(self.dataobj, copy=True), self)


def load_image(value, name: str = "image") -> nib.spatialimages.SpatialImage:
    if isinstance(value, (str, os.PathLike)):
        return nib.load(str(value))
    if isinstance(value, nib.spatialimages.SpatialImage):
        return value
    raise TypeError(f"{name} must be a path or nibabel spatial image")


def new_image(data, reference: nib.spatialimages.SpatialImage, *, affine=None):
    """Create a NIfTI, retaining stored geometry when its grid is unchanged.

    Re-encoding an unchanged qform can round pixdim across an integer field
    of view boundary. Keep the reference NIfTI forms and zooms in that case;
    an explicit new affine or a non-NIfTI reference receives new forms.
    """
    array = np.asarray(data)
    header = nib.Nifti1Header.from_header(reference.header)
    header.set_data_dtype(array.dtype)
    target_affine = reference.affine if affine is None else affine
    image = FNITNifti1Image(array, target_affine, header)
    if (isinstance(reference.header, nib.Nifti1Header)
            and np.array_equal(target_affine, reference.affine)
            and array.shape[:3] == reference.shape[:3]):
        return image
    image.set_qform(target_affine, int(header["qform_code"]))
    # MGH has no NIfTI sform code. Code 0 would discard the supplied affine
    # and make nibabel fall back to a centered, axis-aligned voxel grid.
    image.set_sform(target_affine, max(1, int(header["sform_code"])))
    return image
