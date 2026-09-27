"""Shared FLIRT file inputs and result type."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .._nib import FNITNifti1Image, load_image


def _load_volume(value, name):
    return load_image(value, name)


def _single_frame(volume, name):
    data = np.asanyarray(volume.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{name} must contain one 3D frame")
    if any(size < 2 for size in data.shape):
        raise ValueError(f"{name} dimensions must each contain at least two voxels")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(data, dtype=np.float32, copy=True)


@dataclass(frozen=True)
class FLIRTResult:
    """Reference-grid image and transforms returned by a FLIRT model."""

    moved: FNITNifti1Image
    matrix: np.ndarray
    moving_to_fixed_world: np.ndarray
    fixed_to_moving_world: np.ndarray
    qc: dict

    @property
    def fsl_matrix(self):
        """Moving/input to fixed/reference transform in FSL scaled-mm."""
        return self.matrix

    def save(self, output=None, omat=None):
        """Save the moved image and/or FSL ``.mat`` transform."""
        if (
            output is not None
            and omat is not None
            and Path(output).resolve() == Path(omat).resolve()
        ):
            raise ValueError("output and omat must use different paths")
        if output is not None:
            nib.save(self.moved, str(output))
        if omat is not None:
            np.savetxt(str(omat), self.matrix, fmt="%.12g")
        return self


__all__ = ["FLIRTResult"]
