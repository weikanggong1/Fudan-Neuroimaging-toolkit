"""Brain mask from a 3D EPI reference without external executables."""

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def epi_brain_mask(reference, *, dilation=2):
    """Return a binary reference-grid NIfTI mask.

    Otsu thresholding on positive EPI intensities is followed by the largest
    connected component, cavity filling and a small dilation. This is an
    independent EPI mask; it does not reproduce BET's deformable surface.
    """
    image = nib.load(str(reference)) if not isinstance(reference, (nib.Nifti1Image, nib.Nifti2Image)) else reference
    if image.ndim != 3:
        raise ValueError("reference must be a 3D NIfTI image")
    if dilation < 0:
        raise ValueError("dilation must be nonnegative")
    data = np.asarray(image.dataobj, dtype=np.float32)
    positive = data[np.isfinite(data) & (data > 0)]
    if positive.size < 100:
        raise ValueError("reference has too few positive voxels to estimate a brain mask")
    histogram, edges = np.histogram(positive, bins=512)
    centers = (edges[1:] + edges[:-1]) / 2
    left_count = np.cumsum(histogram)
    right_count = left_count[-1] - left_count
    left_mean = np.cumsum(histogram * centers) / np.maximum(left_count, 1)
    right_mean = (
        np.cumsum((histogram * centers)[::-1]) / np.maximum(right_count[::-1], 1)
    )[::-1]
    score = left_count * right_count * (left_mean - right_mean) ** 2
    threshold = centers[int(np.argmax(score))]
    components, number = ndi.label(data > threshold)
    if number < 1:
        raise ValueError("no foreground component found")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    mask = ndi.binary_fill_holes(components == np.argmax(sizes))
    if dilation:
        mask = ndi.binary_dilation(mask, iterations=dilation)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(mask.astype(np.uint8), image.affine, header)
