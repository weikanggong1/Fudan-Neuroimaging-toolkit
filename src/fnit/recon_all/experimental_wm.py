"""Explicit historical baseline: derive WM and filled volumes from SynthSeg labels.

This approximation is retained for comparison with the earlier Python
recon-all profile. It is not the default Conda reconstruction path.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def synthseg_wm_fill(t1_file: str | Path, aseg: np.ndarray,
                     wm_file: str | Path, filled_file: str | Path) -> None:
    image = nib.load(str(t1_file))
    wm = np.where(np.isin(aseg, (2, 41, 77, 78, 79)), 255, 0).astype(np.uint8)
    filled = np.zeros(aseg.shape, np.uint8)
    for code, label in ((255, 2), (127, 41)):
        components, count = ndi.label(aseg == label)
        if count == 0:
            raise ValueError("hemisphere has no white matter")
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        filled[ndi.binary_fill_holes(components == np.argmax(sizes))] = code
    for path, array in ((wm_file, wm), (filled_file, filled)):
        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.MGHImage(np.ascontiguousarray(array), image.affine, header), str(path))
