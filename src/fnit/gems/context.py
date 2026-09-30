"""Shared native-grid inputs for GEMS subregion recipes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from scipy import ndimage


def _native_labels(value, image: nib.spatialimages.SpatialImage, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        labels = value
    else:
        source = nib.load(str(value)) if isinstance(value, (str, Path)) else value
        if source.shape != image.shape or not np.allclose(source.affine, image.affine, atol=1e-5):
            raise ValueError(f"{name} must have the input T1 shape and affine")
        labels = np.asanyarray(source.dataobj)
    if labels.shape != image.shape:
        raise ValueError(f"{name} must have the input T1 shape")
    return np.asarray(labels, dtype=np.int32)


def build_wmparc_proxy(coarse: np.ndarray, cortical: np.ndarray, *,
                       voxel_sizes: tuple[float, float, float] = (1, 1, 1),
                       max_distance_mm: float = 15) -> np.ndarray:
    """Assign nearby DKT temporal parcels to white matter within each hemisphere.

    FreeSurfer's hippocampal intensity prior samples WM labels 3006, 3007,
    3016 and the corresponding right-side 4000-series labels. The other
    wmparc labels retain their coarse anatomical identity.
    """
    coarse = np.asarray(coarse)
    cortical = np.asarray(cortical)
    if coarse.ndim != 3 or cortical.shape != coarse.shape:
        raise ValueError("coarse and cortical labels must share a 3-D grid")
    proxy = coarse.astype(np.int32, copy=True)
    for wm_id, source_ids, offset in ((2, (1006, 1007, 1016), 2000),
                                      (41, (2006, 2007, 2016), 2000)):
        wm = coarse == wm_id
        source = np.isin(cortical, source_ids)
        if not wm.any() or not source.any():
            continue
        low = np.maximum(np.argwhere(wm | source).min(0) - 1, 0)
        high = np.minimum(np.argwhere(wm | source).max(0) + 2, coarse.shape)
        crop = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        distance, nearest = ndimage.distance_transform_edt(
            ~source[crop], sampling=voxel_sizes, return_indices=True)
        propagated = cortical[crop][tuple(nearest)] + offset
        local = proxy[crop]
        selected = wm[crop] & (distance <= max_distance_mm)
        local[selected] = propagated[selected]
    return proxy


@dataclass
class SubregionContext:
    image: nib.spatialimages.SpatialImage
    data: np.ndarray
    coarse_segmentation: np.ndarray | None
    cortical_parcellation: np.ndarray | None
    wmparc_proxy: np.ndarray | None

    @classmethod
    def prepare(cls, t1, *, need_coarse: bool, need_parc: bool,
                coarse_segmentation=None, cortical_parcellation=None, wmparc=None,
                synthseg_weights=None, synthseg_parc_weights=None,
                device="cuda:0") -> "SubregionContext":
        image = nib.load(str(t1)) if isinstance(t1, (str, Path)) else t1
        data = np.asanyarray(image.dataobj, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError("segment_subregions expects one 3-D T1 image")
        coarse = None if coarse_segmentation is None else _native_labels(coarse_segmentation, image, "coarse_segmentation")
        parc = None if cortical_parcellation is None else _native_labels(cortical_parcellation, image, "cortical_parcellation")
        wm = None if wmparc is None else _native_labels(wmparc, image, "wmparc")
        use_plus = need_parc and parc is None and wm is None
        if (use_plus or need_coarse and coarse is None) and synthseg_weights is None and synthseg_parc_weights is None:
            from ..weights import (MODEL_FILES, WEIGHT_FILES, cache_dir, configured_dir,
                                   download_file, resolve_weights, verify_file)
            names = MODEL_FILES["synthseg-plus" if use_plus else "synthseg"]
            try:
                if not all(verify_file(resolve_weights(name), *WEIGHT_FILES[name][1:])
                           for name in names):
                    raise FileNotFoundError("unverified SynthSeg weights")
            except FileNotFoundError:
                destination = configured_dir() or cache_dir()
                for name in names:
                    download_file(name, destination)
                synthseg_weights = destination
                synthseg_parc_weights = destination
        if need_parc and parc is None and wm is None:
            from ..synthseg_parc import SynthSegPlus
            if isinstance(t1, (str, Path)):
                predicted = SynthSegPlus(weights=synthseg_weights,
                                         parc_weights=synthseg_parc_weights,
                                         device=device)(t1, keep_geometry=True)
            else:
                with TemporaryDirectory(prefix="fnit-subregions-") as directory:
                    source = Path(directory) / "t1.nii.gz"
                    nib.save(image, source)
                    predicted = SynthSegPlus(weights=synthseg_weights,
                                             parc_weights=synthseg_parc_weights,
                                             device=device)(source, keep_geometry=True)
            if coarse is None:
                coarse = _native_labels(predicted.segmentation, image, "SynthSeg segmentation")
            if parc is None:
                parc = _native_labels(predicted.cortical_parcellation, image, "SynthSeg parcellation")
        elif need_coarse and coarse is None:
            from ..synthseg_parc import SynthSeg
            predicted = SynthSeg(weights=synthseg_weights, device=device)(t1, keep_geometry=True)
            coarse = _native_labels(predicted.segmentation, image, "SynthSeg segmentation")
        if need_parc and wm is None:
            if parc is None:
                raise ValueError("cortical_parcellation is required to build wmparc proxy")
            wm = build_wmparc_proxy(
                coarse, parc, voxel_sizes=tuple(np.linalg.norm(image.affine[:3, :3], axis=0)))
        return cls(image, data, coarse, parc, wm)
