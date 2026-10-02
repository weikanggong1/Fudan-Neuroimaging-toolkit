"""Surface data contracts and geometry checks for fsLR32k projection."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class SurfaceHemisphere:
    """Paths for one hemisphere, with matching native and fsLR32k topology."""

    white: str | Path
    pial: str | Path
    midthickness: str | Path
    registered_sphere: str | Path
    native_roi: str | Path
    atlas_sphere: str | Path
    atlas_midthickness: str | Path
    atlas_roi: str | Path


@dataclass(frozen=True)
class SurfaceProjectionResult:
    """The dense timeseries, cortical metrics, volume source and timing."""

    dtseries: Path
    left_metric: Path
    right_metric: Path
    subcortical_volume: Path
    timing_seconds: dict[str, float]
    coverage_report: Path


def _check_gifti_hemisphere(image, name, hemisphere):
    """Reject an explicitly declared opposite hemisphere, including array metadata."""
    expected = str(hemisphere).lower()
    expected = "left" if expected in ("l", "left") else "right"
    declarations = [image.meta, *(array.meta for array in image.darrays)]
    for metadata in declarations:
        value = metadata.get("AnatomicalStructurePrimary", "")
        normalized = "".join(character for character in str(value).lower() if character.isalnum())
        declared = {
            "left": "left", "right": "right",
            "cortexleft": "left", "cortexright": "right",
            "ciftistructurecortexleft": "left", "ciftistructurecortexright": "right",
        }.get(normalized)
        if declared is not None and declared != expected:
            raise ValueError(f"{name} declares the opposite hemisphere: {value}")


def _gifti_count(path, name, *, metric=False, hemisphere=None):
    image = nib.load(str(path))
    if not isinstance(image, nib.GiftiImage):
        raise ValueError(f"{name} must be a GIFTI file")
    if hemisphere is not None:
        _check_gifti_hemisphere(image, name, hemisphere)
    arrays = image.darrays if metric else [item for item in image.darrays if item.intent == 1008]
    if len(arrays) != 1:
        raise ValueError(f"{name} must have exactly one {'metric' if metric else 'pointset'} array")
    data = np.asarray(arrays[0].data)
    if data.ndim != (1 if metric else 2) or (not metric and data.shape[1] != 3):
        raise ValueError(f"{name} has invalid vertex data")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains nonfinite values")
    if metric and (data < 0).any():
        raise ValueError(f"{name} ROI must be nonnegative")
    return len(data), int(np.count_nonzero(data)) if metric else None


def _hemisphere_paths(hemi, name):
    paths = {field: Path(getattr(hemi, field)).expanduser().resolve()
             for field in SurfaceHemisphere.__dataclass_fields__}
    for field, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}.{field}: {path}")
    native = [_gifti_count(paths[field], f"{name}.{field}", hemisphere=name)[0]
              for field in ("white", "pial", "midthickness", "registered_sphere")]
    native_roi, native_nonzero = _gifti_count(paths["native_roi"], f"{name}.native_roi", metric=True, hemisphere=name)
    atlas = [_gifti_count(paths[field], f"{name}.{field}", hemisphere=name)[0]
             for field in ("atlas_sphere", "atlas_midthickness")]
    atlas_roi, atlas_nonzero = _gifti_count(paths["atlas_roi"], f"{name}.atlas_roi", metric=True, hemisphere=name)
    if len(set((*native, native_roi))) != 1 or len(set((*atlas, atlas_roi))) != 1:
        raise ValueError(f"{name} surface and ROI vertex counts do not match")
    if atlas_roi != 32492 or not native_nonzero or not atlas_nonzero:
        raise ValueError(f"{name} requires nonempty native ROI and fsLR32k atlas ROI")

    def triangles(field, vertices):
        image = nib.load(str(paths[field]))
        arrays = [item for item in image.darrays if item.intent == 1009]
        if len(arrays) != 1:
            raise ValueError(f"{name}.{field} must have one triangle array")
        faces = np.asarray(arrays[0].data)
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= vertices):
            raise ValueError(f"{name}.{field} has invalid triangle indices")
        return faces

    native_faces = triangles("white", native[0])
    for field in ("pial", "midthickness", "registered_sphere"):
        if not np.array_equal(triangles(field, native[0]), native_faces):
            raise ValueError(f"{name}.{field} does not share native vertex order")
    atlas_faces = triangles("atlas_sphere", atlas[0])
    if not np.array_equal(triangles("atlas_midthickness", atlas[0]), atlas_faces):
        raise ValueError(f"{name}.atlas_midthickness does not share atlas vertex order")
    return paths
