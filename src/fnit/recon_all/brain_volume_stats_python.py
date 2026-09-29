"""Compute FreeSurfer 8.2 brain-volume measures from matched inputs."""

from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np


MEASURE_NAMES = (
    "BrainSegVol", "BrainSegVolNotVent", "SupraTentorialVol",
    "SupraTentorialVolNotVent", "SubCortGrayVol", "lhCortexVol",
    "rhCortexVol", "CortexVol", "TotalGrayVol",
    "lhCerebralWhiteMatterVol", "rhCerebralWhiteMatterVol",
    "CerebralWhiteMatterVol", "MaskVol", "SupraTentorialVolNotVentVox",
    "BrainSegVolNotVentSurf", "VentricleChoroidVol",
)


def _surface_volume(path: Path) -> float:
    vertices, faces = fsio.read_geometry(str(path))
    a, b, c = (vertices[faces[:, index]] for index in range(3))
    cross = np.cross(b - a, c - a)
    length_squared = np.float32(np.float32(cross[:, 0] ** 2 + cross[:, 1] ** 2) +
                                cross[:, 2] ** 2)
    length = np.sqrt(length_squared.astype(np.float64))
    area = length * .5
    length32 = length.astype(np.float32)
    inverse_length = np.float32(1 / np.where(length32 < np.finfo(np.float32).eps,
                                             np.float32(1), length32))
    normal = np.float32(cross * inverse_length[:, None])
    center = np.float32(np.float32(a + b + c).astype(np.float64) / 3)
    dot = np.float32(np.float32(center[:, 0] * normal[:, 0] +
                                center[:, 1] * normal[:, 1]) +
                     center[:, 2] * normal[:, 2])
    return float(np.sum(dot.astype(np.float64) * area.astype(np.float64)) / 3)


def compute_brain_volume_stats(subject: str | Path,
                               aseg_lut: str | Path) -> dict[str, float]:
    """Replay the ``ComputeBrainVolumeStats2`` branch of recon-all.

    The subject must already contain ``aseg.mgz``, ``brainmask.mgz`` and both
    white/pial surfaces. ``aseg_lut`` is the small ASegStatsLUT.txt data file.
    """
    subject = Path(subject)
    surf = subject / "surf"
    lh_white = _surface_volume(surf / "lh.white")
    rh_white = _surface_volume(surf / "rh.white")
    lh_pial = _surface_volume(surf / "lh.pial")
    rh_pial = _surface_volume(surf / "rh.pial")
    aseg_image = nib.load(str(subject / "mri" / "aseg.mgz"))
    aseg = np.asarray(aseg_image.dataobj).astype(np.int16, copy=False)
    brainmask = np.asarray(nib.load(str(subject / "mri" / "brainmask.mgz")).dataobj)
    if aseg.shape != brainmask.shape:
        raise ValueError("Aseg and brainmask shapes differ")
    voxel = float(np.prod(aseg_image.header.get_zooms()[:3]))
    lut_ids = {int(line.split()[0]) for line in Path(aseg_lut).read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")}
    valid = (np.isin(aseg, tuple(lut_ids | {2, 3, 41, 42})) &
             ~np.isin(aseg, (0, 16, 85)))
    hypo_coordinates = np.argwhere(aseg == 77)
    left_hypo = right_hypo = 0
    if len(hypo_coordinates):
        xfm = subject / "mri/transforms/talairach.xfm"
        lines = xfm.read_text().splitlines()
        index = lines.index("Linear_Transform =")
        transform = np.array([[float(value) for value in line.rstrip(" ;").split()]
                              for line in lines[index + 1:index + 4]])
        scanner_ras = nib.affines.apply_affine(aseg_image.affine, hypo_coordinates)
        mni_x = scanner_ras @ transform[0, :3] + transform[0, 3]
        left_hypo = int(np.count_nonzero(mni_x <= 0))
        right_hypo = len(hypo_coordinates) - left_hypo

    def count(mask: np.ndarray) -> float:
        return float(np.count_nonzero(mask)) * voxel

    callosum = count(valid & np.isin(aseg, (251, 252, 253, 254, 255)))
    subcort = count(valid & np.isin(aseg, (10, 49, 11, 50, 12, 51, 13, 52,
                                           17, 53, 18, 54, 26, 58, 28, 60, 27, 59)))
    cerebellum_gray = count(valid & np.isin(aseg, (8, 47)))
    cerebellum = count(valid & np.isin(aseg, (7, 8, 46, 47)))
    vent = count(valid & np.isin(aseg, (4, 43, 5, 44, 31, 63)))
    tffc = count(valid & np.isin(aseg, (14, 15, 72, 24)))
    brain_seg = count(valid)
    mask = count(brainmask > 0)
    left_gray = lh_pial - lh_white
    right_gray = rh_pial - rh_white
    left_white = count(valid & np.isin(aseg, (2, 78))) + left_hypo * voxel + callosum / 2
    right_white = count(valid & np.isin(aseg, (41, 79))) + right_hypo * voxel + callosum / 2
    supratentorial = brain_seg - cerebellum
    brain_not_vent = brain_seg - vent - tffc
    supratentorial_not_vent = supratentorial - vent - tffc
    values = (brain_seg, brain_seg - vent - tffc, supratentorial,
              supratentorial_not_vent, subcort, left_gray, right_gray,
              left_gray + right_gray, subcort + left_gray + right_gray + cerebellum_gray,
              left_white, right_white, left_white + right_white, mask,
              supratentorial_not_vent / voxel, brain_not_vent, vent)
    return dict(zip(MEASURE_NAMES, values))
