"""FreeSurfer-independent SynthSeg 2.0 + volumetric cortical parcellation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

from .._nib import FNITNifti1Image
from ..weights import resolve_weights
from .labels import PARCELLATION_LABELS, PARCELLATION_NAME_BY_ID
from .pipeline import SynthSegParc
from .segment import SynthSegSegmenter, run_synthseg_parc_t1
from .synthseg import _official_soft_volumes, _segmentation_image


@dataclass
class SynthSegPlusResult:
    """Base anatomy, cortical parcels and their combined output-grid map."""

    segmentation: FNITNifti1Image
    cortical_parcellation: FNITNifti1Image
    combined: FNITNifti1Image
    label_names: dict[int, str]
    volumes_mm3: dict[int, float] | None = None
    total_intracranial_mm3: float | None = None

    def mask(self, label: int | str) -> np.ndarray:
        if isinstance(label, str):
            matches = [idx for idx, name in self.label_names.items() if name == label]
            if len(matches) != 1:
                raise KeyError(label)
            label = matches[0]
        return np.asanyarray(self.combined.dataobj) == int(label)

    def write_volumes_csv(self, source: str | Path, path: str | Path) -> None:
        """Write the FreeSurfer ``mri_synthseg --parc --vol`` column order."""
        if self.volumes_mm3 is None or self.total_intracranial_mm3 is None:
            raise ValueError("Run SynthSegPlus with volumes=True before writing volumes")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["subject", "total intracranial",
                             *(self.label_names[label] for label in self.volumes_mm3)])
            writer.writerow([Path(source).name.replace(".nii.gz", ""),
                             str(np.float32(self.total_intracranial_mm3)),
                             *(str(np.float32(value)) for value in self.volumes_mm3.values())])


def _to_image(data: torch.Tensor, affine: np.ndarray, reference):
    return _segmentation_image(data.to(torch.int32).cpu().numpy(), reference, affine)


def _native(image, reference):
    resampled = resample_from_to(image, (reference.shape[:3], reference.affine), order=0)
    return _segmentation_image(np.asanyarray(resampled.dataobj), reference, reference.affine)


class SynthSegPlus:
    """Run official SynthSeg 2.0 segmentation and ``--parc`` heads in PyTorch.

    The default FNIT contract returns the three label volumes on the original
    input T1 voxel grid. Set ``keep_geometry=False`` to retain SynthSeg's
    RAS-aligned approximately 1-mm inference grid.
    """

    def __init__(self, weights: str | Path | None = None,
                 parc_weights: str | Path | None = None,
                 device: str | torch.device = "cpu"):
        self.device = device
        self.segment_weights = resolve_weights("synthseg_2.0.h5", explicit=weights)
        self.models = self.segment_weights.parent
        parc_explicit = parc_weights
        if parc_explicit is None and weights is not None:
            supplied = Path(weights).expanduser()
            sibling = supplied / "synthseg_parc_2.0.h5" if supplied.is_dir() else supplied.parent / "synthseg_parc_2.0.h5"
            if sibling.is_file():
                parc_explicit = sibling
            elif supplied.is_dir():
                parc_explicit = supplied
        self.parc_weights = resolve_weights("synthseg_parc_2.0.h5", explicit=parc_explicit)
        self.segmentation_labels = self.models / "synthseg_segmentation_labels_2.0.npy"
        self.topology_classes = self.models / "synthseg_topological_classes_2.0.npy"
        if not self.segmentation_labels.is_file():
            self.segmentation_labels = resolve_weights("synthseg_segmentation_labels_2.0.npy", explicit=weights)
        if not self.topology_classes.is_file():
            self.topology_classes = resolve_weights("synthseg_topological_classes_2.0.npy", explicit=weights)

        names_path = self.models / "synthseg_segmentation_names_2.0.npy"
        if not names_path.is_file():
            names_path = resolve_weights("synthseg_segmentation_names_2.0.npy", explicit=weights)
        raw_labels = np.load(self.segmentation_labels)
        self._base_volume_labels = tuple(int(label) for label in np.unique(raw_labels)[1:])
        self._volume_labels = self._base_volume_labels + tuple(
            int(label) for label in PARCELLATION_LABELS[1:])
        names = np.load(names_path)
        unique, indices = np.unique(raw_labels, return_index=True)
        self.label_names = {int(label): str(names[index]) for label, index in zip(unique, indices)}
        self.label_names.update(PARCELLATION_NAME_BY_ID)
        self._segmenter = None
        self._parcellator = None

    @torch.inference_mode()
    def __call__(self, t1: str | Path, *,
                 keep_geometry: bool = True, fast: bool = False,
                 min_pad: int = 128, volumes: bool = False) -> SynthSegPlusResult:
        reference = nib.load(str(t1))
        if self._segmenter is None:
            self._segmenter = SynthSegSegmenter(
                self.segment_weights, self.segmentation_labels, self.device)
        if self._parcellator is None:
            self._parcellator = SynthSegParc(
                self.parc_weights, PARCELLATION_LABELS, self.device)
        result = run_synthseg_parc_t1(
            t1,
            self.segment_weights,
            self.segmentation_labels,
            self.parc_weights,
            PARCELLATION_LABELS,
            device=self.device,
            min_pad=min_pad,
            topology_classes=self.topology_classes,
            fast=fast,
            volumes=volumes,
            segmenter=self._segmenter,
            parcellator=self._parcellator,
        )
        volume_map = None
        intracranial = None
        if volumes:
            base = _official_soft_volumes(result.segmentation_posterior,
                                          result.source_affine,
                                          result.voxel_volume_mm3,
                                          round_decimals=None)
            parcel = result.parcellation_soft_voxels
            left = parcel[:34] / parcel[:34].sum() * base[1 + self._base_volume_labels.index(3)]
            right = parcel[34:] / parcel[34:].sum() * base[1 + self._base_volume_labels.index(42)]
            values = np.around(np.concatenate((base, left, right)), 3)
            intracranial = float(values[0])
            volume_map = dict(zip(self._volume_labels, (float(value) for value in values[1:])))
            result.segmentation_posterior = None
        segmentation = _to_image(result.segmentation, result.affine, reference)
        parc = _to_image(result.parcellation, result.affine, reference)
        combined = _to_image(result.combined, result.affine, reference)
        if keep_geometry:
            segmentation = _native(segmentation, reference)
            parc = _native(parc, reference)
            combined = _native(combined, reference)
        return SynthSegPlusResult(segmentation, parc, combined, dict(self.label_names),
                                  volume_map, intracranial)
