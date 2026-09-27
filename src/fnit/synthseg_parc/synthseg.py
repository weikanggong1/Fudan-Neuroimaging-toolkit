"""Independent 33-class SynthSeg 2.0 inference shared with GPU recon-all."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path

import numpy as np
import torch

from ..weights import resolve_weights
from .postprocess import postprocess_segmentation
from .preprocess import _ras_axes, preprocess_t1
from .segment import SynthSegSegmenter
from .synthseg_io import SynthSegVolume, read_color_lut


# Cross-framework FP32 convolutions can reverse an almost exact SynthSeg tie.
SYNTHSEG_TIE_EPSILON = 2 ** -20
# Small FP32 threshold margin resolves one cross-framework foreground-mask tie.
_SYNTHSEG_FOREGROUND_THRESHOLD = 0.2500001


def _synthseg_index_with_numerical_ties(posterior: torch.Tensor) -> torch.Tensor:
    peak = posterior.amax(dim=0, keepdim=True)
    return (posterior >= peak - SYNTHSEG_TIE_EPSILON).to(torch.uint8).argmax(dim=0)


def _restore_posterior_orientation(posterior: np.ndarray,
                                   reference_affine: np.ndarray) -> np.ndarray:
    """Apply mri_synthseg's axis swaps and flips before its NumPy reduction."""
    floating_affine = np.eye(4)
    reference_axes = _ras_axes(reference_affine)
    floating_axes = _ras_axes(floating_affine)
    floating_affine[:, reference_axes] = floating_affine[:, floating_axes]
    for axis in range(3):
        if floating_axes[axis] != reference_axes[axis]:
            posterior = np.swapaxes(posterior, floating_axes[axis], reference_axes[axis])
            other = np.where(floating_axes == reference_axes[axis])[0]
            floating_axes[other], floating_axes[axis] = (
                floating_axes[axis], floating_axes[other])
    directions = np.sum(floating_affine[:3, :3] * reference_affine[:3, :3], axis=0)
    for axis in range(3):
        if directions[axis] < 0:
            posterior = np.flip(posterior, axis=axis)
            floating_affine[:, axis] *= -1
    return posterior


def _official_soft_volumes(posterior: torch.Tensor, reference_affine: np.ndarray,
                           voxel_volume_mm3: float) -> np.ndarray:
    # Keep channels last and C-contiguous, as in the official Keras output.
    spatial = torch.empty((*posterior.shape[1:], posterior.shape[0]),
                          dtype=posterior.dtype, device="cpu")
    spatial.copy_(posterior.permute(1, 2, 3, 0))
    restored = _restore_posterior_orientation(spatial.numpy(), reference_affine)
    soft = np.sum(restored[..., 1:], axis=(0, 1, 2))
    return np.around(np.concatenate(([np.sum(soft)], soft)) * voxel_volume_mm3, 3)


@dataclass
class SynthSegResult:
    segmentation: SynthSegVolume
    volumes_mm3: dict[int, float]
    total_intracranial_mm3: float
    label_names: dict[int, str]
    near_tie_voxels: int

    def write_volumes_csv(self, source: str | Path, path: str | Path) -> None:
        """Write the FreeSurfer non-parcellated SynthSeg 2.0 volume columns."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["subject", "total intracranial",
                             *(self.label_names[label] for label in self.volumes_mm3)])
            writer.writerow([Path(source).name.replace(".nii.gz", ""),
                             str(np.float32(self.total_intracranial_mm3)),
                             *(str(np.float32(value)) for value in self.volumes_mm3.values())])


class SynthSeg:
    """Run the non-robust, non-parcellated 33-class T1 model without FreeSurfer."""

    def __init__(self, weights: str | Path | None = None, device: str = "cpu",
                 threads: int | None = None):
        self.device = torch.device(device)
        if threads is not None:
            torch.set_num_threads(os.cpu_count() if threads < 0 else threads)
        model = resolve_weights("synthseg_2.0.h5", explicit=weights)
        models = model.parent
        labels_path = models / "synthseg_segmentation_labels_2.0.npy"
        raw_labels = np.load(labels_path)
        unique_labels, unique_indices = np.unique(raw_labels, return_index=True)
        names = np.load(models / "synthseg_segmentation_names_2.0.npy")
        topology = np.load(models / "synthseg_topological_classes_2.0.npy")
        if len(names) != len(topology) or len(names) != len(raw_labels):
            raise ValueError("SynthSeg labels, names and topology classes must align")
        self.label_ids = tuple(int(value) for value in unique_labels[1:])
        self.label_names = {label: str(names[index])
                            for label, index in zip(self.label_ids, unique_indices[1:])}
        self.topology = torch.as_tensor(topology[unique_indices], device=self.device)
        self.segmenter = SynthSegSegmenter(model, labels_path, device=self.device)

    @torch.inference_mode()
    def __call__(self, image: str | Path, *, keep_geometry: bool = False,
                 color_lut: str | Path | None = None) -> SynthSegResult:
        prepared = preprocess_t1(image, device=self.device)
        posterior = self.segmenter.posterior(prepared.image)
        ordinary_labels, posterior = postprocess_segmentation(
            posterior, self.segmenter.labels, self.topology, prepared.content_slices,
            foreground_threshold=_SYNTHSEG_FOREGROUND_THRESHOLD)
        labels = self.segmenter.labels[_synthseg_index_with_numerical_ties(posterior)]
        near_tie_voxels = int(torch.count_nonzero(labels != ordinary_labels))
        aligned_affine = prepared.aligned_affine.copy()
        aligned_affine[:3, 3] += aligned_affine[:3, :3] @ np.asarray(
            [part.start for part in prepared.content_slices])

        data = labels.to(torch.float32).cpu().numpy()
        segmentation = SynthSegVolume(data, aligned_affine)
        if keep_geometry:
            segmentation = segmentation.resample_like(image)
        if color_lut is not None:
            segmentation.labels = read_color_lut(color_lut)

        values = _official_soft_volumes(posterior, prepared.volume_affine,
                                        prepared.voxel_volume_mm3)
        volumes = {label: float(value) for label, value in zip(self.label_ids, values[1:])}
        return SynthSegResult(segmentation, volumes, float(values[0]),
                              self.label_names, near_tie_voxels)
