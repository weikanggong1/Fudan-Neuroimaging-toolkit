"""SynthSeg cortical parcel inference from a preprocessed image and labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .._dmri import configure_device
from .model import ParcUNet


class SynthSegParc:
    """Map an image and 33-class segmentation to 68 cortical parcel IDs.

    Inputs are spatially aligned 3-D tensors. ``image`` is the official
    preprocessing's 0–1 normalized, RAS-aligned, 1-mm image, padded to a
    multiple of 32; ``segmentation`` has SynthSeg label IDs, including 3/42
    for left/right cortex. No volumetric segmentation is estimated here.
    """

    def __init__(self, weights: str | Path, labels: str | Path | np.ndarray, device="cpu"):
        self.device = configure_device(device)
        raw_labels = np.load(labels) if isinstance(labels, (str, Path)) else np.asarray(labels)
        label_ids = np.unique(raw_labels)
        if len(label_ids) != 69 or label_ids[0] != 0:
            raise ValueError("Expected 69 SynthSeg parcellation labels including background")
        self.labels = torch.as_tensor(label_ids.astype(np.int64), device=self.device)
        self.model = ParcUNet().load_h5(weights).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, image: torch.Tensor, segmentation: torch.Tensor,
                 output_segmentation: torch.Tensor | None = None, *,
                 soft_volumes: bool = False,
                 content_slices: tuple[slice, slice, slice] | None = None):
        if image.ndim != 3 or segmentation.shape != image.shape:
            raise ValueError("image and segmentation must be aligned 3-D tensors")
        image = image.to(device=self.device, dtype=torch.float32)
        segmentation = segmentation.to(device=self.device)
        cortex = (segmentation == 3) | (segmentation == 42)
        if output_segmentation is not None and output_segmentation.shape != image.shape:
            raise ValueError("output_segmentation must have the same shape as image")
        inputs = torch.stack((image, (~cortex).to(image.dtype), cortex.to(image.dtype)))[None]
        posterior = self.model(inputs)

        # Official GaussianBlur(sigma=0.5): a normalized 3x3x3 kernel, zero padding.
        axis = torch.arange(-1, 2, device=self.device, dtype=image.dtype)
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))
        kernel = torch.exp(-grid.square().sum(0) / (2 * 0.5 ** 2))
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3, 3)
        posterior = F.conv3d(posterior, kernel.expand(69, 1, 3, 3, 3),
                             padding=1, groups=69)[0]

        # FreeSurfer forces the background channel to zero inside cortex and
        # one outside before argmax; the resulting hard labels match this mask.
        parcel_index = posterior[1:].argmax(dim=0) + 1
        output_cortex = cortex if output_segmentation is None else (
            (output_segmentation.to(self.device) == 3) | (output_segmentation.to(self.device) == 42)
        )
        parcels = torch.where(output_cortex, self.labels[parcel_index], self.labels[0])
        if not soft_volumes:
            return parcels
        if content_slices is None:
            raise ValueError("content_slices are required for soft volumes")
        selected = posterior[(slice(None), *content_slices)]
        spatial = torch.empty((*selected.shape[1:], selected.shape[0]),
                              dtype=selected.dtype, device="cpu")
        spatial.copy_(selected.permute(1, 2, 3, 0))
        probabilities = spatial.numpy()
        probabilities[..., 0] = (~output_cortex[content_slices]).cpu().numpy()
        probabilities /= np.sum(probabilities, axis=-1)[..., None]
        volumes = np.sum(probabilities[..., 1:], axis=(0, 1, 2))
        return parcels, volumes
