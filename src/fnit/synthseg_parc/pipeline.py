"""SynthSeg cortical parcel inference from a preprocessed image and labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .._dmri import configure_device
from .model import ParcUNet
from .cpu_conv import convolution_slabs, cpu_autocast_enabled
from .precision import cuda_tf32_scope, tensor_precision, validate_cudnn_tf32


class SynthSegParc:
    """Map an image and 33-class segmentation to 68 cortical parcel IDs.

    Inputs are spatially aligned 3-D tensors. ``image`` is the official
    preprocessing's 0–1 normalized, RAS-aligned, 1-mm image, padded to a
    multiple of 32; ``segmentation`` has SynthSeg label IDs, including 3/42
    for left/right cortex. No volumetric segmentation is estimated here.
    """

    def __init__(self, weights: str | Path, labels: str | Path | np.ndarray, device="cpu", *,
                 cudnn_tf32: bool | None = True):
        validate_cudnn_tf32(cudnn_tf32)
        self.device = configure_device(device, configure_precision=False)
        self.cudnn_tf32 = cudnn_tf32
        self.precision = None
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
        """Infer parcels under a scoped CUDA policy; caller autocast is preserved.

        cuDNN True is the existing default; False disables cuDNN TF32 for both
        the network and its Gaussian blur, and None inherits the caller flag.
        Construction and CPU calls do not change global CUDA precision.
        Normal and exceptional exits restore both matmul/cuDNN flags.
        ``precision`` records actual network/blur dtype and precision state.
        """
        policy = getattr(self, "cudnn_tf32", True)
        self.precision = {"requested_cudnn_tf32": policy, "forwards": [], "operations": []}
        with cuda_tf32_scope(self.device, policy, self.precision):
            return self._predict(image, segmentation, output_segmentation,
                                 soft_volumes=soft_volumes, content_slices=content_slices)

    def _predict(self, image, segmentation, output_segmentation, *, soft_volumes, content_slices):
        if image.ndim != 3 or segmentation.shape != image.shape:
            raise ValueError("image and segmentation must be aligned 3-D tensors")
        image = image.to(device=self.device, dtype=torch.float32)
        segmentation = segmentation.to(device=self.device)
        cortex = (segmentation == 3) | (segmentation == 42)
        if output_segmentation is not None and output_segmentation.shape != image.shape:
            raise ValueError("output_segmentation must have the same shape as image")
        inputs = torch.stack((image, (~cortex).to(image.dtype), cortex.to(image.dtype)))[None]
        row = tensor_precision(inputs, operation="parcellation_network", model=self.model)
        self.precision["forwards"].append(row)
        posterior = self.model(inputs)
        row["output_dtype"] = str(posterior.dtype)

        # Official GaussianBlur(sigma=0.5): a normalized 3x3x3 kernel, zero padding.
        axis = torch.arange(-1, 2, device=self.device, dtype=image.dtype)
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))
        kernel = torch.exp(-grid.square().sum(0) / (2 * 0.5 ** 2))
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3, 3)
        row = tensor_precision(posterior, operation="parcellation_gaussian_blur")
        self.precision["operations"].append(row)
        if (posterior.device.type == "cpu" and not cpu_autocast_enabled()
                and not torch.backends.mkldnn.enabled):
            posterior = convolution_slabs(posterior, kernel.expand(69, 1, 3, 3, 3),
                                         padding=1, groups=69)[0]
        else:
            posterior = F.conv3d(posterior, kernel.expand(69, 1, 3, 3, 3),
                                 padding=1, groups=69)[0]
        row["output_dtype"] = str(posterior.dtype)

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
