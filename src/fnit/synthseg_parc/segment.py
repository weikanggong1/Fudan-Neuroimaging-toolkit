"""SynthSeg 2.0's 33-class segmentation network and flip ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .._dmri import configure_device
from .model import _Block
from .pipeline import SynthSegParc
from .postprocess import postprocess_segmentation
from .preprocess import preprocess_t1


class SegmentUNet(nn.Module):
    """The official non-robust 1-input, 33-output five-level U-Net."""

    def __init__(self):
        super().__init__()
        widths = (24, 48, 96, 192, 384)
        self.down = nn.ModuleList(_Block(1 if i == 0 else widths[i - 1], width)
                                  for i, width in enumerate(widths))
        self.up = nn.ModuleList(_Block(widths[i + 1] + widths[i], widths[i])
                                for i in (3, 2, 1, 0))
        self.likelihood = nn.Conv3d(24, 33, 1)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or any(size % 32 for size in x.shape[2:]):
            raise ValueError("input must have shape (B, 1, D, H, W), with spatial sizes divisible by 32")
        skips = []
        for level, block in enumerate(self.down):
            x, skip = block(x)
            skips.append(skip)
            if level < 4:
                x = F.max_pool3d(x, 2)
        for level, block in enumerate(self.up):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x, _ = block(torch.cat((skips[3 - level], x), dim=1))
        return torch.softmax(self.likelihood(x), dim=1)

    def load_h5(self, path: str | Path):
        """Read the official Keras layer arrays without TensorFlow."""
        def load_conv(h5, name, layer):
            group = h5[name][name]
            kernel = np.asarray(group["kernel:0"][()], dtype=np.float32).transpose(4, 3, 0, 1, 2)
            bias = np.asarray(group["bias:0"][()], dtype=np.float32)
            if kernel.shape != tuple(layer.weight.shape) or bias.shape != tuple(layer.bias.shape):
                raise ValueError(f"Unexpected weight shape for {name}")
            layer.weight.copy_(torch.from_numpy(kernel.copy()))
            layer.bias.copy_(torch.from_numpy(bias.copy()))

        def load_bn(h5, name, layer):
            group = h5[name][name]
            for source, target in (("gamma:0", layer.weight), ("beta:0", layer.bias),
                                   ("moving_mean:0", layer.running_mean),
                                   ("moving_variance:0", layer.running_var)):
                values = np.asarray(group[source][()], dtype=np.float32)
                if values.shape != tuple(target.shape):
                    raise ValueError(f"Unexpected weight shape for {name}/{source}")
                target.copy_(torch.from_numpy(values.copy()))

        with torch.no_grad(), h5py.File(path, "r") as h5:
            for level, block in enumerate(self.down):
                for conv, layer in enumerate((block.conv0, block.conv1)):
                    load_conv(h5, f"unet_conv_downarm_{level}_{conv}", layer)
                load_bn(h5, f"unet_bn_down_{level}", block.bn)
            for level, block in enumerate(self.up):
                for conv, layer in enumerate((block.conv0, block.conv1)):
                    load_conv(h5, f"unet_conv_uparm_{level + 5}_{conv}", layer)
                load_bn(h5, f"unet_bn_up_{level}", block.bn)
            load_conv(h5, "unet_likelihood", self.likelihood)
        return self


def _blur(posterior):
    axis = torch.arange(-1, 2, device=posterior.device, dtype=posterior.dtype)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))
    kernel = torch.exp(-grid.square().sum(0) / (2 * 0.5 ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3, 3)
    return F.conv3d(posterior, kernel.expand(33, 1, 3, 3, 3), padding=1, groups=33)


class SynthSegSegmenter:
    """Return the hard 33-class map supplied to the official ``--parc`` head."""

    def __init__(self, weights: str | Path, labels: str | Path, device="cpu"):
        self.device = configure_device(device)
        raw_labels = np.load(labels)
        if len(raw_labels) != 55 or len(np.unique(raw_labels)) != 33:
            raise ValueError("Expected SynthSeg 2.0's 55-entry label array with 33 unique IDs")
        unique = np.unique(raw_labels)
        partners = dict(zip(raw_labels[19:37], raw_labels[37:55]))
        partners.update(zip(raw_labels[37:55], raw_labels[19:37]))
        neutral = set(raw_labels[:19])
        self.labels = torch.as_tensor(unique.astype(np.int64), device=self.device)
        self.flip_indices = torch.as_tensor(
            [int(np.flatnonzero(unique == (label if label in neutral else partners[label]))[0])
             for label in unique], device=self.device
        )
        self.model = SegmentUNet().load_h5(weights).to(self.device).eval()

    @torch.inference_mode()
    def posterior(self, image: torch.Tensor, *, flip: bool = True,
                  smooth: bool = True) -> torch.Tensor:
        """Return probabilities, optionally smoothing and averaging left/right."""
        if image.ndim != 3:
            raise ValueError("image must be a preprocessed 3-D tensor")
        x = image.to(device=self.device, dtype=torch.float32)[None, None]
        original = self.model(x)
        if not smooth:
            if flip:
                raise ValueError("unsmoothed probabilities require flip=False")
            return original[0]
        original = _blur(original)
        if not flip:
            return original[0]
        if x.is_cuda:
            torch.cuda.empty_cache()
        flipped = _blur(self.model(torch.flip(x, (2,))))
        flipped = torch.flip(flipped, (2,))[:, self.flip_indices]
        return (0.5 * (original + flipped))[0]

    @torch.inference_mode()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.labels[self.posterior(image).argmax(dim=0)]


@dataclass
class SynthSegParcResult:
    """Labels on the unpadded, RAS-aligned, approximately 1-mm output grid."""

    segmentation: torch.Tensor
    parcellation: torch.Tensor
    combined: torch.Tensor
    affine: np.ndarray
    source_affine: np.ndarray
    source_shape: tuple[int, int, int]


@torch.inference_mode()
def run_synthseg_parc_t1(
    t1: str | Path,
    segment_weights: str | Path,
    segment_labels: str | Path,
    parc_weights: str | Path,
    parc_labels: str | Path,
    *,
    device: str | torch.device = "cpu",
    min_pad: int = 128,
    topology_classes: str | Path | None = None,
    fast: bool = False,
) -> SynthSegParcResult:
    """Run official SynthSeg 2.0 non-robust segmentation and ``--parc`` heads.

    Returned label tensors stay on ``device``. Their ``affine`` maps the
    unpadded RAS-aligned grid to world coordinates; it is generally different
    from the source image's voxel grid. ``combined`` replaces cortical labels
    3/42 with the 68 parcel IDs. ``topology_classes`` defaults to the official
    2.0 array beside ``segment_labels``. ``fast`` skips the left-right ensemble
    and uses FreeSurfer's shorter posterior filter.
    """
    prepared = preprocess_t1(t1, device=device, min_pad=min_pad)
    segmenter = SynthSegSegmenter(segment_weights, segment_labels, device)
    raw_posterior = segmenter.posterior(prepared.image, flip=not fast, smooth=not fast)
    parc_posterior = _blur(raw_posterior[None])[0] if fast else raw_posterior
    raw_segmentation = segmenter.labels[parc_posterior.argmax(0)]
    del parc_posterior
    if topology_classes is None:
        topology_classes = Path(segment_labels).with_name("synthseg_topological_classes_2.0.npy")
    raw_labels = np.load(segment_labels)
    _, unique_indices = np.unique(raw_labels, return_index=True)
    topology = np.load(topology_classes)
    if len(topology) != len(raw_labels):
        raise ValueError("topology classes must align with the segmentation label array")
    topology = torch.as_tensor(topology[unique_indices], device=segmenter.device)
    selection = prepared.content_slices
    segmentation, processed_posterior = postprocess_segmentation(
        raw_posterior, segmenter.labels, topology, selection, fast=fast)
    del raw_posterior, processed_posterior, segmenter
    segmentation_padded = torch.zeros_like(raw_segmentation)
    segmentation_padded[selection] = segmentation
    parcellation = SynthSegParc(parc_weights, parc_labels, device)(
        prepared.image, raw_segmentation, segmentation_padded)[selection]
    combined = torch.where(parcellation != 0, parcellation, segmentation)
    affine = prepared.aligned_affine.copy()
    affine[:3, 3] += affine[:3, :3] @ np.asarray([s.start for s in selection])
    return SynthSegParcResult(segmentation, parcellation, combined, affine,
                             prepared.input_affine, prepared.original_shape)
