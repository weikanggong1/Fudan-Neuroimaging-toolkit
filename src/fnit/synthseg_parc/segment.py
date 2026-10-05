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
from .cpu_conv import CPUInferenceConv3d, convolution_slabs, cpu_autocast_enabled
from .cpu_join import cpu_join_allowed, join_nearest_cpu
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
        self.likelihood = CPUInferenceConv3d(24, 33, 1)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or any(size % 32 for size in x.shape[2:]):
            raise ValueError("input must have shape (B, 1, D, H, W), with spatial sizes divisible by 32")
        skips = []
        for level, block in enumerate(self.down):
            x, skip = block(x)
            skips.append(skip)
            if level < 4:
                x = F.max_pool3d(x, 2)
        skips.pop()
        del skip
        for level, block in enumerate(self.up):
            if cpu_join_allowed(self, skips[-1], x):
                x = block(join_nearest_cpu(skips.pop(), x))[0]
            else:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = block(torch.cat((skips.pop(), x), dim=1))[0]
        logits = self.likelihood(x)
        del x
        return torch.softmax(logits, dim=1)

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
    if (posterior.device.type == "cpu" and not cpu_autocast_enabled()
            and not torch.backends.mkldnn.enabled):
        return convolution_slabs(posterior, kernel.expand(33, 1, 3, 3, 3),
                                 padding=1, groups=33)
    return F.conv3d(posterior, kernel.expand(33, 1, 3, 3, 3), padding=1, groups=33)


class SynthSegSegmenter:
    """Return the hard 33-class map supplied to the official ``--parc`` head."""

    def __init__(self, weights: str | Path, labels: str | Path, device="cpu", *,
                 cudnn_tf32: bool | None = True):
        """加载 33 类 FP32 模型，不更改调用方全局精度。

        weights 为官方 HDF5 权重，labels 为 55 项官方标签数组；device 默认
        CPU。cudnn_tf32 默认 True，保留 CUDA 推理的原默认；False 在实际
        CUDA 前向中关闭 cuDNN TF32，None 保留调用方 cuDNN 设置。CUDA
        matmul 在前向内仍使用既有 TF32 默认。参数不是半精度/autocast 开关。
        标签不符、权重读取/形状错误或设备不可用时抛出异常。构造无影像输出；
        posterior 的精度实测记录位于 precision，无独立原软件 CLI。
        """
        if cudnn_tf32 is not None and not isinstance(cudnn_tf32, bool):
            raise ValueError("cudnn_tf32 must be True, False or None")
        self.device = configure_device(device, configure_precision=False)
        self.cudnn_tf32 = cudnn_tf32
        self.precision = {}
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

    def _forward(self, image: torch.Tensor, pass_name: str) -> torch.Tensor:
        """执行一个前向并记录实际精度；输入/输出为 B×C×D×H×W 张量。

        pass_name 是 original 或 flipped；记录 TF32、输入/模型/输出 dtype
        和 CPU/CUDA autocast 开关及目标 dtype，不开启 autocast。调用方主动
        开启的 autocast 沿用并如实记录。模型失败时保留进入前向的记录后抛出。
        """
        def autocast_state(device_type):
            """查询 CPU/CUDA autocast 开关与目标 dtype，兼容 PyTorch 2.1。"""
            try:
                enabled = torch.is_autocast_enabled(device_type)
            except TypeError:  # torch 2.1 的查询接口没有设备参数。
                enabled = (torch.is_autocast_enabled() if device_type == "cuda"
                           else torch.is_autocast_cpu_enabled())
            if hasattr(torch, "get_autocast_dtype"):
                dtype = torch.get_autocast_dtype(device_type)
            else:
                dtype = (torch.get_autocast_gpu_dtype() if device_type == "cuda"
                         else torch.get_autocast_cpu_dtype())
            return {"enabled": bool(enabled), "dtype": str(dtype)}

        row = {"pass": pass_name, "device": str(image.device),
               "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
               "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
               "input_dtype": str(image.dtype),
               "model_dtypes": sorted({str(value.dtype) for value in self.model.parameters()}),
               "autocast": {name: autocast_state(name) for name in ("cpu", "cuda")}}
        self.precision["forwards"].append(row)
        output = self.model(image)
        row["output_dtype"] = str(output.dtype)
        return output

    @torch.inference_mode()
    def posterior(self, image: torch.Tensor, *, flip: bool = True,
                  smooth: bool = True) -> torch.Tensor:
        """推理并返回 (33,D,H,W) 概率，同时保存实际前向精度记录。

        image 为预处理后的 3-D 强度张量，转到模型设备和 float32；flip/smooth
        默认 True，分别控制左右翻转集成和 0.5 体素高斯平滑。关闭 smooth 时
        必须同时关闭 flip，否则抛 ValueError。输出留在模型设备，概率无量纲，
        网格沿用输入。CUDA 的 TF32 策略仅在该作用域生效，正常或异常退出均
        恢复调用方 matmul/cuDNN 设置。不会开启或关闭调用方 autocast；precision
        记录两次前向的真实开关和 dtype，不能仅据 cudnn_tf32=False 宣称无半精度。
        翻转集成复用内部 flipped 后验缓冲，先相加再乘 0.5；输出是该缓冲的
        第零批次视图，不与输入 image 共享存储。precision 记录是否完成复用及
        集成 dtype。模型、平滑或集成失败时传播原异常，并恢复 CUDA 精度设置。
        """
        if image.ndim != 3:
            raise ValueError("image must be a preprocessed 3-D tensor")
        if not smooth and flip:
            raise ValueError("unsmoothed probabilities require flip=False")
        policy = getattr(self, "cudnn_tf32", True)
        self.precision = {"requested_cudnn_tf32": policy, "forwards": [],
                          "posterior_buffer_reused": False}
        previous_matmul = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        cuda = self.device.type == "cuda"
        try:
            if cuda:
                torch.backends.cuda.matmul.allow_tf32 = True
                if policy is not None:
                    torch.backends.cudnn.allow_tf32 = policy
            x = image.to(device=self.device, dtype=torch.float32)[None, None]
            original = self._forward(x, "original")
            if not smooth:
                return original[0]
            original = _blur(original)
            if not flip:
                return original[0]
            if x.is_cuda:
                # The first 33-channel posterior is only needed after the second
                # pass; keep its exact float32 values off the GPU meanwhile.
                original = original.cpu()
                torch.cuda.empty_cache()
            flipped = _blur(self._forward(torch.flip(x, (2,)), "flipped"))
            flipped = torch.flip(flipped, (2,))[:, self.flip_indices]
            if x.is_cuda:
                original = original.to(x.device)
            # Both posteriors are private buffers. Retain the original FP32
            # addition-then-scaling order without allocating two more volumes.
            flipped.add_(original).mul_(0.5)
            self.precision["posterior_buffer_reused"] = True
            self.precision["posterior_ensemble_dtype"] = str(flipped.dtype)
            return flipped[0]
        finally:
            if cuda:
                torch.backends.cuda.matmul.allow_tf32 = previous_matmul
                torch.backends.cudnn.allow_tf32 = previous_cudnn

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
    segmentation_posterior: torch.Tensor | None = None
    parcellation_soft_voxels: np.ndarray | None = None
    voxel_volume_mm3: float = 1.0


@torch.inference_mode()
def run_synthseg_parc_t1(
    t1: str | Path,
    segment_weights: str | Path,
    segment_labels: str | Path,
    parc_weights: str | Path,
    parc_labels: str | Path | np.ndarray,
    *,
    device: str | torch.device = "cpu",
    min_pad: int = 128,
    topology_classes: str | Path | None = None,
    fast: bool = False,
    volumes: bool = False,
    segmenter: SynthSegSegmenter | None = None,
    parcellator: SynthSegParc | None = None,
) -> SynthSegParcResult:
    """Run official SynthSeg 2.0 non-robust segmentation and ``--parc`` heads.

    Returned label tensors stay on ``device``. Their ``affine`` maps the
    unpadded RAS-aligned grid to world coordinates; it is generally different
    from the source image's voxel grid. ``combined`` replaces cortical labels
    3/42 with the 68 parcel IDs. ``topology_classes`` defaults to the official
    2.0 array beside ``segment_labels``. ``fast`` skips the left-right ensemble
    and uses FreeSurfer's shorter posterior filter.
    ``volumes=True`` retains the postprocessed segmentation probabilities
    and 68 parcellation probability sums for soft-volume reporting.
    """
    prepared = preprocess_t1(t1, device=device, min_pad=min_pad)
    if segmenter is None:
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
    segmentation_posterior = processed_posterior if volumes else None
    del raw_posterior, processed_posterior
    segmentation_padded = torch.zeros_like(raw_segmentation)
    segmentation_padded[selection] = segmentation
    if parcellator is None:
        parcellator = SynthSegParc(parc_weights, parc_labels, device)
    parcel_result = parcellator(
        prepared.image, raw_segmentation, segmentation_padded,
        soft_volumes=volumes, content_slices=selection if volumes else None)
    parcellation_padded, soft_parc = parcel_result if volumes else (parcel_result, None)
    parcellation = parcellation_padded[selection]
    combined = torch.where(parcellation != 0, parcellation, segmentation)
    affine = prepared.aligned_affine.copy()
    affine[:3, 3] += affine[:3, :3] @ np.asarray([s.start for s in selection])
    return SynthSegParcResult(segmentation, parcellation, combined, affine,
                             prepared.input_affine, prepared.original_shape,
                             segmentation_posterior, soft_parc,
                             prepared.voxel_volume_mm3)
