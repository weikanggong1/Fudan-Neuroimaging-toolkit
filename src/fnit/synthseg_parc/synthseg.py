"""Independent 33-class SynthSeg 2.0 inference shared with GPU recon-all."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

from .._dmri import configure_device
from .._nib import FNITNifti1Image, new_image
from ..weights import resolve_weights
from .postprocess import postprocess_segmentation
from .preprocess import _ras_axes, preprocess_t1
from .segment import SynthSegSegmenter


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
                           voxel_volume_mm3: float, *,
                           round_decimals: int | None = 3) -> np.ndarray:
    # Keep channels last and C-contiguous, as in the official Keras output.
    spatial = torch.empty((*posterior.shape[1:], posterior.shape[0]),
                          dtype=posterior.dtype, device="cpu")
    spatial.copy_(posterior.permute(1, 2, 3, 0))
    restored = _restore_posterior_orientation(spatial.numpy(), reference_affine)
    soft = np.sum(restored[..., 1:], axis=(0, 1, 2))
    volumes = np.concatenate(([np.sum(soft)], soft)) * voxel_volume_mm3
    return np.around(volumes, round_decimals) if round_decimals is not None else volumes


def _segmentation_image(data: np.ndarray, reference, affine: np.ndarray):
    """Create the int32 NIfTI header written by FreeSurfer mri_synthseg."""
    image = new_image(np.asarray(data, dtype=np.int32), reference, affine=affine)
    image.set_qform(image.affine, code=0)
    image.set_sform(image.affine, code=2)
    return image


@dataclass
class SynthSegResult:
    """分割、软体积（mm³）和实际前向精度；分割网格见 SynthSeg.__call__。"""
    segmentation: FNITNifti1Image
    volumes_mm3: dict[int, float]
    total_intracranial_mm3: float
    label_names: dict[int, str]
    near_tie_voxels: int
    precision: dict = field(default_factory=dict)

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
                 threads: int | None = None, *, cudnn_tf32: bool | None = True):
        """构造独立 SynthSeg，不调用原软件命令或改变全局 TF32。

        weights 为模型/权重目录，None 使用已声明的默认权重；device 默认 CPU。
        threads 默认 None 保留 PyTorch 线程，负数使用 CPU 核数。cudnn_tf32
        默认 True 保留原 CUDA 卷积默认，False 关闭该模型前向中的 cuDNN TF32，
        None 继承调用方设置；不控制 matmul 或主动启用的 autocast。返回模型
        对象，缺少资源、标签不齐、设备/参数无效或权重加载失败时抛异常。
        原软件对应 mri_synthseg --i T1 --o aseg；完整示例见精度策略文档。
        """
        if cudnn_tf32 is not None and not isinstance(cudnn_tf32, bool):
            raise ValueError("cudnn_tf32 must be True, False or None")
        self.device = configure_device(device, configure_precision=False)
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
        self.segmenter.cudnn_tf32 = cudnn_tf32

    @torch.inference_mode()
    def __call__(self, image: str | Path | nib.spatialimages.SpatialImage, *,
                 keep_geometry: bool = False,
                 color_lut: str | Path | None = None) -> SynthSegResult:
        """从 T1 路径或 nibabel 影像生成分割、体积和实际前向精度。

        image 为单幅 3-D T1，keep_geometry 默认 False 返回约 1-mm RAS 对齐
        网格，True 最近邻恢复输入网格；color_lut 默认 None，可指定存在的
        颜色表路径。返回 SynthSegResult：int32 NIfTI 分割、按标签 mm³ 软体积、
        总颅内容积、名称、近并列体素数及 precision（每次前向 TF32、dtype、
        autocast）。计算失败或颜色表不存在时抛异常；不更改调用方 TF32 状态。
        """
        prepared = preprocess_t1(image, device=self.device)
        # Keep full-volume oneDNN disabled after the observed CPU crash. The
        # convolution layers use restoring local contexts for bounded slabs.
        with torch.backends.mkldnn.flags(
                enabled=self.device.type != "cpu" and torch.backends.mkldnn.enabled):
            posterior = self.segmenter.posterior(prepared.image)
        ordinary_labels, posterior = postprocess_segmentation(
            posterior, self.segmenter.labels, self.topology, prepared.content_slices,
            foreground_threshold=_SYNTHSEG_FOREGROUND_THRESHOLD)
        labels = self.segmenter.labels[_synthseg_index_with_numerical_ties(posterior)]
        near_tie_voxels = int(torch.count_nonzero(labels != ordinary_labels))
        aligned_affine = prepared.aligned_affine.copy()
        aligned_affine[:3, 3] += aligned_affine[:3, :3] @ np.asarray(
            [part.start for part in prepared.content_slices])

        data = labels.to(torch.int32).cpu().numpy()
        reference = (nib.load(str(image)) if isinstance(image, (str, Path)) else image)
        segmentation = _segmentation_image(data, reference, aligned_affine)
        if keep_geometry:
            resampled = resample_from_to(
                segmentation, (prepared.original_shape, prepared.input_affine), order=0)
            segmentation = _segmentation_image(
                np.asanyarray(resampled.dataobj), reference, prepared.input_affine)
        if color_lut is not None:
            color_lut = Path(color_lut)
            if not color_lut.is_file():
                raise FileNotFoundError(color_lut)
            from .color_lut import attach_color_lut
            attach_color_lut(segmentation, color_lut)

        values = _official_soft_volumes(posterior, prepared.volume_affine,
                                        prepared.voxel_volume_mm3)
        volumes = {label: float(value) for label, value in zip(self.label_ids, values[1:])}
        return SynthSegResult(segmentation, volumes, float(values[0]),
                              self.label_names, near_tie_voxels,
                              getattr(self.segmenter, "precision", {}))
