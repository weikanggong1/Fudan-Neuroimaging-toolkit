"""一次 MCFLIRT run 中逐帧重放原插值 kernels 的 CUDA graph。"""

import nibabel as nib
import numpy as np
import torch

from ..flirt.core import _edge_background, _flip_to_radiological
from .sampling import _coordinates, _cubic_coefficients, _sample_cubic, sample_motion_frame
from ._cuda_graph import cuda_graph_capture_enabled


class CudaMotionFrameSampler:
    """固定输入与参考网格的 motion-only 采样，支持 linear/spline。

    spline 的 Graph 捕获原有 PyTorch 运算，不编译或合并浮点表达式；
    linear 沿用原 CUDA 采样函数。
    每次 ``sample_numpy`` 返回独立的 CPU float32 数组，可保留前帧结果。
    实例供一次 run 逐帧调用，不供多个线程同时调用。
    """

    def __init__(self, image, reference, *, device, interpolation):
        if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
            raise TypeError("image must be a NIfTI image")
        if not isinstance(reference, (nib.Nifti1Image, nib.Nifti2Image)):
            raise TypeError("reference must be a NIfTI image")
        if image.ndim not in (3, 4) or reference.ndim != 3:
            raise ValueError("image must have 3D frames and reference must be 3D")
        if min(image.shape[:3]) < 2 or min(reference.shape[:3]) < 1:
            raise ValueError("input spatial axes must have at least two voxels")
        if interpolation not in ("linear", "spline"):
            raise ValueError("interpolation must be linear or spline")
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("CudaMotionFrameSampler requires a CUDA device")
        device_index = (torch.cuda.current_device() if requested_device.index is None
                        else requested_device.index)
        self.device = torch.device("cuda", device_index)
        self.interpolation = interpolation
        self._input_image = image
        self._reference_image = reference
        self.input_shape = tuple(image.shape[:3])
        self.output_shape = tuple(reference.shape[:3])
        self._input_affine = np.array(image.affine, dtype=np.float64, copy=True)
        self._flip_output = np.linalg.det(reference.affine[:3, :3]) > 0
        source_sizes = tuple(float(v) for v in image.header.get_zooms()[:3])
        target_sizes = tuple(float(v) for v in reference.header.get_zooms()[:3])
        # 保留 sample_motion_frame 的 NumPy 两次矩阵乘法及顺序。
        self._source_sampling_inverse = np.diag([
            *(1 / np.asarray(source_sizes)), 1
        ])
        self._target_sampling = np.diag([*target_sizes, 1])
        self._graph = None
        self._capture_enabled = cuda_graph_capture_enabled()
        with torch.cuda.device(self.device):
            self._stream = torch.cuda.Stream(device=self.device)
            if interpolation == "linear":
                return
            with torch.cuda.stream(self._stream):
                self._data = torch.zeros(self.input_shape, dtype=torch.float32,
                                         device=self.device)
                self._pull = torch.eye(4, dtype=torch.float32,
                                       device=self.device)[:3].clone()
                if self._capture_enabled:
                    for _ in range(3):
                        self._sample()
            self._stream.synchronize()
            if self._capture_enabled:
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph, stream=self._stream):
                    self._output = self._sample()
                self._stream.synchronize()

    def _sample(self):
        coordinates = _coordinates(self._pull, self.output_shape, self.device)
        lower = torch.floor(coordinates)
        valid = torch.ones(self.output_shape, dtype=torch.bool, device=self.device)
        for axis, size in enumerate(self.input_shape):
            valid &= (lower[axis] >= -1) & (lower[axis] < size)
        sampled = _sample_cubic(_cubic_coefficients(self._data), coordinates)
        sampled = torch.where(valid, sampled, _edge_background(self._data))
        return sampled.flip(0) if self._flip_output else sampled

    def sample_numpy(self, values, fsl_matrix):
        """将一帧及其 input→reference FSL 矩阵采样为新 CPU 数组。"""
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 3 or values.shape != self.input_shape:
            raise ValueError("values must be one 3D frame on the input grid")
        matrix = np.asarray(fsl_matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("fsl_matrix must have shape (4, 4)")
        if self.interpolation == "linear":
            with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
                return sample_motion_frame(
                    values, self._input_image, self._reference_image, matrix,
                    device=self.device, interpolation="linear",
                ).cpu().numpy()
        data = _flip_to_radiological(values, self._input_affine)
        pull = self._source_sampling_inverse @ np.linalg.inv(matrix)
        pull = pull @ self._target_sampling
        coefficients = torch.as_tensor(pull[:3], dtype=torch.float32, device="cpu")
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            self._data.copy_(torch.as_tensor(data, device="cpu"))
            self._pull.copy_(coefficients)
            if self._graph is None:
                self._output = self._sample()
            else:
                self._graph.replay()
            # GPU→CPU 每次分配独立 Tensor；numpy 保持该 Tensor 的所有权。
            return self._output.cpu().numpy()
