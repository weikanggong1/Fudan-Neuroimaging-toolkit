"""MCFLIRT 坐标搜索中固定重心、重复旋转及采样矩阵的复用。"""

import numpy as np
import torch

from ..flirt.core import _axis_rotation


class RigidAffineComposer:
    """固定重心的六自由度 CPU composer；保留 FLIRT 的逐轴运算。

    每轴只保存最后一个角度及其旋转矩阵，另保存最后一组三轴的
    ``I @ Rx @ Ry @ Rz``。平移在矩阵副本上加入，避免修改缓存。
    参数键使用 float64 原始字节，因此正负零不会共用键。
    """

    def __init__(self, centre):
        if isinstance(centre, torch.Tensor) and centre.device.type != "cpu":
            raise ValueError("MCFLIRT rigid composition requires a CPU centre")
        self.centre = torch.as_tensor(centre, dtype=torch.float64).clone()
        if self.centre.shape != (3,):
            raise ValueError("centre must have shape (3,)")
        self._axis_keys = [None] * 3
        self._axis_rotations = [None] * 3
        self._angles_key = None
        self._rotation = None

    def __call__(self, parameters):
        if isinstance(parameters, torch.Tensor) and parameters.device.type != "cpu":
            raise ValueError("MCFLIRT rigid composition requires CPU parameters")
        parameters = torch.as_tensor(parameters, dtype=torch.float64)
        if parameters.shape != (12,):
            raise ValueError("parameters must have shape (12,)")
        angles_key = parameters[:3].numpy().tobytes()
        if angles_key != self._angles_key:
            for axis in range(3):
                key = parameters[axis].numpy().tobytes()
                if key != self._axis_keys[axis]:
                    self._axis_rotations[axis] = _axis_rotation(
                        parameters[axis], axis, self.centre
                    )
                    self._axis_keys[axis] = key
            # 保留共享标量 composer 的单位矩阵及三个 4×4 乘积；
            # 不合并运算，以免改变 double 的舍入顺序。
            rotation = torch.eye(4, dtype=torch.float64)
            for axis_rotation in self._axis_rotations:
                rotation = rotation @ axis_rotation
            self._rotation = rotation
            self._angles_key = angles_key
        affine = self._rotation.clone()
        affine[:3, 3] += parameters[3:6]
        return affine


class FSLPullCoefficients:
    """固定 voxel size 的 FSL double→float32 pull coefficients。

    仅复用两份对角采样矩阵；每次仍按共享 FLIRT 的顺序计算
    ``moving_sampling_inverse @ inv(affine) @ reference_sampling``，
    然后将前三行缩窄到 float32。
    """

    def __init__(self, moving_voxel_sizes, reference_voxel_sizes):
        self.moving_sampling_inverse = np.diag([
            1 / float(moving_voxel_sizes[0]),
            1 / float(moving_voxel_sizes[1]),
            1 / float(moving_voxel_sizes[2]),
            1,
        ])
        self.reference_sampling = np.diag([
            float(reference_voxel_sizes[0]),
            float(reference_voxel_sizes[1]),
            float(reference_voxel_sizes[2]),
            1,
        ])

    def __call__(self, moving_to_reference):
        affine = np.asarray(moving_to_reference, dtype=np.float64)
        if affine.shape != (4, 4):
            raise ValueError("moving_to_reference must have shape (4, 4)")
        pull = (
            self.moving_sampling_inverse
            @ np.linalg.inv(affine)
            @ self.reference_sampling
        )
        return np.asarray(pull[:3, :], dtype=np.float32)
