"""Isolated FreeSurfer 8.2 EM alignment objective and gradient candidate."""

from dataclasses import dataclass

import numpy as np
from numba import njit

from .mri_em_register import (
    GCA, StableSamples, _vnl_affine_inverse, log_sample_probability,
    regularize_covariance,
)


@njit
def _accumulate_gradient(voxels: np.ndarray, derivative: np.ndarray,
                         mean: np.ndarray, variance: np.ndarray,
                         intensity: np.ndarray, sample_count: int,
                         classifier_present: np.ndarray | None = None) -> np.ndarray:
    """Preserve the float32 MatrixMultiply/MatrixAdd order of the native loop."""

    result = np.zeros((3, 4), np.float32)
    determinant_sum = 0.0
    determinant_count = 0
    for index in range(len(voxels)):
        inverse_variance = np.float32(1.0 / variance[index])
        if classifier_present is None or classifier_present[index]:
            determinant_sum += float(inverse_variance)
            determinant_count += 1
        residual = np.float32(mean[index] - intensity[index])
        for axis in range(3):
            for column in range(4):
                coordinate = np.float32(voxels[index, column]) if column < 3 else np.float32(0.)
                first = np.float32(residual * coordinate)
                second = np.float32(inverse_variance * first)
                product = np.float32(derivative[index, axis] * second)
                result[axis, column] = np.float32(result[axis, column] + product)
    if determinant_count == 0:
        # 源在无有效分类器的试探中以零det归一化，产生Inf/NaN梯度。
        # 保留这一值，让完整步长协议收缩/报告失败；不是丢弃试探。
        return result * np.float32(np.inf)
    average_determinant = determinant_sum / determinant_count
    factor = np.float32(float(np.float32(5e-6)) /
                        (average_determinant * sample_count))
    return result * factor


@dataclass
class EMObjective:
    gca: GCA
    samples: StableSamples
    source: np.ndarray
    clamp: float = 6.0

    def __post_init__(self) -> None:
        node_ids = np.repeat(np.arange(len(self.gca.node_training), dtype=np.int32),
                             np.diff(self.gca.node_offsets))
        self.classifiers = np.full(
            (len(self.gca.node_training), int(self.gca.node_labels.max()) + 1),
            -1, np.int32,
        )
        self.classifiers[node_ids, self.gca.node_labels] = np.arange(
            len(self.gca.node_labels), dtype=np.int32)
        self.variances = regularize_covariance(self.gca).variances

    def cost(self, matrix: np.ndarray) -> float:
        return -log_sample_probability(self.samples, self.source, matrix, clamp=self.clamp)

    def gradient(self, matrix: np.ndarray) -> np.ndarray:
        """Single-input branch of ``computeEMAlignmentGradient`` for the fixed atlas."""

        matrix = np.asarray(matrix, np.float32)
        inverse = _vnl_affine_inverse(matrix)
        prior_to_source = inverse @ np.diag(np.array([
            self.gca.prior_spacing / self.gca.voxel_sizes[0],
            self.gca.prior_spacing / self.gca.voxel_sizes[1],
            self.gca.prior_spacing / self.gca.voxel_sizes[2], 1,
        ], np.float32))
        float_source = np.zeros((len(self.samples.coordinates), 3), np.float32)
        for axis in range(3):
            for prior_axis in range(3):
                float_source[:, axis] = np.float32(
                    float_source[:, axis] + np.float32(
                        prior_to_source[axis, prior_axis] *
                        self.samples.coordinates[:, prior_axis].astype(np.float32)))
            float_source[:, axis] = np.float32(
                float_source[:, axis] + prior_to_source[axis, 3])
        inside = np.all((float_source >= 0) &
                        (float_source <= np.array(self.source.shape) - 1), axis=1)
        voxels = np.where(float_source[inside] < 0,
                          np.ceil(float_source[inside] - .5),
                          np.floor(float_source[inside] + .5)).astype(np.int32)
        atlas_voxels = np.zeros((len(voxels), 3), np.float32)
        for axis in range(3):
            for column in range(3):
                atlas_voxels[:, axis] = np.float32(atlas_voxels[:, axis] +
                    np.float32(matrix[axis, column] * voxels[:, column].astype(np.float32)))
            atlas_voxels[:, axis] = np.float32(atlas_voxels[:, axis] + matrix[axis, 3])
        prior = np.floor(atlas_voxels / self.gca.prior_spacing + .5).astype(np.int32)
        prior = np.clip(prior, 0, np.array(self.gca.prior_shape) - 1)
        node_coordinates = prior // int(self.gca.node_spacing / self.gca.prior_spacing)
        node = np.ravel_multi_index(node_coordinates.T, self.gca.node_shape)
        chosen = self.classifiers[node, self.samples.labels[inside]]
        present = chosen >= 0
        # 源协议：缺少分类器仍以mean=0/inverse_variance=1贡献梯度；
        # 仅已有分类器计入平均协方差行列式，不丢弃该源体素。
        mean = np.zeros(len(chosen), np.float32)
        variance = np.ones(len(chosen), np.float32)
        mean[present] = self.gca.node_means[chosen[present]]
        variance[present] = self.variances[chosen[present]]
        intensity = self.source[voxels[:, 0], voxels[:, 1], voxels[:, 2]].astype(np.float32)

        derivatives = []
        for axis in range(3):
            plus, minus = voxels.copy(), voxels.copy()
            plus[:, axis] += 1
            minus[:, axis] -= 1
            plus_inside = np.all((plus >= 0) & (plus < np.array(self.source.shape)), axis=1)
            minus_inside = np.all((minus >= 0) & (minus < np.array(self.source.shape)), axis=1)
            upper, lower = np.zeros(len(voxels), np.float32), np.zeros(len(voxels), np.float32)
            upper[plus_inside] = self.source[plus[plus_inside, 0],
                                             plus[plus_inside, 1], plus[plus_inside, 2]]
            lower[minus_inside] = self.source[minus[minus_inside, 0],
                                             minus[minus_inside, 1], minus[minus_inside, 2]]
            derivatives.append((upper - lower) / 2)
        derivative = np.stack(derivatives, axis=1)
        return _accumulate_gradient(voxels, derivative, mean, variance,
                                    intensity, len(self.samples.labels), present)
