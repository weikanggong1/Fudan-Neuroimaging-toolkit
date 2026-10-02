"""CUDA graph 采样的算术合同；这些小体积不作为影像 benchmark。"""

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.flirt.core import fsl_affine_from_parameters
from fnit.mcflirt._sampling_cuda import CudaMotionFrameSampler
from fnit.mcflirt.sampling import sample_motion_frame


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for graph sampling"
)


def _images(input_sign, reference_sign):
    generator = np.random.default_rng(45061)
    data = (generator.normal(size=(7, 9, 5, 3)) * 140 + 850).astype(np.float32)
    input_affine = np.diag([input_sign * 1.7, 1.3, 2.2, 1.])
    image = nib.Nifti1Image(data, input_affine)
    image.header.set_zooms((1.7, 1.3, 2.2, .735))
    reference_affine = np.diag([reference_sign * 1.9, 1.1, 2.2000003, 1.])
    reference = nib.Nifti1Image(data[..., 0], reference_affine)
    reference.header.set_zooms((1.9, 1.1, 2.2000003))
    return image, reference, data


def _matrices():
    parameters = np.array([
        .09, -.07, .12, .7, -.2, .4, 1., 1., 1., 0., 0., 0.
    ])
    rigid = fsl_affine_from_parameters(
        torch.as_tensor(parameters), np.array([4.1, 5.2, 3.3]), 6
    ).numpy()
    outside = np.eye(4)
    outside[:3, 3] = (-4., 6., 2.8)
    return np.eye(4), rigid, outside


@pytest.mark.parametrize("interpolation", ["linear", "spline"])
@pytest.mark.parametrize("input_sign,reference_sign", [(-1, -1), (1, 1), (-1, 1), (1, -1)])
def test_graph_keeps_multiframe_bits_orientation_pixdim_and_prior_outputs(
    interpolation, input_sign, reference_sign
):
    image, reference, data = _images(input_sign, reference_sign)
    device = torch.device("cuda", torch.cuda.current_device())
    sampler = CudaMotionFrameSampler(
        image, reference, device=device, interpolation=interpolation
    )
    retained_outputs = []
    expected_outputs = []
    for frame, matrix in enumerate(_matrices()):
        expected = sample_motion_frame(
            data[..., frame], image, reference, matrix, device=device,
            interpolation=interpolation,
        ).cpu().numpy()
        actual = sampler.sample_numpy(data[..., frame], matrix)
        assert actual.shape == reference.shape
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        retained_outputs.append(actual)
        expected_outputs.append(expected.copy())
    for actual, expected in zip(retained_outputs, expected_outputs):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert all(
        not np.shares_memory(first, second)
        for index, first in enumerate(retained_outputs)
        for second in retained_outputs[index + 1:]
    )


def test_graph_rejects_non_3d_reference_and_mismatched_frame():
    image, reference, data = _images(-1, -1)
    device = torch.device("cuda", torch.cuda.current_device())
    with pytest.raises(ValueError, match="reference must be 3D"):
        CudaMotionFrameSampler(image, image, device=device, interpolation="linear")
    sampler = CudaMotionFrameSampler(image, reference, device=device, interpolation="linear")
    with pytest.raises(ValueError, match="3D frame"):
        sampler.sample_numpy(data, np.eye(4))
    with pytest.raises(ValueError, match="3D frame"):
        sampler.sample_numpy(data[:6, ..., 0], np.eye(4))
    with pytest.raises(ValueError, match="fsl_matrix"):
        sampler.sample_numpy(data[..., 0], np.eye(3))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two visible GPUs are required")
def test_explicit_device_restores_other_current_device_and_stream():
    image, reference, data = _images(-1, -1)
    original_device = torch.cuda.current_device()
    desired_index = original_device
    other_index = (original_device + 1) % torch.cuda.device_count()
    desired_device = torch.device("cuda", desired_index)
    with torch.cuda.device(other_index):
        caller_stream = torch.cuda.Stream(device=other_index)
        with torch.cuda.stream(caller_stream):
            sampler = CudaMotionFrameSampler(
                image, reference, device=desired_device, interpolation="spline"
            )
            assert torch.cuda.current_device() == other_index
            assert torch.cuda.current_stream(other_index) == caller_stream
            actual = sampler.sample_numpy(data[..., 0], np.eye(4))
            assert torch.cuda.current_device() == other_index
            assert torch.cuda.current_stream(other_index) == caller_stream
    expected = sample_motion_frame(
        data[..., 0], image, reference, np.eye(4), device=desired_device,
        interpolation="spline",
    ).cpu().numpy()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
