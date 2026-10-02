"""CUDA graph replay keeps the existing MCFLIRT reduction and sampling bits."""

from importlib.util import find_spec

import numpy as np
import pytest
import torch

from fnit.mcflirt.core import _normcorr_reduce


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or find_spec("triton") is None,
    reason="CUDA and Triton are required for the motion-cost executor",
)


def _assert_bits(actual, expected):
    actual = actual.detach().cpu().numpy()
    expected = expected.detach().cpu().numpy()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def _images(device=None):
    generator = torch.Generator().manual_seed(90612)
    reference = (torch.rand((3, 4, 7), generator=generator) * 600).cuda(device)
    moving = (torch.rand((11, 13, 15), generator=generator) * 800).cuda(device)
    return reference, moving


def test_graph_replay_preserves_compiled_cost_and_running_count():
    from fnit.mcflirt._cost_cuda import FusedMotionSampler
    from fnit.mcflirt._cuda_executor import CudaMotionCostExecutor

    reference, moving = _images()
    sizes = (2.1, 1.7, 2.4)
    reducer = torch.compile(_normcorr_reduce, fullgraph=True)
    executor = CudaMotionCostExecutor(reference, moving, sizes, reducer)
    original = FusedMotionSampler(reference, moving, sizes)
    coefficients = np.eye(4, dtype=np.float32)[:3].copy()
    captured_graph = None
    for translation in ((0, 0, 0), (-1.7, .3, .6), (1000, -1000, 1000),
                        (.2, .7, -.4), (0, 0, 0)):
        coefficients[:, 3] = translation
        expected_inputs = original.prepare(coefficients)
        expected = reducer(*expected_inputs).clone()
        actual_inputs = executor.sampler.prepare(coefficients)
        actual = executor.reduce(*actual_inputs)
        _assert_bits(actual, expected)
        for actual_input, expected_input in zip(actual_inputs, expected_inputs):
            _assert_bits(actual_input, expected_input)
        if captured_graph is None:
            captured_graph = executor.graph
        assert executor.graph is captured_graph
        if translation[0] == 1000:
            assert float(actual) == 1.0


def test_frame_update_keeps_graph_addresses_without_stale_frame_values():
    from fnit.mcflirt._cost_cuda import FusedMotionSampler
    from fnit.mcflirt._cuda_executor import CudaMotionCostExecutor

    reference, moving = _images()
    sizes = (1.8, 2.2, 2.0)
    reducer = torch.compile(_normcorr_reduce, fullgraph=True)
    executor = CudaMotionCostExecutor(reference, moving, sizes, reducer)
    coefficients = np.eye(4, dtype=np.float32)[:3]
    addresses = (executor.moving.data_ptr(),
                 executor.sampler.reference_values.data_ptr(),
                 executor.sampler.moving_values.data_ptr(),
                 executor.sampler.weights.data_ptr())
    graph = None
    for frame in (moving, moving * 1.3 + 19, moving.flip(0).contiguous(), moving):
        original = FusedMotionSampler(reference, frame, sizes)
        expected = reducer(*original.prepare(coefficients)).clone()
        executor.set_moving(frame)
        actual = executor.reduce(*executor.sampler.prepare(coefficients))
        _assert_bits(actual, expected)
        _assert_bits(executor.moving, frame)
        if graph is None:
            graph = executor.graph
        assert executor.graph is graph
        assert addresses == (executor.moving.data_ptr(),
                             executor.sampler.reference_values.data_ptr(),
                             executor.sampler.moving_values.data_ptr(),
                             executor.sampler.weights.data_ptr())


def test_executor_rejects_another_grid_or_external_reduction_buffers():
    from fnit.mcflirt._cuda_executor import CudaMotionCostExecutor

    reference, moving = _images()
    executor = CudaMotionCostExecutor(reference, moving, (2, 2, 2),
                                      torch.compile(_normcorr_reduce, fullgraph=True))
    with pytest.raises(ValueError, match="same contiguous moving grid"):
        executor.set_moving(moving[..., :-1])
    with pytest.raises(ValueError, match="sampler buffers"):
        executor.reduce(reference, reference, reference)


def test_two_scales_and_instances_keep_their_outputs_independent():
    from fnit.mcflirt._cost_cuda import FusedMotionSampler
    from fnit.mcflirt._cuda_executor import CudaMotionCostExecutor

    reference, moving = _images()
    reducer = torch.compile(_normcorr_reduce, fullgraph=True)
    sizes = (2.0, 2.1, 2.3)
    references = (reference, reference[:, :, ::2].contiguous())
    executors = [CudaMotionCostExecutor(ref, moving, sizes, reducer)
                 for ref in references]
    coefficients = np.eye(4, dtype=np.float32)[:3].copy()
    outputs = []
    for executor, ref in zip(executors, references):
        original = FusedMotionSampler(ref, moving, sizes)
        expected = reducer(*original.prepare(coefficients)).clone()
        outputs.append(executor.reduce(*executor.sampler.prepare(coefficients)))
        _assert_bits(outputs[-1], expected)
    first_snapshot = outputs[0].clone()
    coefficients[:, 3] = (-1.4, .6, .9)
    executors[1].set_moving(moving.flip(0).contiguous())
    executors[1].reduce(*executors[1].sampler.prepare(coefficients))
    _assert_bits(outputs[0], first_snapshot)
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    assert executors[0].graph is not executors[1].graph


def test_nondefault_stream_is_restored_and_replays_updated_values():
    from fnit.mcflirt._cost_cuda import FusedMotionSampler
    from fnit.mcflirt._cuda_executor import CudaMotionCostExecutor

    reference, moving = _images()
    reducer = torch.compile(_normcorr_reduce, fullgraph=True)
    stream = torch.cuda.Stream(device=moving.device)
    stream.wait_stream(torch.cuda.current_stream(moving.device))
    coefficients = np.eye(4, dtype=np.float32)[:3].copy()
    with torch.cuda.stream(stream):
        executor = CudaMotionCostExecutor(reference, moving, (2, 2, 2), reducer)
        for translation in ((0, 0, 0), (-1.2, .3, .8)):
            coefficients[:, 3] = translation
            expected_sampler = FusedMotionSampler(reference, moving, (2, 2, 2))
            expected = reducer(*expected_sampler.prepare(coefficients)).clone()
            actual = executor.reduce(*executor.sampler.prepare(coefficients))
            assert torch.cuda.current_stream(moving.device) == stream
            _assert_bits(actual, expected)
    stream.synchronize()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
def test_cuda_one_executor_preserves_callers_cuda_zero_device():
    from fnit.mcflirt._cost_cuda import FusedMotionSampler
    from fnit.mcflirt._cuda_executor import CudaMotionCostExecutor

    caller_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        reference, moving = _images(device=1)
        reducer = torch.compile(_normcorr_reduce, fullgraph=True)
        executor = CudaMotionCostExecutor(reference, moving, (2, 2, 2), reducer)
        coefficients = np.eye(4, dtype=np.float32)[:3].copy()
        for translation in ((0, 0, 0), (-1.3, .6, .2)):
            coefficients[:, 3] = translation
            with torch.cuda.device(1):
                original = FusedMotionSampler(reference, moving, (2, 2, 2))
                expected = reducer(*original.prepare(coefficients)).clone()
            assert torch.cuda.current_device() == 0
            actual = executor.reduce(*executor.sampler.prepare(coefficients))
            _assert_bits(actual, expected)
            assert actual.device.index == 1
            assert torch.cuda.current_device() == 0
    finally:
        torch.cuda.set_device(caller_device)


def test_core_accepts_noncontiguous_nifti_frames():
    import nibabel as nib
    from fnit.mcflirt import TorchMCFLIRT

    rng = np.random.default_rng(1803)
    values = rng.uniform(20, 800, (6, 8, 9, 3)).astype(np.float32)
    affine = np.diag([-4.0, 4.0, 4.0, 1.0])
    reference = nib.Nifti1Image(values[..., 0].copy(), affine)
    fits = []
    for order in ('C', 'F'):
        image = nib.Nifti1Image(np.array(values, order=order), affine)
        fits.append(TorchMCFLIRT(device='cuda:0').run(
            image, reference, stages=1, stage_iterations=(1, 0, 0)))
    assert fits[0].cost_evaluations == fits[1].cost_evaluations > 0
    np.testing.assert_array_equal(fits[0].matrices.view(np.uint64),
                                  fits[1].matrices.view(np.uint64))
    np.testing.assert_array_equal(fits[0].parameters.view(np.uint64),
                                  fits[1].parameters.view(np.uint64))
