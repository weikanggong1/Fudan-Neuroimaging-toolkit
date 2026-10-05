"""Ordered CPU interpolation must preserve the reference BBR search costs."""
import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.fmri.bbr import _BBRCost


@pytest.mark.parametrize('threads', [1, 8])
def test_cpu_cost_matches_tensor_for_batch_steps_and_outside_points(threads):
    from numba import get_num_threads
    rng = np.random.default_rng(92)
    image = nib.Nifti1Image(rng.uniform(1, 900, (19, 21, 17)).astype(np.float32),
                            np.diag([-1.1, 1.4, 2.1, 1.]))
    grey = rng.uniform(-3, 29, (71, 3)).astype(np.float32)
    white = grey + rng.normal(0, 1.3, grey.shape).astype(np.float32)
    candidate = _BBRCost(image, grey, white, 'cpu', max_batch_size=3)
    reference = _BBRCost(image, grey, white, 'cpu', max_batch_size=3, use_fused=False)
    matrices = np.repeat(np.eye(4)[None], 17, axis=0)
    matrices[:, :3, :3] += rng.normal(0, .02, (17, 3, 3))
    matrices[:, :3, 3] = rng.normal(0, 2, (17, 3))
    previous = torch.get_num_threads()
    numba_previous = get_num_threads()
    try:
        torch.set_num_threads(threads)
        for step in (1, 2, 7):
            expected = reference.evaluate(matrices, step=step)
            actual = candidate.evaluate(matrices, step=step)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert get_num_threads() == numba_previous
    finally:
        torch.set_num_threads(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_gpu_keeps_existing_fused_backend():
    image = nib.Nifti1Image(np.ones((4, 5, 6), np.float32), np.diag([-1., 1., 1., 1.]))
    points = np.array([[1., 1., 1.], [2., 2., 2.]], np.float32)
    cost = _BBRCost(image, points, points, 'cuda:0')
    assert cost._cpu_updates is None
    assert cost.image.is_cuda


def test_cpu_cost_preserves_gradient_tensor_fallback():
    image = nib.Nifti1Image(np.arange(120, dtype=np.float32).reshape(4, 5, 6) + 1,
                           np.diag([-1., 1., 1., 1.]))
    grey = np.array([[1.2, 2., 2.], [2., 2., 3.]], np.float32)
    white = grey - np.array([.2, 0., 0.], np.float32)
    accelerated = _BBRCost(image, grey, white, 'cpu')
    reference = _BBRCost(image, grey, white, 'cpu', use_fused=False)
    accelerated.image.requires_grad_()
    reference.image.requires_grad_()
    def unexpected(*args):
        raise AssertionError('gradient tensor entered NumPy kernel')
    accelerated._cpu_updates = unexpected
    actual = accelerated.evaluate(np.eye(4), step=1)
    expected = reference.evaluate(np.eye(4), step=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(accelerated.image.grad, reference.image.grad, rtol=0, atol=0)
