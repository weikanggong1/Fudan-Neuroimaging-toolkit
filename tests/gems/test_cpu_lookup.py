"""CPU lookup boundaries and gradients; real-image validation is separate."""
import numba
import numpy as np
import pytest
import torch

from fnit.gems._raster_cpu import lookup_candidates_cpu
from fnit.gems import rasterize


def inputs(count=64):
    generator = torch.Generator().manual_seed(57)
    points = torch.rand(2, count, 3, generator=generator) * 4
    ids = torch.arange(32).reshape(2, 16)
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, -3:] = False
    origins = torch.rand(32, 3, generator=generator)
    inverses = torch.rand(32, 3, 3, generator=generator) * 2 - 1
    singular = torch.zeros(32, dtype=torch.bool)
    singular[3] = True
    rows = torch.arange(2 * count)[::3]
    return points, ids, mask, origins, inverses, singular, rows


def reference(values, tolerance=2e-5):
    points, ids, mask, origins, inverses, singular, rows = values
    relative = points[:, :, None] - origins[ids][:, None]
    w123 = torch.einsum("bcij,bpcj->bpci", inverses[ids], relative)
    weights = torch.cat((1.0 - w123.sum(-1, keepdim=True), w123), -1)
    score = weights.amin(-1).masked_fill((singular[ids] | ~mask)[:, None], -torch.inf)
    maximum, best = score.max(2)
    selected = ids[torch.arange(len(points))[:, None], best].flatten()[rows]
    return selected, (maximum >= -float(tolerance)).flatten()[rows]


@pytest.mark.parametrize("count", [1, 2, 16, 32, 64, 512])
def test_selected_ids_and_coverage_preserve_order(count):
    values = inputs(count)
    actual = lookup_candidates_cpu(*values)
    for candidate, expected in zip(actual, reference(values)):
        assert torch.equal(candidate, expected)


@pytest.mark.parametrize("kind", ["tie", "singular", "masked", "nan"])
def test_first_candidate_and_uncovered_semantics(kind):
    values = list(inputs())
    values[3][:] = 0
    values[4][:] = torch.eye(3)
    if kind == "singular":
        values[5][:] = True
    elif kind == "masked":
        values[2][:] = False
    elif kind == "nan":
        values[4][1, 2, 0] = torch.nan
        values[4][2, 0, 0] = torch.nan
    actual = lookup_candidates_cpu(*values, return_hint_hits=True)
    assert actual[2] is None
    for candidate, expected in zip(actual[:2], reference(values)):
        assert torch.equal(candidate, expected)


def test_empty_rows_and_numba_budget_restore():
    values = list(inputs())
    previous = numba.get_num_threads()
    torch_previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        values[-1] = values[-1][:0]
        selected, covered = lookup_candidates_cpu(*values)
        assert selected.shape == covered.shape == (0,)
        assert selected.dtype == torch.long and covered.dtype == torch.bool
        assert numba.get_num_threads() == previous
    finally:
        torch.set_num_threads(torch_previous)


def test_unsupported_float64_keeps_original_torch_path():
    values = [x.double() if x.is_floating_point() else x for x in inputs()]
    assert lookup_candidates_cpu(*values) is None


def test_caller_autocast_keeps_original_torch_path():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert lookup_candidates_cpu(*inputs()) is None


def test_scalar_tolerance_uses_same_float32_rounding_as_torch():
    # This double scalar rounds upward in FP32. Its exact double value must
    # not move the coverage boundary below the original Torch threshold.
    threshold = np.float32(2e-5)
    previous = np.nextafter(threshold, np.float32(-np.inf))
    tolerance = float(previous) + 0.75 * (float(threshold) - float(previous))
    values = [torch.tensor([[[-threshold, .25, .25],
                             [-previous, .25, .25],
                             [-np.nextafter(threshold, np.float32(np.inf)), .25, .25]]]),
              torch.zeros((1, 1), dtype=torch.long),
              torch.ones((1, 1), dtype=torch.bool), torch.zeros((1, 3)),
              torch.eye(3).reshape(1, 3, 3), torch.zeros(1, dtype=torch.bool),
              torch.arange(3)]
    actual = lookup_candidates_cpu(*values, tolerance=tolerance)
    for candidate, expected in zip(actual, reference(values, tolerance=tolerance)):
        assert torch.equal(candidate, expected)
    assert actual[1].tolist() == [True, True, False]


@pytest.mark.parametrize("compact", [False, True])
def test_original_lookup_and_fused_cpu_preserve_priors_and_gradients(monkeypatch, compact):
    shape = (8, 7, 6)
    vertices = torch.tensor([[.13, .21, .17], [5.13, .21, .17],
                             [.73, 5.21, .17], [.13, .61, 5.17]])
    tetrahedra = torch.tensor([[0, 1, 2, 3]])
    alphas = torch.tensor([[.7, .2, .1], [.1, .7, .2],
                           [.2, .1, .7], [.3, .5, .2]])
    mask = torch.arange(np.prod(shape)).reshape(shape) % 3 != 0
    index = rasterize.build_block_index(vertices.numpy(), tetrahedra.numpy(), shape, block_size=4)
    function = rasterize.rasterize_priors_compact if compact else rasterize.rasterize_priors
    kwargs = {"valid_mask": mask} if compact else {}

    def run():
        current_vertices = vertices.clone().requires_grad_(True)
        current_alphas = alphas.clone().requires_grad_(True)
        priors, covered = function(current_vertices, tetrahedra, current_alphas, shape,
                                    block_index=index, **kwargs)
        coefficient = torch.linspace(.1, 1.3, priors.numel()).reshape_as(priors)
        gradients = torch.autograd.grad((priors.square() * coefficient).sum(),
                                        (current_vertices, current_alphas))
        return priors, covered, gradients

    actual = run()
    if compact:
        from fnit.gems import _raster_cpu_compact
        monkeypatch.setattr(_raster_cpu_compact, "lookup_compact_cpu", lambda *a, **k: None)
    monkeypatch.setattr(rasterize, "lookup_candidates", lambda *a, **k: None)
    original = run()
    assert torch.equal(actual[0], original[0])
    assert torch.equal(actual[1], original[1])
    for candidate, expected in zip(actual[2], original[2]):
        assert torch.equal(candidate, expected)


def test_cuda_dispatch_does_not_enter_cpu_helper(monkeypatch):
    class CudaPoints:
        device = torch.device("cuda:0")
    marker = object()
    monkeypatch.setattr(rasterize, "_lookup_candidates_cuda", lambda *a, **k: marker)
    assert rasterize.lookup_candidates(CudaPoints()) is marker
