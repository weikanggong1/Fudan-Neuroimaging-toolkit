import pytest
import torch

from fnit.gems import _raster_triton


def _geometry(vertices, tetrahedra):
    cells = vertices[tetrahedra]
    origins = cells[:, 0]
    matrices = torch.stack((cells[:, 1] - origins, cells[:, 2] - origins,
                            cells[:, 3] - origins), dim=-1)
    inverse, info = torch.linalg.inv_ex(matrices)
    singular = (info != 0) | (torch.linalg.det(matrices).abs() <= 1e-10)
    return origins, inverse, singular


def _torch_lookup(points, ids, candidate_mask, origins, inverse, singular, point_rows, tolerance):
    with torch.no_grad():
        rel = points[:, :, None, :] - origins[ids][:, None]
        w123 = torch.einsum("bcij,bpcj->bpci", inverse[ids], rel)
        weights = torch.cat((1.0 - w123.sum(-1, keepdim=True), w123), dim=-1)
        scores = weights.amin(-1).masked_fill((singular[ids] | ~candidate_mask)[:, None], -torch.inf)
        best_score, best = scores.max(2)
        batches = torch.arange(len(points), device=points.device)[:, None]
        return ids[batches, best].flatten()[point_rows], (best_score >= -tolerance).flatten()[point_rows]


def _inputs(device="cpu", dtype=torch.float32):
    cell = torch.tensor([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]], device=device, dtype=dtype)
    rotation = cell.new_tensor([[.9736664, -.2279775, 0], [.2279775, .9736664, 0], [0, 0, 1]])
    cell = cell @ rotation.T + cell.new_tensor([1.137, 1.211, 1.317])
    vertices = torch.cat((cell, cell + cell.new_tensor([.51, .32, .22]))).requires_grad_(True)
    tetrahedra = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=device)
    points = cell.new_tensor([[[2, 2, 2], [0, 0, 0], [3, 3, 2], [4, 2, 2], [3, 2, 3], [6, 6, 6], [2, 3, 2]],
                             [[2, 2, 2], [3, 2, 2], [0, 0, 0], [2, 3, 3], [3, 2, 3], [6, 6, 6], [3, 3, 2]]])
    ids = torch.tensor([[0, 1, 0], [1, 0, 0]], device=device)
    candidate_mask = torch.tensor([[True, True, False], [True, False, False]], device=device)
    rows = torch.tensor([0, 2, 4, 6, 8, 10, 13], device=device)
    return vertices, tetrahedra, points, ids, candidate_mask, rows


@pytest.fixture
def cuda():
    if not torch.cuda.is_available() or _raster_triton.triton is None:
        pytest.skip("CUDA and Triton required")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield "cuda"
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cpu_lookup_requests_torch_fallback(dtype):
    vertices, tetrahedra, points, ids, mask, rows = _inputs(dtype=dtype)
    assert _raster_triton.lookup_candidates(points, ids, mask, *_geometry(vertices, tetrahedra), rows) is None


def test_missing_triton_requests_torch_fallback(cuda, monkeypatch):
    vertices, tetrahedra, points, ids, mask, rows = _inputs(cuda)
    monkeypatch.setattr(_raster_triton, "triton", None)
    assert _raster_triton.lookup_candidates(points, ids, mask, *_geometry(vertices, tetrahedra), rows) is None


def test_unsupported_dtype_and_layout_request_torch_fallback(cuda):
    vertices, tetrahedra, points, ids, mask, rows = _inputs(cuda, torch.float64)
    assert _raster_triton.lookup_candidates(points, ids, mask, *_geometry(vertices, tetrahedra), rows) is None
    vertices, tetrahedra, points, ids, mask, rows = _inputs(cuda)
    strided_rows = torch.stack((rows, rows), dim=1)[:, 0]
    assert not strided_rows.is_contiguous()
    assert _raster_triton.lookup_candidates(points, ids, mask, *_geometry(vertices, tetrahedra), strided_rows) is None


def test_lookup_selection_priors_and_gradients_match_torch_fp32(cuda):
    vertices, tetrahedra, points, ids, mask, rows = _inputs(cuda)
    geometry = _geometry(vertices, tetrahedra)
    actual = _raster_triton.lookup_candidates(points, ids, mask, *geometry, rows)
    expected = _torch_lookup(points, ids, mask, *geometry, rows, 2e-5)
    assert actual is not None
    for got, reference in zip(actual, expected):
        assert torch.equal(got, reference)
        assert not got.requires_grad
    alphas = torch.tensor([[.7, .2, .1], [.1, .7, .2], [.2, .1, .7], [.3, .5, .2],
                           [.2, .3, .5], [.6, .1, .3], [.4, .4, .2], [.1, .2, .7]],
                          device=cuda, requires_grad=True)

    def interpolate(selection):
        origins, inverse, _ = geometry
        selected, covered = selection
        rel = points.reshape(-1, 3)[rows] - origins[selected]
        w123 = torch.einsum("pij,pj->pi", inverse[selected], rel)
        weights = torch.cat((1.0 - w123.sum(-1, keepdim=True), w123), -1)
        values = (alphas[tetrahedra[selected]] * weights[..., None]).sum(1).clamp_min(0)
        values = values / values.sum(-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
        return torch.where(covered[:, None], values, 0)

    priors, reference = interpolate(actual), interpolate(expected)
    torch.testing.assert_close(priors, reference, atol=1e-5, rtol=1e-5)
    coefficient = torch.linspace(.2, 1.3, priors.numel(), device=cuda).reshape_as(priors)
    gradient = torch.autograd.grad((priors.square() * coefficient).sum(), (vertices, alphas), retain_graph=True)
    reference_gradient = torch.autograd.grad((reference.square() * coefficient).sum(), (vertices, alphas))
    for got, wanted in zip(gradient, reference_gradient):
        relative = torch.linalg.vector_norm(got - wanted) / torch.linalg.vector_norm(wanted).clamp_min(1e-12)
        assert float(relative) < 1e-4


@pytest.mark.parametrize("tolerance", (0, 2e-5))
def test_lookup_tolerance_and_first_equal_candidate(cuda, tolerance):
    origins = torch.tensor([[.00001, 0, 0], [.00001, 0, 0]], device=cuda)
    inverse = torch.eye(3, device=cuda)[None].repeat(2, 1, 1) / 4
    singular = torch.zeros(2, device=cuda, dtype=torch.bool)
    points = torch.tensor([[[0., 1, 1], [1, 1, 1]]], device=cuda)
    ids = torch.tensor([[1, 0, 1]], device=cuda)
    mask = torch.tensor([[True, True, False]], device=cuda)
    rows = torch.tensor([0, 1], device=cuda)
    actual = _raster_triton.lookup_candidates(points, ids, mask, origins, inverse, singular, rows,
                                             tolerance=tolerance)
    expected = _torch_lookup(points, ids, mask, origins, inverse, singular, rows, tolerance)
    for got, wanted in zip(actual, expected):
        assert torch.equal(got, wanted)
    assert torch.equal(actual[0], torch.tensor([1, 1], device=cuda))
    assert bool(actual[1][0]) == (tolerance > 0)


def test_lookup_masks_singular_and_padded_candidates_and_handles_all_invalid(cuda):
    origins = torch.zeros((2, 3), device=cuda)
    inverse = torch.eye(3, device=cuda)[None].repeat(2, 1, 1)
    inverse[0] = torch.nan
    singular = torch.tensor([True, False], device=cuda)
    points = torch.tensor([[[.2, .2, .2]], [[.2, .2, .2]]], device=cuda)
    ids = torch.tensor([[0, 1, 1], [0, 1, 1]], device=cuda)
    mask = torch.tensor([[True, True, False], [True, False, False]], device=cuda)
    rows = torch.tensor([0, 1], device=cuda)
    actual = _raster_triton.lookup_candidates(points, ids, mask, origins, inverse, singular, rows)
    expected = _torch_lookup(points, ids, mask, origins, inverse, singular, rows, 2e-5)
    for got, wanted in zip(actual, expected):
        assert torch.equal(got, wanted)
    assert torch.equal(actual[0], torch.tensor([1, 0], device=cuda))
    assert torch.equal(actual[1], torch.tensor([True, False], device=cuda))


def test_empty_real_point_mapping_returns_empty_lookup(cuda):
    vertices, tetrahedra, points, ids, mask, rows = _inputs(cuda)
    actual = _raster_triton.lookup_candidates(points, ids, mask, *_geometry(vertices, tetrahedra), rows[:0])
    assert actual[0].shape == actual[1].shape == (0,)
