"""Small arithmetic regression oracles; these are not imaging benchmarks."""

import pytest
import torch

from fnit.amico_noddi import solver


def _old_masked_solve(gram, rhs, mask, *, ridge=0.0, tolerance=1e-13):
    """Frozen pre-fix algorithm, intentionally small enough for CPU fixtures."""
    columns = rhs.shape[-1]
    flat_mask = mask.reshape(-1, columns).bool()
    flat_rhs = rhs.reshape(-1, columns)
    counts = flat_mask.sum(-1)
    width = int(counts.max())
    if width == 0:
        return torch.zeros_like(rhs)
    order = torch.topk(flat_mask.to(torch.uint8), width, dim=-1, sorted=False).indices
    occupied = flat_mask.gather(-1, order)
    if gram.ndim == 2:
        block = gram[order[:, :, None], order[:, None, :]]
    else:
        slots = rhs.shape[-2]
        groups = torch.arange(gram.shape[0], device=rhs.device).repeat_interleave(slots)
        block = gram[groups[:, None, None], order[:, :, None], order[:, None, :]]
    active_block = occupied[:, :, None] & occupied[:, None, :]
    block = block * active_block
    diagonal = torch.diagonal(block, dim1=-2, dim2=-1)
    diagonal.add_(torch.where(occupied, float(ridge), 1.0))
    compact_rhs = flat_rhs.gather(-1, order) * occupied
    factor, info = torch.linalg.cholesky_ex(block, check_errors=False)
    failed = info != 0
    has_failed = bool(failed.any())
    if has_failed:
        identity = torch.eye(width, dtype=block.dtype, device=block.device)
        block[failed] = identity
        compact_rhs[failed] = 0
        factor = torch.linalg.cholesky(block)
    compact = torch.cholesky_solve(compact_rhs[..., None], factor).squeeze(-1)
    result = torch.zeros_like(flat_rhs)
    result.scatter_(-1, order, compact * occupied)
    if has_failed:
        if gram.ndim == 2:
            result[failed] = solver._masked_cg(
                gram, flat_rhs[failed], flat_mask[failed], ridge=ridge, tolerance=tolerance)
        else:
            result[failed] = solver._masked_cg(
                gram[groups[failed]], flat_rhs[failed, None, :],
                flat_mask[failed, None, :], ridge=ridge, tolerance=tolerance)[:, 0]
    return result.reshape_as(rhs)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _positive_system(grouped):
    generator = torch.Generator().manual_seed(442)
    design_shape = (4, 9, 5) if grouped else (9, 5)
    design = torch.randn(design_shape, dtype=torch.float64, generator=generator)
    gram = design.transpose(-2, -1) @ design
    rhs_shape = (4, 5, 5) if grouped else (20, 5)
    rhs = torch.randn(rhs_shape, dtype=torch.float64, generator=generator)
    mask = torch.zeros(rhs_shape, dtype=torch.bool)
    flattened = mask.reshape(-1, 5)
    # Empty first direction group, holes within groups, and varying support.
    flattened[5, [0, 2, 4]] = True
    flattened[9, [1]] = True
    flattened[12, [1, 3]] = True
    flattened[15, [0, 1, 3]] = True
    flattened[18, [4]] = True
    return gram, rhs, mask


@pytest.mark.parametrize("grouped", (False, True))
@pytest.mark.parametrize("ridge", (0.0, 1e-3))
@pytest.mark.parametrize("force_chunks", (False, True))
def test_padded_shared_and_grouped_systems_match_old_oracle(monkeypatch, grouped, ridge, force_chunks):
    gram, rhs, mask = _positive_system(grouped)
    expected = _old_masked_solve(gram, rhs, mask, ridge=ridge)
    if force_chunks:
        monkeypatch.setattr(solver, "_MASKED_SOLVE_WORKSPACE_BYTES", 1)
    actual = solver._masked_solve(gram, rhs, mask, ridge=ridge)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == rhs.dtype
    assert torch.count_nonzero(actual[~mask]) == 0
    assert torch.count_nonzero(actual.reshape(-1, 5)[:5]) == 0


def test_padding_never_enters_cholesky_and_chunks_keep_global_topk_width(monkeypatch):
    gram, rhs, mask = _positive_system(True)
    monkeypatch.setattr(solver, "_MASKED_SOLVE_WORKSPACE_BYTES", 1)
    factorizations, widths = [], []
    original_cholesky = torch.linalg.cholesky_ex
    original_topk = torch.topk

    def capture_cholesky(block, **kwargs):
        factorizations.append(tuple(block.shape))
        return original_cholesky(block, **kwargs)

    def capture_topk(values, width, **kwargs):
        widths.append(width)
        return original_topk(values, width, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", capture_cholesky)
    monkeypatch.setattr(torch, "topk", capture_topk)
    actual = solver._masked_solve(gram, rhs, mask)
    assert len(factorizations) == 5
    assert all(shape == (1, 3, 3) for shape in factorizations)
    assert widths == [3] * 5
    assert torch.isfinite(actual).all()


def test_all_empty_passive_rows_return_exact_zero_without_linalg(monkeypatch):
    gram = torch.eye(5, dtype=torch.float64).expand(3, 5, 5)
    rhs = torch.ones(3, 7, 5, dtype=torch.float64)
    mask = torch.zeros_like(rhs, dtype=torch.bool)

    def unexpected_factorization(*args, **kwargs):
        raise AssertionError("empty passive sets must not allocate a matrix batch")

    monkeypatch.setattr(torch.linalg, "cholesky_ex", unexpected_factorization)
    actual = solver._masked_solve(gram, rhs, mask)
    assert torch.equal(actual, torch.zeros_like(rhs))


@pytest.mark.parametrize("grouped", (False, True))
@pytest.mark.parametrize("ridge", (0.0, 1e-3))
def test_singular_cg_fallback_and_regularization_match_old_oracle(monkeypatch, grouped, ridge):
    singular = torch.tensor([[1., 1., 0.], [1., 1., 0.], [0., 0., 2.]], dtype=torch.float64)
    gram = (torch.stack((2 * singular, singular, torch.diag(torch.tensor(
        [3., 4., 5.], dtype=torch.float64)))) if grouped else singular)
    rhs = torch.zeros((3, 4, 3) if grouped else (12, 3), dtype=torch.float64)
    mask = torch.zeros_like(rhs, dtype=torch.bool)
    flat_rhs, flat_mask = rhs.reshape(-1, 3), mask.reshape(-1, 3)
    for row in (1, 4, 6, 9):
        flat_rhs[row] = torch.tensor([2., 2., 4.], dtype=torch.float64)
        flat_mask[row] = True
    flat_mask[6, 2] = False  # Smaller support still uses global width 3.
    expected = _old_masked_solve(gram, rhs, mask, ridge=ridge)
    monkeypatch.setattr(solver, "_MASKED_SOLVE_WORKSPACE_BYTES", 1)
    calls = []
    original_cg = solver._masked_cg

    def capture_cg(selected_gram, selected_rhs, selected_mask, **kwargs):
        calls.append((tuple(selected_gram.shape), tuple(selected_rhs.shape)))
        return original_cg(selected_gram, selected_rhs, selected_mask, **kwargs)

    monkeypatch.setattr(solver, "_masked_cg", capture_cg)
    actual = solver._masked_solve(gram, rhs, mask, ridge=ridge)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if ridge == 0:
        assert calls
        if grouped:
            assert all(shape == (1, 3, 3) for shape, _ in calls)
        assert all(shape[0] == 1 for _, shape in calls)
    else:
        assert calls == []
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[~mask]) == 0


def test_nonnegative_active_set_support_iterations_and_values_unchanged(monkeypatch):
    generator = torch.Generator().manual_seed(8)
    design = torch.rand((3, 11, 5), generator=generator, dtype=torch.float64)
    signal = torch.rand((3, 4, 11), generator=generator, dtype=torch.float64)
    allowed = torch.ones((3, 4, 5), dtype=torch.bool)
    allowed[0] = False
    allowed[1, 1] = False
    allowed[2, 3] = False
    new_solve = solver._masked_solve
    monkeypatch.setattr(solver, "_masked_solve", _old_masked_solve)
    expected = solver.nonnegative_quadratic(design, signal, allowed=allowed, l1=0.3, l2=1e-3)
    monkeypatch.setattr(solver, "_masked_solve", new_solve)
    monkeypatch.setattr(solver, "_MASKED_SOLVE_WORKSPACE_BYTES", 1)
    actual = solver.nonnegative_quadratic(design, signal, allowed=allowed, l1=0.3, l2=1e-3)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_grouped_cg_broadcasts_one_gram_for_multiple_failed_rows(monkeypatch):
    columns, slots = 20, 10
    gram = torch.zeros((2, columns, columns), dtype=torch.float64)
    gram[0, :2, :2] = 1
    gram[1, :2, :2] = 4  # sqrt(4) is exact, forcing the singular Schur pivot to zero.
    rhs = torch.zeros((2, slots, columns), dtype=torch.float64)
    mask = torch.zeros_like(rhs, dtype=torch.bool)
    flat_rhs, flat_mask = rhs.reshape(-1, columns), mask.reshape(-1, columns)
    for row in (2, 3, 4, 5, 6, 7, 11, 13, 14, 16, 18, 19):
        flat_rhs[row, :2] = 2
        flat_mask[row, :2] = True
    expected = _old_masked_solve(gram, rhs, mask)
    # One Cholesky chunk contains both groups; each CG packet is at most two
    # rows even though its compact Cholesky width is only two out of 20 columns.
    monkeypatch.setattr(solver, "_MASKED_SOLVE_WORKSPACE_BYTES", 12 * columns * 8 * 2)
    original_cg = solver._masked_cg
    packets = []

    def capture_cg(selected_gram, selected_rhs, selected_mask, **kwargs):
        packets.append((tuple(selected_gram.shape), tuple(selected_rhs.shape)))
        return original_cg(selected_gram, selected_rhs, selected_mask, **kwargs)

    monkeypatch.setattr(solver, "_masked_cg", capture_cg)
    actual = solver._masked_solve(gram, rhs, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(packets) == 6
    assert all(shape == (1, columns, columns) for shape, _ in packets)
    assert all(shape == (2, 1, columns) for _, shape in packets)
