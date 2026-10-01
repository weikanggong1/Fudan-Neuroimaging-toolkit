import pytest
import torch

from fnit.gems.optim import CachedLBFGS


def _optimizer(cls, parameters):
    return cls(parameters, lr=1., max_iter=1, history_size=12,
               tolerance_grad=1e-10, tolerance_change=1e-10,
               line_search_fn="strong_wolfe")


def test_multistep_convergence_matches_torch_with_fewer_real_evaluations():
    trajectories, evaluations = [], []
    for cls in (torch.optim.LBFGS, CachedLBFGS):
        position = torch.tensor([4., -3., 2.], requires_grad=True)
        target = position.new_tensor([.5, 2., -1.])
        weights = position.new_tensor([17., 3., .25])
        optimizer = _optimizer(cls, [position])
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad(set_to_none=True)
            objective = ((position - target).square() * weights).sum()
            objective.backward()
            return objective

        trajectory = []
        for _ in range(12):
            optimizer.step(closure)
            trajectory.append(position.detach().clone())
            if cls is CachedLBFGS:
                expected = ((position.detach() - target).square() * weights).sum()
                torch.testing.assert_close(optimizer.accepted_objective, expected)
                assert not optimizer.accepted_objective.requires_grad
                assert optimizer.closure_evaluations == calls
        torch.testing.assert_close(position, target, atol=2e-5, rtol=0)
        trajectories.append(torch.stack(trajectory))
        evaluations.append(calls)
    torch.testing.assert_close(trajectories[1], trajectories[0], atol=2e-6, rtol=2e-6)
    assert evaluations[1] < evaluations[0]


def test_wolfe_rejects_last_trial_and_cache_keeps_selected_gradient():
    # This cusp has no representable stationary point near sqrt(2). Real Wolfe
    # zoom stops at its interval limit, selecting an earlier trial whose gradient
    # has the opposite sign to the last rejected trial. FP64 keeps the two nearby
    # trial coordinates distinguishable; production dtype is not changed.
    runs = []
    for cls in (torch.optim.LBFGS, CachedLBFGS):
        position = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
        optimizer = _optimizer(cls, [position])
        trials = []

        def closure():
            optimizer.zero_grad(set_to_none=True)
            objective = (position.square() - 2).abs().sum()
            objective.backward()
            trials.append((position.detach().clone(), position.grad.clone()))
            return objective

        optimizer.step(closure)
        accepted = position.detach().clone()
        expected_gradient = 2 * accepted * (accepted.square() - 2).sign()
        assert not torch.equal(accepted, trials[-1][0])
        assert expected_gradient.item() * trials[-1][1].item() < 0
        if cls is CachedLBFGS:
            torch.testing.assert_close(optimizer.accepted_objective,
                                       (accepted.square() - 2).abs().sum(), atol=0, rtol=0)
        optimizer.step(closure)
        torch.testing.assert_close(optimizer.state[position]["prev_flat_grad"],
                                   expected_gradient, atol=0, rtol=0)
        runs.append((position.detach().clone(), len(trials)))
    torch.testing.assert_close(runs[1][0], runs[0][0], atol=0, rtol=0)
    assert runs[1][1] == runs[0][1] - 1


def test_projected_gradient_and_multiple_parameter_shapes_are_preserved():
    runs = []
    for cls in (torch.optim.LBFGS, CachedLBFGS):
        position = torch.tensor([[2., -1., 3.]], requires_grad=True)
        scalar = torch.tensor(4., requires_grad=True)
        initial = position.detach().clone()
        direction = position.new_tensor([.6, .8, 0.])
        projection = direction[:, None] @ direction[None, :]
        optimizer = _optimizer(cls, [position, scalar])
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad(set_to_none=True)
            objective = position.square().sum() + 3 * (scalar - 1).square()
            objective.backward()
            position.grad.copy_(position.grad @ projection)
            return objective

        for _ in range(8):
            optimizer.step(closure)
        displacement = position.detach() - initial
        torch.testing.assert_close(displacement @ (torch.eye(3) - projection),
                                   torch.zeros_like(displacement), atol=1e-6, rtol=0)
        if cls is CachedLBFGS:
            assert optimizer.cache_hits == 7
            assert optimizer.closure_evaluations == calls
        runs.append((position.detach().clone(), scalar.detach().clone(), calls))
    torch.testing.assert_close(runs[1][0], runs[0][0])
    torch.testing.assert_close(runs[1][1], runs[0][1])
    assert runs[1][2] == runs[0][2] - 7


@pytest.mark.parametrize("change", ["key", "invalidate", "parameter"])
def test_cache_is_invalidated_when_conditions_or_parameters_change(change):
    position = torch.tensor([4., -3.], requires_grad=True)
    target = torch.zeros_like(position)
    optimizer = _optimizer(CachedLBFGS, [position])
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        objective = (position - target).square().sum()
        objective.backward()
        return objective

    optimizer.step(closure, cache_key=0)
    old_calls = calls
    key = 0
    if change == "key":
        target.add_(2.)
        key = 1
    elif change == "invalidate":
        target.add_(2.)
        optimizer.invalidate_cache()
        assert optimizer.accepted_objective is None
    else:
        with torch.no_grad():
            position.add_(.25)
    expected = (position.detach() - target).square().sum()
    returned = optimizer.step(closure, cache_key=key)
    torch.testing.assert_close(returned, expected, atol=0, rtol=0)
    assert optimizer.cache_hits == 0
    assert optimizer.last_step_evaluations == calls - old_calls
    assert optimizer.closure_evaluations == calls


def test_zero_step_can_cache_initial_gradient_and_new_optimizer_starts_empty():
    position = torch.tensor([2.8], requires_grad=True)
    optimizer = _optimizer(CachedLBFGS, [position])
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        objective = 1e7 + (position - 3).square().sum()
        objective.backward()
        return objective

    initial = position.detach().clone()
    optimizer.step(closure)
    assert float(optimizer.state[position]["t"]) == 0.
    torch.testing.assert_close(position, initial, atol=0, rtol=0)
    first_calls = calls
    optimizer.step(closure)
    assert optimizer.cache_hits == 1
    assert optimizer.last_step_evaluations == calls - first_calls
    rebuilt = _optimizer(CachedLBFGS, [position])
    assert rebuilt.accepted_objective is None
    assert rebuilt.closure_evaluations == 0


def test_failed_closure_discards_accepted_cache():
    position = torch.tensor([4., -3.], requires_grad=True)
    optimizer = _optimizer(CachedLBFGS, [position])

    def closure():
        optimizer.zero_grad(set_to_none=True)
        objective = position.square().sum()
        objective.backward()
        return objective

    optimizer.step(closure)

    def failing_closure():
        raise RuntimeError("trial failed")

    with pytest.raises(RuntimeError, match="trial failed"):
        optimizer.step(failing_closure)
    assert optimizer.accepted_objective is None
    assert optimizer.last_step_evaluations == 1
    hits = optimizer.cache_hits
    optimizer.step(closure)
    assert optimizer.cache_hits == hits


@pytest.mark.parametrize("kwargs", [{"max_iter": 2}, {"line_search_fn": None}])
def test_unsupported_optimizer_modes_are_rejected(kwargs):
    with pytest.raises(ValueError, match="max_iter=1 and strong_wolfe"):
        CachedLBFGS([torch.tensor([1.], requires_grad=True)], **kwargs)
