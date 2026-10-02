import pytest
import torch

from fnit.gems.optim import CachedArmijoLBFGS, CachedLBFGS, PrecisionLBFGS


_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))]


def _optimizer(cls, parameters):
    return cls(parameters, lr=1., max_iter=1, history_size=12,
               tolerance_grad=1e-10, tolerance_change=1e-10,
               line_search_fn="strong_wolfe")


def test_multistep_convergence_matches_torch_with_fewer_real_evaluations():
    trajectories, evaluations = [], []
    for cls in (torch.optim.LBFGS, PrecisionLBFGS, CachedLBFGS):
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
    torch.testing.assert_close(trajectories[1], trajectories[0], atol=0, rtol=0)
    torch.testing.assert_close(trajectories[2], trajectories[0], atol=2e-6, rtol=2e-6)
    assert evaluations[1] == evaluations[0]
    assert evaluations[2] < evaluations[0]


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


@pytest.mark.parametrize("offset", [1e7, 1e10])
def test_double_scalar_cost_preserves_small_decrease_with_float32_parameters(offset):
    # The FP32 scalar version above stalls because its .04 decrease is below
    # one cost ULP. A double final sum must restore the known minimum without
    # changing the parameter or gradient dtype, including on cached evaluations.
    trajectories, evaluations = [], []
    for cls in (torch.optim.LBFGS, CachedLBFGS):
        position = torch.tensor([2.8], dtype=torch.float32, requires_grad=True)
        optimizer = _optimizer(cls, [position])
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad(set_to_none=True)
            objective = (position - 3).square().sum(dtype=torch.float64) + offset
            objective.backward()
            assert objective.dtype == torch.float64
            assert position.grad.dtype == torch.float32
            return objective

        trajectory = []
        for _ in range(3):
            optimizer.step(closure)
            assert position.dtype == torch.float32
            assert position.grad.dtype == torch.float32
            trajectory.append(position.detach().clone())
            if cls is CachedLBFGS:
                expected = ((position.detach() - 3).square().sum(dtype=torch.float64)
                            + offset)
                assert optimizer.accepted_objective.dtype == torch.float64
                assert not optimizer.accepted_objective.requires_grad
                torch.testing.assert_close(optimizer.accepted_objective,
                                           expected, atol=0, rtol=0)
                assert optimizer.closure_evaluations == calls

        torch.testing.assert_close(position, position.new_tensor([3.]), atol=0, rtol=0)
        assert float(optimizer.state[position]["t"]) > 0
        if cls is CachedLBFGS:
            assert optimizer.cache_hits == 2
        trajectories.append(torch.stack(trajectory))
        evaluations.append(calls)

    torch.testing.assert_close(trajectories[1], trajectories[0], atol=0, rtol=0)
    assert evaluations[1] == evaluations[0] - 2


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("cls", [PrecisionLBFGS, CachedLBFGS])
def test_double_state_rejects_cost_increase_hidden_by_float32_armijo(cls, device):
    # The first trial at x=1 increases a FP64 objective by .0005 and has zero
    # gradient. FP32 Wolfe arithmetic rounds its Armijo bound to 1e7 and accepts
    # the rise; FP64 working state must reject it and take a decreasing step.
    outcomes = []
    for double_state in (False, True):
        position = torch.tensor([0.], device=device, dtype=torch.float32, requires_grad=True)
        optimizer = cls([position], lr=1., max_iter=1, history_size=12,
                        tolerance_grad=1e-10, tolerance_change=1e-10,
                        line_search_fn="strong_wolfe",
                        double_precision_state=double_state)

        def value():
            return (-position + 2.0015 * position.square()
                    - 1.001 * position.pow(3)).sum().double() + 1e7 + .04

        def closure():
            optimizer.zero_grad(set_to_none=True)
            objective = value()
            objective.backward()
            assert position.grad.dtype == torch.float32
            return objective

        initial_cost = value().detach()
        optimizer.step(closure)
        accepted_cost = value().detach()
        assert position.dtype == position.grad.dtype == torch.float32
        if cls is CachedLBFGS:
            torch.testing.assert_close(optimizer.accepted_objective,
                                       accepted_cost, atol=0, rtol=0)
        outcomes.append((position.detach().clone(), accepted_cost))
        if double_state:
            assert accepted_cost < initial_cost
            assert accepted_cost <= initial_cost - 1e-4 * position.detach().double().sum()
            assert optimizer.state[position]["d"].dtype == torch.float64

    torch.testing.assert_close(outcomes[0][0], position.new_tensor([1.]), atol=0, rtol=0)
    assert outcomes[0][1] > initial_cost
    assert outcomes[1][0].item() < 1.


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("cls", [PrecisionLBFGS, CachedLBFGS])
def test_double_working_gradient_preserves_cancelling_directional_product(cls, device):
    position = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
    position.grad = position.new_tensor([1e8, 1., -1e8])
    optimizer = cls([position], double_precision_state=True)
    working_gradient = optimizer._gather_flat_grad()
    direction = torch.ones_like(working_gradient)
    assert working_gradient.dtype == torch.float64
    torch.testing.assert_close(working_gradient.dot(direction),
                               working_gradient.new_tensor(1.), atol=0, rtol=0)
    assert position.dtype == position.grad.dtype == torch.float32


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("cls", [PrecisionLBFGS, CachedLBFGS])
def test_double_state_keeps_float32_parameter_grad_and_accepted_cache(cls, device):
    position = torch.tensor([2.8], device=device, dtype=torch.float32, requires_grad=True)
    optimizer = cls([position], lr=1., max_iter=1, history_size=12,
                    tolerance_grad=1e-10, tolerance_change=1e-10,
                    line_search_fn="strong_wolfe", double_precision_state=True)

    def value():
        return (position - 3).square().sum(dtype=torch.float64) + 1e7

    def closure():
        optimizer.zero_grad(set_to_none=True)
        objective = value()
        objective.backward()
        assert position.grad.dtype == torch.float32
        return objective

    for _ in range(3):
        optimizer.step(closure)
        assert position.dtype == position.grad.dtype == torch.float32
        if cls is CachedLBFGS:
            torch.testing.assert_close(optimizer.accepted_objective,
                                       value().detach(), atol=0, rtol=0)
            assert optimizer.accepted_objective.dtype == torch.float64
            assert not optimizer.accepted_objective.requires_grad
    torch.testing.assert_close(position, position.new_tensor([3.]), atol=0, rtol=0)
    assert optimizer.state[position]["d"].dtype == torch.float64
    assert optimizer.state[position]["prev_flat_grad"].dtype == torch.float64
    if cls is CachedLBFGS:
        assert optimizer.cache_hits == 2


@pytest.mark.parametrize("cls", [PrecisionLBFGS, CachedLBFGS])
def test_double_precision_state_rejects_non_bool(cls):
    with pytest.raises(ValueError, match="double_precision_state must be a bool"):
        cls([torch.tensor([1.], requires_grad=True)], double_precision_state="float64")


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("cls", [PrecisionLBFGS, CachedLBFGS])
def test_history_reset_recovers_zero_wolfe_step_near_positive_barrier(cls, device):
    # A valid curvature pair between feasible points couples a nearly constrained
    # coordinate into the quasi-Newton direction. Its finite barrier rejects the
    # line search even though g*d<0; normalized steepest descent remains feasible.
    position = torch.tensor([1e-8, .5], device=device, dtype=torch.float32, requires_grad=True)
    optimizer = cls([position], lr=1., max_iter=1, history_size=12,
                    tolerance_grad=1e-10, tolerance_change=1e-10,
                    line_search_fn="strong_wolfe", double_precision_state=True)

    def value(point):
        regular = (point[1] - 1).square() + 1e-8 * point[0].square()
        barrier = 1e12 * (1 - point[0])
        return torch.where(point[0] > 0, regular, barrier).double()

    def gradient_at(point):
        point = point.detach().clone().requires_grad_(True)
        return torch.autograd.grad(value(point), point)[0].double()

    previous = position.detach() + position.new_tensor([1., -1.])
    displacement = position.detach().double() - previous.double()
    gradient_change = gradient_at(position) - gradient_at(previous)
    curvature = gradient_change.dot(displacement)
    assert curvature > 0
    state = optimizer.state[position]
    state.update(n_iter=1, func_evals=0, old_dirs=[gradient_change],
                 old_stps=[displacement], ro=[1 / curvature],
                 H_diag=curvature / gradient_change.dot(gradient_change),
                 prev_flat_grad=gradient_at(position),
                 d=torch.zeros_like(position, dtype=torch.float64), t=0.,
                 prev_loss=float(value(position)))
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        objective = value(position)
        objective.backward()
        return objective

    initial = position.detach().clone()
    initial_cost = value(initial).detach()
    optimizer.step(closure)
    torch.testing.assert_close(position, initial, atol=0, rtol=0)
    assert float(state["t"]) == 0.
    assert gradient_at(position).dot(state["d"]) < 0
    evaluations = state["func_evals"]
    if cls is CachedLBFGS:
        cache = optimizer._cached
        torch.testing.assert_close(cache[0], initial_cost, atol=0, rtol=0)

    optimizer.reset_history()
    assert state["func_evals"] == evaluations
    if cls is CachedLBFGS:
        assert optimizer._cached is cache
    returned = optimizer.step(closure)
    torch.testing.assert_close(returned, initial_cost, atol=0, rtol=0)
    assert state["n_iter"] == 1
    assert state["t"] > 0
    assert position[0] > 0 and value(position) < initial_cost
    torch.testing.assert_close(position[1], position.new_tensor(1.), atol=0, rtol=0)
    assert position.dtype == position.grad.dtype == torch.float32
    assert state["d"].dtype == torch.float64
    if cls is CachedLBFGS:
        assert optimizer.cache_hits == 1
        assert optimizer.closure_evaluations == calls
        torch.testing.assert_close(optimizer.accepted_objective,
                                   value(position).detach(), atol=0, rtol=0)
    cache_hits = optimizer.cache_hits if cls is CachedLBFGS else 0
    assert state["func_evals"] == calls + cache_hits


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


@pytest.mark.parametrize("device", _DEVICES)
def test_armijo_mesh_converges_monotonically_with_native_parameters_and_bounded_steps(device):
    position = torch.tensor([[2., -1., .7], [-.5, 1.5, -2.]],
                            device=device, dtype=torch.float32, requires_grad=True)
    weights = position.new_tensor([1., 4., 8.])
    optimizer = CachedArmijoLBFGS([position], max_vertex_displacement=.25,
                                  tolerance_grad=1e-8)
    calls = 0

    def value():
        return (position.square() * weights).sum(dtype=torch.float64) + 1e7

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        objective = value()
        objective.backward()
        return objective

    previous_cost = value().detach()
    moved = 0
    for _ in range(60):
        previous = position.detach().clone()
        returned = optimizer.step(closure)
        torch.testing.assert_close(returned, previous_cost, atol=0, rtol=0)
        accepted = value().detach()
        assert accepted <= previous_cost
        delta = position.detach() - previous
        assert delta.norm(dim=1).max() <= .25
        if not torch.equal(position.detach(), previous):
            moved += 1
            assert accepted < previous_cost
            expected_bound = previous_cost + 1e-4 * optimizer.state[position]["prev_flat_grad"].dot(
                delta.flatten().double())
            assert accepted <= expected_bound
        torch.testing.assert_close(optimizer.accepted_objective, accepted, atol=0, rtol=0)
        assert position.dtype == position.grad.dtype == torch.float32
        assert optimizer._cached[1].dtype == torch.float64
        assert len(optimizer.state[position]["old_stps"]) <= 12
        assert all(pair.dtype == torch.float64 for pair in optimizer.state[position]["old_stps"])
        previous_cost = accepted
    assert moved > 5
    torch.testing.assert_close(position, torch.zeros_like(position), atol=3e-4, rtol=0)
    assert optimizer.closure_evaluations == calls
    assert optimizer.cache_hits == 59
    assert optimizer.state[position]["func_evals"] == calls + optimizer.cache_hits


@pytest.mark.parametrize("device", _DEVICES)
def test_armijo_curvature_and_bounds_use_actual_float32_displacement(device):
    # At this coordinate FP32 steps quantize to .0625. The proposed .1 move
    # rounds to .125 and exceeds the bound; a feasible .0625 step is evaluated.
    position = torch.tensor([[1e6, 0., 0.]], device=device, dtype=torch.float32,
                            requires_grad=True)
    target = position.detach().double() - position.new_tensor([[.37, 0., 0.]]).double()
    optimizer = CachedArmijoLBFGS([position], max_vertex_displacement=.1)
    initial = position.detach().clone()
    trials = []

    def value():
        return (position.double() - target).square().sum() + 1e7

    def closure():
        optimizer.zero_grad(set_to_none=True)
        objective = value()
        objective.backward()
        trials.append(position.detach().clone())
        return objective

    initial_cost = value().detach()
    optimizer.step(closure)
    delta = position.detach().double() - initial.double()
    torch.testing.assert_close(delta, delta.new_tensor([[-.0625, 0., 0.]]), atol=0, rtol=0)
    torch.testing.assert_close(optimizer.state[position]["old_stps"][0], delta.flatten(), atol=0, rtol=0)
    assert len(trials) == 2  # Over-bound native rounding is rejected before closure.
    assert value() <= initial_cost + 1e-4 * optimizer.state[position]["prev_flat_grad"].dot(delta.flatten())
    torch.testing.assert_close(optimizer.accepted_objective, value().detach(), atol=0, rtol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_armijo_rejects_infeasible_barrier_and_keeps_finite_decreasing_cache(device):
    position = torch.tensor([[.2, .5, 0.]], device=device, dtype=torch.float32,
                            requires_grad=True)
    target = position.new_tensor([[-.5, 1., 0.]])
    optimizer = CachedArmijoLBFGS([position])
    rejected = 0

    def value():
        return (position - target).square().sum(dtype=torch.float64)

    def closure():
        nonlocal rejected
        optimizer.zero_grad(set_to_none=True)
        if position[0, 0] <= 0:
            rejected += 1
            return position.new_tensor(float("inf"), dtype=torch.float64)
        objective = value()
        objective.backward()
        return objective

    initial_cost = value().detach()
    optimizer.step(closure)
    assert rejected > 0 and position[0, 0] > 0
    assert optimizer.accepted_objective < initial_cost
    torch.testing.assert_close(optimizer.accepted_objective, value().detach(), atol=0, rtol=0)
    torch.testing.assert_close(position.grad, 2 * (position.detach() - target), atol=0, rtol=0)


@pytest.mark.parametrize("invalid", ["cost_nan", "gradient_nan"])
def test_armijo_rejects_nonfinite_trial_cost_or_gradient(invalid):
    position = torch.tensor([[.2, 0., 0.]], requires_grad=True)
    target = position.new_tensor([[.8, 0., 0.]])
    optimizer = CachedArmijoLBFGS([position])
    rejected = 0

    def closure():
        nonlocal rejected
        optimizer.zero_grad(set_to_none=True)
        objective = (position - target).square().sum(dtype=torch.float64)
        objective.backward()
        if position[0, 0] > .5:
            rejected += 1
            if invalid == "cost_nan":
                return objective.detach().new_tensor(float("nan"))
            position.grad.fill_(float("nan"))
        return objective

    optimizer.step(closure)
    assert rejected > 0 and .2 < position[0, 0] <= .5
    assert torch.isfinite(position.grad).all()
    torch.testing.assert_close(optimizer.accepted_objective,
                               (position.detach() - target).square().sum(dtype=torch.float64), atol=0, rtol=0)


@pytest.mark.parametrize("failure", ["inf", "exception"])
def test_armijo_failed_trials_restore_parameters_gradient_cache_and_versions(failure):
    position = torch.zeros((1, 3), requires_grad=True)
    target = position.new_tensor([[1., 0., 0.]])
    optimizer = CachedArmijoLBFGS([position], max_backtracking_steps=5)
    permit_trial = False

    def closure():
        optimizer.zero_grad(set_to_none=True)
        if position[0, 0] > 0 and not permit_trial:
            if failure == "exception":
                raise RuntimeError("invalid trial")
            return position.new_tensor(float("inf"), dtype=torch.float64)
        objective = (position - target).square().sum(dtype=torch.float64)
        objective.backward()
        return objective

    initial = position.detach().clone()
    if failure == "exception":
        with pytest.raises(RuntimeError, match="invalid trial"):
            optimizer.step(closure, cache_key=7)
    else:
        optimizer.step(closure, cache_key=7)
        assert optimizer.last_step_evaluations == 6
        assert optimizer.state[position]["t"] == 0
    torch.testing.assert_close(position, initial, atol=0, rtol=0)
    torch.testing.assert_close(position.grad, 2 * (initial - target), atol=0, rtol=0)
    torch.testing.assert_close(optimizer.accepted_objective,
                               torch.tensor(1., dtype=torch.float64), atol=0, rtol=0)
    assert optimizer._versions == (position._version,)
    assert optimizer._cache_key == 7
    cached = optimizer._cached
    permit_trial = True
    optimizer.step(closure, cache_key=7)
    assert optimizer.cache_hits == 1
    assert optimizer._cached is not cached
    assert optimizer.accepted_objective < 1


@pytest.mark.parametrize("change", ["key", "invalidate", "parameter"])
def test_armijo_invalidates_cache_and_old_curvature_when_objective_changes(change):
    position = torch.tensor([[1., 0., 0.]], requires_grad=True)
    target = torch.zeros_like(position)
    optimizer = CachedArmijoLBFGS([position])

    def closure():
        optimizer.zero_grad(set_to_none=True)
        objective = (position - target).square().sum(dtype=torch.float64)
        objective.backward()
        return objective

    optimizer.step(closure, cache_key=0)
    assert len(optimizer.state[position]["old_stps"]) == 1
    old_pair = optimizer.state[position]["old_stps"][0]
    key = 0
    if change == "key":
        target.fill_(2.)
        key = 1
    elif change == "invalidate":
        target.fill_(2.)
        optimizer.invalidate_cache()
    else:
        with torch.no_grad():
            position.add_(.25)
    initial = (position.detach() - target).square().sum(dtype=torch.float64)
    returned = optimizer.step(closure, cache_key=key)
    torch.testing.assert_close(returned, initial, atol=0, rtol=0)
    assert optimizer.cache_hits == 0
    assert optimizer.state[position]["n_iter"] == 1
    assert all(pair is not old_pair for pair in optimizer.state[position]["old_stps"])
    torch.testing.assert_close(optimizer.accepted_objective,
                               (position.detach() - target).square().sum(dtype=torch.float64), atol=0, rtol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_armijo_refreshes_non_descent_history_before_accepting_step(device):
    position = torch.zeros((1, 3), device=device, requires_grad=True)
    target = position.new_tensor([[1., 0., 0.]])
    optimizer = CachedArmijoLBFGS([position])
    state = optimizer.state[position]
    bad_s = torch.tensor([1., 0., 0.], device=device, dtype=torch.float64)
    state.update(old_stps=[bad_s], old_dirs=[-bad_s],
                 ro=[bad_s.new_tensor(-1.)], H_diag=-1.)
    # The first parameter-version initialization resets history, so start from
    # a valid evaluated cache before injecting this stale non-descent history.
    optimizer._versions = (position._version,)
    optimizer._cached = (position.new_tensor(1., dtype=torch.float64),
                          (-2 * target).flatten().double())

    def closure():
        optimizer.zero_grad(set_to_none=True)
        objective = (position - target).square().sum(dtype=torch.float64)
        objective.backward()
        return objective

    optimizer.step(closure)
    assert state["prev_flat_grad"].dot(state["d"]) < 0
    assert position[0, 0] > 0 and optimizer.accepted_objective < 1
    assert all(pair is not bad_s for pair in state["old_stps"])
    assert all(s.dot(y) > 0 for s, y in zip(state["old_stps"], state["old_dirs"]))


@pytest.mark.parametrize("kwargs", [{"max_vertex_displacement": 0},
                                    {"max_vertex_displacement": float("inf")},
                                    {"max_backtracking_steps": 0},
                                    {"max_backtracking_steps": True},
                                    {"armijo_constant": 1.},
                                    {"double_precision_state": False}])
def test_armijo_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        CachedArmijoLBFGS([torch.zeros((1, 3), requires_grad=True)], **kwargs)
