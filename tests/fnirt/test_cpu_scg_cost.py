"""CPU SCG discarded-cost regression; small fixtures are not benchmarks."""
from dataclasses import asdict

import nibabel as nib
import numpy as np
import pytest
import torch

import fnit.fnirt.registration as module
from fnit.fnirt.optimizer import scaled_conjugate_gradient
from fnit.fnirt.spline import BendingOperator, fsl_control_shape, spline_bases

# The default system computes the complete cost and is the retained oracle.
# Each test compares that same production implementation with its explicit
# CPU derivative-only option; no historical source or private path is loaded.
old = new = module


@pytest.fixture(autouse=True)
def restore_torch_threads():
    previous = torch.get_num_threads()
    yield
    torch.set_num_threads(previous)


def bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8))


def fixture(module, *, device="cpu", scale_fit=False, weighted=True, grad_input=None):
    shape, spacing = (9, 8, 7), (2, 2, 2)
    axes = torch.stack(torch.meshgrid(*(torch.arange(n, dtype=torch.float32, device=device)
                                       for n in shape), indexing="ij"))
    fixed = 10.0 + 0.9 * axes[0] + torch.sin(axes[1]) + 0.3 * axes[2]
    moving = fixed + 0.05 * axes[0].square()
    bases = spline_bases(shape, spacing, (1.0, 1.0, 1.0), device=device, dtype=torch.float64)
    bending = BendingOperator(shape, spacing, (1.0, 1.0, 1.0), device=device,
                              dtype=torch.float64, execution="optimized")
    matrix = torch.eye(4, dtype=torch.float32, device=device)
    moving_mask = torch.ones_like(moving)
    reference_mask = torch.ones(shape, dtype=torch.bool, device=device)
    coordinate_affine = matrix.clone() if grad_input == "coordinate_affine" else None
    if grad_input in ("moving", "fixed", "moving_mask", "coordinate_affine"):
        {"moving": moving, "fixed": fixed, "moving_mask": moving_mask,
         "coordinate_affine": coordinate_affine}[grad_input].requires_grad_(True)
    system = module._LevelSystem(moving, fixed, reference_mask, moving_mask,
        matrix, axes, matrix.clone(), bases, bending, 0.15, weighted, scale_fit,
        coordinate_affine)
    generator = torch.Generator().manual_seed(913)
    coefficients = (torch.randn((3, *fsl_control_shape(shape, spacing)), generator=generator,
                       dtype=torch.float64) * 0.01).to(device)
    scale = torch.tensor(1.125 if scale_fit else 1.0, dtype=torch.float64, device=device)
    if grad_input == "coefficients":
        coefficients.requires_grad_(True)
    elif grad_input == "scale":
        scale.requires_grad_(True)
    counter = {"energy": 0, "normal": 0}
    for name in counter:
        function = getattr(bending, name)
        def tracked(value, _name=name, _function=function):
            counter[_name] += 1
            return _function(value)
        setattr(bending, name, tracked)
    return system, coefficients, scale, counter


def state_equal(actual, expected, skipped=False):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        if skipped and name in ("cost", "bending_energy"):
            assert value is None
        elif isinstance(value, torch.Tensor):
            bits(value, expected[name])
        else:
            assert value == expected[name]


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("scale_fit,weighted", [(False, True), (True, True), (False, False)])
@pytest.mark.parametrize("override", [None, 0.0, 0.03125])
def test_gradient_skip_cost_and_latest_ssd_override_exact(threads, scale_fit, weighted, override):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        original, a, s, old_calls = fixture(old, scale_fit=scale_fit, weighted=weighted)
        candidate, b, t, new_calls = fixture(new, scale_fit=scale_fit, weighted=weighted)
        expected_state, expected = original.gradient(a, s, effective_lambda=override)
        actual_state, actual = candidate.gradient(b, t, effective_lambda=override, _skip_cost=True)
        bits(actual, expected)
        state_equal(actual_state, expected_state, skipped=True)
        assert old_calls == {"energy": 1, "normal": 1}
        assert new_calls == {"energy": 0, "normal": 1}
        # The default entry points retain both costs and the full original state.
        state_equal(candidate.evaluate(b, t, derivatives=True),
                    original.evaluate(a, s, derivatives=True))
        state_equal(candidate.gradient(b, t)[0], original.gradient(a, s)[0])
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("grad_input", ["coefficients", "scale", "moving", "fixed", "moving_mask", "coordinate_affine"])
def test_autograd_request_keeps_original_cost_path(grad_input):
    torch.set_num_threads(1)
    original, a, s, _ = fixture(old, grad_input=grad_input)
    candidate, b, t, calls = fixture(new, grad_input=grad_input)
    expected_state, expected = original.gradient(a, s)
    actual_state, actual = candidate.gradient(b, t, _skip_cost=True)
    bits(actual, expected)
    state_equal(actual_state, expected_state)
    assert calls["energy"] == 1
    assert actual_state["cost"].requires_grad == expected_state["cost"].requires_grad


@pytest.mark.parametrize("weighted", [False, True])
def test_complete_scg_steps_costs_latest_ssd_and_stops_exact(weighted):
    torch.set_num_threads(1)
    reports, vectors, final_states, counters = [], [], [], []
    for module, skip in ((old, False), (new, True)):
        system, coefficients, scale, counter = fixture(module, weighted=weighted)
        latest_ssd = None
        cost_calls, gradient_calls = 0, 0
        def cost(vector):
            nonlocal latest_ssd, cost_calls
            cost_calls += 1
            current, _ = module._unpack(vector, tuple(coefficients.shape[1:]), False)
            state = system.evaluate(current, scale)
            latest_ssd = float(state["ssd"])
            return state["cost"]
        def gradient(vector):
            nonlocal gradient_calls
            gradient_calls += 1
            if latest_ssd is None:
                raise RuntimeError("SCG cost must be evaluated first")
            effective_lambda = 0.15 * latest_ssd if weighted else 0.15
            kwargs = {"_skip_cost": True} if skip else {}
            current, _ = module._unpack(vector, tuple(coefficients.shape[1:]), False)
            return system.gradient(current, scale,
                effective_lambda=effective_lambda, **kwargs)[1]
        vector, report = scaled_conjugate_gradient(cost, gradient,
            module._pack(coefficients), max_iterations=12)
        reports.append(asdict(report)); vectors.append(vector)
        final_coefficients, _ = module._unpack(vector, tuple(coefficients.shape[1:]), False)
        final_states.append(system.evaluate(final_coefficients, scale))
        counters.append((cost_calls, gradient_calls, dict(counter)))
    bits(vectors[1], vectors[0]); assert reports[1] == reports[0]
    state_equal(final_states[1], final_states[0])
    assert counters[1][:2] == counters[0][:2]
    cost_count, gradient_count = counters[1][:2]
    assert gradient_count > 1 and reports[1]["iterations"] > 1
    assert counters[0][2]["energy"] == cost_count + gradient_count + 1
    assert counters[1][2]["energy"] == cost_count + 1


@pytest.mark.parametrize("kind", ["forward_dual", "torch_func_jvp"])
def test_cpu_forward_ad_and_func_keep_original_cost_path(kind, monkeypatch):
    torch.set_num_threads(1)
    original, a, s, old_calls = fixture(old)
    candidate, b, t, new_calls = fixture(new)
    # Use the mature differentiable bending path to isolate this new guard.
    original.bending.execution = candidate.bending.execution = "reference"
    options = []
    evaluate = candidate.evaluate
    def tracked(*args, **kwargs):
        options.append(kwargs.get("_compute_cost", True))
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(candidate, "evaluate", tracked)
    direction = torch.arange(a.numel(), dtype=a.dtype).reshape_as(a) * 0.001
    if kind == "forward_dual":
        with torch.autograd.forward_ad.dual_level():
            old_dual = torch.autograd.forward_ad.make_dual(a, direction)
            new_dual = torch.autograd.forward_ad.make_dual(b, direction)
            assert not old_dual.requires_grad and not new_dual.requires_grad
            expected_state, expected = original.gradient(old_dual, s)
            actual_state, actual = candidate.gradient(new_dual, t, _skip_cost=True)
            assert actual_state["cost"] is not None
            for actual_value, expected_value in ((actual, expected), (actual_state["cost"], expected_state["cost"])):
                actual_parts = torch.autograd.forward_ad.unpack_dual(actual_value)
                expected_parts = torch.autograd.forward_ad.unpack_dual(expected_value)
                bits(actual_parts.primal, expected_parts.primal)
                assert actual_parts.tangent is not None and expected_parts.tangent is not None
                bits(actual_parts.tangent, expected_parts.tangent)
    else:
        def expected_function(value):
            return original.gradient(value, s)[1]
        def actual_function(value):
            assert torch._C._functorch.is_functorch_wrapped_tensor(value)
            assert not value.requires_grad
            return candidate.gradient(value, t, _skip_cost=True)[1]
        expected_parts = torch.func.jvp(expected_function, (a,), (direction,))
        actual_parts = torch.func.jvp(actual_function, (b,), (direction,))
        for actual_value, expected_value in zip(actual_parts, expected_parts):
            bits(actual_value, expected_value)
    assert options and all(options)
    assert new_calls["energy"] == old_calls["energy"] == 1


def test_original_failures_and_cpu_private_guard():
    original, a, s, _ = fixture(old)
    candidate, b, t, _ = fixture(new)
    original.reference_mask.zero_(); candidate.reference_mask.zero_()
    for system, coefficients, scale in ((original, a, s), (candidate, b, t)):
        kwargs = {"_skip_cost": True} if system is candidate else {}
        with pytest.raises(RuntimeError, match="fewer than eight"):
            system.gradient(coefficients, scale, **kwargs)
    candidate, b, t, _ = fixture(new)
    with pytest.raises(ValueError, match="CPU derivative-only"):
        candidate.evaluate(b, t, derivatives=False, _compute_cost=False)
    for system, coefficients, scale in ((original, a, s), (candidate, b, t)):
        system.reference_mask.fill_(True)
        def failure(value):
            raise RuntimeError("energy failure")
        system.bending.energy = failure
        with pytest.raises(RuntimeError, match="energy failure"):
            system.gradient(coefficients, scale)


@pytest.mark.parametrize("threads", [1, 8])
def test_cpu_complete_public_api_uses_skip_only_in_scg_and_keeps_outputs(threads, monkeypatch):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        axes = np.indices((9, 8, 7), dtype=np.float32)
        fixed_data = 10.0 + axes[0] * 0.4 + np.sin(axes[1]) + axes[2] * 0.3
        moving_data = fixed_data + 0.04 * axes[0] * axes[0]
        fixed = nib.Nifti1Image(fixed_data.astype(np.float32), np.eye(4))
        moving = nib.Nifti1Image(moving_data.astype(np.float32), np.eye(4))
        from fnit._transforms import AffineTransform
        initial = AffineTransform(np.eye(4), source=moving, target=fixed, space="world")
        kwargs = dict(subsampling=(1, 1), maximum_iterations=(1, 4),
            input_fwhm_mm=(0.0, 0.0), reference_fwhm_mm=(0.0, 0.0),
            regularization=(0.15, 0.03), estimate_intensity=(False, False),
            apply_reference_mask=(False, False), minimization_methods=("lm", "scg"),
            process_stages=(1, 2), warp_resolution_schedule_mm=((4.0,) * 3, (2.0,) * 3),
            jacobian_range=(0.01, 100.0))
        function = module._LevelSystem.gradient
        def full_cost(system, *args, **options):
            options["_skip_cost"] = False
            return function(system, *args, **options)
        with monkeypatch.context() as baseline_context:
            baseline_context.setattr(module._LevelSystem, "gradient", full_cost)
            original = module.TorchFNIRT(device="cpu", config=module.GMFNIRTConfig(**kwargs))(moving, fixed, initial)
        calls = []
        def tracked(system, *args, **options):
            assert options.get("_skip_cost") is True
            calls.append(options["effective_lambda"])
            return function(system, *args, **options)
        monkeypatch.setattr(new._LevelSystem, "gradient", tracked)
        candidate = new.TorchFNIRT(device="cpu", config=new.GMFNIRTConfig(**kwargs))(moving, fixed, initial)
        assert len(calls) > 1
        for name in ("coefficients", "moved", "full_pull_jacobian", "nonlinear_jacobian", "modulated_gm"):
            expected, actual = getattr(original, name), getattr(candidate, name)
            if isinstance(expected, np.ndarray):
                assert np.array_equal(np.ascontiguousarray(actual).view(np.uint8), np.ascontiguousarray(expected).view(np.uint8))
            else:
                assert np.array_equal(np.ascontiguousarray(actual.dataobj).view(np.uint8), np.ascontiguousarray(expected.dataobj).view(np.uint8))
                assert np.array_equal(actual.affine, expected.affine)
                assert actual.header.binaryblock == expected.header.binaryblock
        def numeric_qc(value):
            if isinstance(value, dict):
                return {key: numeric_qc(item) for key, item in value.items() if key != "elapsed_seconds"}
            if isinstance(value, list):
                return [numeric_qc(item) for item in value]
            return value
        assert numeric_qc(candidate.qc) == numeric_qc(original.qc)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("scale_fit", [False, True])
def test_cuda_cost_and_gradient_body_unchanged(scale_fit):
    original, a, s, _ = fixture(old, device="cuda", scale_fit=scale_fit)
    candidate, b, t, calls = fixture(new, device="cuda", scale_fit=scale_fit)
    expected_state, expected = original.gradient(a, s)
    actual_state, actual = candidate.gradient(b, t, _skip_cost=True)
    bits(actual, expected); state_equal(actual_state, expected_state)
    assert calls["energy"] == 1
