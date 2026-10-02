"""Execution selection, option forwarding and search scheduling contracts."""

from contextlib import nullcontext
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
import torch

import fnit.cli as toolkit_cli
import fnit.flirt as flirt_module
import fnit.flirt.cli as flirt_cli
import fnit.flirt.standalone as standalone
from fnit._nib import FNITNifti1Image
from fnit.flirt import TorchFLIRT
from fnit.flirt import batched, core


@pytest.fixture(autouse=True)
def restore_global_torch_settings():
    matmul, cudnn = (torch.backends.cuda.matmul.allow_tf32,
                     torch.backends.cudnn.allow_tf32)
    threads = torch.get_num_threads()
    yield
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn
    torch.set_num_threads(threads)


@pytest.mark.parametrize("device,requested,expected", (
    ("cpu", "auto", "reference"), ("cpu", "batched", "batched"),
    ("cpu", "reference", "reference"), ("cuda:0", "auto", "batched"),
    ("cuda:0", "reference", "reference"), ("cuda:0", "batched", "batched"),
))
def test_execution_selection_without_starting_registration(monkeypatch, device, requested, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model = TorchFLIRT(device=device, execution=requested,
                       candidate_batch_size=32, memory_budget_gb=7.5)

    assert model.execution == expected
    assert model.candidate_batch_size == 32
    assert model.memory_budget_gb == 7.5


def test_default_device_and_execution_follow_cuda_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = TorchFLIRT()
    assert model.device.type == "cpu"
    assert model.execution == "reference"
    with pytest.raises(RuntimeError, match="CUDA"):
        TorchFLIRT(device="cuda:0")


@pytest.mark.parametrize("requested,expected", (("auto", "reference"), ("reference", "reference")))
def test_cuda_nmi_without_triton_uses_reference(monkeypatch, requested, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core, "find_spec", lambda name: None)

    model = TorchFLIRT(device="cuda:0", dof=6, cost="normmi", execution=requested)

    assert model.execution == expected


def test_explicit_cuda_batched_nmi_without_triton_reports_required_dependency(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="Triton.*execution='reference'"):
        TorchFLIRT(device="cuda:0", dof=6, cost="normmi", execution="batched")


def test_cpu_batched_nmi_does_not_require_triton(monkeypatch):
    def unexpected_lookup(name):
        raise AssertionError("CPU execution must not require a CUDA compiler")

    monkeypatch.setattr(core, "find_spec", unexpected_lookup)
    assert TorchFLIRT(device="cpu", dof=6, cost="normmi",
                      execution="batched").execution == "batched"


@pytest.mark.parametrize("options", (
    {"execution": "fast"}, {"candidate_batch_size": 0},
    {"candidate_batch_size": -1}, {"candidate_batch_size": 1.5},
    {"memory_budget_gb": 0}, {"memory_budget_gb": -1},
    {"memory_budget_gb": float("nan")}, {"memory_budget_gb": float("inf")},
))
def test_execution_options_reject_invalid_values(options):
    with pytest.raises(ValueError):
        TorchFLIRT(device="cpu", **options)


def test_python_wrapper_forwards_execution_options_and_writes_matrix(monkeypatch, tmp_path):
    calls = {}
    result = SimpleNamespace(matrix=np.eye(4))

    class FakeModel:
        def __init__(self, **options):
            calls["constructor"] = options

        def __call__(self, *images, **options):
            calls["registration"] = (images, options)
            return result

    monkeypatch.setattr(standalone, "TorchFLIRT", FakeModel)
    output_matrix = tmp_path / "input_to_reference.mat"

    actual = standalone.run_flirt(
        "moving.nii.gz", "reference.nii.gz", omat=output_matrix,
        init="initial.mat", device="cpu", dof=6, cost="normmi",
        execution="batched", candidate_batch_size=17, memory_budget_gb=6.25,
    )

    assert actual is result
    assert calls["constructor"] == {
        "device": "cpu", "dof": 6, "cost": "normmi", "execution": "batched",
        "candidate_batch_size": 17, "memory_budget_gb": 6.25,
    }
    assert calls["registration"] == (("moving.nii.gz", "reference.nii.gz"),
                                        {"init": "initial.mat", "inweight": None, "refweight": None})
    np.testing.assert_array_equal(np.loadtxt(output_matrix), np.eye(4))


@pytest.mark.parametrize("entrypoint", ("standalone", "toolkit"))
def test_cli_forwards_execution_options(monkeypatch, entrypoint):
    calls = []

    def capture(*images, **options):
        calls.append((images, options))

    arguments = ["-in", "moving.nii.gz", "-ref", "reference.nii.gz",
                 "-omat", "transform.mat", "-dof", "6", "-cost", "normmi",
                 "--device", "cuda:0", "--execution", "reference",
                 "--candidate-batch-size", "31", "--memory-budget-gb", "9.5"]
    if entrypoint == "standalone":
        monkeypatch.setattr(flirt_cli, "run_flirt", capture)
        assert flirt_cli.main(arguments) == 0
    else:
        monkeypatch.setattr(flirt_module, "run_flirt", capture)
        toolkit_cli.main(["flirt", *arguments, "--threads", "1"])

    assert len(calls) == 1
    images, options = calls[0]
    assert images == ("moving.nii.gz", "reference.nii.gz")
    assert options["execution"] == "reference"
    assert options["candidate_batch_size"] == 31
    assert options["memory_budget_gb"] == 9.5
    assert options["device"] == "cuda:0"
    assert options["applyxfm"] is False
    assert options["usesqform"] is False


@pytest.mark.parametrize("transform_arguments,expected_init,expected_qform", (
    (["-usesqform"], None, True),
    (["-init", "input_to_reference.mat"], "input_to_reference.mat", False),
))
def test_root_and_dedicated_cli_forward_the_same_applyxfm_options(
    monkeypatch, transform_arguments, expected_init, expected_qform
):
    calls = []

    def capture(*images, **options):
        calls.append((images, options))

    monkeypatch.setattr(flirt_cli, "run_flirt", capture)
    monkeypatch.setattr(flirt_module, "run_flirt", capture)
    arguments = ["-in", "moving.nii.gz", "-ref", "reference.nii.gz",
                 "-out", "moved.nii.gz", "-omat", "saved.mat", "-applyxfm",
                 *transform_arguments, "--device", "cpu", "--execution", "reference",
                 "--candidate-batch-size", "31", "--memory-budget-gb", "9.5", "--overwrite"]

    assert flirt_cli.main(arguments) == 0
    toolkit_cli.main(["flirt", *arguments, "--threads", "1"])

    expected = (("moving.nii.gz", "reference.nii.gz"), {
        "output": "moved.nii.gz", "omat": "saved.mat", "init": expected_init,
        "inweight": None, "refweight": None, "dof": 12, "cost": "corratio",
        "applyxfm": True, "usesqform": expected_qform, "device": "cpu",
        "execution": "reference", "candidate_batch_size": 31,
        "memory_budget_gb": 9.5, "overwrite": True,
    })
    assert calls == [expected, expected]


def test_root_and_dedicated_applyxfm_write_identical_images_and_matrices(tmp_path):
    import nibabel as nib

    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    image = FNITNifti1Image(np.arange(120, dtype=np.float32).reshape(4, 5, 6), affine)
    reference = FNITNifti1Image(np.zeros((4, 5, 6), dtype=np.float32), affine)
    moving_path, reference_path = tmp_path / "moving.nii.gz", tmp_path / "reference.nii.gz"
    image.save(moving_path)
    reference.save(reference_path)
    outputs = []
    for entrypoint in ("standalone", "toolkit"):
        moved_path, matrix_path = tmp_path / f"{entrypoint}.nii.gz", tmp_path / f"{entrypoint}.mat"
        arguments = ["-in", str(moving_path), "-ref", str(reference_path),
                     "-out", str(moved_path), "-omat", str(matrix_path),
                     "-applyxfm", "-usesqform", "--device", "cpu"]
        if entrypoint == "standalone":
            assert flirt_cli.main(arguments) == 0
        else:
            toolkit_cli.main(["flirt", *arguments])
        outputs.append((nib.load(moved_path), np.loadtxt(matrix_path)))

    np.testing.assert_array_equal(outputs[0][0].dataobj, outputs[1][0].dataobj)
    np.testing.assert_array_equal(outputs[0][0].affine, outputs[1][0].affine)
    assert outputs[0][0].header.binaryblock == outputs[1][0].header.binaryblock
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


def test_dedicated_cli_argument_import_does_not_load_torch_or_registration():
    code = ("import sys; import fnit.flirt.cli; "
            "assert 'torch' not in sys.modules; "
            "assert 'fnit.flirt.core' not in sys.modules; "
            "assert 'fnit.flirt.standalone' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_model_passes_execution_options_to_engine_and_qc(monkeypatch):
    selected = {}

    class FakeEngine:
        cost_evaluations = 2
        phase_timings = {"angular_coarse": .01}
        phase_cost_evaluations = {"angular_coarse": 2}
        batch_evaluations = 1
        host_result_transfers = 1

        def __init__(self, *images, **options):
            selected.update(options)

        def run(self, qsform, dof):
            return .25, np.eye(4)

        def phase(self, name):
            return nullcontext()

    monkeypatch.setattr(core, "_DefaultFLIRTEngine", FakeEngine)
    monkeypatch.setattr(core, "_resample_output",
                        lambda moving, shape, *args, **kwargs: torch.zeros(shape))
    image = FNITNifti1Image(np.ones((3, 4, 5), dtype=np.float32), np.eye(4))

    result = TorchFLIRT(device="cpu", execution="batched",
                       candidate_batch_size=13, memory_budget_gb=4.25)(image, image)

    assert selected["execution"] == result.qc["execution"] == "batched"
    assert selected["candidate_batch_size"] == result.qc["candidate_batch_size"] == 13
    assert selected["memory_budget_gb"] == result.qc["memory_budget_gb"] == 4.25
    assert result.qc["batched_host_result_transfers"] == 1
    assert result.qc["phase_timings_seconds"] == {"angular_coarse": .01}


def test_candidate_cache_preserves_order_and_deduplicates_only_exact_affines(monkeypatch):
    calls = []

    class FakeEvaluator:
        def __init__(self, reference, **options):
            pass

        def __call__(self, matrices):
            calls.append(matrices.copy())
            return torch.as_tensor(matrices[:, 0, 3].copy())

    monkeypatch.setattr(batched, "BatchedAffineCost", FakeEvaluator)
    engine = object.__new__(core._DefaultFLIRTEngine)
    engine.initial_matrix = np.eye(4)
    engine.initial_matrix[0, 3] = 4
    engine.level = SimpleNamespace(cost=SimpleNamespace(moving=object(), smooth_size=1))
    engine._cache = {}
    engine._batch_source = None
    engine.candidate_batch_size = 8
    engine.memory_budget_gb = 4.
    engine.host_result_transfers = engine.batch_evaluations = engine.cost_evaluations = 0
    engine._active_phase = "test"
    engine.phase_cost_evaluations = {}
    first = np.eye(4)
    first[0, 0], first[0, 3] = 2, 1
    second = first.copy()
    second[0, 3] += 1e-10

    actual = engine.cost_many([first, second, first])
    cached = engine.cost_many([second, first])

    assert len(calls) == 1
    assert calls[0].shape == (2, 4, 4)
    np.testing.assert_array_equal(calls[0], np.stack([first, second]) @ engine.initial_matrix)
    assert actual == [9., 9. + 1e-10, 9.]
    assert cached == [9. + 1e-10, 9.]
    assert engine.cost_many([]) == []
    assert engine.cost_evaluations == 2
    assert engine.batch_evaluations == engine.host_result_transfers == 1


@pytest.mark.parametrize("dof", (6, 12))
def test_batched_schedule_keeps_serial_candidate_and_perturbation_order(dof):
    def run(execution):
        engine = object.__new__(core._DefaultFLIRTEngine)
        engine.execution = execution
        engine.angular_search = True
        engine.initial_matrix = np.eye(4)
        engine.level = SimpleNamespace(centre=np.array([1.7, -.4, 2.3]))
        engine.phase_timings = {}
        engine._active_phase = "preparation"
        engine.bound_guess = (10., 1.)
        trace, scales = [], []

        def cost(matrix):
            return float(np.square(matrix[:3, 3]).sum()
                         + np.square(matrix[:3, :3] - np.eye(3)).sum())

        def set_scale(scale):
            engine.requested_scale = scale
            scales.append(scale)

        def candidates(sign):
            output = []
            for value in range(1, 6):
                matrix = np.eye(4)
                matrix[0, 3] = sign * value
                output.append((float(value), matrix))
            return output

        def optimize(rows, selected_dof, maximum_iterations, perturbation=None):
            output = []
            for _, matrix in rows:
                if perturbation is not None:
                    parameters = core.fsl_parameters_from_affine(matrix, engine.level.centre)
                    parameters += perturbation
                    matrix = engine._parameters_to_matrix(parameters, 12)
                trace.append((engine.requested_scale, selected_dof, maximum_iterations,
                              matrix.tobytes()))
                output.append((cost(matrix), matrix.copy()))
            return output

        engine.set_scale = set_scale
        engine.angular_candidates = lambda selected_dof: (candidates(1), candidates(-1))
        engine._measure = lambda rows: [(cost(matrix), matrix.copy()) for _, matrix in rows]
        engine._optimize = optimize
        engine.cost = cost
        qsform = np.eye(4)
        qsform[0, 3] = .5
        result = engine.run(qsform, dof=dof)
        return result, trace, scales

    (reference_cost, reference_matrix), reference_trace, reference_scales = run("reference")
    (batched_cost, batched_matrix), batched_trace, batched_scales = run("batched")

    assert batched_cost == reference_cost
    np.testing.assert_array_equal(batched_matrix, reference_matrix)
    assert batched_trace == reference_trace
    assert batched_scales == reference_scales == [8, 4, 2, 1]
    assert sum(scale == 4 for scale, *_ in reference_trace) == 7 + 4 * 10
    refinement_dofs = [selected for scale, selected, *_ in reference_trace if scale == 2]
    assert refinement_dofs == ([6] if dof == 6 else [7, 9, 12])
    assert [selected for scale, selected, *_ in reference_trace if scale == 1] == [dof, dof]
