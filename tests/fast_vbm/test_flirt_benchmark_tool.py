"""Regression tests for profiling hooks; synthetic arrays are unit fixtures only."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.flirt import core


@pytest.fixture
def benchmark_tool():
    path = Path(__file__).resolve().parents[2] / "tools/benchmark_flirt_gpu.py"
    specification = importlib.util.spec_from_file_location("flirt_benchmark_tool", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def resample_arguments(device):
    data = np.arange(12 * 13 * 14, dtype=np.float32).reshape(12, 13, 14)
    sampling = np.diag([2., 2., 2., 1.])
    return data, data.shape, sampling, sampling, np.eye(4), (2, 2, 2), (2, 2, 2), torch.device(device)


def test_instrumentation_preserves_resampling_callable(benchmark_tool, tmp_path):
    arguments = resample_arguments("cpu")
    expected = core._resample_output(*arguments)
    # Installing cost hooks used to overwrite the resampling closure's callable.
    with benchmark_tool.Instrumentation(core, torch, tmp_path) as instrumentation:
        actual = core._resample_output(*arguments)
    assert torch.equal(actual, expected)
    assert instrumentation.report()["stages"]["final_resampling"]["calls"] == 1


def test_paired_matrix_score_uses_header_pixdim_for_both_artifacts(benchmark_tool, tmp_path):
    shape = (7, 8, 9)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    moving_affine = np.array([[-2., .5, 0, 0], [0, 2, 0, 0],
                              [0, 0, 2, 0], [0, 0, 0, 1]])
    fixed_affine = np.diag([-2., 2., 2., 1.])
    moving = nib.Nifti1Image(data, moving_affine)
    fixed = nib.Nifti1Image(data, fixed_affine)
    moving.header.set_zooms((2, 2, 2))
    matrix_path, moved_path = tmp_path / "fsl.mat", tmp_path / "fsl.nii.gz"
    np.savetxt(matrix_path, np.eye(4))
    nib.save(fixed, moved_path)
    # Identical saved scaled-mm matrices must have zero displacement. With
    # equal header pixdim, this matrix maps identical voxel indices. Do not
    # trust legacy result.world fields built with affine norms instead.
    result = SimpleNamespace(
        matrix=np.eye(4), moved=fixed, moving_to_fixed_world=np.eye(4),
        qc={"cost_value": 0., "cost_evaluations": 1},
    )
    metrics = benchmark_tool.paired_metrics(
        result, moving, fixed, matrix_path, moved_path, None, core,
    )
    assert metrics["world_grid_displacement_mm"]["rms"] == 0


def test_stage_timers_cover_complete_registration(benchmark_tool):
    arguments = resample_arguments("cpu")
    image = nib.Nifti1Image(arguments[0], arguments[2])
    with benchmark_tool.StageTimings(core, 12) as timers:
        result = core.TorchFLIRT(device="cpu", angular_search=False)(image, image)
    stages = timers.report()["stages"]
    assert result.moved.shape == image.shape
    assert stages["angular_search"]["cost_evaluations"] > 0
    assert stages["final_resampling"]["calls"] == 1
    assert any(name.startswith("local_optimization/") for name in stages)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA profiler regression")
def test_cuda_full_profile_checkpoints_and_resamples(benchmark_tool, tmp_path):
    arguments = resample_arguments("cuda:0")
    values = torch.as_tensor(arguments[0], device="cuda:0")
    cost = core.FSLCorrelationRatio(values, values, (2, 2, 2), (2, 2, 2), bins=16, smooth_size=2)
    expected = core._resample_output(*arguments).cpu()
    with benchmark_tool.Instrumentation(core, torch, tmp_path, profile_full=True) as instrumentation:
        cost(np.eye(4))
        actual = core._resample_output(*arguments).cpu()
    assert torch.equal(actual, expected)
    checkpoint = json.loads((tmp_path / "profile.progress.private.json").read_text())
    assert checkpoint["full_profile_event_windows"] == 1
    assert checkpoint["kernel_launch_runtime_api_count"] > 0
    assert sum(instrumentation.report()["cost_evaluations_by_type_and_scale"].values()) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA profiler activity regression")
def test_cuda_only_activity_keeps_runtime_counters():
    value = torch.ones((50, 50), device="cuda:0")
    (value @ value).sum().item()  # warm the same CUDA operations
    counters = []
    for activities in ([torch.profiler.ProfilerActivity.CUDA],
                       [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]):
        with torch.profiler.profile(activities=activities) as profiler:
            (value @ value).sum().item()
        counters.append({name: sum(event.name == name for event in profiler.events())
                         for name in ("cudaLaunchKernel", "cudaMemcpyAsync", "cudaStreamSynchronize")})
    assert counters[0] == counters[1]
    assert counters[0]["cudaLaunchKernel"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA profile-only artifacts")
def test_profile_only_saves_result_for_exact_gate(benchmark_tool, tmp_path, monkeypatch):
    image = nib.Nifti1Image(resample_arguments("cpu")[0], np.eye(4))
    input_path = tmp_path / "input.nii.gz"
    matrix_path = tmp_path / "identity.mat"
    nib.save(image, input_path)
    np.savetxt(matrix_path, np.eye(4))
    fixture_result = core.FLIRTResult(image, np.eye(4), np.eye(4), np.eye(4),
                                      {"cost_value": 0., "cost_evaluations": 1})

    class ProfileFixture(core.TorchFLIRT):
        def __call__(self, *args, **kwargs):
            # Exercise CUDA recording; registration numerics have separate tests.
            torch.ones(8, device=self.device).sum().cpu()
            return fixture_result

    monkeypatch.setattr(core, "TorchFLIRT", ProfileFixture)
    arguments = SimpleNamespace(
        source_root=str(Path(core.__file__).resolve().parents[3]),
        moving=str(input_path), reference=str(input_path), fsl_matrix=str(matrix_path),
        fsl_moved=str(input_path), mask=None, init=None, output_dir=str(tmp_path / "profile"),
        device="cuda:0", dof=12, cost="corratio", execution="default",
        candidate_batch_size=None, threads=1, warm_repeats=0, profile_only=True,
        profile_costs=0, profile_full=True, official_command=None, official_wall_seconds=None,
    )
    benchmark_tool.worker(arguments)
    output = Path(arguments.output_dir)
    assert np.array_equal(np.loadtxt(output / "profiled.mat"), fixture_result.matrix)
    assert np.array_equal(np.asarray(nib.load(output / "profiled.nii.gz").dataobj),
                          np.asarray(image.dataobj))
    assert json.loads((output / "report.private.json").read_text())["runs"] == []
