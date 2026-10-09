"""CPU controls for the exact comparator and timing tool, not image benchmarks."""

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def benchmark():
    source = (Path(__file__).resolve().parents[2] /
              "tools/benchmark_connectome_tracking_exact.py")
    name = "_tracking_exact_benchmark_cpu_test"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(benchmark, *, empty=False, mean_fa=None):
    points = (torch.empty((0, 3), dtype=torch.float32) if empty else
              torch.tensor([[0., 1., 2.], [3., 4., 5.]], dtype=torch.float32))
    return benchmark.Snapshot(
        packed_points=points,
        paths=() if empty else (points,),
        point_counts=torch.empty(0, dtype=torch.int64) if empty else torch.tensor([2]),
        endpoints=(torch.empty((0, 2, 3)) if empty else
                   torch.stack((points[0], points[-1]))[None]),
        lengths_mm=torch.empty(0) if empty else torch.tensor([5.]),
        accepted_seeds=torch.empty((0, 3)) if empty else points[:1].clone(),
        mean_fa=mean_fa,
        seeds_attempted=3,
    )


def test_equal_values_with_different_dtype_are_rejected(benchmark):
    reference = _snapshot(benchmark, mean_fa=torch.tensor([.5]))
    for field in ("packed_points", "point_counts", "endpoints", "lengths_mm",
                  "accepted_seeds", "mean_fa"):
        value = getattr(reference, field)
        dtype = torch.int32 if value.dtype == torch.int64 else torch.float64
        actual = replace(reference, **{field: value.to(dtype)})
        comparison = benchmark.strict_compare(reference, actual)
        assert comparison["torch_equal"][field]
        assert not comparison["dtype_shape_bytes_equal"][field]
        assert not comparison["all_equal"]
    points = reference.packed_points.double()
    comparison = benchmark.strict_compare(
        reference, replace(reference, packed_points=points, paths=(points,)))
    assert comparison["torch_equal"]["paths"]
    assert not comparison["dtype_shape_bytes_equal"]["paths"]
    assert not comparison["all_equal"]


def test_signed_zero_in_a_path_is_rejected_despite_equal_values(benchmark):
    reference = _snapshot(benchmark)
    points = reference.packed_points.clone()
    points[0, 0] = -0.
    actual = replace(reference, packed_points=points, paths=(points,))
    comparison = benchmark.strict_compare(reference, actual)
    assert comparison["torch_equal"]["paths"]
    assert not comparison["dtype_shape_bytes_equal"]["paths"]
    assert comparison["first_mismatched_raw_path_indices"] == [0]
    assert not comparison["all_equal"]


def test_empty_snapshots_are_equal_and_hashable(benchmark):
    for mean_fa in (None, torch.empty(0)):
        reference = _snapshot(benchmark, empty=True, mean_fa=mean_fa)
        actual = _snapshot(benchmark, empty=True, mean_fa=mean_fa)
        assert benchmark.strict_compare(reference, actual)["all_equal"]
        summary = benchmark.output_summary(actual)
        assert summary["accepted_streamlines"] == summary["total_path_points"] == 0
        assert len(summary["output_sha256"]) == 64
        assert summary["streamlines"] == []


def test_path_point_counts_and_seed_mismatches_are_rejected(benchmark):
    reference = _snapshot(benchmark)
    shorter = reference.packed_points[:1].clone()
    actual = replace(reference, packed_points=shorter, paths=(shorter,),
                     point_counts=torch.tensor([1]))
    comparison = benchmark.strict_compare(reference, actual)
    assert comparison["path_count_equal"]
    assert not comparison["torch_equal"]["point_counts"]
    assert not comparison["all_equal"]
    assert not benchmark.strict_compare(reference, replace(
        reference, accepted_seeds=reference.accepted_seeds + 1))["all_equal"]
    assert not benchmark.strict_compare(reference, replace(
        reference, seeds_attempted=reference.seeds_attempted + 1))["all_equal"]


def test_optional_fa_presence_must_match(benchmark):
    reference = _snapshot(benchmark)
    actual = replace(reference, mean_fa=torch.tensor([.5]))
    for expected, observed in ((reference, actual), (actual, reference)):
        comparison = benchmark.strict_compare(expected, observed)
        assert not comparison["torch_equal"]["mean_fa"]
        assert not comparison["dtype_shape_bytes_equal"]["mean_fa"]
        assert not comparison["all_equal"]


def test_kernel_summary_excludes_annotations_ops_and_memory_events(benchmark):
    cuda = torch.autograd.DeviceType.CUDA
    cpu = torch.autograd.DeviceType.CPU

    def event(name, device_type, duration_us, annotation=False):
        return SimpleNamespace(name=name, key=name, count=1, device_type=device_type,
                               device_time_total=duration_us,
                               self_device_time_total=duration_us,
                               cpu_time_total=0., self_cpu_time_total=0.,
                               is_user_annotation=annotation)

    events = [event("kernel_a", cuda, 1000.), event("kernel_b", cuda, 2000.),
              event("aten::op", cpu, 3000.),
              event("tracking/candidate", cuda, 3000., annotation=True),
              event("Memcpy DtoH (Device -> Pageable)", cuda, 4000.),
              event("Memset (Device)", cuda, 5000.),
              event("[CUDA memcpy HtoD]", cuda, 6000.),
              event("[CUDA memset]", cuda, 7000.)]
    profiler = SimpleNamespace(events=lambda: events, key_averages=lambda: events)
    summary = benchmark.profile_summary(profiler)
    assert summary["kernel_count"] == 2
    assert summary["kernel_device_seconds"] == pytest.approx(.003)
    top = {row["name"]: row for row in benchmark.profile_top_events(profiler)}
    assert top["tracking/candidate"]["is_user_annotation"]
    assert top["kernel_a"]["device_type"] == str(cuda)
    assert top["aten::op"]["device_type"] == str(cpu)


def test_cpu_run_once_records_scalar_timings_and_no_gpu(benchmark, monkeypatch, tmp_path):
    reference = _snapshot(benchmark)
    tracks = SimpleNamespace(paths=reference.paths, endpoints=reference.endpoints,
                             lengths_mm=reference.lengths_mm, mean_fa=None,
                             seeds_attempted=3, accepted_seeds=reference.accepted_seeds)
    calls = []

    def tracking(*, sentinel, n_seeds):
        assert torch.is_inference_mode_enabled()
        calls.append((sentinel, n_seeds))
        return tracks

    monkeypatch.setattr(benchmark.os, "getloadavg", lambda: (1., 2., 3.), raising=False)
    trace = tmp_path / "cpu.chrome.json"
    record, snapshot = benchmark.run_once(
        SimpleNamespace(probabilistic_tractography=tracking), "candidate", "profile", 0,
        {"sentinel": "unit-control"}, {"n_seeds": 3}, torch.device("cpu"), trace_path=trace)
    assert calls == [("unit-control", 3)]
    assert benchmark.strict_compare(reference, snapshot)["all_equal"]
    for key in ("tracking_seconds", "tracking_process_cpu_seconds", "path_pack_seconds",
                "d2h_seconds", "profile_export_seconds"):
        assert isinstance(record[key], float) and math.isfinite(record[key]) and record[key] >= 0
    assert record["gpu_before"] is record["gpu_after"] is None
    assert record["system_load_before"] == record["system_load_after"] == [1., 2., 3.]
    assert all(value is None for value in record["tracking_memory"].values())
    assert record["profile_summary"]["kernel_count"] == 0
    assert record["profile_summary"]["kernel_device_seconds"] == 0
    assert trace.is_file()
    json.dumps(record)


def _source_pair(tmp_path):
    tracking = "from .fod import tracking_sh_precomputed, real_sh\n" \
               "def probabilistic_tractography():\n    return real_sh()\n"
    paths = {}
    for variant in ("baseline", "candidate"):
        directory = tmp_path / variant
        directory.mkdir()
        paths[variant] = directory / "tracking.py"
        paths[variant].write_text(tracking)
        paths[variant + "_fod"] = directory / "fod.py"
        paths[variant + "_fod"].write_text(
            f"def real_sh():\n    return {variant!r}\n"
            f"def tracking_sh_precomputed():\n    return {variant!r}\n")
    return paths


def test_paired_fod_sources_bind_only_to_their_own_functions(benchmark, tmp_path):
    paths = _source_pair(tmp_path)
    modules, fods = benchmark.load_source_pairs(
        paths["baseline"], paths["candidate"],
        baseline_fod_path=paths["baseline_fod"],
        candidate_fod_path=paths["candidate_fod"])
    assert fods["baseline"] is not fods["candidate"]
    assert modules["baseline"].__package__ != modules["candidate"].__package__
    assert benchmark.file_sha256(paths["baseline_fod"]) != benchmark.file_sha256(paths["candidate_fod"])
    for variant in ("baseline", "candidate"):
        assert modules[variant].probabilistic_tractography() == variant
        for name in ("real_sh", "tracking_sh_precomputed"):
            assert getattr(modules[variant], name) is getattr(fods[variant], name)
    identity = benchmark.fod_source_identity(modules, fods)
    assert identity["fod_mode"] == "paired"
    assert not identity["fixed_fod_function_identity_equal"]
    assert not any(identity["cross_variant_fod_function_identity"].values())
    assert all(all(values.values()) for values in identity["fod_function_identity"].values())


def test_default_shared_fod_preserves_existing_loader_and_identity(benchmark, tmp_path):
    paths = _source_pair(tmp_path)
    for loader in (benchmark.load_sources, benchmark.load_source_pairs):
        result = loader(paths["baseline"], paths["candidate"], paths["candidate_fod"])
        modules = result[0]
        fods = (result[1] if isinstance(result[1], dict) else
                {"baseline": result[1], "candidate": result[1]})
        assert fods["baseline"] is fods["candidate"]
        assert modules["baseline"].probabilistic_tractography() == "candidate"
        assert modules["candidate"].probabilistic_tractography() == "candidate"
        identity = benchmark.fod_source_identity(modules, fods)
        assert identity["fod_mode"] == "shared"
        assert identity["fixed_fod_function_identity_equal"]
        assert all(identity["cross_variant_fod_function_identity"].values())
    # Omitting the source still selects candidate's neighbouring fod.py.
    modules, fods = benchmark.load_source_pairs(paths["baseline"], paths["candidate"])
    assert Path(fods["baseline"].__file__) == paths["candidate_fod"]
    assert benchmark.fod_source_identity(modules, fods)["fixed_fod_function_identity_equal"]


def test_paired_loader_rejects_tracking_that_overrides_its_fod_function(benchmark, tmp_path):
    paths = _source_pair(tmp_path)
    with paths["baseline"].open("a") as stream:
        stream.write("def real_sh():\n    return 'wrong binding'\n")
    with pytest.raises(ValueError, match="baseline did not bind its FOD function real_sh"):
        benchmark.load_source_pairs(
            paths["baseline"], paths["candidate"],
            baseline_fod_path=paths["baseline_fod"],
            candidate_fod_path=paths["candidate_fod"])


@pytest.mark.parametrize("extra", [
    ["--baseline-fod-module", "baseline_fod"],
    ["--candidate-fod-module", "candidate_fod"],
    ["--fod-module", "candidate_fod", "--baseline-fod-module", "baseline_fod",
     "--candidate-fod-module", "candidate_fod"],
])
def test_cli_rejects_unpaired_or_conflicting_fod_sources(benchmark, tmp_path, capsys, extra):
    paths = _source_pair(tmp_path)
    arguments = ["--baseline-tracking", str(paths["baseline"]),
                 "--candidate-tracking", str(paths["candidate"]),
                 "--fod", str(paths["baseline"]), "--five-tissue", str(paths["baseline"]),
                 "--gmwmi", str(paths["baseline"]), "--output", str(tmp_path / "report.json")]
    arguments += [str(paths[value]) if value in paths else value for value in extra]
    with pytest.raises(SystemExit) as failure:
        benchmark.parse_args(arguments)
    assert failure.value.code == 2
    error = capsys.readouterr().err
    assert "must be supplied together" in error or "cannot be combined" in error


@pytest.mark.parametrize("mode", ["default", "shared", "paired"])
def test_cli_resolves_only_the_selected_fod_sources(benchmark, tmp_path, mode):
    paths = _source_pair(tmp_path)
    arguments = ["--baseline-tracking", str(paths["baseline"]),
                 "--candidate-tracking", str(paths["candidate"]),
                 "--fod", str(paths["baseline"]), "--five-tissue", str(paths["baseline"]),
                 "--gmwmi", str(paths["baseline"]), "--output", str(tmp_path / "report.json")]
    if mode == "shared":
        arguments += ["--fod-module", str(paths["baseline_fod"])]
    elif mode == "paired":
        arguments += ["--baseline-fod-module", str(paths["baseline_fod"]),
                      "--candidate-fod-module", str(paths["candidate_fod"])]
    args = benchmark.parse_args(arguments)
    if mode == "paired":
        assert args.fod_module is None
        assert args.baseline_fod_module == paths["baseline_fod"].resolve()
        assert args.candidate_fod_module == paths["candidate_fod"].resolve()
    else:
        expected = paths["baseline_fod"] if mode == "shared" else paths["candidate_fod"]
        assert args.fod_module == expected.resolve()
        assert args.baseline_fod_module is args.candidate_fod_module is None
