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
