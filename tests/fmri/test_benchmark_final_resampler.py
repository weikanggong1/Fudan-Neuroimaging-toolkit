"""Benchmark hooks follow the active pipeline entry point without changing calls."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "final_resampler_benchmark", root / "validation/fmri/benchmark_combined_e2e.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.mark.parametrize("entry_point", ["resample_world", "_resample_final_volume"])
def test_observer_measures_all_outputs_at_actual_owner_and_preserves_calls(tmp_path, entry_point):
    captured = []
    result = object()

    def sampler(*args, **kwargs):
        captured.append((args, kwargs))
        return result

    def unused_legacy(*args, **kwargs):
        raise AssertionError("The migrated pipeline must not observe its legacy alias")

    owner = SimpleNamespace(**{entry_point: sampler})
    if entry_point == "_resample_final_volume":
        owner.resample_world = unused_legacy
    observer = driver.Observer(tmp_path)
    observer.observe_final_volume_resampling(owner)
    arguments = (object(), object(), object(), object())
    options = [
        {"backend": "fnirt", "interpolation": "nearest"},
        {"backend": "fnirt", "output_mask": object(), "boundary": "periodic"},
        {"backend": "synthmorph", "motion_pull_world": object(), "coordinate_precision": "fmriprep"},
        {"backend": "fnirt", "motion_pull_world": object(), "pre_affine_pull_ras": object()},
    ]
    labels = ("clean_mni_mask_resampling", "clean_mni_resampling",
              "preproc_t1w_resampling", "preproc_mni_resampling")
    for kwargs in options:
        assert getattr(owner, entry_point)(*arguments, **kwargs) is result
    assert len(captured) == 4
    for (actual_args, actual_kwargs), expected in zip(captured, options):
        assert all(actual is original for actual, original in zip(actual_args, arguments))
        assert actual_kwargs == expected
    assert set(observer.calls) == set(labels)
    assert all(observer.calls[label]["calls"] == 1 for label in labels)
    assert all(observer.calls[label]["capture_seconds"] == 0 for label in labels)
    observer.restore()
    assert getattr(owner, entry_point) is sampler
    if entry_point == "_resample_final_volume":
        assert owner.resample_world is unused_legacy


def test_source_inventory_detects_shared_world_sampler_changes(tmp_path):
    source = tmp_path / "src/fnit/_world_resampling.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen shared sampler version1\n")
    first = driver.source_hashes(tmp_path)
    relative = "src/fnit/_world_resampling.py"
    assert first[relative] == hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(b"changed shared sampler version2\n")
    second = driver.source_hashes(tmp_path)
    assert first[relative] != second[relative]


def test_missing_or_invalid_final_owner_is_not_silently_ignored():
    with pytest.raises(AttributeError, match="no final resampling"):
        driver._helpers.final_volume_resampler_name(SimpleNamespace())
    with pytest.raises(TypeError, match="not callable"):
        driver._helpers.final_volume_resampler_name(
            SimpleNamespace(_resample_final_volume=None, resample_world=lambda: None))
