"""Reference-driver contracts; these small cases are not a benchmark."""

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "official_mmorf_driver", Path(__file__).with_name("run_official_mmorf.py")
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def templates(directory, affine):
    paths = [directory / name for name in ("FA.nii.gz", "T1.nii.gz", "tensor.nii.gz")]
    for path, frames in zip(paths, ((), (), (6,))):
        nib.save(nib.Nifti1Image(np.ones((3, 4, 5) + frames, np.float32), affine), path)
    return paths


def test_config_has_single_T1_scalar_and_matches_current_five_levels(tmp_path):
    from fnit.mmorf import MMORFConfig

    config = MMORFConfig()
    output = tmp_path / "single_T1_tensor.ini"
    driver.write_config(output, t1_brain=tmp_path / "T1.nii.gz",
                        native_tensor=tmp_path / "tensor.nii.gz",
                        t1_reference=tmp_path / "T1_template.nii.gz",
                        tensor_reference=tmp_path / "tensor_template.nii.gz",
                        t1_matrix=tmp_path / "T1.mat", tensor_matrix=tmp_path / "FA.mat",
                        identity=tmp_path / "identity.mat", output_dir=tmp_path)
    entries = [line.split("=", 1) for line in output.read_text().splitlines()]
    assert sum(key.strip() == "img_mov_scalar" for key, _ in entries) == 1
    values = {key.strip(): value.strip() for key, value in entries}
    scales = [int(value) for value in values["warp_scaling"].split()]
    resolution = float(values["warp_res_init"])
    resolutions = []
    for scale in scales:
        resolution /= scale
        resolutions.append(resolution)
    assert tuple(resolutions) == config.warp_resolution_mm
    assert tuple(map(float, values["fwhm_mov_scalar"].split())) == config.smoothing_mm
    assert tuple(map(float, values["lambda_reg"].split())) == config.regularization
    assert tuple(map(float, values["lambda_scalar"].split())) == (1,) * 5
    assert tuple(map(float, values["lambda_tensor"].split())) == (1,) * 5
    assert tuple(config.iterations) == (int(values["optimiser_max_it_lowres"]),) * 5


def test_radiological_template_grid_accepts_and_records_json(tmp_path):
    paths = templates(tmp_path, np.diag([-2.0, 2.0, 2.0, 1.0]))
    record = driver.require_template_contract(*paths)
    assert record["reference_affine_determinant"] < 0
    assert record["orthogonal_reference_grid"]
    json.dumps(record, allow_nan=False)
    json.dumps(driver.image_record(paths[0]), allow_nan=False)


def test_positive_determinant_reference_requires_conversion(tmp_path):
    paths = templates(tmp_path, np.diag([2.0, 2.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="radiological"):
        driver.require_template_contract(*paths)


def test_sheared_reference_is_rejected_before_native_execution(tmp_path):
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    affine[0, 1] = 0.2
    paths = templates(tmp_path, affine)
    with pytest.raises(ValueError, match="sheared"):
        driver.require_template_contract(*paths)


def test_templates_with_different_grids_are_rejected(tmp_path):
    paths = templates(tmp_path, np.diag([-2.0, 2.0, 2.0, 1.0]))
    image = nib.load(paths[0])
    affine = image.affine.copy()
    affine[0, 3] += 1
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj), affine), paths[0])
    with pytest.raises(ValueError, match="share one grid"):
        driver.require_template_contract(*paths)


def test_retry_guard_only_allows_diagnosed_cuda_constructor_failure(tmp_path):
    error = "terminate after thrust::system::detail::bad_alloc cudaErrorMemoryAllocation"
    assert driver.startup_retry_allowed(error, -6, tmp_path)
    assert not driver.startup_retry_allowed(error, -15, tmp_path)
    assert not driver.startup_retry_allowed("unrelated exception", -6, tmp_path)
    for marker in driver.REGISTRATION_STARTED_MARKERS:
        assert not driver.startup_retry_allowed(error + marker, -6, tmp_path)
    (tmp_path / "mmorf_warp.nii.gz").write_bytes(b"saved output must prevent restart")
    assert not driver.startup_retry_allowed(error, -6, tmp_path)


def test_retry_is_bounded_and_retains_all_attempts(tmp_path, monkeypatch):
    runner = driver.Runner(tmp_path, {}, {})
    calls = []

    def fail(arguments, *, name, monitor_cuda):
        calls.append(name)
        (tmp_path / (name + ".log")).write_text(
            "thrust::system::detail::bad_alloc cudaErrorMemoryAllocation")
        runner.commands.append({"name": name, "exit_code": -6})
        raise RuntimeError("constructor failed")

    monkeypatch.setattr(runner, "run", fail)
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError):
        runner.run_mmorf(["original_binary", "--config", "unchanged.ini"])
    assert len(calls) == len(runner.commands) == 3
    assert runner.report["startup_retry_policy"]["all_attempts_included_in_processing_time"]
    assert runner.commands[-1]["startup_retry"]["retry_allowed"] is False
