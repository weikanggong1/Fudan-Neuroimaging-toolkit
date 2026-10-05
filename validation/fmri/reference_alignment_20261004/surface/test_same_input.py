"""Constructed contracts only; these tests are not MRI benchmark evidence."""
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("surface_same_input", Path(__file__).with_name("run_same_input.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_input_drift_is_rejected(tmp_path):
    image = tmp_path / "input"
    image.write_bytes(b"original")
    specs = {"t1w_bold": {"path": str(image), "sha256": driver.sha256(image)}}
    assert driver.verify_specs(specs)["t1w_bold"] == specs["t1w_bold"]["sha256"]
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Input SHA-256 mismatch: t1w_bold"):
        driver.verify_specs(specs)


def test_fresh_output_cannot_contain_original_inputs(tmp_path):
    input_file = tmp_path / "original" / "bold.nii"
    input_file.parent.mkdir()
    input_file.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        driver.require_fresh_output(input_file.parent, [input_file])
    with pytest.raises(ValueError, match="protected root"):
        driver.require_fresh_output(tmp_path / "original" / "new", [], [input_file.parent])
    with pytest.raises(ValueError, match="protected input"):
        driver.require_fresh_output(tmp_path / "absent", [tmp_path / "absent" / "bold.nii"])
    assert driver.require_fresh_output(tmp_path / "fresh", [input_file]) == tmp_path / "fresh"


@pytest.mark.parametrize("frames,tr", [(179, 2.1), (180, 2.4)])
def test_partial_or_changed_time_axis_is_rejected(tmp_path, frames, tr):
    image = nib.Nifti1Image(np.ones((2, 2, 2, frames), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("mm", "sec")
    image.header.set_zooms((1, 1, 1, tr))
    path = tmp_path / "bold.nii"
    nib.save(image, path)
    with pytest.raises(ValueError, match="frame axis|original TR"):
        driver.validate_time_axes({"expected_frames": 180, "tr_seconds": 2.1},
                                  {k: str(path) for k in ("raw_bold", "t1w_bold", "mni_bold")})


def test_reference_cannot_enable_gpu_or_use_unbound_container():
    manifest = {"reference": {"container": {"path": "/tmp/reference.sif"},
                "container_prefix": ["singularity", "exec", "--cleanenv", "--nv", "/tmp/reference.sif"]}}
    with pytest.raises(ValueError, match="CPU-only"):
        driver.reference_prefix(manifest, 4)
    manifest["reference"]["container_prefix"] = ["singularity", "exec", "--cleanenv", "/tmp/other.sif"]
    with pytest.raises(ValueError, match="SHA-bound"):
        driver.reference_prefix(manifest, 4)


def test_difference_reports_changed_value_and_enforces_shape_and_finite():
    candidate = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    reference = candidate.copy()
    reference[1, 1] += 0.25
    result = driver.array_difference(candidate, reference)
    assert result["different_values"] == 1
    assert result["maximum_absolute_error"] == .25
    assert not result["within_declared_absolute_tolerance"]
    with pytest.raises(ValueError, match="shapes"):
        driver.array_difference(candidate, reference[:, :2])
    reference[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        driver.array_difference(candidate, reference)


def test_axis_mismatch_rejected_even_if_signal_equal(tmp_path):
    cortex = nib.cifti2.cifti2_axes.BrainModelAxis.from_surface([0, 1], 3, name="CORTEX_LEFT")
    a = tmp_path / "a.dtseries.nii"
    b = tmp_path / "b.dtseries.nii"
    for path, tr in ((a, 2.1), (b, 2.4)):
        time = nib.cifti2.cifti2_axes.SeriesAxis(0, tr, 2)
        nib.save(nib.Cifti2Image(np.ones((2, 2), dtype=np.float32),
                 header=nib.Cifti2Header.from_axes((time, cortex))), path)
    with pytest.raises(ValueError, match="CIFTI axes"):
        driver.compare_files(a, b)


def test_generic_loader_compares_gifti_without_nifti_only_options(tmp_path):
    paths = [tmp_path / name for name in ("FNIT.func.gii", "reference.func.gii")]
    for path in paths:
        nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(np.asarray([1, 2, 3], dtype=np.float32))]), path)
    difference = driver.compare_files(*paths)
    assert difference["shape"] == [1, 3]
    assert difference["different_values"] == 0
    assert difference["maximum_absolute_error"] == 0
