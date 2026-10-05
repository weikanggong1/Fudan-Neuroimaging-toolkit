"""Reference selection/intensity contracts; fixtures are not MRI benchmarks."""

import hashlib
import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri.reference import prepare_bold_reference, select_reference_volumes


def image_with_signal(signal, shape=(4, 5, 6)):
    spatial = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 20
    data = spatial[..., None] * np.asarray(signal, dtype=np.float32)
    image = nib.Nifti1Image(data, np.diag([2., 2., 2., 1.]))
    image.header.set_zooms((2., 2., 2., 2.1))
    image.header.set_xyzt_units("mm", "sec")
    return image


def test_no_or_one_initial_outlier_uses_last_twenty_of_first_forty():
    signal = np.linspace(.95, 1.05, 180, dtype=np.float32)
    normal = select_reference_volumes(image_with_signal(signal))
    signal[0] = 10
    one = select_reference_volumes(image_with_signal(signal))
    assert normal.algorithm_dummy_scans == 0
    assert one.algorithm_dummy_scans == 1
    assert normal.selected_indices == one.selected_indices == tuple(range(20, 40))


def test_two_initial_outliers_are_selected_and_manual_skip_does_not_change_mask():
    signal = np.linspace(.95, 1.05, 180, dtype=np.float32)
    signal[:2] = (10, 9)
    automatic = select_reference_volumes(image_with_signal(signal))
    manual = select_reference_volumes(image_with_signal(signal), dummy_scans=0)
    assert automatic.algorithm_dummy_scans == manual.algorithm_dummy_scans == 2
    assert automatic.selected_indices == manual.selected_indices == (0, 1)
    assert automatic.skip_vols == 2 and manual.skip_vols == 0


def test_short_series_window_and_noncontiguous_later_outlier():
    signal = np.linspace(.9, 1.1, 8, dtype=np.float32)
    signal[3] = 30
    result = select_reference_volumes(image_with_signal(signal))
    assert result.algorithm_dummy_scans == 0
    assert result.selected_indices == tuple(range(8))


def test_zero_mad_retains_upstream_nan_inf_stop_semantics():
    # Upstream's score<=3.5 never holds when MAD==0: it does not add an epsilon.
    result = select_reference_volumes(image_with_signal(np.ones(50)))
    assert result.algorithm_dummy_scans == 40
    assert result.selected_indices == tuple(range(40))


def test_clip_drift_and_even_median_match_locked_numpy_control(tmp_path):
    image = image_with_signal([.94, 1.02, .98, 1.06])
    data = np.asarray(image.dataobj).copy()
    data[0, 0, 0, :] = [-10, 100000, 1, 2]
    image = nib.Nifti1Image(data, image.affine, image.header)
    source = tmp_path / "raw.nii.gz"
    nib.save(image, source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    result = prepare_bold_reference(source, tmp_path / "reference", device="cpu",
                                    motion_correction=False, spatial_chunk_size=7)
    # Use the upstream's nibabel selected-volume layout. Spatial float32 mean
    # reductions can differ by ulps if a control invents another stride order.
    loaded = nib.load(source)
    selected = nib.concat_images(
        frame for index, frame in enumerate(nib.four_to_three(loaded))
        if index in result.selected_indices
    ).get_fdata(dtype="float32")
    clipped = np.clip(selected, 0., np.percentile(selected, 99.8))
    drift = clipped.mean(axis=(0, 1, 2))
    drift /= drift.max()
    clipped /= drift
    expected = np.median(np.maximum(clipped, 0), axis=3)
    np.testing.assert_array_equal(nib.load(result.reference).dataobj, expected)
    np.testing.assert_array_equal(nib.load(result.selected_volumes).dataobj, clipped)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert nib.load(source).shape[3] == 4
    header = nib.load(result.reference)
    assert header.shape == data.shape[:3]
    assert header.get_data_dtype() == np.dtype("float32")
    np.testing.assert_array_equal(header.affine, image.affine)
    metadata = json.loads(result.metadata.read_text())
    assert metadata["discarded_input_frames"] == 0
    assert metadata["reference_motion"]["backend"] is None
    assert metadata["upstream"]["motion_backend_equivalence"] == "not_assessed"


def test_single_selected_frame_uses_original_unclipped_values(tmp_path):
    image = image_with_signal([1., 1.02, .98, 1.04])
    values = np.asarray(image.dataobj).copy()
    values[0, 0, 0, 3] = -100
    image = nib.Nifti1Image(values, image.affine, image.header)
    result = prepare_bold_reference(image, tmp_path / "one", device="cpu",
                                    zero_dummy_masked=1, motion_correction=False)
    assert result.selected_indices == (3,)
    np.testing.assert_array_equal(nib.load(result.reference).dataobj, values[..., 3])
    assert np.asarray(nib.load(result.selected_volumes).dataobj).min() >= 0
    assert result.drift == (1.,)


@pytest.mark.parametrize("single_4d", [False, True])
def test_single_input_bypasses_clipping_and_motion(tmp_path, monkeypatch, single_4d):
    import fnit.fmri.reference as module
    monkeypatch.setattr(module, "TorchMCFLIRT", lambda **kw: pytest.fail("single frame used motion"))
    values = np.zeros((3, 4, 5), dtype=np.float32)
    values[0, 0, 0] = -12
    image = nib.Nifti1Image(values[..., None] if single_4d else values, np.eye(4))
    result = prepare_bold_reference(image, tmp_path / "one", device="cpu")
    np.testing.assert_array_equal(nib.load(result.reference).dataobj, values)
    assert result.algorithm_dummy_scans == 1


def test_motion_routes_normalized_selected_series_and_first_selected_target(tmp_path, monkeypatch):
    import fnit.fmri.reference as module
    seen = {}

    class Motion:
        def __init__(self, *, device):
            seen["device"] = str(device)

        def run(self, series, *, reference, **options):
            seen.update(series=series, reference=reference, options=options)
            return SimpleNamespace(corrected=series, matrices=np.repeat(np.eye(4)[None], series.shape[3], axis=0),
                                   parameters=np.zeros((series.shape[3], 6)), cost_evaluations=123)

    monkeypatch.setattr(module, "TorchMCFLIRT", Motion)
    image = image_with_signal(np.linspace(.95, 1.05, 180))
    result = prepare_bold_reference(image, tmp_path / "motion", device="cpu", stage_iterations=(2, 3, 4))
    assert seen["series"].shape[3] == 20
    np.testing.assert_array_equal(seen["reference"].dataobj, seen["series"].dataobj[..., 0])
    assert seen["options"] == {"stages": 3, "stage_iterations": (2, 3, 4), "resample": True,
                               "interpolation": "spline"}
    report = json.loads(result.metadata.read_text())
    assert report["reference_motion"]["backend"] == "fnit.TorchMCFLIRT"
    assert report["reference_motion"]["cost_evaluations"] == 123
    assert len(report["reference_motion"]["matrices"]) == 20


@pytest.mark.parametrize("kwargs", [dict(n_volumes=0), dict(zero_dummy_masked=0),
                                    dict(dummy_scans=-1), dict(dummy_scans=181),
                                    dict(dummy_scans=True), dict(nonnegative="yes")])
def test_selection_rejects_invalid_contracts(kwargs):
    with pytest.raises(ValueError):
        select_reference_volumes(image_with_signal(np.linspace(.9, 1.1, 180)), **kwargs)


def test_zero_signal_and_nonfinite_inputs_fail_without_complete_metadata(tmp_path):
    zero = nib.Nifti1Image(np.zeros((4, 5, 6, 4), dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="zero global signal"):
        prepare_bold_reference(zero, tmp_path / "zero", device="cpu", motion_correction=False)
    assert not (tmp_path / "zero" / "bold_reference.json").exists()
    values = np.asarray(zero.dataobj).copy()
    values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        select_reference_volumes(nib.Nifti1Image(values, zero.affine))


def test_output_collision_requires_explicit_overwrite_and_preserves_other_files(tmp_path):
    image = image_with_signal(np.linspace(.9, 1.1, 6))
    result = prepare_bold_reference(image, tmp_path / "reference", device="cpu", motion_correction=False)
    user_file = result.reference.parent / "user_notes.txt"
    user_file.write_text("keep")
    with pytest.raises(FileExistsError):
        prepare_bold_reference(image, user_file.parent, device="cpu", motion_correction=False)
    rerun = prepare_bold_reference(image, user_file.parent, device="cpu", motion_correction=False, overwrite=True)
    assert rerun.reference.is_file() and user_file.read_text() == "keep"
    with pytest.raises(ValueError, match="overwrite its input"):
        prepare_bold_reference(result.reference, result.reference.parent, device="cpu", overwrite=True)


def test_output_symlink_cannot_redirect_overwrite_to_input(tmp_path):
    image = image_with_signal(np.linspace(.9, 1.1, 6))
    source = tmp_path / "raw.nii.gz"
    nib.save(image, source)
    before = source.read_bytes()
    output = tmp_path / "reference"
    output.mkdir()
    (output / "bold_reference.nii.gz").symlink_to(source)
    with pytest.raises(ValueError, match="symlinks"):
        prepare_bold_reference(source, output, device="cpu", overwrite=True)
    assert source.read_bytes() == before


def test_failed_overwrite_invalidates_former_complete_manifest(tmp_path, monkeypatch):
    import fnit.fmri.reference as module
    image = image_with_signal(np.linspace(.9, 1.1, 6))
    result = prepare_bold_reference(image, tmp_path / "reference", device="cpu", motion_correction=False)
    user_file = result.reference.parent / "user_notes.txt"
    user_file.write_text("keep")

    def failed(*args, **kwargs):
        raise RuntimeError("motion failure")

    monkeypatch.setattr(module, "TorchMCFLIRT", failed)
    with pytest.raises(RuntimeError, match="motion failure"):
        prepare_bold_reference(image, user_file.parent, device="cpu", overwrite=True)
    assert not result.metadata.exists() and user_file.read_text() == "keep"


@pytest.mark.parametrize("output_name", ["bold_reference.nii.gz", "bold_reference_selected.nii.gz"])
def test_file_backed_image_input_cannot_be_overwritten_or_invalidate_prior_metadata(tmp_path, output_name):
    output = tmp_path / "reference"
    output.mkdir()
    source = output / output_name
    nib.save(image_with_signal(np.linspace(.9, 1.1, 6)), source)
    metadata = output / "bold_reference.json"
    metadata.write_text('{"status":"complete","sentinel":"keep"}')
    before = (source.read_bytes(), metadata.read_bytes())
    with pytest.raises(ValueError, match="overwrite its input"):
        prepare_bold_reference(nib.load(source), output, device="cpu", motion_correction=False, overwrite=True)
    assert (source.read_bytes(), metadata.read_bytes()) == before


@pytest.mark.parametrize("as_image", [False, True])
def test_hardlinked_output_cannot_overwrite_input(tmp_path, as_image):
    source = tmp_path / "raw.nii.gz"
    nib.save(image_with_signal(np.linspace(.9, 1.1, 6)), source)
    output = tmp_path / "reference"
    output.mkdir()
    alias = output / "bold_reference.nii.gz"
    alias.hardlink_to(source)
    metadata = output / "bold_reference.json"
    metadata.write_text('{"status":"complete","sentinel":"keep"}')
    before = (source.read_bytes(), alias.read_bytes(), metadata.read_bytes())
    with pytest.raises(ValueError, match="overwrite its input"):
        prepare_bold_reference(nib.load(source) if as_image else source, output,
                               device="cpu", motion_correction=False, overwrite=True)
    assert (source.read_bytes(), alias.read_bytes(), metadata.read_bytes()) == before


def test_real_mature_motion_on_small_asymmetric_fixture(tmp_path):
    # Exercise the actual optimizer/spline route, rather than a mocked algorithm.
    axes = np.indices((12, 12, 12), dtype=np.float32)
    base = np.exp(-((axes[0] - 5)**2 / 8 + (axes[1] - 6)**2 / 12 + (axes[2] - 5)**2 / 16)) * 1000
    signal = np.stack((base, base * 1.01, base * .99), axis=3).astype(np.float32)
    image = nib.Nifti1Image(signal, np.diag([2., 2., 2., 1.]))
    result = prepare_bold_reference(image, tmp_path / "actual_motion", device="cpu")
    produced = np.asarray(nib.load(result.reference).dataobj)
    assert produced.shape == base.shape and np.isfinite(produced).all()
    assert produced.min() >= 0
    report = json.loads(result.metadata.read_text())
    assert report["reference_motion"]["cost_evaluations"] > 0
    assert len(report["reference_motion"]["matrices"]) == 3
