"""Regression and modality contracts derived from HCP's MATLAB/shell operators."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.msm import features


def _demean(array, axis):
    return array - np.mean(array, axis=axis, keepdims=True)


def _nodes(data, maps, weight=None):
    if weight is None:
        design, observations = _demean(maps, 0), _demean(data, 0)
    else:
        root = np.sqrt(weight)[:, None]
        design, observations = _demean(maps * root, 0), _demean(data, 0) * root
    return _demean((np.linalg.pinv(design) @ observations).T, 0)


def _beta(data, nodes):
    return (np.linalg.pinv(nodes) @ _demean(data.T, 0)).T


def _axis(count):
    return nib.cifti2.BrainModelAxis.from_surface(np.arange(count), count, name="CortexLeft")


def _cifti(path, values, axis, *, timeseries=False):
    first = (nib.cifti2.SeriesAxis(0, 0.8, values.shape[1]) if timeseries else
             nib.cifti2.ScalarAxis([f"component {i + 1}" for i in range(values.shape[1])]))
    nib.save(nib.Cifti2Image(values.T.astype(np.float32),
                           nib.Cifti2Header.from_axes((first, axis))), path)
    return path


def test_cortical_area_axis_roundtrip_preserves_vertex_order(tmp_path):
    cortex = _axis(5)
    volume = nib.cifti2.BrainModelAxis.from_mask(np.ones((2, 2, 2), dtype=bool),
                                               affine=np.eye(4), name="ThalamusLeft")
    cortical_slice = (cortex + volume)[:5]
    path = _cifti(tmp_path / "area.dscalar.nii", np.ones((5, 1)), cortical_slice)
    serialized = nib.load(path).header.get_axis(1)
    assert cortical_slice.volume_shape is None and serialized.volume_shape is None
    assert features._same_axis(cortical_slice, serialized)
    reversed_axis = nib.cifti2.BrainModelAxis.from_surface(np.arange(4, -1, -1), 5, name="CortexLeft")
    assert not features._same_axis(cortical_slice, reversed_axis)


def test_voxel_axis_still_requires_matching_affine_and_grid():
    mask = np.ones((2, 2, 2), dtype=bool)
    first = nib.cifti2.BrainModelAxis.from_mask(mask, affine=np.eye(4), name="ThalamusLeft")
    affine = np.eye(4); affine[0, 3] = 2
    shifted = nib.cifti2.BrainModelAxis.from_mask(mask, affine=affine, name="ThalamusLeft")
    assert not features._same_axis(first, shifted)


def test_wrn_constant_bold_is_rejected_before_smoothing(tmp_path):
    axis = _axis(5)
    data = np.arange(30).reshape(5, 6).astype(np.float64)
    data[1] = 0
    bold = _cifti(tmp_path / "bold.dtseries.nii", data, axis, timeseries=True)
    reference = _cifti(tmp_path / "maps.dscalar.nii", np.arange(20).reshape(5, 4), axis)
    vn = _cifti(tmp_path / "vn.dscalar.nii", np.ones((5, 1)), axis)
    with pytest.raises(ValueError, match="1 constant BOLD time series"):
        features.run_msmall_regression(bold, reference, tmp_path / "output",
                                       variance_normalization=vn, method="WRN", device="cpu")


@pytest.mark.parametrize("timeseries,frames", [(False, 3), (True, 1)])
def test_clean_bold_requires_series_axis_and_multiple_frames(tmp_path, timeseries, frames):
    bad_bold = _cifti(tmp_path / "bad.nii", np.ones((5, frames)), _axis(5), timeseries=timeseries)
    with pytest.raises(ValueError, match="SeriesAxis and at least two timepoints"):
        features._bold_cifti(bad_bold)


def test_dr_is_two_distinct_demean_regressions():
    rng = np.random.default_rng(421)
    maps = rng.normal(size=(45, 6)) + 7
    data = maps @ rng.normal(size=(6, 29)) + rng.normal(size=(45, 29))
    expected_nodes = _nodes(data, maps)
    expected_maps = _beta(data, expected_nodes)
    actual_maps, actual_nodes, weights = features._regression(
        torch.as_tensor(data), torch.as_tensor(maps), method="DR")
    np.testing.assert_allclose(actual_nodes, expected_nodes, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual_maps, expected_maps, rtol=2e-12, atol=2e-12)
    assert weights is None


def test_wrn_preserves_source_weighting_and_leading_zero_columns():
    rng = np.random.default_rng(42)
    basis = rng.normal(size=(60, 21))
    data = basis @ rng.normal(size=(21, 64)) + rng.normal(size=(60, 64)) * 0.1
    reference = basis[:, :8]
    low_maps = [basis[:, :dimension] for dimension in range(7, 22)]
    area = np.linspace(0.8, 1.2, 53)
    full_area = np.r_[area, np.ones(7)]
    corrs = np.zeros((60, 18))
    for index, low in enumerate(low_maps, 3):
        nodes = _nodes(data, low, full_area)
        maps = _beta(data, nodes)
        nodes = _nodes(data, maps, full_area)
        maps = _beta(data, nodes)
        corrs[:, index] = [np.arctanh(np.corrcoef(maps[row], low[row])[0, 1]) for row in range(60)]
    fisher = np.mean(corrs, 1)
    spatial = np.maximum(np.mean(fisher) + fisher - fisher * 0.8, 0) ** 3
    nodes = _nodes(data, reference, full_area * spatial)
    maps = _beta(data, nodes)
    nodes = _nodes(data, maps, full_area)
    maps = _beta(data, nodes)
    expected = ((maps - maps[:53].mean(0)) / maps[:53].std(0, ddof=1)
                * reference[:53].std(0, ddof=1) + reference[:53].mean(0))
    actual, actual_nodes, actual_spatial = features._regression(
        torch.as_tensor(data), torch.as_tensor(reference), method="WRN",
        cortical_area=torch.as_tensor(area), low_maps=[torch.as_tensor(value) for value in low_maps],
        smooth=lambda values: values * 0.8)
    np.testing.assert_allclose(actual_spatial, spatial, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(actual_nodes, nodes, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


def test_vn_regresses_retained_signal_instead_of_temporal_sd(tmp_path):
    rng = np.random.default_rng(840)
    mixing = rng.normal(size=(32, 3))
    data = rng.normal(size=(13, 32)) + np.arange(13)[:, None] * mixing[:, 0]
    clean = _cifti(tmp_path / "clean.dtseries.nii", data, _axis(13), timeseries=True)
    mixing_file = tmp_path / "melodic_mix"
    np.savetxt(mixing_file, mixing)
    labels = tmp_path / "classification.txt"
    labels.write_text("existing classification\n[2, 3]\n")
    output = features.compute_msmall_variance_normalization(
        clean, mixing_file, labels, tmp_path / "vn.dscalar.nii", device="cpu")
    observed = np.asarray(nib.load(output).dataobj)[0]
    stored_data = data.astype(np.float32).astype(np.float64)
    standardized = _demean(mixing, 0) / mixing.std(0, ddof=1)
    centered = _demean(stored_data, 1)
    signal = standardized[:, :1]
    residual = centered - (signal @ (np.linalg.pinv(signal) @ centered.T)).T
    expected = np.maximum(residual.std(1, ddof=1), 0.001).astype(np.float32)
    np.testing.assert_array_equal(observed, expected)
    assert not np.allclose(observed, centered.std(1, ddof=1))
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["signal_component_count"] == 1
    assert report["noise_cleanup_performed"] is False


def test_wrn_refuses_to_infer_vn_or_drop_lowdim_templates(tmp_path):
    rng = np.random.default_rng(51)
    clean = _cifti(tmp_path / "clean.dtseries.nii", rng.normal(size=(25, 30)), _axis(25), timeseries=True)
    maps = _cifti(tmp_path / "maps.dscalar.nii", rng.normal(size=(25, 4)), _axis(25))
    with pytest.raises(ValueError, match="explicitly supplied variance_normalization"):
        features.run_msmall_regression(clean, maps, tmp_path / "out", variance_normalization=None, device="cpu")
    vn = _cifti(tmp_path / "vn.dscalar.nii", np.ones((25, 1)), _axis(25))
    with pytest.raises(ValueError, match="all 15"):
        features.run_msmall_regression(clean, maps, tmp_path / "out", variance_normalization=vn,
                                      vertex_area=vn, left_midthickness=clean, right_midthickness=clean,
                                      low_dimensional_maps=[], device="cpu")


def test_dr_cifti_preserves_component_order_and_weights(tmp_path):
    rng = np.random.default_rng(331)
    clean = _cifti(tmp_path / "clean.dtseries.nii", rng.normal(size=(29, 35)), _axis(29), timeseries=True)
    maps = _cifti(tmp_path / "maps.dscalar.nii", rng.normal(size=(29, 6)), _axis(29))
    result = features.run_msmall_regression(clean, maps, tmp_path / "out", variance_normalization=None,
                                           method="DR", component_indices=[2, 5], device="cpu")
    weights = np.asarray(nib.load(result.component_weights).dataobj)
    np.testing.assert_array_equal(weights[:, 0], [0, 1, 0, 0, 1, 0])
    assert nib.load(result.spatial_maps).header.get_axis(0).name.tolist() == [f"component {i + 1}" for i in range(6)]
    assert json.loads(result.report.read_text())["variance_normalization_supplied"] is False


def test_regression_rejects_same_shape_with_different_grayordinate_order(tmp_path):
    rng = np.random.default_rng(90)
    clean = _cifti(tmp_path / "clean.dtseries.nii", rng.normal(size=(9, 20)), _axis(9), timeseries=True)
    reversed_axis = nib.cifti2.BrainModelAxis.from_surface(np.arange(8, -1, -1), 9, name="CortexLeft")
    maps = _cifti(tmp_path / "maps.dscalar.nii", rng.normal(size=(9, 3)), reversed_axis)
    with pytest.raises(ValueError, match="different grayordinate grids"):
        features.run_msmall_regression(clean, maps, tmp_path / "out", variance_normalization=None, method="DR", device="cpu")


def _surface(path, count):
    points = np.column_stack((np.arange(count), np.ones(count), np.zeros(count))).astype(np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(points, intent="NIFTI_INTENT_POINTSET"),
                                   nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE")]), path)
    return path


def _metric(path, data, names=None):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    features._save_metric(path, data, names or [f"map {i + 1}" for i in range(data.shape[1])])
    return path


def test_prepare_ca_keeps_subject_myelin_and_drops_unused_rsn(tmp_path):
    source = _surface(tmp_path / "source.surf.gii", 5)
    reference = _surface(tmp_path / "reference.surf.gii", 6)
    rsn = np.arange(12, dtype=np.float32).reshape(6, 2)
    weights = np.tile([1, 0], (6, 1))
    kwargs = dict(source_rsn=_metric(tmp_path / "srsn.gii", rsn[:5]),
                  reference_rsn=_metric(tmp_path / "rrsn.gii", rsn),
                  source_rsn_weights=_metric(tmp_path / "sw.gii", weights[:5]),
                  reference_rsn_weights=_metric(tmp_path / "rw.gii", weights),
                  subject_myelin=_metric(tmp_path / "smyelin.gii", np.arange(5) + 2),
                  reference_myelin=_metric(tmp_path / "rmyelin.gii", np.arange(6) + 3),
                  subject_myelin_bias=_metric(tmp_path / "bias.gii", np.ones(5) * 0.3),
                  source_roi=_metric(tmp_path / "sroi.gii", [1, 1, 1, 1, 0]),
                  reference_roi=_metric(tmp_path / "rroi.gii", [1, 1, 1, 1, 1, 0]))
    prepared = features.prepare_msmall_inputs(source, reference, tmp_path / "prepared", **kwargs)
    values, names = features._metric(prepared.source_features)
    report = json.loads((tmp_path / "prepared/features.json").read_text())
    assert names == ["map 1", "Myelin", "Medial wall"]
    assert report["dropped_zero_reference_weight_columns"] == 1
    expected = (((np.arange(5) + 2 - np.float32(0.3)) / report["myelin_scale"]).astype(np.float32)
                * np.array([1, 1, 1, 1, 0]) * report["rsn_scale"]).astype(np.float32)
    np.testing.assert_array_equal(values[:, 1], expected)
    assert not np.array_equal(values[:4, 1], np.arange(6)[:4] + 3)
    with pytest.raises(ValueError, match="provided together"):
        features.prepare_msmall_inputs(source, reference, tmp_path / "bad", **kwargs,
                                       source_topography=kwargs["source_rsn"])


def test_prepare_rejects_component_order_even_with_equal_shapes(tmp_path):
    source = _surface(tmp_path / "source.surf.gii", 4)
    reference = _surface(tmp_path / "reference.surf.gii", 4)
    rsn = np.arange(8).reshape(4, 2)
    with pytest.raises(ValueError, match="names/order"):
        features.prepare_msmall_inputs(source, reference, tmp_path / "out",
            modalities="C",
            source_rsn=_metric(tmp_path / "s.gii", rsn, ["B", "A"]),
            reference_rsn=_metric(tmp_path / "r.gii", rsn, ["A", "B"]),
            source_rsn_weights=None, reference_rsn_weights=None, subject_myelin=None,
            reference_myelin=None, subject_myelin_bias=None, source_roi=None, reference_roi=None)


def test_c_mode_is_explicit_and_default_ca_does_not_drop_missing_myelin(tmp_path):
    source = _surface(tmp_path / "source.surf.gii", 4)
    reference = _surface(tmp_path / "reference.surf.gii", 4)
    rsn = np.arange(8).reshape(4, 2)
    kwargs = dict(source_rsn=_metric(tmp_path / "s.gii", rsn),
                  reference_rsn=_metric(tmp_path / "r.gii", rsn),
                  source_rsn_weights=_metric(tmp_path / "sw.gii", np.ones((4, 2))),
                  reference_rsn_weights=_metric(tmp_path / "rw.gii", np.ones((4, 2))),
                  source_roi=_metric(tmp_path / "sr.gii", [1, 1, 1, 0]),
                  reference_roi=_metric(tmp_path / "rr.gii", [1, 1, 1, 0]))
    with pytest.raises(ValueError, match="explicit subject myelin"):
        features.prepare_msmall_inputs(source, reference, tmp_path / "bad", **kwargs)
    prepared = features.prepare_msmall_inputs(source, reference, tmp_path / "c", modalities="C", **kwargs)
    _, names = features._metric(prepared.source_features)
    assert names == ["map 1", "map 2", "Medial wall"]
    report = json.loads((tmp_path / "c/features.json").read_text())
    assert report["modalities"] == "C" and report["subject_myelin_supplied"] is False


def test_modality_stats_use_population_sd_and_seven_digit_source_rounding():
    values = np.asarray([[0, 2], [2, 7], [5, 10]], dtype=np.float32)
    expected = np.asarray([float(format(float(value), ".7g"))
                           for value in values.astype(np.float64).std(0, ddof=0).astype(np.float32)])
    np.testing.assert_array_equal(features._source_stdev(values), expected)
    assert not np.allclose(expected, values.std(0, ddof=1))
