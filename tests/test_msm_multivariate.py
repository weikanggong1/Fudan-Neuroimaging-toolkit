"""MSMAll vector-feature/weight contracts; fixtures are not benchmarks."""

from dataclasses import replace

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.msm.config_msmall import MSMAllConfig
from fnit.msm.msmall import (
    MSMAllInputs, _combine_weights, _feature_weights, _read_features,
    _register_msmall_one, _same_sphere_coordinates, _variance_normalize_columns, _weighted_vector_cost,
)
from fnit.msm.msmsulc import _ico, _variance_normalize


def write_features(path, values, matrix=False):
    values = np.asarray(values, dtype=np.float32)
    arrays = [values] if matrix else list(values.T)
    nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(array) for array in arrays]), path)


def write_sphere(path, points, faces):
    nib.save(nib.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(np.asarray(points, np.float32), intent=1008),
        nib.gifti.GiftiDataArray(np.asarray(faces, np.int32), intent=1009),
    ]), path)


def canonical_text(levels):
    config = MSMAllConfig.coarse() if levels == 1 else MSMAllConfig()
    options = {"simval": config.simval, "it": config.iterations,
               "CPgrid": config.control_grid, "SGgrid": config.sampling_grid,
               "datagrid": config.data_grid}
    lines = ["--" + key + "=" + ",".join(map(str, values)) for key, values in options.items()]
    lines += ["--lambda=" + ("0.00001" if levels == 1 else "0.00001,0.0075,0.01"),
              "--opt=" + ",".join(["DISCRETE"] * levels),
              "--sigma_in=" + ",".join(["0"] * levels),
              "--sigma_ref=" + ",".join(["0"] * levels),
              "--regoption=3", "--regexp=2", "--dopt=HOCR", "--VN", "--rescaleL",
              "--triclique", "--k_exponent=2", "--bulkmod=1.6", "--shearmod=0.4"]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("levels", [1, 3])
def test_official_discrete_schedule_parses_without_affine(tmp_path, levels):
    path = tmp_path / "MSMAll.conf"; path.write_text(canonical_text(levels))
    config = MSMAllConfig.from_file(path)
    assert config == (MSMAllConfig.coarse() if levels == 1 else MSMAllConfig.refine())
    assert config.shear_modulus == float(np.float32(.4))
    assert config.regularization[0] == float(np.float32(.00001))


@pytest.mark.parametrize("extra", ["--patchwise", "--cprange=2", "--IN", "--simval=4"])
def test_changed_likelihood_is_rejected(tmp_path, extra):
    path = tmp_path / "MSMAll.conf"; path.write_text(canonical_text(1) + extra + "\n")
    with pytest.raises(ValueError):
        MSMAllConfig.from_file(path)


@pytest.mark.parametrize("missing", ["opt", "sigma_in", "sigma_ref", "dopt", "regoption"])
def test_config_file_cannot_silently_default_scientific_options(tmp_path, missing):
    path = tmp_path / "MSMAll.conf"
    path.write_text("\n".join(line for line in canonical_text(1).splitlines()
                               if not line.startswith("--" + missing + "=")))
    with pytest.raises(ValueError, match="explicitly select"):
        MSMAllConfig.from_file(path)


@pytest.mark.parametrize("values", [{"iterations": (10, 0, 15)},
                                    {"regularization": (1e-5, float("nan"), .01)},
                                    {"control_grid": (2., 3, 4)},
                                    {"simval": (2, 8, 2)}, {"bulk_modulus": float("inf")}])
def test_invalid_config_never_reaches_optimizer(values):
    with pytest.raises(ValueError):
        MSMAllConfig(**values)


@pytest.mark.parametrize("matrix", [False, True])
def test_metric_columns_preserve_vertex_and_feature_order(tmp_path, matrix):
    path = tmp_path / "features.func.gii"
    values = np.arange(15).reshape(5, 3)
    write_features(path, values, matrix)
    np.testing.assert_array_equal(_read_features(path, 5), values)
    with pytest.raises(ValueError, match="row per sphere"):
        _read_features(path, 6)


def test_cost_weight_rows_do_not_broadcast_and_overlap_is_averaged():
    source = np.array([[.2], [.4]])
    reference = np.array([[.8, .3, .6], [.6, .5, .9]])
    combined = _combine_weights(source, reference)
    np.testing.assert_array_equal(combined, [[.5, .3, .6], [.5, .5, .9]])
    np.testing.assert_array_equal(_feature_weights(source, 3), [[.2, 1., 1.], [.4, 1., 1.]])


def scalar_pearson_cost(first, second, weights):
    total = sum(float(value) for value in weights)
    ma = sum(float(w) * float(a) for w, a in zip(weights, first))
    mb = sum(float(w) * float(b) for w, b in zip(weights, second))
    if total > 0:
        ma /= total; mb /= total
    covariance = sum(float(w) * (float(a) - ma) * (float(b) - mb)
                     for w, a, b in zip(weights, first, second))
    va = sum(float(w) * (float(a) - ma) ** 2 for w, a in zip(weights, first))
    vb = sum(float(w) * (float(b) - mb) ** 2 for w, b in zip(weights, second))
    if total > 0:
        covariance /= total; va /= total; vb /= total
    correlation = 0 if va == 0 or vb == 0 else covariance / (np.sqrt(va) * np.sqrt(vb))
    return 1 - (1 + correlation) * .5


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_pearson_is_across_features_at_each_vertex(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    first = np.array([[1., 3., -2., 4.], [0., 1., 4., 5.], [2., 2., 2., 2.]])
    second = np.array([[2., 4., -1., 0.], [5., 4., 1., 0.], [1., 2., 3., 4.]])
    weights = np.array([[1., .3, .9, .4], [0., 0., 0., 0.], [1., 1., 1., 1.]])
    expected = [scalar_pearson_cost(a, b, w) for a, b, w in zip(first, second, weights)]
    actual = _weighted_vector_cost(*(torch.tensor(values, device=device) for values in (first, second, weights)))
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=2e-15)
    assert actual[1] == actual[2] == .5


def test_vn_keeps_independent_spatial_welford_order_for_each_feature():
    values = np.array([[2., 9., 7.], [1., 2., 7.], [5., 0., 7.], [3., 1., 7.]])
    actual = _variance_normalize_columns(values)
    for column in range(3):
        np.testing.assert_array_equal(actual[:, column], _variance_normalize(values[:, column]))


def test_initial_identity_matches_upstream_point_tolerance():
    first = np.zeros((2, 3), dtype=np.float64)
    assert _same_sphere_coordinates(first, first + 0.999999e-8)
    assert not _same_sphere_coordinates(first, first + 1e-8)
    assert not _same_sphere_coordinates(first, first[:1])


def prepared_case(tmp_path):
    points, faces = _ico(1)
    sphere = tmp_path / "sphere.surf.gii"; write_sphere(sphere, points, faces)
    features = tmp_path / "features.func.gii"
    write_features(features, points / 100)
    return MSMAllInputs(sphere, features, sphere, features)


def test_initial_sphere_must_preserve_source_topology(tmp_path):
    entry = prepared_case(tmp_path)
    points, faces = _ico(1); bad = tmp_path / "bad.surf.gii"
    write_sphere(bad, points, faces[:, ::-1])
    with pytest.raises(ValueError, match="vertex order and topology"):
        _register_msmall_one(replace(entry, initial_sphere=bad), tmp_path / "result", device="cpu")


def test_low_resolution_discrete_registration_has_no_rigid_stage(tmp_path, monkeypatch):
    import fnit.msm._affine as affine
    def forbidden(*args, **kwargs):
        raise AssertionError("MSMAll must not add a rigid initialization")
    monkeypatch.setattr(affine, "_affine_initialization", forbidden)
    config = MSMAllConfig(simval=(2,), iterations=(1,), control_grid=(1,),
                         sampling_grid=(2,), data_grid=(1,), regularization=(1e-5,))
    path, report = _register_msmall_one(prepared_case(tmp_path), tmp_path / "result", device="cpu", config=config)
    assert path.is_file() and report["feature_count"] == 3
    assert len(report["stages"]) == 1 and report["stages"][0]["maximum_iterations"] == 1
    assert "affine_seconds" not in report


def test_only_one_cost_weight_file_follows_official_unweighted_behavior(tmp_path):
    entry = prepared_case(tmp_path)
    points, _ = _ico(1)
    weight_path = tmp_path / "source_weights.func.gii"
    write_features(weight_path, ((points[:, 0] + 100) / 200)[:, None])
    config = MSMAllConfig(simval=(2,), iterations=(1,), control_grid=(1,),
                         sampling_grid=(2,), data_grid=(1,), regularization=(1e-5,))
    first, _ = _register_msmall_one(entry, tmp_path / "unweighted", device="cpu", config=config)
    second, report = _register_msmall_one(replace(entry, source_weights=weight_path),
                                         tmp_path / "one_weight", device="cpu", config=config)
    assert not report["weighted_cost"]
    np.testing.assert_array_equal(nib.load(first).darrays[0].data, nib.load(second).darrays[0].data)
