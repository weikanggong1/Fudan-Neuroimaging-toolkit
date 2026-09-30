import json
import sys

import h5py
import nibabel as nib
import numpy as np
import pytest
import torch
from scipy.stats import norm, t as student_t
from sklearn.decomposition import sparse_encode
from sklearn.utils.extmath import randomized_svd

import fnit.bigflica.streaming as streaming_module
import fnit.bigflica.cli as bigflica_cli
import fnit.bigflica.flica_torch as flica_torch_module
import fnit.bigflica.flica_vb as flica_vb_module
import fnit.bigflica.pipeline as pipeline_module
import fnit.bigflica.pipeline_gpu as pipeline_gpu_module
from fnit.bigflica import apply_model
from fnit.bigflica.dicl_torch import _randomized_svd_dictionary, _sparse_codes_lars
from fnit.bigflica.flica_torch import _estimate_dd_eigenvalues, _scaled_inverse
from fnit.bigflica.flica_vb import fit_eigenspectrum, flica_init_params, update_H
from fnit.bigflica.pipeline import (_check_flica_output, _fit_flica,
                                   _flica_directory, _read_vector,
                                   _write_maps, fit_mmigp, run_bigflica)
from fnit.bigflica.stats_torch import t_to_z_gpu
from fnit.bigflica.streaming import (
    _eigen_residuals, convert_normalized_store_float32, fit_mmigp_streaming,
    prepare_modalities,
)


def test_rejects_image_mask_affine_mismatch(tmp_path):
    image = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), image)
    shifted = np.eye(4)
    shifted[0, 3] = 2
    mask_image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), shifted)
    with pytest.raises(ValueError, match="grid differ"):
        _read_vector(image, mask_image, np.ones((2, 2, 2), dtype=bool))


def test_apply_uses_frozen_mask_and_loadings(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    new_subject = tmp_path / "new"
    new_subject.mkdir()
    mask = np.ones((2, 2, 1), dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), model / "vbm_mask.nii.gz")
    nib.save(nib.Nifti1Image(np.array([2, 3, 0, 0], dtype=np.float32).reshape(2, 2, 1),
                            np.eye(4)), new_subject / "vbm.nii.gz")
    np.save(model / "vbm_mean.npy", np.zeros(4, dtype=np.float32))
    np.save(model / "vbm_std.npy", np.ones(4, dtype=np.float32))
    np.save(model / "vbm_loadings.npy", np.array([[1, 0, 0, 0], [0, 1, 0, 0]],
                                                 dtype=np.float32))
    (model / "model.json").write_text(json.dumps({
        "subjects": ["train"], "n_components": 2,
        "modalities": {"vbm": {"image": "vbm.nii.gz", "mask": "vbm_mask.nii.gz"}},
    }))
    output_file = tmp_path / "new_course.tsv"
    np.testing.assert_allclose(apply_model(model, new_subject, ridge=0,
                                           output_file=output_file), [2, 3])
    assert output_file.is_file()
    (tmp_path / "train").mkdir()
    with pytest.raises(ValueError, match="absent from training"):
        apply_model(model, tmp_path / "train")


def test_lars_codes_match_sklearn_objective():
    generator = np.random.default_rng(7)
    samples = generator.normal(size=(16, 6))
    dictionary = generator.normal(size=(9, 6))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    expected = sparse_encode(samples, dictionary, algorithm="lasso_lars", alpha=0.4)
    actual = _sparse_codes_lars(torch.as_tensor(samples),
                               torch.as_tensor(dictionary), 0.4).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_lars_mixed_finished_rows_match_sklearn():
    generator = np.random.default_rng(17)
    samples = generator.normal(size=(32, 10))
    samples[::4] = 0
    dictionary = generator.normal(size=(40, 10))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    expected = sparse_encode(samples, dictionary, algorithm="lasso_lars", alpha=0.8)
    actual = _sparse_codes_lars(torch.as_tensor(samples),
                               torch.as_tensor(dictionary), 0.8).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_dicl_randomized_svd_initialization_matches_sklearn(tmp_path):
    generator = np.random.default_rng(23)
    projected = generator.normal(size=(128, 48))
    mean = projected.mean(axis=0)
    std = projected.std(axis=0)
    standardized = (projected - mean) / std
    _, singular_values, right = randomized_svd(
        standardized, n_components=12, n_oversamples=10, n_iter=4,
        random_state=0, transpose=False, flip_sign=True)
    expected = singular_values[:, None] * right
    with h5py.File(tmp_path / "projected.h5", "w") as file:
        data = file.create_dataset("data", data=projected)
        actual = _randomized_svd_dictionary(
            data, torch.as_tensor(standardized, device="cuda", dtype=torch.float64),
            torch.as_tensor(mean, device="cuda", dtype=torch.float64),
            torch.as_tensor(std, device="cuda", dtype=torch.float64), 12,
            np.random.RandomState(0), 64).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


def test_gpu_t_to_z_matches_student_distribution():
    t_values = np.array([-10., -2., 0., 1., 5., 10.])
    expected = np.sign(t_values) * norm.isf(student_t.sf(np.abs(t_values), 14))
    actual = t_to_z_gpu(t_values, 14, device="cpu")
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_cpu_streaming_mmigp_matches_exact_small_reference(tmp_path):
    generator = np.random.default_rng(19)
    matrices = {name: generator.normal(size=(16, size))
                for name, size in (("vbm", 23), ("fa", 31))}
    expected_u, expected_projected = fit_mmigp(matrices, 5, device="cpu")
    store = tmp_path / "normalized"
    store.mkdir()
    for name, matrix in matrices.items():
        with h5py.File(store / f"{name}.h5", "w") as file:
            file.create_dataset("data", data=matrix,
                                chunks=(matrix.shape[0], 7))
    actual_u, projected_dir = fit_mmigp_streaming(
        store, list(matrices), 5, tmp_path / "projected", device="cpu",
        feature_block=7, max_gpu_gb=3)
    diagnostics = json.loads((projected_dir / "eigen_diagnostics.json").read_text())
    assert diagnostics["converged"] and diagnostics["relative_residual"] < 1e-8
    assert len(diagnostics["pair_relative_residuals"]) == 5
    assert diagnostics["max_pair_relative_residual"] < 1e-8
    assert all(diagnostics[name] >= 0 for name in
               ("covariance_s", "eigensolver_s", "projection_s"))
    np.testing.assert_allclose(
        diagnostics["adjacent_eigenvalue_gaps"],
        -np.diff(diagnostics["eigenvalues"]), atol=1e-12)
    assert diagnostics["boundary_eigenvalue_gap"] >= 0
    assert diagnostics["eigenvalue_gap_source"] == "exact_spectrum"
    orientation = np.sign((expected_u * actual_u).sum(axis=0))
    np.testing.assert_allclose(actual_u * orientation, expected_u,
                               atol=1e-8, rtol=1e-8)
    for name in matrices:
        with h5py.File(projected_dir / f"{name}_projected.h5", "r") as file:
            np.testing.assert_allclose(file["data"][:] * orientation,
                                       expected_projected[name], atol=1e-8, rtol=1e-8)


def test_mmigp_convergence_checks_each_eigenpair():
    covariance = torch.eye(64, dtype=torch.float64) * 100
    vectors = torch.eye(64, dtype=torch.float64)
    values = torch.full((64,), 100.0, dtype=torch.float64)
    values[-1] -= 3e-6
    overall, per_pair = _eigen_residuals(covariance, values, vectors)
    assert overall < 1e-8
    assert float(per_pair.max()) > 1e-8


def test_flica_subjectwise_noise_updates_each_h_covariance():
    spatial = np.array([[1., 2.], [.5, -1.], [2., .5]])
    images = np.array([[2., 1., 3.], [0., -1., 2.], [1., 3., -2.]])
    weights = np.array([1., .75])
    noise_precision = np.array([.7, 1.2, 2.3])
    weight_moment = np.array([[1.5, .2], [.2, .8]])
    spatial_moment = spatial.T @ spatial
    state = {
        "opts": {"lambda_dims": "R"}, "NH": 3, "K": 1, "L": 2, "R": 3,
        "Y": [images], "X": [spatial], "H": np.zeros((2, 3)),
        "W": [np.matrix(weights[None, :])], "eta": np.array([[2.], [3.]]),
        "Gmat": np.ones(3), "DD": np.ones(1),
        "Lambda": [np.matrix(noise_precision[:, None])],
        "lambda_R": [np.matrix(noise_precision[:, None])],
        "WtW": [np.matrix(weight_moment)],
        "XtDX": [np.matrix(spatial_moment)],
    }
    actual = update_H(state)
    for subject in range(3):
        precision = (np.diag(state["eta"][:, 0]) + noise_precision[subject] *
                     weight_moment * spatial_moment.T)
        rhs = (weights * (spatial.T @ images[:, subject]) *
               noise_precision[subject])
        np.testing.assert_allclose(actual["H"][:, subject],
                                   np.linalg.solve(precision, rhs), rtol=1e-6)
        np.testing.assert_allclose(actual["H_colcov"][:, :, subject],
                                   np.linalg.inv(precision), rtol=1e-6)


def test_flica_explicit_subject_initializer_matches_matlab_formula():
    initial_h = np.array([[1., 0., 2., 1.], [0., 1., 1., -1.]])
    images = [np.array([[1., 2., 3., 1.], [2., -1., 1., 4.]]),
              np.array([[3., 0., 1., 2.], [-2., 1., 2., 3.]])]
    opts = {"num_components": 2, "maxits": 1, "lambda_dims": "o",
            "initH": initial_h, "dof_per_voxel": np.array([1., 2.])}
    _, posterior, constants = flica_init_params(images, opts)
    np.testing.assert_allclose(posterior["H"], initial_h / np.sqrt(1.5))
    for index, image in enumerate(images):
        expected = image * np.sqrt(constants["DD"][index]) @ np.linalg.pinv(initial_h)
        np.testing.assert_allclose(posterior["X"][index], expected)


def test_flica_dd_estimator_matches_source_near_gamma_one():
    gamma = 0.99995
    grid = np.arange((1 - np.sqrt(gamma)) ** 2,
                     (1 + np.sqrt(gamma)) ** 2, 0.001)
    density = (np.sqrt((grid - grid.min()) * (grid.max() - grid)) /
               (2 * gamma * np.pi * grid))
    cumulative = np.cumsum(density) * 0.001
    cumulative[-1] = 1
    spectrum = np.interp(np.linspace(0, 1, 100), cumulative, grid)[::-1] * 200
    selected = np.full(100, np.nan)
    selected[[24, 74]] = spectrum[[24, 74]]
    expected = 100 / (fit_eigenspectrum(selected)[0] * 200)
    actual = _estimate_dd_eigenvalues(torch.as_tensor(spectrum), 200, 100)
    np.testing.assert_allclose(actual.item(), expected, rtol=1e-12)


def test_flica_lambda_option_and_existing_output_guard(tmp_path):
    modalities = {"vbm": {"image": "vbm.nii.gz", "mask": str(tmp_path / "mask.nii.gz")}}
    with pytest.raises(ValueError, match="flica_lambda_dims"):
        run_bigflica(tmp_path, modalities, tmp_path / "out", 2,
                     flica_lambda_dims="invalid")
    with pytest.raises(ValueError, match="compressed"):
        run_bigflica(tmp_path, modalities, tmp_path / "out", 2,
                     use_mmigp_dicl=False, flica_lambda_dims="R")
    scalar = _flica_directory(tmp_path, 2, "o")
    subjectwise = _flica_directory(tmp_path, 2, "R")
    assert scalar != subjectwise
    scalar.mkdir()
    (scalar / "model.json").write_text(json.dumps({
        "input_signature": "old", "source_modalities": {"vbm": {}}}))
    _check_flica_output(scalar, "old", ["vbm"])
    with pytest.raises(ValueError, match="fresh output_dir"):
        _check_flica_output(scalar, "new", ["vbm"])
    with pytest.raises(ValueError, match="fresh output_dir"):
        _check_flica_output(scalar, "old", ["fa"])


def test_public_compressed_run_routes_subjectwise_noise(tmp_path, monkeypatch):
    root = tmp_path / "subjects"
    for index in range(4):
        directory = root / f"s{index}"
        directory.mkdir(parents=True)
        (directory / "vbm.nii.gz").write_bytes(b"")
    mask = tmp_path / "mask.nii.gz"
    mask.write_bytes(b"")
    observed = []
    monkeypatch.setattr(pipeline_module, "_load_mask",
                        lambda path: (None, np.ones((1, 1, 1), dtype=bool)))
    monkeypatch.setattr(pipeline_module, "_device", lambda requested: torch.device("cuda"))

    def gpu_run(*args):
        observed.append((args[-5], args[-1]))
        return tmp_path / "model"

    monkeypatch.setattr(pipeline_gpu_module, "run_bigflica_gpu", gpu_run)
    result = run_bigflica(root, {"vbm": {"image": "vbm.nii.gz", "mask": str(mask)}},
                           tmp_path / "out", 1, migp_dim=3, dicl_dim=2,
                           device="cuda:0", flica_lambda_dims="R")
    assert result == tmp_path / "model"
    assert observed == [(19.0, "R")]
    existing = _flica_directory(tmp_path / "out", 1, "R")
    existing.mkdir(parents=True)
    (existing / "model.json").write_text(json.dumps({
        "input_signature": "other", "source_modalities": {"zstat1": {}}}))
    with pytest.raises(ValueError, match="fresh output_dir"):
        run_bigflica(root, {"vbm": {"image": "vbm.nii.gz", "mask": str(mask)}},
                     tmp_path / "out", 1, migp_dim=3, dicl_dim=2,
                     device="cuda:0", flica_lambda_dims="R")
    assert observed == [(19.0, "R")]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_flica_lambda_option_reaches_compressed_fit(tmp_path, monkeypatch, device):
    h = np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]])
    x = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]])
    dictionaries = {"vbm": x @ h, "fa": (x * 2) @ h}
    expected = {"H": h, "X": [x, x * 2], "W": [np.ones(2), np.ones(2)],
                "H_PCs": np.ones((3, 2))}
    observed = []
    if device == "cpu":
        def cpu_initialize(data, options):
            observed.append(options["lambda_dims"])
            return {}, {}, {}

        monkeypatch.setattr(flica_vb_module, "flica_init_params", cpu_initialize)
        monkeypatch.setattr(flica_vb_module, "flica_iterate",
                            lambda *args: expected)
    else:
        monkeypatch.setattr(pipeline_module, "_device", lambda requested: torch.device("cuda"))

        def gpu_initialize(data, components, *, device, lambda_dims):
            observed.append(lambda_dims)
            return {}, {}, {}

        monkeypatch.setattr(flica_torch_module, "initialize_flica_torch", gpu_initialize)
        monkeypatch.setattr(flica_torch_module, "iterate_flica_torch",
                            lambda *args, **kwargs: expected)
    output_dir = tmp_path / device.replace(":", "_")
    _fit_flica(dictionaries, 2, 1, output_dir, device, "R")
    assert observed == ["R"]
    diagnostics = json.loads((output_dir / "flica_reconstruction.json").read_text())
    assert diagnostics["flica_lambda_dims"] == "R"


def test_cli_passes_flica_lambda_option(tmp_path, monkeypatch):
    config = tmp_path / "modalities.json"
    config.write_text(json.dumps({"modalities": {"vbm": {"image": "vbm.nii.gz",
                                                           "mask": "mask.nii.gz"}}}))
    observed = {}

    def fit(*args, **kwargs):
        observed.update(kwargs)
        return tmp_path / "model"

    monkeypatch.setattr(bigflica_cli, "run_bigflica", fit)
    monkeypatch.setattr(sys, "argv", ["fnit-bigflica", "fit", "--subjects-root",
                                      str(tmp_path), "--config", str(config),
                                      "--output-dir", str(tmp_path / "out"),
                                      "--n-components", "2", "--flica-lambda-dims", "R"])
    bigflica_cli.main()
    assert observed["flica_lambda_dims"] == "R"
    assert observed["max_gpu_gb"] == 19.0


def test_flica_maps_keep_float_z_values_with_uint8_mask(tmp_path):
    mask_image = nib.Nifti1Image(np.ones((2, 2, 1), dtype=np.uint8), np.eye(4))
    mask = np.ones((2, 2, 1), dtype=bool)
    values = np.array([[0.03125], [-0.0475], [0.5], [-0.875]], dtype=np.float32)
    _write_maps("vbm", values, mask_image, mask, tmp_path, top_voxels=2)
    output = tmp_path / "vbm"
    continuous = nib.load(output / "component-001_zstat.nii.gz")
    thresholded = nib.load(output / "component-001_top-2.nii.gz")
    assert continuous.get_data_dtype() == np.dtype("float32")
    assert thresholded.get_data_dtype() == np.dtype("float32")
    np.testing.assert_allclose(continuous.get_fdata(dtype=np.float32)[mask],
                               values[:, 0], atol=1e-7)
    np.testing.assert_allclose(thresholded.get_fdata(dtype=np.float32)[mask],
                               [0, 0, 0.5, -0.875], atol=1e-7)


def test_float32_normalized_cache_and_cpu_mmigp(tmp_path):
    generator = np.random.default_rng(27)
    store = tmp_path / "normalized"
    store.mkdir()
    (store / "manifest.json").write_text(json.dumps({
        "signature": "normalized-float64", "subjects": ["s1", "s2"],
        "modalities": ["vbm", "fa"],
    }))
    for name, width in (("vbm", 23), ("fa", 29)):
        data = generator.normal(size=(16, width))
        with h5py.File(store / f"{name}.h5", "w") as file:
            file.create_dataset("data", data=data, chunks=(8, 7))
            file.create_dataset("mean", data=np.zeros(width, dtype=np.float64))
            file.create_dataset("std", data=np.ones(width, dtype=np.float64))
            file.create_dataset("valid_rows", data=np.ones(16, dtype=bool))
            file.attrs["signature"] = "normalized-float64"
    fp32_store = convert_normalized_store_float32(store, tmp_path / "normalized_fp32")
    fp32_manifest = json.loads((fp32_store / "manifest.json").read_text())
    assert fp32_manifest["signature"] != "normalized-float64"
    assert fp32_manifest["normalized_dtype"] == "float32"
    for name in ("vbm", "fa"):
        with h5py.File(store / f"{name}.h5", "r") as original, \
             h5py.File(fp32_store / f"{name}.h5", "r") as converted:
            assert converted["data"].dtype == np.dtype("float32")
            np.testing.assert_array_equal(converted["data"][:],
                                          original["data"][:].astype(np.float32))
            np.testing.assert_array_equal(converted["mean"][:], original["mean"][:])
            assert converted.attrs["signature"] == fp32_manifest["signature"]
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        actual_u, projected_dir = fit_mmigp_streaming(
            fp32_store, ["vbm", "fa"], 5, tmp_path / "mmigp_fp32",
            device="cpu", feature_block=7, max_gpu_gb=3)
        assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    reference_u, reference_dir = fit_mmigp_streaming(
        store, ["vbm", "fa"], 5, tmp_path / "mmigp_fp64",
        device="cpu", feature_block=7, max_gpu_gb=3)
    diagnostics = json.loads((projected_dir / "eigen_diagnostics.json").read_text())
    assert diagnostics["input_dtype"] == diagnostics["compute_dtype"] == "float32"
    assert not diagnostics["tf32_enabled"]
    assert diagnostics["converged"]
    assert actual_u.dtype == np.dtype("float32")
    np.testing.assert_allclose(actual_u, reference_u, atol=2e-4, rtol=2e-4)
    for name in ("vbm", "fa"):
        with h5py.File(projected_dir / f"{name}_projected.h5", "r") as fp32, \
             h5py.File(reference_dir / f"{name}_projected.h5", "r") as fp64:
            assert fp32["data"].dtype == np.dtype("float32")
            np.testing.assert_allclose(fp32["data"][:], fp64["data"][:],
                                       atol=2e-4, rtol=2e-4)


def test_mmigp_strict_float32_flag_after_device_selection(tmp_path, monkeypatch):
    store = tmp_path / "normalized"
    store.mkdir()
    with h5py.File(store / "vbm.h5", "w") as file:
        file.create_dataset("data", data=np.ones((3, 4), dtype=np.float32))
    observed = []

    def inspect_precision(*args, **kwargs):
        observed.append((torch.backends.cuda.matmul.allow_tf32,
                         kwargs["backend"].type))
        return np.empty((3, 2), dtype=np.float32), tmp_path

    monkeypatch.setattr(streaming_module, "_fit_mmigp_streaming_impl", inspect_precision)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False
    try:
        fit_mmigp_streaming(store, ["vbm"], 2, tmp_path / "output", device="cpu")
        assert observed == [(False, "cpu")]
        assert torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def test_direct_float32_normalization_matches_float64_cast(tmp_path):
    subjects_root = tmp_path / "subjects"
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    subjects = ["s1", "s2", "s3"]
    for index, subject in enumerate(subjects):
        directory = subjects_root / subject
        directory.mkdir(parents=True)
        values = (np.arange(8, dtype=np.float32).reshape(2, 2, 2) +
                  np.float32(index + 0.5))
        nib.save(nib.Nifti1Image(values, np.eye(4)), directory / "vbm.nii.gz")
    specs = {"vbm": {"image": "vbm.nii.gz", "mask": str(mask_path)}}
    full = prepare_modalities(subjects_root, specs, subjects,
                              tmp_path / "normalized_fp64", feature_block=4)
    compact = prepare_modalities(subjects_root, specs, subjects,
                                 tmp_path / "normalized_fp32", feature_block=4,
                                 normalized_dtype="float32")
    with h5py.File(full / "vbm.h5", "r") as fp64, \
         h5py.File(compact / "vbm.h5", "r") as fp32:
        assert fp32["data"].dtype == np.dtype("float32")
        assert fp32.attrs["normalized_dtype"] == "float32"
        np.testing.assert_array_equal(fp32["data"][:],
                                      fp64["data"][:].astype(np.float32))
        np.testing.assert_array_equal(fp32["mean"][:], fp64["mean"][:])
        np.testing.assert_array_equal(fp32["std"][:], fp64["std"][:])
    assert (json.loads((full / "manifest.json").read_text())["signature"] !=
            json.loads((compact / "manifest.json").read_text())["signature"])
    assert json.loads((compact / "manifest.json").read_text())["normalized_dtype"] == "float32"


def test_flica_inverse_reports_singular_matrix_without_host_check():
    invertible = torch.tensor([[3., 1.], [1., 2.]], dtype=torch.float64)
    inverse, info = _scaled_inverse(invertible)
    assert int(info) == 0
    np.testing.assert_allclose(inverse.numpy(), np.linalg.inv(invertible.numpy()),
                               atol=1e-12)
    singular = torch.tensor([[1., 1.], [1., 1.]], dtype=torch.float64)
    _, info = _scaled_inverse(singular)
    assert int(info) != 0
