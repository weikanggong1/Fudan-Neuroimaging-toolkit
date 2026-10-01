"""API and quality-gate regressions; generated arrays are not benchmarks."""

import inspect
import json

import h5py
import numpy as np
import pytest
import torch

import fnit.bigflica.pipeline as pipeline
import fnit.bigflica.pipeline_gpu as pipeline_gpu


def _subject_tree(tmp_path):
    root = tmp_path / "subjects"
    for index in range(5):
        directory = root / f"s{index}"
        directory.mkdir(parents=True)
        (directory / "vbm.nii.gz").write_bytes(b"")
    mask = tmp_path / "mask.nii.gz"
    mask.write_bytes(b"")
    return root, {"vbm": {"image": "vbm.nii.gz", "mask": str(mask)}}


@pytest.mark.parametrize("noise", ["o", "R"])
def test_public_raw_routes_noise_and_checks_existing_output(tmp_path, monkeypatch, noise):
    root, modalities = _subject_tree(tmp_path)
    monkeypatch.setattr(pipeline, "_load_mask",
                        lambda path: (None, np.ones((1, 1, 1), dtype=bool)))
    monkeypatch.setattr(pipeline, "_device", lambda value: torch.device("cuda"))
    observed = []
    result_dir = pipeline._flica_directory(tmp_path / "output", 2, noise)

    def raw_backend(*args, **kwargs):
        observed.append(inspect.signature(original_backend).bind(*args, **kwargs).arguments)
        return result_dir

    original_backend = pipeline_gpu.run_bigflica_raw_gpu
    monkeypatch.setattr(pipeline_gpu, "run_bigflica_raw_gpu", raw_backend)
    options = {} if noise == "o" else {"flica_lambda_dims": "R"}
    result = pipeline.run_bigflica(root, modalities, tmp_path / "output", 2,
                                  device="cuda:0", use_mmigp_dicl=False,
                                  flica_max_iter=np.int64(7), **options)
    assert result == result_dir
    assert len(observed) == 1
    assert observed[0]["flica_lambda_dims"] == noise
    assert observed[0]["flica_max_iter"] == 7
    assert type(observed[0]["flica_max_iter"]) is int
    assert json.loads(json.dumps(observed[0]["flica_max_iter"])) == 7
    assert observed[0]["n_components"] == 2
    assert observed[0]["max_gpu_gb"] == 19.0
    result_dir.mkdir()
    (result_dir / "model.json").write_text(json.dumps({
        "input_signature": "different", "source_modalities": modalities}))
    with pytest.raises(ValueError, match="fresh output_dir"):
        pipeline.run_bigflica(root, modalities, tmp_path / "output", 2,
                             device="cuda:0", use_mmigp_dicl=False, **options)
    assert len(observed) == 1


def _fitted_state():
    h = np.array([[1., -2., 0.5, 1.5, 0.], [-0.5, 1., 2., 0., 1.]])
    x = [np.array([[1., 2.], [-1., 0.5], [0.25, -2.]]),
         np.array([[2., -1.], [0.5, 1.5], [-2., 0.25], [1., 1.]])]
    w = [np.array([0.7, -1.3]), np.array([1.2, 0.4])]
    return {"H": h, "X": x, "W": w,
            "H_PCs": np.array([[0.3, 0.8], [0.7, 0.2], [0., 0.]])}


def _raw_backend_stubs(tmp_path, monkeypatch, fitted):
    store = tmp_path / "normalized"
    store.mkdir()
    names = ["vbm", "fa"]
    source_norms = []
    for index, name in enumerate(names):
        matrix = np.arange(5 * fitted["X"][index].shape[0], dtype=float).reshape(5, -1) + 1
        source_norms.append(float(np.square(matrix).sum()))
        with h5py.File(store / f"{name}.h5", "w") as file:
            file.create_dataset("data", data=matrix)
    observed = {}
    monkeypatch.setattr(pipeline_gpu, "prepare_modalities", lambda *args, **kwargs: store)

    def initialize(matrices, components, *, device, max_gpu_gb, lambda_dims):
        observed["initialize"] = (components, device, max_gpu_gb, lambda_dims)
        for index, matrix in enumerate(matrices):
            matrix.squared_sum = source_norms[index]
            assert matrix.shape == (fitted["X"][index].shape[0], 5)
        return {}, {}, {}

    def iterate(matrices, priors, posteriors, constants, max_iter, *, device):
        observed["iterate"] = (max_iter, device)
        return fitted

    original_save = pipeline_gpu._save_gpu_results

    def save(*args, **kwargs):
        observed["save"] = inspect.signature(original_save).bind(*args, **kwargs).arguments
        return pipeline._flica_directory(args[2], args[6], kwargs["flica_lambda_dims"])

    monkeypatch.setattr(pipeline_gpu, "initialize_flica_raw", initialize)
    monkeypatch.setattr(pipeline_gpu, "iterate_flica_torch", iterate)
    monkeypatch.setattr(pipeline_gpu, "_save_gpu_results", save)
    specs = {name: {"image": f"{name}.nii.gz", "mask": f"{name}_mask.nii.gz"}
             for name in names}
    arguments = (tmp_path, specs, tmp_path / "output", [f"s{i}" for i in range(5)],
                 {}, "fixed-input-signature", 2, 7, 100, 0, "cuda:0", 19.0, 2)
    return arguments, observed, source_norms


@pytest.mark.parametrize("noise", ["o", "R"])
def test_raw_backend_passes_noise_to_initialization_and_saved_model(tmp_path, monkeypatch, noise):
    fitted = _fitted_state()
    arguments, observed, source_norms = _raw_backend_stubs(tmp_path, monkeypatch, fitted)
    options = {} if noise == "o" else {"flica_lambda_dims": "R"}
    result = pipeline_gpu.run_bigflica_raw_gpu(*arguments, **options)
    assert result == pipeline._flica_directory(tmp_path / "output", 2, noise)
    assert observed["initialize"] == (2, "cuda:0", 19.0, noise)
    assert observed["iterate"] == (7, "cuda:0")
    saved = observed["save"]
    assert saved["flica_lambda_dims"] == noise
    assert saved["flica_max_iter"] == 7
    assert saved["u"] is None
    assert saved["mmigp_dir"] is None
    assert saved["migp_dim"] is None
    assert saved["dicl_dim"] is None
    strengths = sum(np.square(x * w).sum(axis=0)
                    for x, w in zip(fitted["X"], fitted["W"]))
    order = np.argsort(strengths)[::-1]
    np.testing.assert_array_equal(saved["h_migp"], fitted["H"].T[:, order])
    np.testing.assert_array_equal(saved["contribution"], fitted["H_PCs"][:2, order])
    diagnostics = json.loads((result / "flica_reconstruction.json").read_text())
    assert diagnostics["flica_lambda_dims"] == noise
    assert diagnostics["component_rank"] == 2
    assert diagnostics["requested_components"] == 2


@pytest.mark.parametrize("failure", ["rank", "modality"])
def test_raw_fit_rejection_writes_diagnostics_before_maps_or_model(tmp_path, monkeypatch, failure):
    fitted = _fitted_state()
    if failure == "rank":
        fitted["H"][1] = 2 * fitted["H"][0]
    else:
        fitted["W"][1] *= 1e-9
    arguments, observed, _ = _raw_backend_stubs(tmp_path, monkeypatch, fitted)
    with pytest.raises(ValueError, match="collapsed or pruned"):
        pipeline_gpu.run_bigflica_raw_gpu(*arguments, flica_lambda_dims="R")
    output = pipeline._flica_directory(tmp_path / "output", 2, "R")
    diagnostics = json.loads((output / "flica_reconstruction.json").read_text())
    assert diagnostics["flica_lambda_dims"] == "R"
    assert "save" not in observed
    assert not (output / "model.json").exists()
    assert not (output / "maps").exists()
    if failure == "rank":
        assert diagnostics["component_rank"] == 1
    else:
        assert diagnostics["component_rank"] == 2
        assert diagnostics["per_modality_ratio"]["fa"] < 1e-6


def test_gram_quality_metrics_match_independent_dense_reconstruction(tmp_path):
    fitted = _fitted_state()
    names = ["vbm", "fa"]
    source_norms = [15., 30.]
    dense = [(x * w) @ fitted["H"] for x, w in zip(fitted["X"], fitted["W"])]
    dense_sums = np.array([np.square(value).sum() for value in dense])
    actual_strengths = pipeline._check_flica_fit(fitted, names, source_norms, 2,
                                               tmp_path, "R")
    diagnostics = json.loads((tmp_path / "flica_reconstruction.json").read_text())
    for name, source, norm_sq in zip(names, source_norms, dense_sums):
        assert diagnostics["per_modality_ratio"][name] == pytest.approx(np.sqrt(norm_sq / source))
    assert diagnostics["overall_ratio"] == pytest.approx(np.sqrt(dense_sums.sum() / sum(source_norms)))
    expected_strengths = sum(np.square(x * w).sum(axis=0)
                             for x, w in zip(fitted["X"], fitted["W"]))
    np.testing.assert_allclose(actual_strengths, expected_strengths, rtol=1e-13)
    np.testing.assert_allclose(diagnostics["component_row_norms"],
                               np.linalg.norm(fitted["H"], axis=1), rtol=1e-13)


def test_updated_fit_preserves_older_model_instead_of_overwriting(tmp_path):
    directory = tmp_path / "components_2"
    directory.mkdir()
    path = directory / "model.json"
    previous = {"input_signature": "same-input", "source_modalities": {"vbm": {}}}
    path.write_text(json.dumps(previous))
    original_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="algorithm version.*fresh output_dir"):
        pipeline._check_flica_output(directory, "same-input", ["vbm"],
                                     pipeline._FLICA_ALGORITHM_VERSION)
    assert path.read_bytes() == original_bytes
    previous["flica_algorithm_version"] = pipeline._FLICA_ALGORITHM_VERSION
    path.write_text(json.dumps(previous))
    pipeline._check_flica_output(directory, "same-input", ["vbm"],
                                 pipeline._FLICA_ALGORITHM_VERSION)


@pytest.mark.parametrize("invalid", [0, -1, True, np.bool_(False), 1.5,
                                      np.float64(2), "10", None])
def test_invalid_iterations_fail_before_input_io(tmp_path, monkeypatch, invalid):
    def unexpected_io(*args, **kwargs):
        pytest.fail("Input I/O or device setup occurred before iteration validation")

    for name in ("_load_mask", "_file_record", "_device"):
        monkeypatch.setattr(pipeline, name, unexpected_io)
    with pytest.raises(ValueError, match="flica_max_iter must be a positive integer"):
        pipeline.run_bigflica(tmp_path / "missing", {"vbm": {}}, tmp_path / "output", 2,
                             flica_max_iter=invalid, use_mmigp_dicl=False)
