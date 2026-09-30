"""Numerical and functional checks; generated images are not benchmark evidence."""

import csv
import json

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.superbigflica.model import SupervisedComponents, SupervisedObjective
from fnit.superbigflica.pipeline import apply_model, run_superbigflica


torch.set_num_threads(1)


def test_forward_matches_fixed_encoder_reconstruction_and_prediction_equations():
    model = SupervisedComponents([3, 2], 2, [1, 3], dropout=0).double()
    spatial = [np.array([[.2, -.5], [1.2, .3], [-.8, .7]]),
               np.array([[.1, 1.4], [-.6, .2]])]
    modality_logits = np.array([[1.3, -.2], [.4, .9]])
    prediction_weight = np.array([[.5, 1., -.3, .2], [-.2, .4, .8, -.7]])
    prediction_bias = np.array([.1, -.2, .3, .4])
    running_mean, running_var = np.array([.2, -.1]), np.array([1.5, .7])
    batch_weight, batch_bias = np.array([1.2, .8]), np.array([-.3, .2])
    images = [np.array([[1., 2., 3.], [-1., .2, 1.1], [.5, -.3, .7]]),
              np.array([[.4, -.7], [1., 2.], [-.2, .3]])]
    with torch.no_grad():
        for parameter, value in zip(model.spatial, spatial):
            parameter.copy_(torch.from_numpy(value))
        model.modality_logits.copy_(torch.from_numpy(modality_logits))
        model.prediction_weight.copy_(torch.from_numpy(prediction_weight))
        model.prediction_bias.copy_(torch.from_numpy(prediction_bias))
        model.normalization.running_mean.copy_(torch.from_numpy(running_mean))
        model.normalization.running_var.copy_(torch.from_numpy(running_var))
        model.normalization.weight.copy_(torch.from_numpy(batch_weight))
        model.normalization.bias.copy_(torch.from_numpy(batch_bias))
    model.eval()
    logits = modality_logits - modality_logits.max(axis=1, keepdims=True)
    weights = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    encoded = sum((image @ loading) * weights[:, index]
                  for index, (image, loading) in enumerate(zip(images, spatial))) / len(images)
    latent = ((encoded - running_mean) / np.sqrt(running_var + model.normalization.eps)
              * batch_weight + batch_bias)
    rebuilt, actual_latent, actual_prediction = model([torch.from_numpy(image) for image in images])
    np.testing.assert_allclose(actual_latent.detach(), latent, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_prediction.detach(), latent @ prediction_weight + prediction_bias,
                               rtol=1e-12, atol=1e-12)
    for index, estimate in enumerate(rebuilt):
        np.testing.assert_allclose(estimate.detach(), (latent * weights[:, index]) @ spatial[index].T,
                                   rtol=1e-12, atol=1e-12)
    assert model([torch.from_numpy(image) for image in images], reconstruct=False)[0] == []


def test_continuous_complete_loss_matches_independent_four_term_equations():
    model = SupervisedComponents([3, 2], 2, [1, 1], dropout=0).double()
    objective = SupervisedObjective(2, [{"type": "continuous"}] * 2, [1, 1], .7).double()
    spatial = [np.array([[.1, -.2], [.3, .4], [-.5, .6]]), np.array([[.7, -.8], [.9, .2]])]
    prediction_weights = np.array([[.4, -.5], [-.2, .8]])
    raw = [np.array([.8, 1.3]), np.array([.7, 1.2]), np.array([1.4, .9]),
           np.array([1.1, .6]), np.array([.8, 1.5])]
    images = [np.arange(9, dtype=float).reshape(3, 3) / 7,
              np.arange(6, dtype=float).reshape(3, 2) / 5]
    rebuilt = [images[0] + .2, images[1] - .3]
    predicted = np.array([[.1, .4], [-.5, 1.2], [.8, -.3]])
    labels = np.array([[.4, -.1], [.2, .6], [-.3, .9]])
    with torch.no_grad():
        for parameter, value in zip(model.spatial, spatial):
            parameter.copy_(torch.from_numpy(value))
        model.prediction_weight.copy_(torch.from_numpy(prediction_weights))
        for parameter, value in zip(objective.scales, raw):
            parameter.copy_(torch.from_numpy(value))
    squared = [value ** 2 for value in raw]
    fraction = 3 / 10
    reconstruction = sum(np.mean((estimate - image) ** 2) / (2 * squared[0][i] ** 2)
                         for i, (estimate, image) in enumerate(zip(rebuilt, images)))
    reconstruction += np.log1p(squared[0]).sum()
    sparsity = fraction * sum(np.abs(loading).mean() / squared[1][i]
                              for i, loading in enumerate(spatial)) + 2 * np.log1p(squared[1]).sum()
    supervision = np.mean((predicted - labels) ** 2 / (2 * squared[2] ** 2))
    supervision += np.log1p(squared[2]).sum()
    regularization = np.mean(fraction * np.abs(prediction_weights) / squared[3])
    regularization += 2 * np.log1p(squared[3]).sum()
    regularization += np.mean(fraction * prediction_weights ** 2 / (2 * squared[4] ** 2))
    regularization += np.log1p(squared[4]).sum()
    expected_terms = np.array([reconstruction, sparsity, supervision, regularization])
    loss, terms = objective(model, [torch.from_numpy(value) for value in images],
                            [torch.from_numpy(value) for value in rebuilt], torch.from_numpy(predicted),
                            torch.from_numpy(labels), 10)
    np.testing.assert_allclose(terms.detach(), expected_terms, rtol=1e-12, atol=1e-12)
    # balance is an intentionally float32 model buffer, even when converted to float64.
    expected_loss = expected_terms @ objective.balance.detach().numpy()
    assert loss.item() == pytest.approx(expected_loss, rel=1e-12)
    quadratic_scale_only = (np.mean((predicted - labels) ** 2 / (2 * squared[2]))
                            + np.log1p(squared[2]).sum())
    assert abs(supervision - quadratic_scale_only) > .01


def test_categorical_loss_masks_missing_labels_without_mutating_them():
    model = SupervisedComponents([2], 2, [3], dropout=0).double()
    objective = SupervisedObjective(1, [{"type": "categorical"}], [3]).double()
    images = [torch.tensor([[.3, .2], [.7, -.2], [-.1, .5]], dtype=torch.float64)]
    predictions = torch.tensor([[0., 1., 2.], [100., -200., 5.], [2., 0., -1.]],
                               dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[0.], [float("nan")], [2.]], dtype=torch.float64)
    original = labels.clone()
    loss, terms = objective(model, images, images, predictions, labels, 3)
    observed = predictions.detach().numpy()[[0, 2]]
    shifted = observed - observed.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    expected = -np.mean(log_probabilities[[0, 1], [0, 2]]) + np.log(2)
    assert terms[2].item() == pytest.approx(expected, rel=1e-12)
    loss.backward()
    torch.testing.assert_close(labels, original, equal_nan=True)
    assert torch.equal(predictions.grad[1], torch.zeros(3, dtype=torch.float64))
    assert torch.isfinite(predictions.grad).all()
    missing = torch.full_like(labels, float("nan"))
    _, missing_terms = objective(model, images, images, predictions, missing, 3)
    assert missing_terms[2].item() == 0


def _mixed_inputs(tmp_path):
    root = tmp_path / "subjects"
    root.mkdir()
    generator = np.random.default_rng(218)
    modalities, masks = {}, {}
    affine = np.array([[2., 0., 0., -6.], [0., 2., 0., 4.], [0., 0., 2., -2.], [0., 0., 0., 1.]])
    for name, shape, count in (("vbm", (2, 2, 2), 6), ("fa", (3, 2, 2), 5)):
        mask = np.zeros(np.prod(shape), dtype=np.uint8)
        mask[:count] = 1
        masks[name] = mask.reshape(shape)
        mask_path = tmp_path / f"{name}_mask.nii.gz"
        nib.save(nib.Nifti1Image(masks[name], affine), mask_path)
        modalities[name] = {"image": f"{name}.nii.gz", "mask": str(mask_path)}
    rows = []
    for index in range(13):
        subject_id = f"s{index:03d}"
        directory = root / subject_id
        directory.mkdir()
        for name, mask in masks.items():
            image = generator.normal(size=mask.shape).astype(np.float32)
            image += np.float32(.15 * index)
            nib.save(nib.Nifti1Image(image, affine), directory / f"{name}.nii.gz")
        split = "train" if index < 7 else "validation" if index < 10 else "test"
        rows.append([subject_id, str(22 + 1.7 * index), "ABC"[index % 3], split])
    table = tmp_path / "phenotypes.csv"
    _write_phenotypes(table, rows)
    return root, modalities, masks, affine, table, rows


def _write_phenotypes(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "score", "group", "split"])
        writer.writerows(rows)


def _fit(root, modalities, table, output):
    return run_superbigflica(root, modalities, table,
                            {"score": "continuous", "group": "categorical"}, output,
                            n_components=2, split_column="split", max_epochs=3,
                            batch_size=3, learning_rate=.001, dropout=.1,
                            random_state=42, device="cpu", feature_block=3, top_voxels=2)


def test_mixed_pipeline_reload_outputs_and_test_label_independence(tmp_path):
    root, modalities, masks, affine, table, rows = _mixed_inputs(tmp_path)
    output = _fit(root, modalities, table, tmp_path / "model")
    metadata = json.loads((output / "model.json").read_text())
    assert metadata["schema_version"] == 1 and metadata["method"] == "superbigflica"
    assert metadata["counts"] == {"train": 7, "validation": 3, "test": 3}
    assert metadata["output_sizes"] == [1, 3]
    assert metadata["dtype"] == "float32" and metadata["tf32"] is True
    assert metadata["targets"][1]["classes"] == ["A", "B", "C"]
    assert 1 <= metadata["best_epoch"] <= 3
    courses = np.load(output / "subj_course.npy")
    assert courses.shape == (13, 2) and courses.dtype == np.float32
    with (output / "predictions.csv").open() as stream:
        predicted = {row["subject_id"]: row for row in csv.DictReader(stream)}
    for index in (7, 12):
        subject_id = rows[index][0]
        result_path = tmp_path / f"{subject_id}_prediction.json"
        reloaded = apply_model(output, root / subject_id, device="cpu", output_file=result_path)
        np.testing.assert_allclose(reloaded["components"], courses[index], rtol=1e-5, atol=1e-6)
        assert reloaded["predictions"]["score"]["value"] == pytest.approx(
            float(predicted[subject_id]["score__prediction"]), rel=1e-5, abs=1e-5)
        group = reloaded["predictions"]["group"]
        assert group["label"] == predicted[subject_id]["group__prediction"]
        assert sum(group["probabilities"].values()) == pytest.approx(1., abs=1e-6)
        for class_index, label in enumerate(("A", "B", "C")):
            assert group["probabilities"][label] == pytest.approx(
                float(predicted[subject_id][f"group__prob_{class_index:03d}"]), rel=1e-5, abs=1e-6)
        assert json.loads(result_path.read_text()) == reloaded
    with pytest.raises(ValueError, match="absent from training"):
        apply_model(output, root / rows[0][0], device="cpu")
    for name, mask in masks.items():
        values = np.load(output / f"{name}_zstat.npy")
        assert values.shape == (int(mask.sum()), 2) and values.dtype == np.float32
        assert np.isfinite(values).all()
        for component in range(2):
            image = nib.load(output / "maps" / name / f"component-{component + 1:03d}_zstat.nii.gz")
            data = np.asarray(image.dataobj)
            assert data.dtype == np.float32 and image.shape == mask.shape
            np.testing.assert_array_equal(data[mask == 0], 0)
            np.testing.assert_allclose(image.affine, affine)
            np.testing.assert_array_equal(data[mask > 0], values[:, component])
    history = list(csv.DictReader((output / "history.csv").open()))
    assert metadata["best_validation_loss"] == min(float(row["validation_loss"]) for row in history)
    for row in rows:
        if row[3] == "test":
            row[1] = "10000"
            row[2] = {"A": "B", "B": "C", "C": "A"}[row[2]]
    _write_phenotypes(table, rows)
    second = _fit(root, modalities, table, tmp_path / "changed_test_labels")
    second_metadata = json.loads((second / "model.json").read_text())
    assert second_metadata["best_epoch"] == metadata["best_epoch"]
    assert second_metadata["best_validation_loss"] == metadata["best_validation_loss"]
    first_state = torch.load(output / "model.pt", weights_only=True)
    second_state = torch.load(second / "model.pt", weights_only=True)
    for section in ("model", "objective"):
        for name, value in first_state[section].items():
            torch.testing.assert_close(value, second_state[section][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="new, empty output_dir"):
        _fit(root, modalities, table, output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_training_frozen_prediction_and_spatial_maps_match_cpu_statistics(tmp_path):
    import h5py
    from fnit.bigflica.pipeline import _spatial_z

    root, modalities, masks, _, table, rows = _mixed_inputs(tmp_path)
    output = run_superbigflica(
        root, modalities, table, {'score': 'continuous', 'group': 'categorical'},
        tmp_path / 'cuda_model', n_components=2, split_column='split', max_epochs=2,
        batch_size=3, random_state=42, device='cuda:0', max_gpu_gb=1,
        feature_block=3, top_voxels=2)
    courses = np.load(output / 'subj_course.npy')
    predicted = apply_model(output, root / rows[12][0], device='cuda:0')
    np.testing.assert_allclose(predicted['components'], courses[12], rtol=2e-4, atol=2e-5)
    metadata = json.loads((output / 'model.json').read_text())
    assert metadata['peak_gpu_allocated_gib'] < 1
    for name in modalities:
        with h5py.File(output / 'input_store' / f'{name}.h5', 'r') as file:
            reference = _spatial_z(courses[:7].astype(np.float64),
                                    file['data'][:7].T.astype(np.float64))
        np.testing.assert_allclose(np.load(output / f'{name}_zstat.npy'), reference,
                                   rtol=1e-5, atol=1e-5)


def test_rank_deficient_courses_are_rejected_before_saving_model(tmp_path):
    root, modalities, masks, affine, table, rows = _mixed_inputs(tmp_path)
    for index, row in enumerate(rows):
        for name, mask in masks.items():
            # Every masked feature carries the same single subject direction.
            vector = (index + 1) * np.arange(1, int(mask.sum()) + 1, dtype=np.float32)
            image = np.zeros(mask.shape, dtype=np.float32)
            image[mask > 0] = vector
            nib.save(nib.Nifti1Image(image, affine), root / row[0] / f"{name}.nii.gz")
    destination = tmp_path / "rank_deficient"
    with pytest.raises(ValueError, match="rank deficient"):
        run_superbigflica(root, modalities, table,
                         {"score": "continuous", "group": "categorical"}, destination,
                         n_components=2, split_column="split", max_epochs=1,
                         batch_size=3, dropout=0, random_state=42, device="cpu",
                         feature_block=3, top_voxels=2)
    assert not (destination / "model.pt").exists()
    assert not (destination / "model.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_spatial_block_budget_clamps_and_matches_cpu_statistics(tmp_path):
    from fnit.bigflica.pipeline import _spatial_z
    import h5py

    root, modalities, masks, affine, table, rows = _mixed_inputs(tmp_path)
    backend = torch.device("cuda:0")
    torch.cuda.synchronize(backend)
    # Leave room for CUDA libraries and this tiny model while forcing block clamping.
    budget_gib = (torch.cuda.memory_allocated(backend) + 64 * 2 ** 20) / 2 ** 30
    requested_block = 1_000_000_000
    output = run_superbigflica(root, modalities, table,
                              {"score": "continuous", "group": "categorical"},
                              tmp_path / "cuda_model", n_components=2,
                              split_column="split", max_epochs=1, batch_size=3,
                              dropout=0, random_state=42, device="cuda:0",
                              max_gpu_gb=budget_gib, feature_block=requested_block, top_voxels=2)
    metadata = json.loads((output / "model.json").read_text())
    assert metadata["feature_block"] == requested_block
    assert 1 <= metadata["spatial_feature_block"] < requested_block
    assert 1 <= metadata["statistic_block"] < requested_block
    assert metadata["spatial_statistics_dtype"] == "float64"
    assert metadata["peak_gpu_allocated_gib"] <= budget_gib
    courses = np.load(output / "subj_course.npy")[:7].astype(np.float64)
    for name in modalities:
        with h5py.File(output / "input_store" / f"{name}.h5") as handle:
            expected = _spatial_z(courses, handle["data"][:7].T.astype(np.float64))
        np.testing.assert_allclose(np.load(output / f"{name}_zstat.npy"), expected,
                                   rtol=1e-4, atol=1e-4)
