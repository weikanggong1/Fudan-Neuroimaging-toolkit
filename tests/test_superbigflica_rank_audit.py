"""Reproductions for component-scale invariance; these are numerical unit tests."""

import numpy as np
import pytest
import torch

from fnit.bigflica.pipeline import _spatial_z
from fnit.bigflica.stats_torch import SpatialRegression, t_to_z_gpu
from fnit.superbigflica.pipeline import _statistics_courses


def _inputs():
    generator = np.random.default_rng(123)
    courses = generator.normal(size=(3000, 3)).astype(np.float32)
    images = generator.normal(size=(3000, 7))
    images += courses.astype(np.float64) @ generator.normal(scale=.2, size=(3, 7))
    return courses, images


@pytest.mark.parametrize("factor", [1e-5, 1e-8])
def test_independent_small_component_units_keep_rank_and_cpu_statistics(factor):
    courses, images = _inputs()
    scaled = courses.copy()
    scaled[:, 1] *= factor
    unchanged = scaled.copy()
    legacy_design = np.column_stack((scaled, np.ones(len(scaled), dtype=np.float32)))
    assert np.linalg.matrix_rank(legacy_design) < 4  # Reproduces the old float32 gate.
    reference = _statistics_courses(courses)
    normalized = _statistics_courses(scaled)
    assert normalized.dtype == np.float64
    np.testing.assert_array_equal(scaled, unchanged)  # Saved courses and model units stay untouched.
    np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(normalized.std(axis=0), 1, atol=1e-14)
    assert np.linalg.matrix_rank(np.column_stack((normalized, np.ones(len(normalized))))) == 4
    np.testing.assert_allclose(_spatial_z(normalized, images.T), _spatial_z(reference, images.T),
                               rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("kind", ["duplicate", "float32_roundoff", "constant"])
def test_true_or_source_precision_rank_deficiency_is_still_rejected(kind):
    courses, _ = _inputs()
    if kind == "duplicate":
        courses[:, 1] = courses[:, 0]
    elif kind == "float32_roundoff":
        courses[:, 1] = np.float32(2.3) * courses[:, 0]
        # Casting creates tiny independent-looking residuals that are numerical roundoff.
        assert np.linalg.matrix_rank(np.column_stack((courses.astype(np.float64),
                                                      np.ones(len(courses))))) == 4
    else:
        courses[:, 1] = 1
    with pytest.raises(ValueError, match="rank deficient"):
        _statistics_courses(courses)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("factor", [1e-5, 1e-8])
def test_small_component_units_keep_gpu_statistics_and_match_cpu(factor):
    courses, images = _inputs()
    reference = _statistics_courses(courses)
    courses[:, 1] *= factor
    normalized = _statistics_courses(courses)
    baseline_regression = SpatialRegression(reference, device="cuda:0")
    regression = SpatialRegression(normalized, device="cuda:0")
    baseline = t_to_z_gpu(baseline_regression.t(images.T), baseline_regression.df, device="cuda:0")
    actual = t_to_z_gpu(regression.t(images.T), regression.df, device="cuda:0")
    np.testing.assert_allclose(actual, baseline, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(actual, _spatial_z(normalized, images.T), rtol=3e-6, atol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("df", [1, 50, 2979, 39979])
def test_gpu_t_to_z_matches_cpu_in_normal_and_underflow_tail(df):
    from scipy.stats import norm, t as student_t

    positive = np.array([.01, .1, 1., 5., 20., 100., 1000.])
    values = np.concatenate((-positive[::-1], [0.], positive))
    probability = np.clip(2 * student_t.sf(np.abs(values), df),
                          np.finfo(np.float64).tiny, 1.)
    reference = np.sign(values) * norm.isf(probability / 2)
    actual = t_to_z_gpu(values, df, device="cuda:0")
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=3e-6)
    if df >= 2979:
        assert probability[-1] == np.finfo(np.float64).tiny
        # The former GPU 1e-300 probability floor clipped this to z=37.06579.
        assert actual[-1] == pytest.approx(reference[-1], abs=3e-6)


def test_public_fit_and_apply_remain_float32_with_float64_global_default(tmp_path):
    import csv
    import json

    import nibabel as nib

    from fnit.superbigflica import apply_model, run_superbigflica

    generator = np.random.default_rng(218)
    root = tmp_path / "subjects"
    root.mkdir()
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), mask)
    table = tmp_path / "phenotypes.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "score", "group", "split"])
        for index in range(13):
            directory = root / f"s{index:03d}"
            directory.mkdir()
            image = generator.normal(size=(2, 2, 2)).astype(np.float32)
            nib.save(nib.Nifti1Image(image, np.eye(4)), directory / "image.nii.gz")
            split = "train" if index < 7 else "validation" if index < 10 else "test"
            writer.writerow([directory.name, 22 + 1.7 * index, "ABC"[index % 3], split])
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        output = run_superbigflica(
            root, {"image": {"image": "image.nii.gz", "mask": str(mask)}}, table,
            {"score": "continuous", "group": "multiclass"}, tmp_path / "model",
            n_components=2, split_column="split", max_epochs=1, batch_size=3,
            dropout=0, random_state=42, device="cpu", feature_block=3,
            top_voxels=2, make_plots=False)
        state = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        for group in state.values():
            assert all(value.dtype == torch.float32 for value in group.values()
                       if value.is_floating_point())
        courses = np.load(output / "subj_course.npy")
        assert courses.dtype == np.float32
        reloaded = apply_model(output, root / "s012", device="cpu")
        np.testing.assert_allclose(reloaded["components"], courses[12], rtol=1e-5, atol=1e-6)
        with (output / "predictions.csv").open(encoding="utf-8") as stream:
            predictions = {row["subject_id"]: row for row in csv.DictReader(stream)}
        assert reloaded["predictions"]["score"]["value"] == pytest.approx(
            float(predictions["s012"]["score__prediction"]), rel=1e-5, abs=1e-5)
        assert json.loads((output / "model.json").read_text())["dtype"] == "float32"
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(default_dtype)
