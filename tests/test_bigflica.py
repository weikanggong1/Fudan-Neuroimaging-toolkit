import json

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy.stats import norm, t as student_t
from sklearn.decomposition import sparse_encode

from fnit.bigflica import apply_model
from fnit.bigflica.dicl_torch import _sparse_codes_lars
from fnit.bigflica.pipeline import _read_vector
from fnit.bigflica.stats_torch import t_to_z_gpu


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


def test_gpu_t_to_z_matches_student_distribution():
    t_values = np.array([-10., -2., 0., 1., 5., 10.])
    expected = np.sign(t_values) * norm.isf(student_t.sf(np.abs(t_values), 14))
    actual = t_to_z_gpu(t_values, 14, device="cpu")
    np.testing.assert_allclose(actual, expected, atol=1e-6)
