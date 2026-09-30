"""Reproduce geometry and class-label collisions in saved-model plots."""

import nibabel as nib
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fnit.superbigflica import plotting


def test_brain_slices_preserve_physical_aspect_on_anisotropic_grid(tmp_path, monkeypatch):
    shape = (3, 5, 7)
    affine = np.diag([2., 3., 5., 1.])
    mask = np.ones(shape, dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask, affine), tmp_path / 'vbm_mask.nii.gz')
    np.save(tmp_path / 'vbm_mean.npy', np.arange(np.prod(shape), dtype=np.float32) + 1)
    maps = tmp_path / 'maps' / 'vbm'
    maps.mkdir(parents=True)
    values = np.zeros(shape, dtype=np.float32)
    values[1, 2, 3] = 4
    nib.save(nib.Nifti1Image(values, affine), maps / 'component-001_top-1.nii.gz')
    metadata = {'modalities': {'vbm': {'mask': 'vbm_mask.nii.gz', 'n_features': mask.size}},
                'top_voxels': 1}
    captured = []
    monkeypatch.setattr(plotting, '_save', lambda figure, destination: captured.append(figure))
    plotting._brain_figure(tmp_path, metadata, np.array([0]), np.array([.5]),
                           'training_signal', 'score', tmp_path / 'brain')
    figure = captured[0]
    image_axes = [axis for axis in figure.axes if len(axis.images) == 2]
    assert len(image_axes) == 3
    # After rotation, each plane's height and width must retain its dimensions in mm.
    physical_ratios = [(7 * 5) / (5 * 3), (7 * 5) / (3 * 2), (5 * 3) / (3 * 2)]
    actual_ratios = [axis.get_aspect() * axis.images[0].get_array().shape[0]
                     / axis.images[0].get_array().shape[1] for axis in image_axes]
    import matplotlib.pyplot as plt
    plt.close(figure)
    np.testing.assert_allclose(actual_ratios, physical_ratios)


def test_multiclass_roc_preserves_labels_that_match_aggregate_names(tmp_path):
    observed = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.array([[.1, .7, .2], [.4, .4, .2], [.2, .5, .3],
                              [.5, .3, .2], [.1, .2, .7], [.6, .2, .2]])
    classes = ['macro_auc', 'micro_auc', 'other']
    result = plotting._roc_figure(observed, probabilities, classes, 'groups', tmp_path / 'roc')
    expected = {label: roc_auc_score(observed == index, probabilities[:, index])
                for index, label in enumerate(classes)}
    assert result['class_auc'] == pytest.approx(expected)
    assert result['macro_auc'] == pytest.approx(np.mean(list(expected.values())))
    assert result['micro_auc'] == pytest.approx(roc_auc_score(
        np.eye(len(classes))[observed].ravel(), probabilities.ravel()))
