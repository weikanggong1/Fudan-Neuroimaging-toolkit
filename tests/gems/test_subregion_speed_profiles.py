"""Fast fitting retains the fine-grid model, masks and all smoothing stages."""
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems.context import SubregionContext
from fnit.gems.recipes import HippoAmygdalaRecipe, ThalamusRecipe
from fnit.gems.recipes.base import GEMSRecipe


def _atlas():
    points = np.asarray([[3, 3, 3], [19, 3, 3], [3, 19, 3], [3, 3, 19]], float)
    return GEMSAtlas(points, points.copy(), np.asarray([[0, 1, 2, 3]]),
                     np.full((4, 3), 1 / 3, np.float32), .05,
                     np.ones((4, 3), bool), np.asarray([0, 8109, 8226]),
                     ("Unknown", "Left-CL", "Right-MDm"))


def _world(points, affine):
    return points @ affine[:3, :3].T + affine[:3, 3]


def test_fast_retains_fine_input_mesh_and_all_thalamic_groups(tmp_path):
    class Recipe(ThalamusRecipe):
        def prepare_working_image(self, context):
            return context.image, context.coarse_segmentation, {}
        def _fit(self, atlas, data, affine, classes, schedule, **kwargs):
            self.calls.append((atlas, data, affine, classes, schedule, kwargs))
            return atlas, SimpleNamespace(affine=affine)
    affine = np.asarray([[0, -.5, 0, 12], [.5, 0, 0, -3], [0, 0, .5, 7], [0, 0, 0, 1.]])
    data = np.arange(30 * 28 * 26, dtype=np.float32).reshape(30, 28, 26)
    image = nib.Nifti1Image(data, affine)
    context = SubregionContext(image, data, np.full(image.shape, 10, np.int32), None, None)
    recipe = Recipe("thalamus", tmp_path)
    recipe.calls = []
    recipe.set_optimization_profile("fast")
    atlas = _atlas()
    _, output_image, report = recipe.fit_intensity_mesh(atlas, context, torch.device("cpu"))
    assert len(recipe.calls) == 1 and output_image is image
    fitted, values, matrix, classes, schedule, options = recipe.calls[0]
    np.testing.assert_array_equal(values, data)
    np.testing.assert_allclose(matrix, affine)
    np.testing.assert_allclose(_world(fitted.vertices, matrix), _world(atlas.vertices, affine))
    np.testing.assert_allclose(fitted.reference_vertices, atlas.reference_vertices)
    np.testing.assert_array_equal(classes, [0, 13, 13])
    np.testing.assert_array_equal(recipe.intensity_groups(fitted, 1), [0, 13, 14])
    assert schedule == recipe.image_schedule == ((1.5, 7), (1.125, 5), (.75, 5), (0., 3))
    assert options["stage_offset"] == 0 and options["mesh_iterations"] == 20
    assert report["outer_em_iterations"] == [7, 5, 5, 3]
    assert "coarse_working_image" not in report


@pytest.mark.parametrize("resolution", [.5, .33333])
def test_real_crop_keeps_world_geometry_hyperprior_and_stage_statistics(tmp_path, monkeypatch, resolution):
    import fnit.gems.recipes.base as base
    import fnit.gems.smoothing as smoothing
    calls = []
    class Recipe(GEMSRecipe):
        resolution_mm = resolution
        image_schedule = ((1.5, 7), (.75, 5), (0., 3))
        em_iterations = 1
        def prepare_working_image(self, context):
            return context.image, context.coarse_segmentation, {}
        def intensity_groups(self, atlas, stage):
            return np.arange(3)
        def gaussian_hyperparameters(self, context, atlas, classes):
            return np.asarray([20, 40, 60], np.float32), np.asarray([10, 50, 25], np.float32)
    class Engine:
        def __init__(self, atlas, **kwargs):
            self.atlas = atlas
            assert kwargs["block_size"] == 8
        def __call__(self, image, **kwargs):
            displacement = np.linalg.solve(affine[:3, :3], [.2, 0., 0.]) if not calls else np.zeros(3)
            fit = SimpleNamespace(vertices=torch.as_tensor(self.atlas.vertices + displacement),
                                  gaussian_parameters=None,
                                  optimization_stats={"mesh_evaluations": 3, "mesh_steps": 2})
            calls.append((self.atlas, fit, kwargs))
            return fit
    monkeypatch.setattr(base, "TorchGEMS", Engine)
    sigma_calls = []
    def smooth(atlas, classes, sigma, **kwargs):
        sigma_calls.append(sigma)
        return atlas.alphas
    monkeypatch.setattr(smoothing, "smooth_atlas_alphas", smooth)
    affine = np.asarray([[0, -resolution, 0, 12], [resolution, 0, 0, -3],
                         [0, 0, resolution, 7], [0, 0, 0, 1.]])
    image = nib.Nifti1Image(np.ones((80, 74, 72), np.float32), affine)
    context = SubregionContext(image, np.asarray(image.dataobj), np.full(image.shape, 10, np.int32), None, None)
    shift = np.eye(4); shift[:3, 3] = 24
    atlas = _atlas().transformed(shift)
    recipe = Recipe("test", tmp_path)
    recipe._reference_vertices = atlas.reference_vertices.copy()
    recipe.set_optimization_profile("fast")
    fit, _, _ = recipe.fit_intensity_mesh(atlas, context, torch.device("cpu"))
    assert np.all((np.linalg.inv(affine) @ fit.affine)[:3, 3] > 0)
    np.testing.assert_allclose(_world(fit.vertices.numpy(), fit.affine),
                               _world(atlas.vertices, affine) + [.2, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(_world(calls[-1][0].reference_vertices, fit.affine),
                               _world(atlas.reference_vertices, affine), atol=1e-12)
    np.testing.assert_array_equal(calls[0][2]["n_hyper"], [10, 50, 25])
    assert sigma_calls == [1.5, .75]
    assert [call[2]["outer_iterations"] for call in calls] == [7, 5, 3]
    assert all(call[2]["fit_alpha_stages"][0][1] == 20 for call in calls)
    assert all(call[2]["deformation_stop"] == .005 and call[2]["cost_stop_patience"] == 3 for call in calls)
    stats = fit.optimization_stats
    assert stats["mesh_evaluations"] == 9 and stats["mesh_steps"] == 6
    assert [stage["stage_index"] for stage in stats["stages"]] == [1, 2, 3]
    assert [stage["resolution_mm"] for stage in stats["stages"]] == [resolution] * 3
    np.testing.assert_allclose(stats["total_seconds"], stats["preparation_seconds"] + stats["gems_fit_seconds"] + stats["post_fit_seconds"])


def test_balanced_profile_preserves_existing_defaults(tmp_path):
    thalamus = ThalamusRecipe("thalamus", tmp_path)
    hippo = HippoAmygdalaRecipe("left", tmp_path)
    assert thalamus.optimization_profile == hippo.optimization_profile == "balanced"
    assert thalamus.image_schedule == ((1.5, 7), (1.125, 5), (.75, 5), (0., 3))
    assert hippo.image_schedule == ((1.5, 7), (.75, 5), (0., 3))
    assert thalamus.mesh_iterations == hippo.mesh_iterations == 30
    assert thalamus.fast_mesh_iterations == hippo.fast_mesh_iterations == 20
    with pytest.raises(ValueError, match="optimization profile"):
        hippo.set_optimization_profile("invalid")
