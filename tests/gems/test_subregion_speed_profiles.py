from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems.context import SubregionContext
from fnit.gems.deformation import ashburner_prior
from fnit.gems.recipes import HippoAmygdalaRecipe, ThalamusRecipe
from fnit.gems.recipes.base import (GEMSRecipe, coarsen_working_image,
                                    physical_neighborhood, spherical_neighborhood)


def _atlas():
    reference = np.asarray([[3, 3, 3], [19, 3, 3], [3, 19, 3], [3, 3, 19]], float)
    vertices = reference.copy()
    vertices[1] += [1, .2, -.1]
    return GEMSAtlas(reference, vertices, np.asarray([[0, 1, 2, 3]]),
                     np.full((4, 3), 1 / 3, np.float32), .05,
                     np.ones((4, 3), bool), np.asarray([0, 8109, 8226]),
                     ("Unknown", "Left-CL", "Right-MDm"))


def _world(points, affine):
    return points @ affine[:3, :3].T + affine[:3, 3]


def test_coarse_grid_keeps_physical_centre_orientation_and_constant_intensity():
    affine = np.asarray([[0, -.5, 0, 12], [.5, 0, 0, -3], [0, 0, .5, 7], [0, 0, 0, 1.]])
    image = nib.Nifti1Image(np.full((17, 21, 15), 83, np.float32), affine)
    coarse = coarsen_working_image(image, 1.)
    np.testing.assert_allclose(np.linalg.norm(coarse.affine[:3, :3], axis=0), 1.)
    np.testing.assert_allclose(coarse.affine[:3, :3] / 1., affine[:3, :3] / .5)
    np.testing.assert_allclose(
        _world((np.asarray(image.shape) - 1)[None] / 2, image.affine),
        _world((np.asarray(coarse.shape) - 1)[None] / 2, coarse.affine))
    np.testing.assert_allclose(np.asarray(coarse.dataobj), 83)
    with pytest.raises(ValueError, match="must not be finer"):
        coarsen_working_image(image, .25)


def test_grid_roundtrip_preserves_vertices_reference_and_physical_prior_mass():
    fine_affine = np.diag([.5, .5, .5, 1.])
    fine_affine[:3, 3] = [8, -4, 3]
    image = nib.Nifti1Image(np.ones((30, 28, 26), np.float32), fine_affine)
    coarse = coarsen_working_image(image, 1.)
    atlas = _atlas()
    mapped = atlas.transformed(np.linalg.inv(coarse.affine) @ fine_affine,
                                transform_reference=True)
    np.testing.assert_allclose(_world(mapped.vertices, coarse.affine),
                               _world(atlas.vertices, fine_affine))
    np.testing.assert_allclose(_world(mapped.reference_vertices, coarse.affine),
                               _world(atlas.reference_vertices, fine_affine))
    restored = mapped.transformed(np.linalg.inv(fine_affine) @ coarse.affine)
    np.testing.assert_allclose(restored.vertices, atlas.vertices)
    np.testing.assert_allclose(restored.reference_vertices, atlas.reference_vertices)
    tetrahedra = torch.as_tensor(atlas.tetrahedra)
    def energy(current):
        return ashburner_prior(torch.as_tensor(current.vertices),
                               torch.as_tensor(current.reference_vertices),
                               tetrahedra, current.stiffness)[0].item()
    np.testing.assert_allclose(energy(atlas) * abs(np.linalg.det(fine_affine[:3, :3])),
                               energy(mapped) * abs(np.linalg.det(coarse.affine[:3, :3])),
                               rtol=1e-10)


def test_erosion_radius_has_the_same_physical_extent():
    np.testing.assert_array_equal(physical_neighborhood(2.5, [.5] * 3),
                                  spherical_neighborhood(5))
    coarse = physical_neighborhood(2.5, [1.] * 3)
    points = np.argwhere(coarse) - (np.asarray(coarse.shape) - 1) / 2
    assert np.max(np.linalg.norm(points, axis=1)) <= 2.5
    assert coarse[3, 3, 3] and not coarse[0, 3, 3]


def test_fast_uses_coarse_then_original_grid_and_keeps_thalamic_second_component(tmp_path):
    class Recipe(ThalamusRecipe):
        def prepare_working_image(self, context):
            return context.image, context.coarse_segmentation, {}

        def _fit(self, atlas, data, affine, classes, schedule, **kwargs):
            self.calls.append((atlas, affine, classes, schedule, kwargs))
            if len(self.calls) == 1:
                # A 0.2-mm translation learned on the coarse grid must survive
                # conversion to the fine grid, without moving the reference.
                displacement = np.linalg.solve(affine[:3, :3], [.2, 0., 0.])
                atlas = atlas.with_vertices(atlas.vertices + displacement)
            return atlas, SimpleNamespace(affine=affine)

    image = nib.Nifti1Image(np.ones((30, 28, 26), np.float32), np.diag([.5, .5, .5, 1.]))
    context = SubregionContext(image, np.asarray(image.dataobj),
                               np.full(image.shape, 10, np.int32), None, None)
    recipe = Recipe("thalamus", tmp_path)
    recipe.calls = []
    recipe.set_optimization_profile("fast")
    atlas = _atlas()
    # The input atlas to this method is on the native/context grid.
    _, output_image, report = recipe.fit_intensity_mesh(atlas, context, torch.device("cpu"))
    assert output_image is image and len(recipe.calls) == 2
    coarse, fine = recipe.calls
    np.testing.assert_allclose(np.linalg.norm(coarse[1][:3, :3], axis=0), 1.)
    np.testing.assert_allclose(fine[1], image.affine)
    np.testing.assert_allclose(_world(fine[0].vertices, fine[1]),
                               _world(atlas.vertices, image.affine) + [.2, 0., 0.])
    np.testing.assert_allclose(_world(fine[0].reference_vertices, fine[1]),
                               _world(atlas.reference_vertices, image.affine))
    assert list(coarse[2]) == [0, 13, 13]
    assert list(fine[2]) == [0, 13, 13]
    assert coarse[3] == ((1.5, 2),)
    assert coarse[4]["mesh_iterations"] == 8
    assert coarse[4]["warm_start"] is True
    assert fine[3] == recipe.fast_image_schedule
    assert fine[4]["stage_offset"] == 0
    assert report["outer_em_iterations"] == [3, 3, 2, 2]
    assert report["mesh_iterations_per_outer"] == 12
    assert report["coarse_working_image"]["hyper_count_scale"] == .125
    np.testing.assert_array_equal(fine[0].label_ids, atlas.label_ids)


def test_coarse_fit_preserves_hyperprior_mass_and_smoothing_width(tmp_path, monkeypatch):
    import fnit.gems.recipes.base as base
    import fnit.gems.smoothing as smoothing
    captured = {}
    class Recipe(GEMSRecipe):
        resolution_mm = .5
        em_iterations = 1
        def intensity_groups(self, atlas, stage):
            return np.arange(3)
        def gaussian_hyperparameters(self, context, atlas, classes):
            captured["hyper_vertices"] = atlas.vertices.copy()
            return np.asarray([20, 40, 60], np.float32), np.asarray([10, 50, 25], np.float32)
    class Engine:
        def __init__(self, atlas, **kwargs):
            self.atlas = atlas
        def __call__(self, image, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(vertices=torch.as_tensor(self.atlas.vertices), gaussian_parameters=None)
    def smooth(atlas, classes, sigma, **kwargs):
        captured["sigma"] = sigma
        return atlas.alphas
    monkeypatch.setattr(base, "TorchGEMS", Engine)
    monkeypatch.setattr(smoothing, "smooth_atlas_alphas", smooth)
    recipe = Recipe("test", tmp_path)
    recipe.set_optimization_profile("fast")
    atlas = _atlas()
    recipe._reference_vertices = atlas.reference_vertices.copy()
    recipe._fit(atlas, np.ones((25, 25, 25), np.float32), np.eye(4), np.arange(3),
                ((1.5, 1),), synthetic=False, device=torch.device("cpu"),
                working_resolution_mm=1., mesh_iterations=12)
    # Count times voxel volume is invariant, including pseudo-observations.
    np.testing.assert_allclose(captured["n_hyper"].numpy() * 1.**3,
                               np.asarray([10, 50, 25]) * .5**3)
    assert captured["sigma"] * 1. == 1.5 * .5
    np.testing.assert_allclose(np.ptp(captured["hyper_vertices"], axis=0) * .5,
                               np.ptp(atlas.vertices, axis=0) * 1.)
    assert captured["fit_alpha_stages"][0][1] == 12
    assert captured["deformation_stop"] == .005
    assert captured["cost_stop_patience"] == 3
    assert captured["outer_relative_cost_stop"] == 1e-5


def test_actual_nonzero_coarse_crop_restores_current_and_reference_world_coordinates(tmp_path, monkeypatch):
    import fnit.gems.recipes.base as base
    import fnit.gems.smoothing as smoothing
    calls = []
    class Recipe(ThalamusRecipe):
        def prepare_working_image(self, context):
            return context.image, context.coarse_segmentation, {}
        def gaussian_hyperparameters(self, context, atlas, classes):
            count = int(classes.max()) + 1
            return np.full(count, 80, np.float32), np.full(count, 10, np.float32)
    class Engine:
        def __init__(self, atlas, **kwargs):
            self.atlas = atlas
        def __call__(self, image, **kwargs):
            # _fit performs the actual crop/uncrop. Only the costly solver is
            # replaced by a known displacement on the first (1-mm) grid.
            displacement = [.2, 0., 0.] if not calls else [0., 0., 0.]
            result = SimpleNamespace(vertices=torch.as_tensor(self.atlas.vertices + displacement),
                                     gaussian_parameters=None,
                                     optimization_stats={"mesh_evaluations": len(calls) + 1,
                                                         "mesh_steps": 2})
            calls.append((self.atlas, result))
            return result
    monkeypatch.setattr(base, "TorchGEMS", Engine)
    monkeypatch.setattr(smoothing, "smooth_atlas_alphas", lambda atlas, *args, **kwargs: atlas.alphas)
    affine = np.diag([.5, .5, .5, 1.])
    affine[:3, 3] = [8, -4, 3]
    image = nib.Nifti1Image(np.ones((80, 74, 72), np.float32), affine)
    context = SubregionContext(image, np.asarray(image.dataobj),
                               np.full(image.shape, 10, np.int32), None, None)
    atlas = _atlas().transformed(np.asarray([[1, 0, 0, 24], [0, 1, 0, 24],
                                            [0, 0, 1, 24], [0, 0, 0, 1.]]))
    recipe = Recipe("thalamus", tmp_path)
    recipe._reference_vertices = atlas.reference_vertices.copy()
    recipe.set_optimization_profile("fast")
    fit, _, _ = recipe.fit_intensity_mesh(atlas, context, torch.device("cpu"))
    coarse_image = coarsen_working_image(image, 1.)
    crop_transform = np.linalg.inv(coarse_image.affine) @ calls[0][1].affine
    assert np.all(crop_transform[:3, 3] > 0)
    np.testing.assert_allclose(_world(fit.vertices.numpy(), fit.affine),
                               _world(atlas.vertices, affine) + [.2, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(_world(calls[-1][0].reference_vertices, fit.affine),
                               _world(atlas.reference_vertices, affine), atol=1e-12)
    stats = fit.optimization_stats
    assert stats["mesh_evaluations"] == 15 and stats["mesh_steps"] == 10
    assert [stage["stage_index"] for stage in stats["stages"]] == [1, 1, 2, 3, 4]
    assert [stage["warm_start"] for stage in stats["stages"]] == [True, False, False, False, False]
    assert [stage["block_size"] for stage in stats["stages"]] == [4, 8, 8, 8, 8]
    assert [stage["resolution_mm"] for stage in stats["stages"]] == [1., .5, .5, .5, .5]
    np.testing.assert_allclose(stats["total_seconds"], stats["preparation_seconds"] +
                               stats["gems_fit_seconds"] + stats["post_fit_seconds"])


def test_balanced_profile_preserves_existing_defaults(tmp_path):
    thalamus = ThalamusRecipe("thalamus", tmp_path)
    hippo = HippoAmygdalaRecipe("left", tmp_path)
    assert thalamus.optimization_profile == hippo.optimization_profile == "balanced"
    assert thalamus.image_schedule == ((1.5, 7), (1.125, 5), (.75, 5), (0., 3))
    assert hippo.image_schedule == ((1.5, 7), (.75, 5), (0., 3))
    assert thalamus.mesh_iterations == hippo.mesh_iterations == 30
    with pytest.raises(ValueError, match="optimization profile"):
        hippo.set_optimization_profile("invalid")
