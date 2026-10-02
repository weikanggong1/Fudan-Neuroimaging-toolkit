"""Keep stable defaults and permit a single validation recipe override."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems.context import SubregionContext
from fnit.gems.recipes.base import GEMSRecipe


@pytest.mark.parametrize("name,override", [
    ("thalamus", None), ("hippo-amygdala-left", None),
    ("hippo-amygdala-right", None), ("brainstem", None),
    ("hippo-amygdala-right", False),
])
@pytest.mark.parametrize("synthetic", [False, True])
def test_stability_controls_preserve_defaults_and_allow_recipe_override(
        tmp_path, monkeypatch, name, override, synthetic):
    import fnit.gems.recipes.base as base
    import fnit.gems.smoothing as smoothing
    points = np.asarray([[3, 3, 3], [19, 3, 3], [3, 19, 3], [3, 3, 19]], float)
    atlas = GEMSAtlas(points, points.copy(), np.asarray([[0, 1, 2, 3]]),
                      np.full((4, 2), .5, np.float32), .05,
                      np.ones((4, 3), bool), np.asarray([0, 10]), ("Unknown", "ROI"))
    classes = np.asarray([0, 1])
    engine_options, smoothing_options = [], []

    class Recipe(GEMSRecipe):
        def intensity_groups(self, atlas, stage):
            return classes

        def synthetic_means(self, atlas, classes):
            return np.asarray([1, 2], np.float32)

        def gaussian_hyperparameters(self, context, atlas, classes):
            return np.asarray([1, 2], np.float32), np.ones(2, np.float32)

    class Engine:
        def __init__(self, atlas, **kwargs):
            self.atlas = atlas

        def __call__(self, image, **kwargs):
            engine_options.append(kwargs)
            return SimpleNamespace(vertices=torch.as_tensor(self.atlas.vertices, dtype=torch.float32),
                                   gaussian_parameters=None, optimization_stats={
                                       key: kwargs[key] for key in
                                       ("stable_mesh_fitting", "precise_mesh_matrices",
                                        "mesh_line_search")})

    def smooth(atlas, classes, sigma, **kwargs):
        smoothing_options.append(kwargs)
        return atlas.alphas

    monkeypatch.setattr(base, "TorchGEMS", Engine)
    monkeypatch.setattr(smoothing, "smooth_atlas_alphas", smooth)
    data = np.ones((28, 28, 28), np.float32)
    affine = np.eye(4)
    context = SubregionContext(nib.Nifti1Image(data, affine), data,
                              np.full(data.shape, 10, np.int32), None, None)
    recipe = Recipe(name, tmp_path)
    recipe._reference_vertices = atlas.reference_vertices.copy()
    recipe.set_optimization_profile("fast")
    default_stable = name == "thalamus" or name.startswith("hippo-amygdala-")
    assert recipe.stable_mesh_fitting == default_stable
    if override is not None:
        recipe.stable_mesh_fitting = override
    _, result = recipe._fit(atlas, data, affine, classes, ((1., 2), (0., 2)),
                synthetic=synthetic, context=context, device=torch.device("cpu"))
    expected = default_stable if override is None else override
    assert len(engine_options) == 2 and len(smoothing_options) == 1
    assert all(options["stable_mesh_fitting"] == expected for options in engine_options)
    assert all(options["precise_mesh_matrices"] == expected for options in engine_options)
    assert all(options["mesh_line_search"] == ("backtracking" if expected else "strong_wolfe")
               for options in engine_options)
    assert smoothing_options[0]["stable_vertex_statistics"] == expected

    assert all(options["double_data_cost_accumulation"] == (expected or not synthetic)
               for options in engine_options)
    stages = result.optimization_stats["stages"]
    assert [stage["synthetic"] for stage in stages] == [synthetic, synthetic]
    assert [stage["stable_vertex_statistics"] for stage in stages] == [expected, False]
    assert all(stage["stable_mesh_fitting"] == expected for stage in stages)
    assert all(stage["precise_mesh_matrices"] == expected for stage in stages)
    assert all(stage["mesh_line_search"] == ("backtracking" if expected else "strong_wolfe")
               for stage in stages)
