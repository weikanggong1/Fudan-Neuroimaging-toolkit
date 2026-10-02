"""Brainstem integration checks; these fixtures are not performance benchmarks."""

import numpy as np
import pytest
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems import brainstem


def _atlas_and_coarse():
    vertices = np.asarray([[x, y, z] for x in (3.13, 20.13)
                           for y in (3.21, 20.21) for z in (3.31, 20.31)])
    tetrahedra = np.asarray([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
                            [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])
    foreground = np.asarray([.15, .25, .35, .45, .55, .65, .75, .85], np.float32)
    movable = np.ones_like(vertices, dtype=bool)
    movable[0, 0] = False
    atlas = GEMSAtlas(vertices, vertices.copy(), tetrahedra,
                      np.stack((foreground, 1 - foreground), 1), .05, movable,
                      np.asarray([173, 0]), ("Midbrain", "Unknown"))
    coarse = np.full((25, 25, 25), 2, np.int32)
    coarse[8:18, 8:18, 8:18] = 16
    return atlas, coarse


@pytest.mark.parametrize("optimizer_name", ("adam", "lbfgs"))
def test_brainstem_compact_fit_preserves_dense_objective_and_fixed_steps(monkeypatch,
                                                                       optimizer_name):
    atlas, coarse = _atlas_and_coarse()
    actual, report = brainstem.fit_brainstem_segmentation(
        atlas, coarse, device="cpu", iterations=4, optimizer_name=optimizer_name)
    assert report["mesh_solver"]["valid_voxels"] > 0
    assert report["mesh_solver"]["mesh_steps"] == 4
    assert report["mesh_solver"]["compact"]
    assert report["mesh_solver"]["shared_geometry"]
    assert report["mesh_solver"]["analytic_prior"]
    assert report["min_jacobian"] > 0
    if optimizer_name == "lbfgs":
        assert report["mesh_solver"]["accepted_cache_hits"] > 0

    dense_raster = brainstem.rasterize_priors
    reference_prior = brainstem.ashburner_prior

    def dense_masked(vertices, tetrahedra, alphas, shape, *, valid_mask, **kwargs):
        kwargs.pop("current_geometry", None)
        priors, covered = dense_raster(vertices, tetrahedra, alphas, shape, **kwargs)
        return priors[:, valid_mask], covered[valid_mask]

    def autograd_prior(vertices, reference, tetrahedra, stiffness, **kwargs):
        return reference_prior(vertices, reference, tetrahedra, stiffness)

    # Run the former dense objective, autograd prior and uncached optimizer
    # with the same integration path, erosion mask, target and mobility flags.
    monkeypatch.setattr(brainstem, "rasterize_priors_compact", dense_masked)
    monkeypatch.setattr(brainstem, "ashburner_prior", autograd_prior)
    monkeypatch.setattr(brainstem, "CachedLBFGS", torch.optim.LBFGS)
    expected, expected_report = brainstem.fit_brainstem_segmentation(
        atlas, coarse, device="cpu", iterations=4, optimizer_name=optimizer_name)
    np.testing.assert_allclose(actual.vertices, expected.vertices, atol=1e-4, rtol=1e-5)
    assert report["mask_dice"] == pytest.approx(expected_report["mask_dice"], abs=1e-6)
    assert report["min_jacobian"] == pytest.approx(expected_report["min_jacobian"], abs=1e-5)
    assert actual.vertices[0, 0] == float(np.float32(atlas.reference_vertices[0, 0]))


def test_brainstem_compact_empty_interior_remains_finite():
    atlas, coarse = _atlas_and_coarse()
    coarse = coarse[:7, :7, :7]
    fitted, report = brainstem.fit_brainstem_segmentation(
        atlas, coarse, device="cpu", iterations=0)
    assert np.isfinite(fitted.vertices).all()
    assert np.isfinite(report["mask_dice"])
    assert report["mesh_solver"]["mesh_steps"] == 0


@pytest.mark.parametrize("optimizer_name", ("adam", "lbfgs"))
def test_stable_brainstem_preserves_objective_mask_budget_and_mobility(monkeypatch,
                                                                    optimizer_name):
    atlas, coarse = _atlas_and_coarse()
    baseline, baseline_report = brainstem.fit_brainstem_segmentation(
        atlas, coarse, device="cpu", iterations=4, optimizer_name=optimizer_name)
    original_geometry = brainstem.prepare_current_geometry
    original_prior = brainstem.ashburner_prior
    ordered_layouts, prior_dtypes = [], []

    def observed_geometry(vertices, tetrahedra, **options):
        assert vertices.dtype == torch.float32
        if torch.is_grad_enabled():
            assert options["deterministic_gradient"] is True
            ordered_layouts.append(options["vertex_reduction"])
        return original_geometry(vertices, tetrahedra, **options)

    def observed_prior(*args, **options):
        result = original_prior(*args, **options)
        if torch.is_grad_enabled():
            assert options["double_accumulation"] is True
            prior_dtypes.append(result[0].dtype)
        return result

    monkeypatch.setattr(brainstem, "prepare_current_geometry", observed_geometry)
    monkeypatch.setattr(brainstem, "ashburner_prior", observed_prior)
    actual, report = brainstem.fit_brainstem_segmentation(
        atlas, coarse, device="cpu", iterations=4, optimizer_name=optimizer_name,
        stable_mesh_fitting=True)
    stats = report["mesh_solver"]
    assert ordered_layouts and len({id(layout) for layout in ordered_layouts}) == 1
    assert set(prior_dtypes) == {torch.float64}
    assert stats["stable_mesh_fitting"] and stats["ordered_point_to_tetrahedron_gradient"]
    assert stats["data_cost_accumulation"] == stats["prior_cost_accumulation"] == "float64"
    assert stats["optimizer"] == optimizer_name
    assert stats["optimizer_state_precision"] == ("float64" if optimizer_name == "lbfgs" else "float32")
    assert stats["mesh_steps"] == baseline_report["mesh_solver"]["mesh_steps"] == 4
    assert stats["valid_voxels"] == baseline_report["mesh_solver"]["valid_voxels"]
    # GEMSAtlas stores host coordinates as NumPy float64 in both policies;
    # the live optimization tensor above stays FP32.
    assert actual.vertices.dtype == baseline.vertices.dtype == atlas.vertices.dtype
    assert actual.vertices[0, 0] == float(np.float32(atlas.reference_vertices[0, 0]))
    assert report["min_jacobian"] > 0
    np.testing.assert_allclose(actual.vertices, baseline.vertices, atol=1e-4, rtol=1e-5)
    assert report["mask_dice"] == pytest.approx(baseline_report["mask_dice"], abs=1e-6)


@pytest.mark.parametrize("initial_tf32", (False, True))
def test_stable_brainstem_restores_caller_precision_after_error(monkeypatch, initial_tf32):
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    deterministic = torch.are_deterministic_algorithms_enabled()
    observed = []

    def failing_fit(*args, **options):
        observed.append(torch.backends.cuda.matmul.allow_tf32)
        assert options["stable_mesh_fitting"] is True
        raise RuntimeError("controlled synthetic fit error")

    monkeypatch.setattr(brainstem, "_fit_brainstem_segmentation", failing_fit)
    try:
        torch.backends.cuda.matmul.allow_tf32 = initial_tf32
        with pytest.raises(RuntimeError, match="controlled synthetic fit error"):
            brainstem.fit_brainstem_segmentation(None, None, device="cuda:0",
                                                stable_mesh_fitting=True)
        assert observed == [False]
        assert torch.backends.cuda.matmul.allow_tf32 is initial_tf32
        assert torch.are_deterministic_algorithms_enabled() is deterministic
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
