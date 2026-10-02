import gzip
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.gems import GEMSAtlas, TorchGEMS, ashburner_prior, rasterize_priors
from fnit.gems.gaussian import (GaussianParameters, gaussian_log_likelihood,
                                label_posterior, update_gaussians)


def _atlas():
    vertices = np.asarray([[1,1,1],[6,1,1],[1,6,1],[1,1,6]], dtype=float)
    tetra = np.asarray([[0,1,2,3]], dtype=np.int64)
    alphas = np.asarray([[1,0],[0,1],[0,1],[0,1]], dtype=np.float32)
    return GEMSAtlas(vertices, vertices, tetra, alphas, .1,
                     np.ones((4,3), bool), np.asarray([0, 10]), ("Unknown", "ROI"))


def test_reference_deformation_prior_is_zero():
    atlas = _atlas()
    v = torch.tensor(atlas.vertices, dtype=torch.float32)
    cost, jac = ashburner_prior(v, v, torch.tensor(atlas.tetrahedra), atlas.stiffness)
    assert torch.allclose(cost, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(jac, torch.ones_like(jac), atol=1e-6)


def test_sliding_boundary_rotates_with_atlas_affine():
    from fnit.gems.deformation import sliding_boundary_projectors
    flags = torch.tensor([[True, False, False], [True, True, False],
                           [False, False, False], [True, True, True]])
    transform = torch.tensor([[0., -2., 0.], [1., 0., 0.], [0., 0., 3.]])
    projection = sliding_boundary_projectors(flags, transform)
    gradient = torch.tensor([1., 2., 3.])
    expected = torch.tensor([[0., 2., 0.], [1., 2., 0.], [0., 0., 0.], [1., 2., 3.]])
    torch.testing.assert_close(projection @ gradient, expected)


def test_rasterized_priors_sum_to_one():
    atlas = _atlas()
    priors, covered = rasterize_priors(
        torch.tensor(atlas.vertices, dtype=torch.float32),
        torch.tensor(atlas.tetrahedra), torch.tensor(atlas.alphas), (8,8,8))
    assert priors.shape == (2,8,8,8)
    assert covered.any()
    assert torch.allclose(priors.sum(0), torch.ones(8,8,8), atol=1e-6)


def test_batched_raster_assignment_reconstructs_interpolated_priors():
    atlas = _atlas()
    vertices = torch.tensor(np.concatenate((atlas.vertices, atlas.vertices + [11, 9, 7])),
                            dtype=torch.float32)
    tetrahedra = torch.tensor(np.concatenate((atlas.tetrahedra, atlas.tetrahedra + 4)))
    alphas = torch.tensor(np.concatenate((atlas.alphas, atlas.alphas[:, ::-1])))
    priors, covered, cells, weights = rasterize_priors(
        vertices, tetrahedra, alphas, (19, 17, 15),
        background_channel=None, return_assignment=True)
    interpolation = (alphas[cells[covered]] * weights[covered][..., None]).sum(1)
    torch.testing.assert_close(priors[:, covered].T, interpolation, atol=1e-6, rtol=1e-6)
    assert not covered[9, 8, 7]
    assert torch.count_nonzero(priors[:, ~covered]) == 0


def test_raster_gradient_matches_finite_difference_inside_a_tetrahedron():
    atlas = _atlas()
    vertices = torch.tensor(atlas.vertices + [.13, .21, .31],
                            dtype=torch.float64, requires_grad=True)
    tetrahedra = torch.tensor(atlas.tetrahedra)
    alphas = torch.tensor(atlas.alphas, dtype=torch.float64)
    def value(positions):
        priors, _ = rasterize_priors(positions, tetrahedra, alphas, (8, 8, 8))
        return priors[:, 2, 2, 2]
    assert torch.autograd.gradcheck(value, (vertices,))


def test_cached_spatial_index_matches_fresh_after_mesh_movement():
    from fnit.gems.rasterize import build_block_index
    atlas = _atlas()
    vertices = torch.tensor(atlas.vertices, dtype=torch.float32)
    tetrahedra = torch.tensor(atlas.tetrahedra)
    alphas = torch.tensor(atlas.alphas)
    index = build_block_index(atlas.vertices, atlas.tetrahedra, (8, 8, 8), margin=2)
    rasterize_priors(vertices, tetrahedra, alphas, (8, 8, 8), block_index=index)
    moved = vertices + torch.tensor([.7, -.4, .2])
    cached, covered = rasterize_priors(moved, tetrahedra, alphas, (8, 8, 8), block_index=index)
    fresh, fresh_covered = rasterize_priors(moved, tetrahedra, alphas, (8, 8, 8))
    torch.testing.assert_close(cached, fresh)
    assert torch.equal(covered, fresh_covered)


def test_compact_gaussian_statistics_match_full_zero_masked_image():
    image = torch.arange(125, dtype=torch.float32).reshape(5, 5, 5)
    image[0] = 0
    responsibilities = torch.stack((torch.full_like(image, .3), torch.full_like(image, .7)))
    hyper = {"mean_hyper": torch.tensor([40., 90.]), "n_hyper": torch.tensor([10., 5.])}
    full = update_gaussians(image, responsibilities, **hyper)
    valid = image != 0
    compact = update_gaussians(image[valid].reshape(-1, 1, 1),
                               responsibilities[:, valid].reshape(2, -1, 1, 1), **hyper)
    torch.testing.assert_close(compact.means, full.means)
    torch.testing.assert_close(compact.covariances, full.covariances)


def test_atlas_smoothing_preserves_normalized_vertex_alphas():
    from fnit.gems.smoothing import smooth_atlas_alphas

    fitted = smooth_atlas_alphas(_atlas(), np.asarray([0, 1]), 1.0)
    assert fitted.shape == (4, 2)
    np.testing.assert_allclose(fitted.sum(1), 1, atol=1e-5)
    assert np.isfinite(fitted).all()


def test_em_returns_native_grid_labels():
    atlas = _atlas()
    image = torch.ones(8,8,8)
    image[2:5,2:5,2:5] = 2
    result = TorchGEMS(atlas)(image, em_iterations=2)
    assert result.labels.shape == image.shape
    assert set(torch.unique(result.labels).tolist()) <= {0, 10}
    assert result.min_jacobian > 0


def test_zero_intensity_voxels_have_no_reported_label():
    image = torch.zeros(8, 8, 8)
    image[2:5, 2:5, 2:5] = 2
    result = TorchGEMS(_atlas())(image, em_iterations=2)
    assert torch.all(result.labels[image == 0] == 0)


def test_multistage_fit_restores_original_anatomical_priors():
    atlas = _atlas()
    image = torch.ones(8, 8, 8)
    staged = TorchGEMS(atlas)(
        image, em_iterations=1, fit_alpha_stages=[
            (torch.full((4, 2), 0.5), 1),
            (torch.as_tensor(atlas.alphas), 1),
        ])
    assert len(staged.objective_history) == 3
    expected, _ = rasterize_priors(staged.vertices, torch.as_tensor(atlas.tetrahedra),
                                    torch.as_tensor(atlas.alphas), image.shape)
    torch.testing.assert_close(staged.priors, expected)


def test_refreshes_spatial_index_without_changing_static_mesh():
    atlas = _atlas()
    from dataclasses import replace
    atlas = replace(atlas, can_move=np.zeros_like(atlas.can_move))
    image = torch.ones(8, 8, 8)
    ordinary = TorchGEMS(atlas)(image, em_iterations=1, deform_iterations=2)
    refreshed = TorchGEMS(atlas)(image, em_iterations=1, deform_iterations=2,
                                  index_refresh_interval=1, index_margin=3)
    torch.testing.assert_close(refreshed.priors, ordinary.priors)
    torch.testing.assert_close(refreshed.posterior, ordinary.posterior)


def test_mesh_fit_may_stop_when_cost_converges():
    atlas = _atlas()
    image = torch.ones(8, 8, 8)
    from fnit.gems.gaussian import GaussianParameters
    fixed = GaussianParameters(torch.ones(2, 1), torch.ones(2, 1, 1))
    result = TorchGEMS(atlas)(image, em_iterations=1, deform_iterations=8,
                               deform_lr=0, fixed_gaussians=fixed,
                               relative_cost_stop=1e-10)
    assert len(result.objective_history) == 3


def test_atlas_mask_erosion_restricts_boundary_voxels():
    image = torch.ones(8, 8, 8)
    ordinary = TorchGEMS(_atlas())(image, mask_to_atlas=True)
    eroded = TorchGEMS(_atlas())(image, mask_to_atlas=True,
                                  atlas_mask_erosion=1)
    assert int((eroded.labels != 0).sum()) < int((ordinary.labels != 0).sum())


def test_zero_intensity_voxel_keeps_atlas_prior_for_soft_volume():
    image = torch.ones(8, 8, 8)
    image[2, 2, 2] = 0
    fixed = GaussianParameters(torch.tensor([[0.], [10.]]),
                               torch.ones(2, 1, 1))
    fit = TorchGEMS(_atlas())(image, fixed_gaussians=fixed)
    torch.testing.assert_close(fit.posterior[:, 2, 2, 2], fit.priors[:, 2, 2, 2])


def test_lbfgs_mesh_fit_keeps_positive_tetrahedra():
    image = torch.ones(8, 8, 8)
    image[2:5, 2:5, 2:5] = 2
    result = TorchGEMS(_atlas())(image, em_iterations=2, deform_iterations=2,
                                  deform_optimizer="lbfgs", deform_lr=0.5)
    assert result.min_jacobian > 0
    assert len(result.objective_history) == 3


def test_outer_em_cycles_each_fit_the_mesh():
    from dataclasses import replace
    atlas = replace(_atlas(), can_move=np.zeros((4, 3), bool))
    image = torch.ones(8, 8, 8)
    result = TorchGEMS(atlas)(
        image, em_iterations=100, em_relative_cost_stop=1e-5,
        deform_iterations=2, outer_iterations=3,
        mean_hyper=torch.tensor([1., 2.]), n_hyper=torch.tensor([10., 10.]))
    assert len(result.objective_history) == 7
    assert torch.isfinite(result.gaussian_parameters.means).all()
    torch.testing.assert_close(result.vertices, torch.tensor(atlas.vertices, dtype=torch.float32))


def test_compact_mesh_fit_matches_dense_valid_voxel_path(monkeypatch):
    import fnit.gems.core as core

    image = torch.zeros(8, 8, 8)
    image[1:6, 1:6, 1:6] = 1
    image[2:5, 2:5, 2:5] = 2
    settings = dict(em_iterations=8, em_relative_cost_stop=1e-5,
                    deform_iterations=3, outer_iterations=2,
                    deform_optimizer="lbfgs", deform_lr=0.5,
                    boundary_transform=np.eye(3),
                    mean_hyper=torch.tensor([1., 2.]), n_hyper=torch.tensor([10., 10.]))
    compact = TorchGEMS(_atlas())(image, **settings)

    def dense_selected(vertices, tetrahedra, alphas, shape, *, valid_mask, **kwargs):
        priors, covered = core.rasterize_priors(vertices, tetrahedra, alphas, shape, **kwargs)
        return priors[:, valid_mask], covered[valid_mask]

    monkeypatch.setattr(core, "rasterize_priors_compact", dense_selected)
    dense = TorchGEMS(_atlas())(image, **settings)
    torch.testing.assert_close(compact.vertices, dense.vertices, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(compact.posterior, dense.posterior, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(compact.gaussian_parameters.means, dense.gaussian_parameters.means)
    torch.testing.assert_close(compact.gaussian_parameters.covariances, dense.gaussian_parameters.covariances)
    assert torch.equal(compact.labels, dense.labels)


def test_compact_em_with_empty_valid_mask_keeps_soft_atlas_prior():
    fit = TorchGEMS(_atlas())(torch.zeros(8, 8, 8), em_iterations=2,
                              em_relative_cost_stop=1e-5)
    assert torch.count_nonzero(fit.labels) == 0
    torch.testing.assert_close(fit.posterior, fit.priors)


def test_sliding_mesh_stops_inner_and_outer_iterations_when_nodes_do_not_move():
    from dataclasses import replace
    atlas = replace(_atlas(), can_move=np.zeros((4, 3), bool))
    result = TorchGEMS(atlas)(
        torch.ones(8, 8, 8), em_iterations=100, em_relative_cost_stop=1e-5,
        deform_iterations=30, outer_iterations=7, deform_optimizer="lbfgs",
        boundary_transform=np.eye(3),
        mean_hyper=torch.tensor([1., 2.]), n_hyper=torch.tensor([10., 10.]))
    assert len(result.objective_history) == 2
    torch.testing.assert_close(result.vertices, torch.tensor(atlas.vertices, dtype=torch.float32))


def test_grouped_labels_keep_distinct_anatomical_posteriors():
    base = _atlas()
    alphas = np.column_stack((base.alphas[:, 0],
                              0.8 * base.alphas[:, 1], 0.2 * base.alphas[:, 1]))
    atlas = GEMSAtlas(base.reference_vertices, base.vertices, base.tetrahedra,
                      alphas, base.stiffness, base.can_move,
                      np.asarray([0, 10, 20]), ("Unknown", "A", "B"))
    result = TorchGEMS(atlas)(torch.ones(8, 8, 8),
                              label_classes=np.asarray([0, 1, 1]), em_iterations=2)
    inside = (result.priors[1] > 1e-4) & (result.priors[2] > 1e-4)
    assert inside.any()
    torch.testing.assert_close(result.posterior[1][inside] / result.posterior[2][inside],
                               result.priors[1][inside] / result.priors[2][inside],
                               rtol=1e-4, atol=1e-4)
    image = torch.full((8, 8, 8), 100.0)
    image[1:6, 1:6, 1:6] = 10
    shifted_groups = TorchGEMS(atlas)(image, label_classes=np.asarray([2, 0, 0]),
                                     em_iterations=3)
    assert shifted_groups.gaussian_parameters.means[1, 0] > shifted_groups.gaussian_parameters.means[0, 0] + 20


def test_zero_intensity_voxels_are_excluded_from_mesh_cost():
    priors = torch.tensor([[0.8, 0.5], [0.2, 0.5]])
    likelihood = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    posterior, cost = label_posterior(priors, likelihood, torch.arange(2),
                                       valid_mask=torch.tensor([True, False]))
    expected = -torch.logsumexp(priors[:, 0].log() + likelihood[:, 0], dim=0)
    torch.testing.assert_close(cost, expected)
    assert posterior.shape == priors.shape


def test_brainstem_hyperprior_uses_native_coarse_tissue_median():
    from fnit.gems.brainstem import brainstem_gaussian_hyperparameters

    data = np.full((9, 9, 9), 50, dtype=np.float32)
    labels = np.zeros(data.shape, dtype=np.int16)
    labels[2:7, 2:7, 2:7] = 16
    data[labels == 16] = 110
    means, counts = brainstem_gaussian_hyperparameters(
        nib.Nifti1Image(data, np.eye(4)), labels,
        np.asarray([10, 0]), np.asarray([0, 175]))
    assert means[0] == 110
    assert counts[0] > 10


def test_gaussian_hyperprior_updates_mean_and_variance():
    image = torch.tensor([[[100., 120.]]])
    responsibilities = torch.ones((1, 1, 1, 2))
    params = update_gaussians(image, responsibilities,
                              mean_hyper=torch.tensor([90.]), n_hyper=torch.tensor([2.]))
    expected_mean = (100 + 120 + 2 * 90) / 4.01
    expected_variance = (((100 - expected_mean) ** 2 + (120 - expected_mean) ** 2
                          + 2 * (expected_mean - 90) ** 2) / 2.01 + 0.01)
    torch.testing.assert_close(params.means[0, 0], torch.tensor(expected_mean))
    torch.testing.assert_close(params.covariances[0, 0, 0],
                               torch.tensor(expected_variance))
    multi = update_gaussians(image, torch.cat((responsibilities, responsibilities)),
                             mean_hyper=torch.tensor([90., 80.]),
                             n_hyper=torch.tensor([2., 2.]))
    assert multi.means.shape == (2, 1)
    assert multi.covariances.shape == (2, 1, 1)


def test_single_modality_likelihood_matches_normal_logpdf():
    image = torch.tensor([[[10., 12., 20.]]])
    params = GaussianParameters(torch.tensor([[11.], [18.]]),
                                torch.tensor([[[4.]], [[9.]]]))
    actual = gaussian_log_likelihood(image, params)
    expected = torch.distributions.Normal(
        params.means[:, 0, None, None, None],
        params.covariances[:, 0, 0, None, None, None].sqrt(),
    ).log_prob(image)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_reads_freesurfer_atlasmesh_text(tmp_path):
    mesh = tmp_path / "AtlasMesh.gz"
    text = """Number of points: 4
Number of cells: 1
Number of labels: 2
Number of meshes: 1
Reference position
0 1 1 1
1 6 1 1
2 1 6 1
3 1 1 6
K: 0.1
Position 0
0 1 1 1
1 6 1 1
2 1 6 1
3 1 1 6
Cells:
0 TETRAHEDRON 0 1 2 3
Point parameters:
0 65535 0 true true true true
1 0 65535 true true true true
2 0 65535 true true true true
3 0 65535 true true true true
"""
    with gzip.open(mesh, "wt") as f:
        f.write(text)
    lut = tmp_path / "compressionLookupTable.txt"
    lut.write_text("0 0 Unknown 0 0 0 255\n10 1 ROI 1 2 3 255\n")
    atlas = GEMSAtlas.from_freesurfer(mesh, lut)
    assert atlas.tetrahedra.tolist() == [[0,1,2,3]]
    assert atlas.label_ids.tolist() == [0,10]
    np.testing.assert_allclose(atlas.alphas.sum(1), 1)


def test_label_centroid_affine_recovers_translation():
    from fnit.gems.initialize import estimate_label_centroid_affine
    # Construct a cube split into five tetrahedra would be preferable for full
    # affine fitting; here directly exercise four shared labels by four tiny
    # translated atlas copies is outside the minimal raster unit. The API must
    # at least reject an underdetermined single-label initialization.
    atlas = _atlas()
    target = np.zeros((8,8,8), dtype=np.int32)
    target[2:5,2:5,2:5] = 10
    try:
        estimate_label_centroid_affine(atlas, target, min_shared_labels=4)
    except ValueError as error:
        assert "need at least 4" in str(error)
    else:
        raise AssertionError("underdetermined centroid affine should be rejected")


def test_mask_affine_aligns_a_shifted_structure():
    from fnit.gems.initialize import estimate_mask_affine

    source = np.zeros((24, 24, 24), dtype=np.float32)
    source[6:15, 8:14, 4:18] = 1
    labels = np.zeros((40, 40, 40), dtype=np.int16)
    labels[10:19, 13:19, 7:21] = 16
    affine, score = estimate_mask_affine(
        nib.Nifti1Image(source, np.eye(4)),
        nib.Nifti1Image(np.zeros(labels.shape, dtype=np.float32), np.eye(4)),
        labels, [16], device="cpu", max_iterations=25)
    np.testing.assert_allclose(affine[:3, 3], [4, 5, 3], atol=1.5)
    assert score > 0.8
