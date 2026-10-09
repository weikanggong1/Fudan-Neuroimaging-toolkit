"""静态 placement GPU 转写保留原逐邻接加法及 float32 QR；此处不是 benchmark。"""
import numpy as np
from scipy.spatial import ConvexHull
import torch

from fnit.recon_all.place_surface_regularization_torch import PlacementRegularizationTorch, _inverse5
from fnit.recon_all.place_surface_curvature import _inverse5_qr, quadratic_curvature, tangent_basis, two_ring_neighbors
from fnit.recon_all.place_surface_gradient_average import average_signed_gradients
from fnit.recon_all.place_surface_smoothing import _ordered_neighbors
from fnit.recon_all.place_surface_spring import spring_gradient


def _inputs():
    rng = np.random.default_rng(712)
    xyz = rng.normal(size=(48, 3)).astype(np.float32)
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    xyz *= (2.0 + rng.uniform(-.1, .1, size=(48, 1))).astype(np.float32)
    faces = ConvexHull(xyz).simplices.astype(np.int32)
    normal = xyz / np.linalg.norm(xyz, axis=1)[:, None]
    ripped = np.zeros(48, dtype=np.bool_)
    ripped[[2, 11, 27]] = True
    neighbors, valid, _ = _ordered_neighbors(faces, len(xyz))
    offsets, candidates = two_ring_neighbors(faces, len(xyz), ordered_neighbors=(neighbors, valid))
    gradient = rng.normal(size=xyz.shape).astype(np.float32)
    return xyz, normal, faces, ripped, neighbors, valid, offsets, candidates, gradient


def test_source_qr_batch_retains_float32_order():
    rng = np.random.default_rng(13)
    design = rng.normal(size=(7, 14, 5)).astype(np.float32)
    gram = np.zeros((7, 5, 5), dtype=np.float32)
    for row in range(14):
        gram += design[:, row, :, None] * design[:, row, None, :]
    expected = np.stack([_inverse5_qr(value) for value in gram])
    actual = _inverse5(torch.from_numpy(gram)).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_regularizer_matches_cpu_order_with_ripped_vertices_and_after_average():
    xyz, normal, faces, ripped, neighbors, valid, offsets, candidates, gradient = _inputs()
    context = PlacementRegularizationTorch(neighbors=neighbors, valid=valid, offsets=offsets,
                                           candidates=candidates, ripped=ripped, device="cpu", chunk_size=9)
    for after in (None, gradient * np.float32(.1)):
        averaged = average_signed_gradients(gradient, faces, ripped, 4,
                                            ordered_neighbors=(neighbors, valid))
        if after is not None:
            averaged = np.float32(averaged + after)
        nspring = spring_gradient(xyz, normal, faces, ripped, weight=.3, direction="normal",
                                  ordered_neighbors=(neighbors, valid))
        tspring = spring_gradient(xyz, normal, faces, ripped, weight=.3, direction="tangent",
                                  ordered_neighbors=(neighbors, valid))
        scalar = quadratic_curvature(xyz, normal, tangent_basis(normal), ripped, offsets, candidates)
        expected = np.float32(np.float32(np.float32(averaged + nspring) +
                                        np.float32(scalar[:, None] * normal)) + tspring)
        actual = context.regularize(vertices=xyz, normals=normal, gradient=gradient,
                                    iterations=4, after_average=after)
        np.testing.assert_array_equal(actual, expected)


def test_invalid_device_and_adjacency_are_rejected():
    _, _, _, ripped, neighbors, valid, offsets, candidates, _ = _inputs()
    for kwargs in ({"device": "cuda"}, {"device": "cpu", "chunk_size": 0}):
        try:
            PlacementRegularizationTorch(neighbors=neighbors, valid=valid, offsets=offsets,
                                         candidates=candidates, ripped=ripped, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid device/chunk_size accepted")


def test_singular_active_curvature_fails_before_optimizer_acceptance():
    xyz, normal, _, ripped, neighbors, valid, offsets, candidates, gradient = _inputs()
    context = PlacementRegularizationTorch(neighbors=neighbors, valid=valid, offsets=offsets,
                                           candidates=candidates, ripped=ripped, device="cpu")
    try:
        context.regularize(vertices=np.zeros_like(xyz), normals=normal, gradient=gradient,
                           iterations=0)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("singular active curvature silently reached optimizer")
