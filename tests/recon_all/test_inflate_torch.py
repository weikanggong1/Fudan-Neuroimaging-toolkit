"""原规则inflation与sulc语义的结构测试；不替代真实表面benchmark。"""
import numpy as np
import pytest
import torch

from fnit.recon_all.inflate_python import inflate_updates, matrix, two_ring_neighbors
from fnit.recon_all.inflate_torch import TorchInflationContext
from fnit.recon_all.inflate_topology import inflate_neighbor_tables
from fnit.recon_all.place_surface_normals import FaceNormalTopology
from fnit.recon_all.smooth_surface_python import ordered_neighbors


def tetrahedron():
    xyz = np.asarray([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]], np.float32)
    faces = np.asarray([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], np.int32)
    return xyz, faces


@pytest.mark.parametrize("duplicate", [False, True])
def test_integer_topology_preserves_source_order_and_duplicates(duplicate):
    _, faces = tetrahedron()
    if duplicate:
        faces = np.concatenate((faces,faces[[2,0]]),axis=0)
    # An extra isolated vertex exercises complete empty-row preservation.
    original = ordered_neighbors(faces,5)
    expected = (*matrix(original),*matrix(two_ring_neighbors(original)))
    actual = inflate_neighbor_tables(normal_topology=FaceNormalTopology(faces,5))
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_sulc_before_normals_update_and_full_schedule():
    xyz, faces = tetrahedron()
    sulc = np.zeros(len(xyz), np.float32)
    report = {}
    expected = inflate_updates(xyz=xyz, faces=faces, niterations=1, rms_target=0,
                               sulc=sulc, diagnostics=report)
    context = TorchInflationContext(faces=faces, nvertices=len(xyz), device="cpu")
    actual = context.integrate(vertices=torch.from_numpy(xyz), niterations=1, rms_target=0)
    assert actual["steps"] == report["steps"] == 6
    np.testing.assert_allclose(actual["coordinates"].numpy(), expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(actual["sulc"].numpy(), sulc, rtol=0, atol=1e-6)
    assert np.any(sulc != 0)
    np.testing.assert_array_equal(xyz, tetrahedron()[0])


def test_sulc_buffer_contract_and_no_mutation():
    xyz, faces = tetrahedron()
    with pytest.raises(ValueError, match="sulc"):
        inflate_updates(xyz=xyz, faces=faces, sulc=np.ones(len(xyz), np.float32))
    context = TorchInflationContext(faces=faces, nvertices=len(xyz), device="cpu")
    with pytest.raises(ValueError, match="positive"):
        context.integrate(vertices=torch.from_numpy(xyz), niterations=0)
    with pytest.raises(ValueError, match="finite"):
        context.integrate(vertices=torch.full_like(torch.from_numpy(xyz), float("nan")))
    np.testing.assert_array_equal(xyz, tetrahedron()[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ordered_reuse_matches_cpu_sulc():
    xyz, faces = tetrahedron()
    cpu = TorchInflationContext(faces=faces, nvertices=len(xyz), device="cpu")
    gpu = TorchInflationContext(faces=faces, nvertices=len(xyz), device="cuda:0")
    expected = cpu.integrate(vertices=torch.from_numpy(xyz), niterations=1, rms_target=0)
    actual = gpu.integrate(vertices=torch.from_numpy(xyz).to("cuda:0"), niterations=1, rms_target=0)
    np.testing.assert_allclose(actual["coordinates"].cpu().numpy(), expected["coordinates"].numpy(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(actual["sulc"].cpu().numpy(), expected["sulc"].numpy(), rtol=0, atol=1e-6)
