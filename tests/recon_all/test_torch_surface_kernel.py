from __future__ import annotations

import pytest
import torch

from fnit.recon_all.torch_surface_kernel import edge_lengths, face_geometry, vertex_area_normals


def _mesh(device: str = "cpu"):
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], device=device)
    return vertices, faces


def test_face_geometry_and_vertex_accumulation_cpu():
    vertices, faces = _mesh()
    areas, normals, centers = face_geometry(vertices, faces)
    assert areas.shape == (4,)
    assert torch.all(areas > 0)
    assert torch.allclose(torch.linalg.vector_norm(normals, dim=-1), torch.ones(4))
    assert centers.shape == (4, 3)
    vertex_area, vertex_normals = vertex_area_normals(vertices, faces)
    assert vertex_area.shape == (4,)
    assert torch.all(vertex_area > 0)
    assert torch.allclose(torch.linalg.vector_norm(vertex_normals, dim=-1), torch.ones(4))
    lengths = edge_lengths(vertices, faces)
    assert lengths.shape == (4, 3)
    assert torch.all(lengths > 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_batched_cuda_matches_cpu():
    cpu_v, cpu_f = _mesh()
    gpu_v, gpu_f = cpu_v.cuda(), cpu_f.cuda()
    cpu_out = face_geometry(cpu_v, cpu_f)
    gpu_out = face_geometry(gpu_v, gpu_f)
    for expected, actual in zip(cpu_out, gpu_out):
        assert torch.allclose(expected, actual.cpu(), rtol=1e-6, atol=1e-6)
    batched_v = torch.stack((gpu_v, gpu_v + 2.0), dim=0)
    batched_areas, batched_normals = vertex_area_normals(batched_v, gpu_f)
    assert batched_areas.shape == (2, 4)
    assert batched_normals.shape == (2, 4, 3)
    assert torch.isfinite(batched_areas).all()

