"""Focused geometric checks for the FIRST mesh rasterizer."""

import torch

from fnit.connectome.first_mesh_pve import first_vertices_to_voxel, mesh_to_pve, read_first_vtk


def test_read_first_triangle_vtk(tmp_path):
    path = tmp_path / "mesh.vtk"
    path.write_text("# vtk DataFile Version 1.0\n\nASCII\nDATASET POLYDATA\n"
                    "POINTS 3 float\n0 0 0\n1 0 0\n0 1 0\n"
                    "POLYGONS 1 4\n3 0 1 2\n")
    vertices, faces = read_first_vtk(path)
    torch.testing.assert_close(vertices, torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float64))
    torch.testing.assert_close(faces, torch.tensor([[0, 1, 2]]))


def test_first_to_voxel_handles_left_handed_nifti_affine():
    affine = torch.diag(torch.tensor([-1., 1., 1., 1.], dtype=torch.float64))
    affine[0, 3] = 5
    vertices = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
    voxel = first_vertices_to_voxel(vertices, affine, (6, 6, 6))
    torch.testing.assert_close(voxel, vertices)


def test_closed_cube_has_full_interior_and_empty_exterior():
    vertices = torch.tensor([[1.5, 1.5, 1.5], [4.5, 1.5, 1.5],
                             [4.5, 4.5, 1.5], [1.5, 4.5, 1.5],
                             [1.5, 1.5, 4.5], [4.5, 1.5, 4.5],
                             [4.5, 4.5, 4.5], [1.5, 4.5, 4.5]], dtype=torch.float64)
    faces = torch.tensor([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                          [0, 1, 5], [0, 5, 4], [3, 6, 2], [3, 7, 6],
                          [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    pve = mesh_to_pve(vertices, faces, (7, 7, 7))
    assert pve[3, 3, 3] == 1
    assert pve[0, 0, 0] == 0
    assert ((pve >= 0) & (pve <= 1)).all()
