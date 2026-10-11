"""Analytical check for the conditional spherical-registration update."""

import torch

from fnit.recon_all.mris_register_nonlinear import (
    apply_spherical_gradient, average_gradients, average_gradients_once,
    area_gradient_add, distance_gradient, face_area_normals,
    original_chord_distances, sphere_arc_distances,
    ordered_neighbors_from_faces,
)


def test_apply_spherical_gradient_projects_once():
    positions = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    gradients = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    result = apply_spherical_gradient(positions, gradients, 10.0)
    assert result.shape == positions.shape
    torch.testing.assert_close(torch.linalg.vector_norm(result, dim=1),
                               torch.full((2,), 100.0), rtol=0, atol=1e-5)
    torch.testing.assert_close(result[0, [0, 1]], result[1, [1, 0]],
                               rtol=0, atol=0)


def test_ordered_neighbors_and_one_gradient_average():
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    neighbors, degrees = ordered_neighbors_from_faces(faces, 4)
    assert [neighbors[i, :degrees[i]].tolist() for i in range(4)] == [
        [2, 1, 3], [0, 2], [1, 0, 3], [2, 0],
    ]
    gradient = torch.tensor([[1., 0., 0.], [2., 0., 0.],
                             [4., 0., 0.], [8., 0., 0.]])
    result = average_gradients_once(gradient, neighbors, degrees)
    torch.testing.assert_close(result[:, 0], torch.tensor([3.75, 7/3, 3.75, 13/3]))


def test_repeated_gradient_average_agrees_with_one_step():
    faces = torch.tensor([[0, 1, 2]])
    neighbors, degrees = ordered_neighbors_from_faces(faces, 3)
    gradient = torch.tensor([[1., 0., 0.], [2., 0., 0.], [4., 0., 0.]])
    first = average_gradients_once(gradient, neighbors, degrees)
    torch.testing.assert_close(average_gradients(gradient, neighbors, degrees, 1), first)
    torch.testing.assert_close(average_gradients(gradient, neighbors, degrees, 2),
                               average_gradients_once(first, neighbors, degrees))


def test_distance_gradient_tangent_force():
    faces = torch.tensor([[0, 1, 2]])
    neighbors, degrees = ordered_neighbors_from_faces(faces, 3)
    positions = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    normals = torch.tensor([[0., 0., 1.]] * 3)
    current = torch.full((3, 2), 2.)
    original = torch.ones((3, 2))
    gradient = distance_gradient(positions, normals, neighbors, degrees,
                                 current, original, 2., 1., 1., 5.)
    torch.testing.assert_close(gradient[0], torch.tensor([2.5, 2.5, 0.]), rtol=0, atol=0)


def test_ordered_original_and_sphere_distances():
    faces = torch.tensor([[0, 1, 2]])
    neighbors, degrees = ordered_neighbors_from_faces(faces, 3)
    original = torch.tensor([[0., 0., 0.], [3., 4., 0.], [0., 0., 5.]])
    chords = original_chord_distances(original, neighbors, degrees)
    torch.testing.assert_close(chords[0], torch.tensor([5., 5.]), rtol=0, atol=0)
    sphere = torch.tensor([[100., 0., 0.], [0., 100., 0.], [0., 0., 100.]])
    arcs = sphere_arc_distances(sphere, neighbors, degrees)
    torch.testing.assert_close(arcs[0, 0], arcs[0, 1], rtol=0, atol=0)
    assert bool(torch.all(torch.isfinite(arcs)))


def test_equal_face_areas_leave_gradient_unchanged():
    faces = torch.tensor([[0, 1, 2]])
    positions = torch.tensor([[100., 0., 0.], [0., 100., 0.], [0., 0., 100.]])
    areas, normals = face_area_normals(positions, faces)
    initial = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    result = area_gradient_add(initial, positions, faces, areas, areas, normals,
                               1.0, 1.0)
    torch.testing.assert_close(result, initial, rtol=0, atol=0)



def test_inverted_spherical_face_has_negative_area_and_outward_normal():
    positions = torch.tensor([[100., 0., 0.], [0., 100., 0.], [0., 0., 100.]])
    faces = torch.tensor([[0, 2, 1]])
    absolute_area, inward_normal = face_area_normals(positions, faces)
    signed_area, outward_normal = face_area_normals(positions, faces,
                                                    signed_sphere=True)
    torch.testing.assert_close(signed_area, -absolute_area, rtol=0, atol=0)
    torch.testing.assert_close(outward_normal, -inward_normal, rtol=0, atol=0)


def test_tangent_basis_has_source_orientation_on_axis_normals():
    from fnit.recon_all.mris_register_nonlinear import tangent_basis

    e1, e2 = tangent_basis(torch.tensor([[0., 0., 1.], [1., 0., 0.]]))
    torch.testing.assert_close(e1, torch.tensor([[-1., 0., 0.], [0., -1., 0.]]),
                               rtol=0, atol=0)
    torch.testing.assert_close(e2, torch.tensor([[0., -1., 0.], [0., 0., -1.]]),
                               rtol=0, atol=0)


def test_zero_correlation_force_preserves_gradient():
    from fnit.recon_all.mris_register_nonlinear import (
        correlation_gradient_add, tangent_basis,
    )

    positions = torch.tensor([[100., 0., 0.], [0., 100., 0.], [0., 0., 100.]])
    e1, e2 = tangent_basis(positions / 100)
    before = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    after = correlation_gradient_add(before, positions, torch.zeros(3), e1, e2,
                                     torch.zeros((8, 4)), torch.ones((8, 4)), 1.0)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_collinear_corner_contributes_zero_without_poisoning_vertex_normals():
    from fnit.recon_all.mris_register_nonlinear import sphere_vertex_normals

    positions = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
                              [0., 1., 0.], [0., 0., 1.]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [1, 3, 4],
                          [2, 4, 3], [0, 4, 2]])
    normals = sphere_vertex_normals(positions, faces)
    without_zero_corner = sphere_vertex_normals(positions, faces[1:])
    assert bool(torch.isfinite(normals).all())
    torch.testing.assert_close(normals, without_zero_corner, rtol=0, atol=0)


def test_zero_incident_normal_sum_fails_before_atlas_sampling():
    import pytest
    from fnit.recon_all.mris_register_nonlinear import sphere_vertex_normals

    with pytest.raises(ValueError, match="Zero vertex-normal sums"):
        sphere_vertex_normals(torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
                              torch.tensor([[0, 1, 2]]))


def test_zero_and_tiny_face_normals_follow_source_fzero_without_changing_area():
    positions = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                              [2., 0., 0.], [0., 1e-8, 0.]])
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]])
    expected_areas = torch.tensor([0., 0.5e-8])
    expected_normals = torch.tensor([[0., 0., 0.], [0., 0., 1e-8]])
    for signed_sphere in (False, True):
        areas, normals = face_area_normals(positions, faces, signed_sphere=signed_sphere)
        assert bool(torch.isfinite(normals).all())
        torch.testing.assert_close(areas, expected_areas, rtol=0, atol=0)
        torch.testing.assert_close(normals, expected_normals, rtol=0, atol=0)
