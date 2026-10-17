"""检查相同试步弧距和 CUDA 工作缓冲复用，不替代真实配准回归。"""

from unittest.mock import patch

import pytest
import torch

from fnit.recon_all.mris_register_average_numba import RegistrationGradientAverager
from fnit.recon_all.mris_register_line_search import first_registration_sse
from fnit.recon_all.mris_register_nonlinear import (
    face_area_normals, ordered_neighbors_from_faces, original_chord_distances,
    sphere_arc_distances,
)


def _objective_inputs():
    positions = torch.tensor([[100., 0., 0.], [0., 100., 0.],
                              [0., 0., 100.], [-100., 0., 0.]])
    faces = torch.tensor([[0, 1, 2], [1, 3, 2], [0, 2, 3], [0, 3, 1]])
    neighbors, degrees = ordered_neighbors_from_faces(faces=faces, nvertices=4)
    original_distances = original_chord_distances(
        positions=positions * .8, neighbors=neighbors, degrees=degrees)
    original_areas, _ = face_area_normals(positions=positions * .8, faces=faces)
    return dict(positions=positions, faces=faces, neighbors=neighbors,
                degrees=degrees, original_distances=original_distances,
                original_face_areas=original_areas,
                curvature=torch.tensor([.1, -.5, 1., 2.]),
                target_mean=torch.arange(128., dtype=torch.float32).reshape(16, 8) / 50,
                target_variance=torch.ones((16, 8)),
                original_area=100000., total_area=125663.703125)


@pytest.mark.parametrize("scale", [1., .999, 1.001])
@pytest.mark.parametrize("return_terms", [False, True])
def test_objective_reuses_same_trial_arcs_without_changing_any_term(scale, return_terms):
    arguments = _objective_inputs()
    arguments["positions"] = arguments["positions"] * scale
    arguments["return_terms"] = return_terms
    expected = first_registration_sse(**arguments)
    arcs = sphere_arc_distances(arguments["positions"], arguments["neighbors"],
                                arguments["degrees"])
    saved = arcs.clone()
    with patch("fnit.recon_all.mris_register_line_search.sphere_arc_distances",
               side_effect=AssertionError("cached trial arcs must not be recomputed")):
        actual = first_registration_sse(**arguments, current_distances=arcs)
    assert actual == expected
    torch.testing.assert_close(arcs, saved, atol=0, rtol=0)


@pytest.mark.parametrize("wrong", [torch.ones((4, 2)), torch.ones((4, 3), dtype=torch.float64)])
def test_objective_rejects_incompatible_cached_arcs(wrong):
    with pytest.raises(ValueError, match="current_distances"):
        first_registration_sse(**_objective_inputs(), current_distances=wrong)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires explicit CUDA target")
@pytest.mark.parametrize("input_device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("averaging_device", ["cuda", "cuda:0"])
def test_cuda_buffers_reuse_storage_preserve_outputs_and_all_ordered_rounds(input_device,
                                                                          averaging_device):
    # Isolated vertex and inactive -1 padding exercise masked loads, not truncation.
    neighbors = torch.tensor([[1, 2], [0, 2], [1, 0], [0, 0], [-1, -1]])
    degrees = torch.tensor([2, 2, 2, 1, 0])
    expected_average = RegistrationGradientAverager(neighbors=neighbors, degrees=degrees,
                                                   device="cpu")
    average = RegistrationGradientAverager(neighbors=neighbors, degrees=degrees,
                                          device=averaging_device)
    pointers = None
    previous_output, previous_values = None, None
    for index, iterations in enumerate((1, 4, 17, 0, 1)):
        # CPU view is intentionally noncontiguous before moving to the chosen device.
        base_gradient = torch.arange(30., dtype=torch.float32).reshape(5, 6) / (index + 2)
        cpu_gradient = base_gradient[:, ::2]
        assert not cpu_gradient.is_contiguous()
        gradient = cpu_gradient.to(device=input_device)
        saved = gradient.clone()
        expected = expected_average(gradient=cpu_gradient, iterations=iterations)
        actual = average(gradient=gradient, iterations=iterations)
        torch.cuda.synchronize(device="cuda:0")
        assert actual.device == gradient.device
        assert actual.data_ptr() != gradient.data_ptr()
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        assert torch.equal(actual.cpu().view(torch.int32), expected.view(torch.int32))
        torch.testing.assert_close(gradient, saved, atol=0, rtol=0)
        if previous_output is not None:
            torch.testing.assert_close(previous_output, previous_values, atol=0, rtol=0)
        current_pointers = tuple(buffer.data_ptr() for buffer in average._cuda_buffers)
        if pointers is None:
            pointers = current_pointers
        assert current_pointers == pointers
        previous_output, previous_values = actual, actual.clone()
