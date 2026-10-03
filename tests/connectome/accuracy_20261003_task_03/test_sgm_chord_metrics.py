"""The ACT exit metric must use internal incoming chords before downsampling."""
import math
import torch
from fnit.connectome.fod import tracking_sh_precomputed
from fnit.connectome.tracking import _ifod2_sgm_chord_metrics

def test_mid_and_end_use_the_immediately_preceding_internal_vertex():
    angle = math.pi / 4
    radius = 1.25 / angle
    start = torch.tensor([[0., 0., 0.]])
    middle = torch.tensor([[radius * (1 - math.cos(angle / 2)), 0.,
                            radius * math.sin(angle / 2)]])
    end = torch.tensor([[radius * (1 - math.cos(angle)), 0., radius * math.sin(angle)]])
    points = torch.cat((middle, end))
    incoming = torch.cat((middle - start, end - middle))
    # A delta FOD makes tangent, incoming internal chord, and full-arc chord
    # measurably different while preserving the actual SH evaluation path.
    coefficients = tracking_sh_precomputed(torch.tensor([[1., 0., 0.]]), 8)
    observed = []
    def sample_fod(selected_points):
        observed.append(selected_points.clone())
        return coefficients.expand(len(selected_points), -1)
    actual = _ifod2_sgm_chord_metrics(points, incoming, sample_fod, lmax=8)
    expected_directions = torch.tensor([[math.sin(angle / 4), 0., math.cos(angle / 4)],
                                        [math.sin(3 * angle / 4), 0., math.cos(3 * angle / 4)]])
    expected = (coefficients * tracking_sh_precomputed(expected_directions, 8)).sum(-1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(observed[0], points)
    tangents = torch.tensor([[math.sin(angle / 2), 0., math.cos(angle / 2)],
                             [math.sin(angle), 0., math.cos(angle)]])
    wrong_tangent = (coefficients * tracking_sh_precomputed(tangents, 8)).sum(-1)
    assert not torch.allclose(actual, wrong_tangent, atol=1e-3, rtol=1e-3)
    wrong_whole_arc = (coefficients * tracking_sh_precomputed(
        torch.cat((middle - start, end - start)), 8)).sum(-1)
    assert not torch.allclose(actual, wrong_whole_arc, atol=1e-3, rtol=1e-3)
