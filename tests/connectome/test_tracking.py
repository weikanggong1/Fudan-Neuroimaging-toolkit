import math

import pytest
import torch

from fnit.connectome.fod import real_sh
from fnit.connectome.tracking import _sphere, probabilistic_tractography


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_straight_fod_tracks_connect_opposite_gm_caps(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    shape = (22, 11, 11)
    tissue = torch.zeros(shape, dtype=torch.int16, device=device)
    tissue[1, 1:10, 1:10] = 1
    tissue[2:20, 1:10, 1:10] = 2
    tissue[20, 1:10, 1:10] = 1
    directions = _sphere(256, torch.device(device))
    target = (5 / (4 * math.pi)) * directions[:, 0].pow(4)
    coefficient = torch.linalg.lstsq(real_sh(directions, 4), target).solution
    fod = torch.zeros((*shape, 15), dtype=torch.float32, device=device)
    fod[tissue == 2] = coefficient
    fa = torch.full(shape, 0.6, dtype=torch.float32, device=device)

    tracks = probabilistic_tractography(
        fod, torch.eye(4, device=device), tissue, n_seeds=100, fa=fa,
        seed=11, cutoff=0.01, power=2.0, max_angle_degrees=20,
        sphere_samples=256,
    )
    assert len(tracks.paths) > 10
    assert tracks.endpoints.shape == (len(tracks.paths), 2, 3)
    assert torch.all((tracks.endpoints[..., 0] < 3) |
                     (tracks.endpoints[..., 0] > 18))
    assert torch.all(tracks.lengths_mm >= 5)
    assert torch.allclose(tracks.mean_fa, torch.full_like(tracks.mean_fa, 0.6), atol=1e-5)
    crossing = ((tracks.endpoints[:, 0, 0] < 3) & (tracks.endpoints[:, 1, 0] > 18)) | (
        (tracks.endpoints[:, 1, 0] < 3) & (tracks.endpoints[:, 0, 0] > 18)
    )
    assert int(crossing.sum()) > 5


def test_first_step_into_gm_terminates_track():
    from fnit.connectome.tracking import _grow

    fod = torch.ones((5, 1, 1, 1), dtype=torch.float32)
    tissues = torch.zeros((5, 1, 1), dtype=torch.int16)
    tissues[1, 0, 0] = 2
    tissues[2, 0, 0] = 1
    path, count, ended = _grow(
        torch.tensor([[1., 0., 0.]]), torch.tensor([[1., 0., 0.]]),
        fod, tissues, torch.eye(4), torch.tensor([[1., 0., 0.]]),
        torch.ones((1, 1)), torch.Generator().manual_seed(0),
        step_mm=1., max_steps=3, max_angle_degrees=45., cutoff=.1, power=.5,
    )
    assert bool(ended.item())
    assert int(count.item()) == 2
    torch.testing.assert_close(path[0, 1], torch.tensor([2., 0., 0.]))


def test_direction_below_cutoff_cannot_continue():
    from fnit.connectome.tracking import _grow

    fod = torch.zeros((5, 1, 1, 2), dtype=torch.float32)
    fod[..., 0] = 0.01  # allowed forward direction
    fod[..., 1] = 1.0   # high-amplitude reverse direction, excluded by angle
    tissues = torch.full((5, 1, 1), 2, dtype=torch.int16)
    _, count, ended = _grow(
        torch.tensor([[1., 0., 0.]]), torch.tensor([[1., 0., 0.]]),
        fod, tissues, torch.eye(4),
        torch.tensor([[1., 0., 0.], [-1., 0., 0.]]), torch.eye(2),
        torch.Generator().manual_seed(0), step_mm=1., max_steps=3,
        max_angle_degrees=45., cutoff=.1, power=.5,
    )
    assert int(count.item()) == 1
    assert not bool(ended.item())
