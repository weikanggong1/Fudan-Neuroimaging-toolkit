"""Analytic GMWMI interface checks for the optional ACT seeding path."""

import pytest
import torch

from fnit.connectome.tracking import sample_gmwmi_seeds


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_weighted_gmwmi_seeds_project_to_planar_interface(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    five = torch.zeros((9, 9, 9, 5), dtype=torch.float32, device=device)
    five[:4, :, :, 0] = 1
    five[4:, :, :, 2] = 1
    gmwmi = torch.zeros((9, 9, 9), dtype=torch.float32, device=device)
    gmwmi[3:5, 3:6, 3:6] = 1
    seeds = sample_gmwmi_seeds(
        gmwmi, five, torch.eye(4, device=device), 64,
        torch.Generator(device=device).manual_seed(7),
    )
    assert seeds.shape == (64, 3)
    assert bool(torch.isfinite(seeds).all())
    torch.testing.assert_close(seeds[:, 0], torch.full((64,), 3.5, device=device), atol=.01, rtol=0)


def test_gmwmi_requires_nonempty_nonnegative_weights():
    five = torch.zeros((3, 3, 3, 5))
    with pytest.raises(ValueError):
        sample_gmwmi_seeds(torch.zeros((3, 3, 3)), five, torch.eye(4), 1,
                           torch.Generator().manual_seed(0))

@pytest.mark.parametrize("device", ["cuda"])
def test_tractography_accepts_separate_fod_and_anatomy_grids(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import math
    from fnit.connectome.anatomy import gmwmi_from_five_tissue
    from fnit.connectome.fod import real_sh
    from fnit.connectome.tracking import probabilistic_tractography

    shape = (22, 11, 11)
    five = torch.zeros((*shape, 5), dtype=torch.float32, device=device)
    five[1, 1:10, 1:10, 0] = 1
    five[2:20, 1:10, 1:10, 2] = 1
    five[20, 1:10, 1:10, 0] = 1
    five = five.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)
    anatomy_affine = torch.diag(torch.tensor([.5, .5, .5, 1.], device=device))
    anatomy_affine[:3, 3] = -.25
    gmwmi = gmwmi_from_five_tissue(five)
    index = torch.arange(256, device=device, dtype=torch.float32)
    z = 1 - 2 * (index + .5) / 256
    angle = index * (math.pi * (3 - math.sqrt(5)))
    directions = torch.stack(((1 - z.square()).sqrt() * angle.cos(),
                              (1 - z.square()).sqrt() * angle.sin(), z), dim=-1)
    target = (5 / (4 * math.pi)) * directions[:, 0].pow(4)
    coefficient = torch.linalg.lstsq(real_sh(directions, 4), target).solution
    fod = torch.zeros((*shape, 15), dtype=torch.float32, device=device)
    fod[2:20, 1:10, 1:10] = coefficient
    fa = torch.full(shape, .6, dtype=torch.float32, device=device)
    tracks = probabilistic_tractography(
        fod, torch.eye(4, device=device), five, anatomy_affine,
        gmwmi, n_seeds=100, lmax=4, fa=fa, seed=11, cutoff=.01,
        power=2., max_angle_degrees=20,
    )
    assert len(tracks.paths) > 1
    assert tracks.endpoints.shape == (len(tracks.paths), 2, 3)
    assert tracks.accepted_seeds.shape == (len(tracks.paths), 3)
    assert torch.allclose(tracks.mean_fa, torch.full_like(tracks.mean_fa, .6), atol=1e-5)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_arc_requires_midpoint_fod_above_cutoff(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import math
    from fnit.connectome.tracking import _grow

    fod = torch.zeros((3, 1, 1, 1), dtype=torch.float32, device=device)
    fod[0, 0, 0, 0] = math.sqrt(4 * math.pi)
    fod[2, 0, 0, 0] = math.sqrt(4 * math.pi)
    five = torch.zeros((3, 1, 1, 5), dtype=torch.float32, device=device)
    five[..., 2] = 1
    path, counts, ended, lengths, reached_wm = _grow(
        torch.tensor([[0., 0., 0.]], device=device),
        torch.tensor([[1., 0., 0.]], device=device), fod, five,
        torch.eye(4, device=device), torch.eye(4, device=device),
        torch.Generator(device=device).manual_seed(0), lmax=0,
        proposals_per_step=1,
        step_mm=2., max_steps=1, max_angle_degrees=45., cutoff=.5, power=.5,
    )
    assert int(counts.item()) == 1
    assert not bool(ended.item())
    assert bool(reached_wm.item())
    assert float(lengths.item()) == 0
    torch.testing.assert_close(path[0, 0], torch.zeros(3, device=device))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_act_sgm_state_continues_then_exits_or_reaches_cgm(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from fnit.connectome.tracking import _act_sample_state

    tissue = torch.eye(5, device=device)[[1, 2, 0]]
    live = torch.ones(1, dtype=torch.bool, device=device)
    off = torch.zeros_like(live)
    continuing, cortical, exit_sgm, in_sgm, reached_wm, visited = _act_sample_state(
        tissue[0:1], live, off, off, off,
    )
    assert bool(continuing.item()) and bool(visited.item()) and bool(in_sgm.item())
    assert not bool(cortical.item()) and not bool(exit_sgm.item())
    continuing, cortical, exit_sgm, _, _, _ = _act_sample_state(
        tissue[1:2], continuing, in_sgm, off, reached_wm,
    )
    assert bool(exit_sgm.item()) and not bool(continuing.item())
    continuing, cortical, exit_sgm, in_sgm, reached_wm, _ = _act_sample_state(
        tissue[0:1], live, off, live, off,
    )
    continuing, cortical, exit_sgm, in_sgm, reached_wm, _ = _act_sample_state(
        tissue[1:2], continuing, in_sgm, live, reached_wm,
    )
    assert bool(continuing.item()) and bool(reached_wm.item())
    assert not bool(exit_sgm.item()) and not bool(in_sgm.item())
    continuing, cortical, exit_sgm, in_sgm, reached_wm, _ = _act_sample_state(
        tissue[0:1], live, off, off, off,
    )
    continuing, cortical, exit_sgm, _, _, _ = _act_sample_state(
        tissue[2:3], continuing, in_sgm, off, reached_wm,
    )
    assert bool(cortical.item()) and not bool(continuing.item())
    assert not bool(exit_sgm.item())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_act_sgm_tracks_to_cgm_and_truncates_at_minimum_fod(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import math
    from fnit.connectome.tracking import _grow

    fod = torch.full((12, 3, 3, 1), math.sqrt(4 * math.pi), device=device)
    five = torch.zeros((12, 3, 3, 5), device=device)
    five[..., 2] = 1
    five[4:8, ..., 1] = 1
    five[4:8, ..., 2] = 0
    five[8:, ..., 0] = 1
    five[8:, ..., 2] = 0
    seeds = torch.tensor([[2., 1., 1.]], device=device)
    direction = torch.tensor([[1., 0., 0.]], device=device)
    path, count, ended, length, reached_wm = _grow(
        seeds, direction, fod, five, torch.eye(4, device=device),
        torch.eye(4, device=device), torch.Generator(device=device).manual_seed(0),
        lmax=0, proposals_per_step=1, step_mm=1., max_steps=10,
        max_angle_degrees=0., cutoff=.1, power=.5,
    )
    assert bool(ended.item()) and bool(reached_wm.item())
    assert float(path[0, count.item() - 1, 0]) >= 7.5
    assert float(length.item()) >= 5.5

    five[..., 0] = 0
    five[..., 2] = 1
    five[3:6, ..., 1] = 1
    five[3:6, ..., 2] = 0
    fod[3, ..., 0] *= .7
    fod[4, ..., 0] *= .4
    fod[5, ..., 0] *= .7
    seeds[:, 0] = 1
    path, count, ended, length, _ = _grow(
        seeds, direction, fod, five, torch.eye(4, device=device),
        torch.eye(4, device=device), torch.Generator(device=device).manual_seed(0),
        lmax=0, proposals_per_step=1, step_mm=1., max_steps=10,
        max_angle_degrees=0., cutoff=.1, power=.5,
    )
    assert bool(ended.item())
    torch.testing.assert_close(path[0, count.item() - 1, 0],
                               torch.tensor(4., device=device), atol=.01, rtol=0)
    torch.testing.assert_close(length, torch.tensor([3.], device=device), atol=.01, rtol=0)
