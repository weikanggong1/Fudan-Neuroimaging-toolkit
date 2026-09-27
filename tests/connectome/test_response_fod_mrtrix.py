"""Known-signal checks for the official MRtrix response and FOD equations."""

import math

import pytest
import torch

from fnit.connectome.fod import (
    _mrtrix_msmt_design, fit_mrtrix_msmt_csd, fit_mrtrix_two_tissue_csd,
)
from fnit.connectome.response import estimate_mrtrix_selected_response


def _gradients() -> torch.Tensor:
    generator = torch.Generator().manual_seed(71)
    directions = torch.randn((72, 3), generator=generator)
    directions = torch.nn.functional.normalize(directions, dim=1)
    gradients = torch.zeros((145, 4), dtype=torch.float64)
    gradients[0, :3] = torch.tensor([0., 0., 1.])
    gradients[0, 3] = 5
    gradients[1:73, :3] = directions
    gradients[1:73, 3] = 1000
    gradients[73:, :3] = directions
    gradients[73:, 3] = 2000
    return gradients


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_fixed_response_msmt_recovers_feasible_known_signal(device):
    gradient = _gradients().to(device)
    shell = torch.tensor([5., 1000., 2000.], device=device)
    wm_response = torch.tensor(
        [[9165.9, 0, 0, 0, 0, 0],
         [4945.1, -1945.4, 404.5, -58.0, 8.0, -6.3],
         [3412.4, -1945.0, 699.1, -175.7, 34.5, -7.6]],
        device=device,
    )
    gm_response = torch.tensor([18097.1, 8349.7, 4592.7], device=device)
    csf_response = torch.tensor([26239.4, 979.9, 389.7], device=device)
    design, _ = _mrtrix_msmt_design(
        gradient, shell, wm_response, gm_response, csf_response,
    )
    truth = torch.zeros(47, dtype=torch.float64, device=device)
    truth[0], truth[45], truth[46] = 0.25, 0.15, 0.04
    signal = (design @ truth).to(torch.float32).reshape(1, 1, 1, -1)
    wm, gm, csf = fit_mrtrix_msmt_csd(
        signal, gradient, shell, wm_response, gm_response, csf_response,
        torch.ones((1, 1, 1), dtype=torch.bool, device=device), batch_size=1,
    )
    recovered = torch.cat((wm.flatten(), gm.flatten(), csf.flatten()))
    assert torch.allclose(recovered, truth.to(torch.float32), atol=3e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_two_tissue_icls_is_independent_of_batch_partition(device):
    gradient = _gradients().to(device)
    bvalues = gradient[:, 3]
    signal = torch.stack((
        1000 * torch.exp(-0.0008 * bvalues),
        900 * torch.exp(-0.0012 * bvalues),
    )).to(torch.float32).reshape(2, 1, 1, -1)
    shell = torch.tensor([5., 1000., 2000.], device=device)
    wm_response = torch.tensor(
        [[9165.9, 0.], [4945.1, -1945.4], [3412.4, -1945.0]], device=device,
    )
    csf_response = torch.tensor([26239.4, 979.9, 389.7], device=device)
    mask = torch.ones((2, 1, 1), dtype=torch.bool, device=device)
    first = fit_mrtrix_two_tissue_csd(
        signal, gradient, shell, wm_response, csf_response, mask,
        lmax=2, batch_size=1,
    )
    second = fit_mrtrix_two_tissue_csd(
        signal, gradient, shell, wm_response, csf_response, mask,
        lmax=2, batch_size=2,
    )
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right, rtol=0, atol=1e-7)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_selected_isotropic_response_uses_raw_mrtrix_zonal_scale(device):
    signal = torch.tensor([100., 50., 25.], device=device).reshape(1, 1, 1, 3)
    gradient = torch.tensor(
        [[0., 0., 1., 5.], [1., 0., 0., 1000.], [0., 1., 0., 2000.]],
        device=device,
    )
    mask = torch.ones((1, 1, 1), dtype=torch.bool, device=device)
    response = estimate_mrtrix_selected_response(
        signal, gradient, torch.tensor([5., 1000., 2000.], device=device),
        mask, isotropic=True,
    )
    expected = signal.flatten().to(torch.float64) * math.sqrt(4 * math.pi)
    assert torch.allclose(response.flatten(), expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_dhollander_preparation_erodes_faces_and_weights_shells(device):
    from fnit.connectome.response import prepare_mrtrix_dhollander_sdm

    signal = torch.empty((5, 5, 5, 4), device=device, dtype=torch.float32)
    signal[..., 0] = 100
    signal[..., 1] = 50
    signal[..., 2:] = 25
    gradient = torch.tensor(
        [[0., 0., 1., 5.], [1., 0., 0., 1000.],
         [0., 1., 0., 2000.], [0., 0., 1., 2000.]],
        device=device,
    )
    mask = torch.ones((5, 5, 5), device=device, dtype=torch.bool)
    eroded, safe, sdm = prepare_mrtrix_dhollander_sdm(
        signal, gradient, torch.tensor([5., 1000., 2000.], device=device),
        mask, erosion_passes=1,
    )
    assert int(eroded.sum()) == 27
    assert torch.equal(eroded, safe)
    expected = (math.log(2) + 2 * math.log(4)) / 3
    assert torch.allclose(sdm[safe], torch.full((27,), expected, device=device), atol=1e-6)
    assert int(torch.count_nonzero(sdm[~safe])) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_dhollander_iwls_tensor_recovers_known_eigenvector(device):
    from fnit.connectome.response import fit_mrtrix_dhollander_tensor

    gradient = _gradients().to(device)
    gx, gy, gz, b = gradient.unbind(dim=1)
    d11, d22, d33 = 0.0015, 0.0006, 0.0004
    exponent = -b * (d11 * gx.square() + d22 * gy.square() + d33 * gz.square())
    signal = (1000 * exponent.exp()).to(torch.float32).reshape(1, 1, 1, -1)
    mask = torch.ones((1, 1, 1), device=device, dtype=torch.bool)
    fa, vec = fit_mrtrix_dhollander_tensor(signal, gradient, mask, batch_size=1)
    eigenvalues = torch.tensor([d11, d22, d33], device=device)
    expected_fa = math.sqrt(
        1.5 * float(((eigenvalues - eigenvalues.mean()).square().sum())
                    / eigenvalues.square().sum())
    )
    assert abs(float(fa[0, 0, 0]) - expected_fa) < 1e-6
    assert abs(float(vec[0, 0, 0, 0])) > 0.99999


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_mrtrix_shell_labels_preserve_round_vs_truncate(device):
    from fnit.connectome.response import mrtrix_shell_centres

    bvalues = [5] * 5 + [998, 999, 999, 1000, 1000] + [1996, 1997, 1998, 1998, 1999]
    gradient = torch.zeros((len(bvalues), 4), device=device, dtype=torch.float64)
    gradient[:, 3] = torch.tensor(bvalues, device=device, dtype=torch.float64)
    means, selection, header, sizes = mrtrix_shell_centres(gradient)
    assert torch.equal(means, torch.tensor([5., 999.2, 1997.6], device=device, dtype=torch.float64))
    assert torch.equal(selection, torch.tensor([5., 999., 1998.], device=device, dtype=torch.float64))
    assert torch.equal(header, torch.tensor([5., 999., 1997.], device=device, dtype=torch.float64))
    assert sizes.tolist() == [5, 5, 5]


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
@pytest.mark.parametrize("lmax", [2, 6])
def test_mrtrix_fod_peak_amplitude_recovers_zonal_peak(device, lmax):
    from fnit.connectome.fod import mrtrix_fod_peak_amplitude, real_sh

    coefficient = torch.zeros((2, (lmax + 1) * (lmax + 2) // 2), device=device)
    coefficient[:, 0] = 1.5
    coefficient[:, 3] = torch.tensor([2., 1.], device=device)
    z = torch.tensor([[0., 0., 1.]], device=device)
    expected = (coefficient.double() * real_sh(z.double(), lmax)).sum(dim=1)
    actual = mrtrix_fod_peak_amplitude(coefficient, lmax)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_mrtrix_tensor_marks_all_nonpositive_voxel_nan(device):
    from fnit.connectome.response import fit_mrtrix_dhollander_tensor

    gradient = _gradients().to(device)
    gx, gy, gz, b = gradient.unbind(dim=1)
    valid = (1000 * (-b * (0.0015 * gx.square() + 0.0006 * gy.square()
                          + 0.0004 * gz.square())).exp()).float()
    signal = torch.stack((valid, -torch.ones_like(valid))).reshape(2, 1, 1, -1)
    mask = torch.ones((2, 1, 1), device=device, dtype=torch.bool)
    fa, direction = fit_mrtrix_dhollander_tensor(signal, gradient, mask)
    assert torch.isfinite(fa[0, 0, 0])
    assert torch.isnan(fa[1, 0, 0])
    assert torch.isnan(direction[1, 0, 0]).all()
