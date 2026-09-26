import torch

from fnit.connectome.dti import fit_tensor_fa


def test_tensor_fa_recovers_anisotropy_and_respects_mask():
    directions = torch.nn.functional.normalize(
        torch.tensor([
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [1., 1., 0.], [1., 0., 1.], [0., 1., 1.],
            [1., -1., 0.], [1., 0., -1.], [0., 1., -1.],
        ]), dim=-1,
    )
    bvecs = torch.cat((torch.zeros(1, 3), directions))
    bvals = torch.cat((torch.zeros(1), torch.full((9,), 1000.)))
    diffusion = torch.tensor([[0.0015, 0., 0.], [0., 0.0003, 0.], [0., 0., 0.0003]])
    atten = torch.exp(-1000 * torch.einsum('ni,ij,nj->n', directions, diffusion, directions))
    dwi = torch.cat((torch.ones(1), atten)).reshape(1, 1, 1, -1).expand(2, 1, 1, -1)
    mask = torch.tensor([[[True]], [[False]]])
    fa = fit_tensor_fa(dwi, bvals, bvecs, mask=mask)
    expected = (1.5 * ((torch.diag(diffusion) - torch.diag(diffusion).mean()) ** 2).sum()
                / (torch.diag(diffusion) ** 2).sum()).sqrt()
    assert torch.allclose(fa[0, 0, 0], expected, atol=1e-4)
    assert fa[1, 0, 0] == 0
