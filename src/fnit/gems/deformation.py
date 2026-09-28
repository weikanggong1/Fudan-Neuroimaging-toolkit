"""FreeSurfer GEMS/Ashburner tetrahedral deformation prior in PyTorch."""

from __future__ import annotations

import torch


def ashburner_prior(
    vertices: torch.Tensor,
    reference_vertices: torch.Tensor,
    tetrahedra: torch.Tensor,
    stiffness: float,
    *,
    invalid_penalty: float = 1e12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the KVL GEMS tetrahedral deformation-prior cost and Jacobians.

    This is the cost implemented in FreeSurfer
    ``kvlAtlasMeshPositionCostAndGradientCalculator``::

      K * V_ref * (1 + detJ) *
      (tr(J^T J) + tr(J^-T J^-1) - 6).

    PyTorch autograd supplies its derivative, avoiding a second hand-written
    gradient implementation.  A non-positive determinant receives a finite
    barrier cost so optimizers can reject the step.
    """
    if tetrahedra.numel() == 0:
        zero = vertices.sum() * 0
        return zero, torch.empty((0,), device=vertices.device, dtype=vertices.dtype)
    cells = tetrahedra.long()
    ref = reference_vertices[cells]
    cur = vertices[cells]
    dm = torch.stack((ref[:, 1]-ref[:, 0], ref[:, 2]-ref[:, 0], ref[:, 3]-ref[:, 0]), -1)
    ds = torch.stack((cur[:, 1]-cur[:, 0], cur[:, 2]-cur[:, 0], cur[:, 3]-cur[:, 0]), -1)
    ref_det = torch.linalg.det(dm)
    ref_volume = ref_det / 6.0
    # Atlas files are consistently oriented in KVL; retain magnitude in case a
    # converted atlas uses the opposite global orientation.
    ref_volume = ref_volume.abs()
    j = ds @ torch.linalg.inv(dm)
    detj = torch.linalg.det(j)
    safe_det = detj.clamp_min(torch.finfo(vertices.dtype).eps)
    frob = j.square().sum(dim=(-2, -1))
    invj = torch.linalg.inv(j + torch.eye(3, device=j.device, dtype=j.dtype)[None] *
                            (detj <= 0).to(j.dtype)[:, None, None] * 1e-6)
    inv_frob = invj.square().sum(dim=(-2, -1))
    per_tet = float(stiffness) * ref_volume * (1.0 + safe_det) * (frob + inv_frob - 6.0)
    per_tet = torch.where(detj > 0, per_tet,
                          per_tet.detach() * 0 + float(invalid_penalty) + (-detj) * float(invalid_penalty))
    return per_tet.sum(), detj
