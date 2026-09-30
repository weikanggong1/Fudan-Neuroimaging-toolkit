"""FreeSurfer GEMS/Ashburner tetrahedral deformation prior in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def sliding_boundary_projectors(can_move: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Project image-space gradients onto each transformed atlas boundary.

    The allowed directions are the affine's columns selected by the original
    atlas mobility flags, as in KVL's sliding boundary condition.
    """
    matrices = torch.zeros((8, 3, 3), device=transform.device, dtype=transform.dtype)
    matrices[7] = torch.eye(3, device=transform.device, dtype=transform.dtype)
    for pattern in range(1, 7):
        allowed = [(pattern & bit) != 0 for bit in (4, 2, 1)]
        basis, _ = torch.linalg.qr(transform[:, allowed], mode="reduced")
        matrices[pattern] = basis @ basis.T
    indices = (can_move.long() * can_move.new_tensor([4, 2, 1], dtype=torch.long)).sum(1)
    return matrices[indices]


@dataclass(frozen=True)
class ReferenceGeometry:
    inverse_edges: torch.Tensor
    volumes: torch.Tensor


@torch.no_grad()
def prepare_deformation_reference(reference_vertices: torch.Tensor,
                                  tetrahedra: torch.Tensor) -> ReferenceGeometry:
    """Cache fixed reference geometry in its current coordinates, dtype and device.

    Rebuild this cache after changing the reference mesh or tetrahedron ordering.
    Reference vertices are treated as constants in the cached path.
    """
    ref = reference_vertices[tetrahedra.long()]
    edges = torch.stack((ref[:, 1]-ref[:, 0], ref[:, 2]-ref[:, 0],
                         ref[:, 3]-ref[:, 0]), -1)
    volumes = (torch.linalg.det(edges) / 6.0).abs()
    return ReferenceGeometry(torch.linalg.inv(edges), volumes)


def ashburner_prior(
    vertices: torch.Tensor,
    reference_vertices: torch.Tensor,
    tetrahedra: torch.Tensor,
    stiffness: float,
    *,
    invalid_penalty: float = 1e12,
    reference_geometry: ReferenceGeometry | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the KVL GEMS tetrahedral deformation-prior cost and Jacobians.

    This is the cost implemented in FreeSurfer
    ``kvlAtlasMeshPositionCostAndGradientCalculator``::

      K * V_ref * (1 + detJ) *
      (tr(J^T J) + tr(J^-T J^-1) - 6).

    PyTorch autograd supplies its derivative, avoiding a second hand-written
    gradient implementation.  A non-positive determinant receives a finite
    barrier cost so optimizers can reject the step.

    ``reference_geometry`` may reuse fixed reference matrices across evaluations.
    """
    if tetrahedra.numel() == 0:
        zero = vertices.sum() * 0
        return zero, torch.empty((0,), device=vertices.device, dtype=vertices.dtype)
    cells = tetrahedra.long()
    cur = vertices[cells]
    ds = torch.stack((cur[:, 1]-cur[:, 0], cur[:, 2]-cur[:, 0], cur[:, 3]-cur[:, 0]), -1)
    if reference_geometry is None:
        ref = reference_vertices[cells]
        dm = torch.stack((ref[:, 1]-ref[:, 0], ref[:, 2]-ref[:, 0], ref[:, 3]-ref[:, 0]), -1)
        # Retain magnitude if an atlas uses the opposite global orientation.
        ref_volume = (torch.linalg.det(dm) / 6.0).abs()
        inverse_reference = torch.linalg.inv(dm)
    else:
        ref_volume = reference_geometry.volumes
        inverse_reference = reference_geometry.inverse_edges
    j = ds @ inverse_reference
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
