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
    edges: torch.Tensor | None = None


@dataclass(frozen=True)
class CurrentGeometry:
    origins: torch.Tensor
    edges: torch.Tensor
    inverse_edges: torch.Tensor
    determinants: torch.Tensor
    singular: torch.Tensor
    deterministic_gradient: bool = False


@dataclass(frozen=True)
class VertexReduction:
    """Fixed corner ordering for a shared-vertex gradient reduction.

    Rebuild after changing the vertex IDs, their ordering, or the vertex count.
    The source tensor is retained so in-place changes cannot reuse stale offsets.
    """
    order: torch.Tensor
    offsets: torch.Tensor
    vertex_count: int
    source_ids: torch.Tensor
    source_version: int


@torch.no_grad()
def prepare_vertex_reduction(vertex_ids: torch.Tensor, vertex_count: int) -> VertexReduction:
    """Sort valid, nonnegative vertex IDs stably, entirely on their device."""
    ids = vertex_ids.reshape(-1)
    if ids.dtype != torch.long:
        raise ValueError("vertex IDs must have torch.long dtype")
    if not isinstance(vertex_count, int) or isinstance(vertex_count, bool) or vertex_count < 0:
        raise ValueError("vertex_count must be a nonnegative integer")
    order = torch.argsort(ids, stable=True)
    boundaries = torch.arange(vertex_count + 1, device=ids.device, dtype=torch.long)
    offsets = torch.searchsorted(ids[order], boundaries)
    return VertexReduction(order, offsets, vertex_count, ids, ids._version)


def ordered_vertex_sum(vertex_ids: torch.Tensor, contributions: torch.Tensor,
                       vertex_count: int, reduction: VertexReduction | None = None) -> torch.Tensor:
    """Sum corner contributions stably in FP64, then restore their dtype.

    ``contributions`` is [number of IDs, channels]. Stable sorting preserves the input
    corner order within each vertex. The segment sum assigns a separate
    output element to each (vertex, channel), including zeros for isolated vertices.
    FP64 accumulation reduces cancellation error. Vertex gradients retain the
    contributions' original dtype and no tensor is copied to the host.
    """
    ids = vertex_ids.reshape(-1)
    if (contributions.ndim != 2 or contributions.shape[0] != ids.numel()
            or contributions.shape[1] < 1 or contributions.device != ids.device):
        raise ValueError("contributions must be [number of vertex IDs, channels] on the IDs device")
    if reduction is None:
        reduction = prepare_vertex_reduction(ids, vertex_count)
    elif (reduction.vertex_count != vertex_count or reduction.source_ids.device != ids.device
          or reduction.source_ids.shape != ids.shape or reduction.source_ids.stride() != ids.stride()
          or reduction.source_ids.data_ptr() != ids.data_ptr()
          or reduction.source_ids._version != reduction.source_version):
        raise ValueError("vertex reduction changed; rebuild it for the current vertex IDs")
    # The IDs originate from a validated tetrahedral mesh. Their stable sorted
    # offsets partition all rows, so the segment operator needs no host checks.
    summed = torch.segment_reduce(contributions[reduction.order].to(torch.float64), "sum",
                                  offsets=reduction.offsets, axis=0, unsafe=True)
    return summed.to(contributions.dtype)


class _OrderedRowGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, row_ids, reduction):
        ctx.save_for_backward(row_ids)
        ctx.reduction = reduction
        ctx.values_shape = tuple(values.shape)
        return values[row_ids]

    @staticmethod
    def backward(ctx, gradient):
        row_ids, = ctx.saved_tensors
        channels = 1
        for width in ctx.values_shape[1:]:
            channels *= width
        if row_ids.numel() == 0 or channels == 0:
            result = gradient.new_zeros(ctx.values_shape)
        else:
            result = ordered_vertex_sum(
                row_ids.reshape(-1), gradient.reshape(row_ids.numel(), channels),
                ctx.values_shape[0], ctx.reduction).reshape(ctx.values_shape)
        return result, None, None


def ordered_row_gather(values: torch.Tensor, row_ids: torch.Tensor,
                       reduction: VertexReduction | None = None) -> torch.Tensor:
    """Gather first-axis rows with a fixed, FP64 backward accumulation.

    The forward is the same as ``values[row_ids]``. Repeated valid nonnegative
    row IDs accumulate gradients in their input order, using the existing
    ordered segment sum, and return ``values``' original gradient dtype.
    ``row_ids`` can have any shape; all its entries must be valid row indices.
    A layout can be shared by gathers using the same unchanged ID tensor.
    No global deterministic setting or cuBLAS workspace is required.
    """
    if values.ndim < 1 or row_ids.dtype != torch.long or values.device != row_ids.device:
        raise ValueError("row gather requires values and long row IDs on the same device")
    if not torch.is_grad_enabled() or not values.requires_grad:
        return values[row_ids]
    if reduction is None:
        reduction = prepare_vertex_reduction(row_ids.reshape(-1), len(values))
    return _OrderedRowGather.apply(values, row_ids, reduction)


class _OrderedVertexGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, tetrahedra, reduction):
        ctx.save_for_backward(tetrahedra)
        ctx.reduction = reduction
        ctx.vertex_count = vertices.shape[0]
        return vertices[tetrahedra]

    @staticmethod
    def backward(ctx, gradient):
        tetrahedra, = ctx.saved_tensors
        result = ordered_vertex_sum(tetrahedra.reshape(-1), gradient.reshape(-1, 3),
                                    ctx.vertex_count, ctx.reduction)
        return result, None, None


class _InverseEdges(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edges):
        inverse, info = torch.linalg.inv_ex(edges, check_errors=False)
        # A singular cell has no inverse derivative. Zero its unused inverse so
        # a masked raster cell cannot introduce NaNs into the shared backward.
        inverse = torch.where((info == 0)[:, None, None], inverse, 0)
        ctx.save_for_backward(inverse)
        ctx.mark_non_differentiable(info)
        return inverse, info

    @staticmethod
    def backward(ctx, gradient, _):
        inverse, = ctx.saved_tensors
        transposed = inverse.transpose(-1, -2)
        return -(transposed @ gradient @ transposed)


class _Determinant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix):
        ctx.save_for_backward(matrix)
        return torch.linalg.det(matrix)

    @staticmethod
    def backward(ctx, gradient):
        matrix, = ctx.saved_tensors
        return gradient[:, None, None] * _cofactor(matrix)


def prepare_current_geometry(vertices: torch.Tensor,
                             tetrahedra: torch.Tensor, *,
                             deterministic_gradient: bool = False,
                             vertex_reduction: VertexReduction | None = None) -> CurrentGeometry:
    """Share differentiable current geometry within one mesh evaluation.

    Rebuild after every vertex change. The object retains this evaluation's
    autograd graph and must not be cached across optimizer steps.
    ``deterministic_gradient`` replaces the shared-vertex scatter backward with
    a fixed tetrahedron/corner reduction. Its layout may be reused while the
    tetrahedra remain unchanged; FP64 accumulation returns the original dtype.
    """
    cells_index = tetrahedra.long()
    if deterministic_gradient:
        if vertex_reduction is None:
            vertex_reduction = prepare_vertex_reduction(cells_index.reshape(-1), vertices.shape[0])
        cells = _OrderedVertexGather.apply(vertices, cells_index, vertex_reduction)
    else:
        cells = vertices[cells_index]
    origins = cells[:, 0]
    edges = torch.stack((cells[:, 1]-origins, cells[:, 2]-origins,
                         cells[:, 3]-origins), -1)
    inverse, info = _InverseEdges.apply(edges)
    determinants = _Determinant.apply(edges)
    singular = (info != 0) | (determinants.abs() <= 1e-10)
    return CurrentGeometry(origins, edges, inverse, determinants, singular,
                           deterministic_gradient=bool(deterministic_gradient))


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
    return ReferenceGeometry(torch.linalg.inv(edges), volumes, edges)


def _cofactor(matrix):
    return torch.stack((torch.linalg.cross(matrix[..., 1], matrix[..., 2]),
                        torch.linalg.cross(matrix[..., 2], matrix[..., 0]),
                        torch.linalg.cross(matrix[..., 0], matrix[..., 1])), -1)


class _AnalyticPrior(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edges, inverse_reference, reference_edges, inverse_edges,
                volumes, stiffness, invalid_penalty, double_accumulation):
        j = edges @ inverse_reference
        detj = torch.linalg.det(j)
        invj = reference_edges @ inverse_edges
        # The invalid branch is a determinant barrier, not inverse energy.
        invj = torch.where((detj > 0)[:, None, None], invj, 0)
        safe_det = detj.clamp_min(torch.finfo(edges.dtype).eps)
        distortion = j.square().sum((-2, -1)) + invj.square().sum((-2, -1)) - 6
        scale = float(stiffness) * volumes
        energy = scale * (1 + safe_det) * distortion
        cost = torch.where(detj > 0, energy, float(invalid_penalty) * (1 - detj)).sum(
            dtype=torch.float64 if double_accumulation else edges.dtype)
        ctx.save_for_backward(j, invj, detj, distortion, scale, inverse_reference)
        ctx.invalid_penalty = float(invalid_penalty)
        return cost, detj

    @staticmethod
    def backward(ctx, cost_gradient, jacobian_gradient):
        j, invj, detj, distortion, scale, inverse_reference = ctx.saved_tensors
        cofactor = _cofactor(j)
        transposed = invj.transpose(-1, -2)
        distortion_gradient = 2 * (j - transposed @ invj @ transposed)
        eps = torch.finfo(j.dtype).eps
        gradient = scale[:, None, None] * (
            (1 + detj.clamp_min(eps))[:, None, None] * distortion_gradient
            + (distortion * (detj >= eps))[:, None, None] * cofactor)
        gradient = torch.where((detj > 0)[:, None, None], gradient,
                               -ctx.invalid_penalty * cofactor)
        if cost_gradient is not None:
            gradient = gradient * cost_gradient
        else:
            gradient = torch.zeros_like(gradient)
        if jacobian_gradient is not None:
            gradient = gradient + jacobian_gradient[:, None, None] * cofactor
        return gradient @ inverse_reference.transpose(-1, -2), None, None, None, None, None, None, None


def ashburner_prior(
    vertices: torch.Tensor,
    reference_vertices: torch.Tensor,
    tetrahedra: torch.Tensor,
    stiffness: float,
    *,
    invalid_penalty: float = 1e12,
    reference_geometry: ReferenceGeometry | None = None,
    current_geometry: CurrentGeometry | None = None,
    analytic_gradient: bool = False,
    double_accumulation: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the KVL GEMS tetrahedral deformation-prior cost and Jacobians.

    This is the cost implemented in FreeSurfer
    ``kvlAtlasMeshPositionCostAndGradientCalculator``::

      K * V_ref * (1 + detJ) *
      (tr(J^T J) + tr(J^-T J^-1) - 6).

    The default path uses PyTorch autograd. A non-positive determinant receives
    a finite barrier cost so optimizers can reject the step.

    ``reference_geometry`` may reuse fixed reference matrices across evaluations.
    ``current_geometry`` shares edges and their inverse with rasterization.
    ``analytic_gradient`` differentiates current edges directly while treating
    the reference mesh as fixed; it supports the same Jacobian barrier and clamp.
    ``double_accumulation`` only promotes the final scalar sum; per-tetrahedron
    arithmetic, geometry, Jacobians and vertex gradients retain their dtype.
    """
    if tetrahedra.numel() == 0:
        zero = vertices.sum(dtype=torch.float64 if double_accumulation else vertices.dtype) * 0
        return zero, torch.empty((0,), device=vertices.device, dtype=vertices.dtype)
    cells = tetrahedra.long()
    if current_geometry is None and analytic_gradient:
        current_geometry = prepare_current_geometry(vertices, tetrahedra)
    if current_geometry is None:
        cur = vertices[cells]
        ds = torch.stack((cur[:, 1]-cur[:, 0], cur[:, 2]-cur[:, 0], cur[:, 3]-cur[:, 0]), -1)
    else:
        ds = current_geometry.edges
    if reference_geometry is None:
        ref = reference_vertices[cells]
        dm = torch.stack((ref[:, 1]-ref[:, 0], ref[:, 2]-ref[:, 0], ref[:, 3]-ref[:, 0]), -1)
        # Retain magnitude if an atlas uses the opposite global orientation.
        ref_volume = (torch.linalg.det(dm) / 6.0).abs()
        inverse_reference = torch.linalg.inv(dm)
    else:
        ref_volume = reference_geometry.volumes
        inverse_reference = reference_geometry.inverse_edges
        dm = reference_geometry.edges
        if dm is None and current_geometry is not None:
            ref = reference_vertices[cells]
            dm = torch.stack((ref[:, 1]-ref[:, 0], ref[:, 2]-ref[:, 0], ref[:, 3]-ref[:, 0]), -1)
    if analytic_gradient:
        return _AnalyticPrior.apply(ds, inverse_reference.detach(), dm.detach(),
                                    current_geometry.inverse_edges.detach(), ref_volume.detach(),
                                    float(stiffness), float(invalid_penalty), bool(double_accumulation))
    j = ds @ inverse_reference
    detj = (torch.linalg.det(j) if current_geometry is None else _Determinant.apply(j))
    safe_det = detj.clamp_min(torch.finfo(vertices.dtype).eps)
    frob = j.square().sum(dim=(-2, -1))
    if current_geometry is None:
        invj = torch.linalg.inv(j + torch.eye(3, device=j.device, dtype=j.dtype)[None] *
                                (detj <= 0).to(j.dtype)[:, None, None] * 1e-6)
    else:
        invj = dm @ current_geometry.inverse_edges
        invj = torch.where((detj > 0)[:, None, None], invj, 0)
    inv_frob = invj.square().sum(dim=(-2, -1))
    per_tet = float(stiffness) * ref_volume * (1.0 + safe_det) * (frob + inv_frob - 6.0)
    per_tet = torch.where(detj > 0, per_tet,
                          per_tet.detach() * 0 + float(invalid_penalty) + (-detj) * float(invalid_penalty))
    return per_tet.sum(dtype=torch.float64 if double_accumulation else vertices.dtype), detj
