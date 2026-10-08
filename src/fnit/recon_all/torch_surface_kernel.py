"""PyTorch surface geometry kernels for experimental recon-all stages.

This module contains device-friendly, topology-preserving primitives only.  It
does not replace FreeSurfer topology repair or white/pial placement: those
algorithms have ordering and validity constraints that still require their
validated backends.  The kernels are useful for GPU acceleration of the
deterministic geometry and metric portions once a stage has frozen its mesh.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _as_device_tensor(value: torch.Tensor, device: Optional[torch.device]) -> torch.Tensor:
    """Return a contiguous tensor on ``device`` without changing precision."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("vertices/faces must be torch.Tensor objects")
    if value.ndim not in (2, 3):
        raise ValueError("geometry tensors must be rank 2 or rank 3")
    return value.to(device=device) if device is not None else value


def face_geometry(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute triangle areas, unit normals, and face centers in parallel.

    Parameters
    ----------
    vertices:
        ``(V, 3)`` or ``(B, V, 3)`` floating point coordinates in a single
        surface RAS frame, in millimetres.
    faces:
        ``(F, 3)`` or ``(B, F, 3)`` integer vertex indices.  Batched vertices
        may share one face table.
    device:
        Optional target device, normally ``cuda`` for the experimental GPU
        path.  ``None`` preserves the input device.

    Returns
    -------
    areas, normals, centers:
        ``(F,)``/``(B,F)`` areas in mm², unit normals, and centers in mm.  A
        zero-area face receives a zero normal rather than a NaN.

    Notes
    -----
    This is a pure geometry kernel.  It deliberately does not mutate vertex
    positions, repair topology, or reproduce ``mris_fix_topology``.
    """

    v = _as_device_tensor(vertices, device)
    f = _as_device_tensor(faces, v.device).long()
    if v.shape[-1] != 3 or f.shape[-1] != 3:
        raise ValueError("vertices/faces must end in a 3-vector")
    if v.ndim == 2:
        tri = v[f]
    else:
        if f.ndim == 2:
            tri = v[:, f]
        elif f.shape[0] != v.shape[0]:
            raise ValueError("batched faces must have the same batch size")
        else:
            batch, n_faces = f.shape[:2]
            tri = v.gather(
                1,
                f.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, 3),
            ).reshape(batch, n_faces, 3, 3)
    e01 = tri[..., 1, :] - tri[..., 0, :]
    e02 = tri[..., 2, :] - tri[..., 0, :]
    cross = torch.cross(e01, e02, dim=-1)
    norm = torch.linalg.vector_norm(cross, dim=-1)
    areas = norm * 0.5
    normals = cross / norm.clamp_min(torch.finfo(cross.dtype).eps).unsqueeze(-1)
    normals = torch.where(norm.unsqueeze(-1) > 0, normals, torch.zeros_like(normals))
    centers = tri.mean(dim=-2)
    return areas, normals, centers


def vertex_area_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate area-weighted vertex normals and one-third face areas.

    The operation is fully vectorized and works with a shared face table for
    batched meshes.  It is suitable for post-placement metrics and collision
    prechecks; callers must retain the existing topology/placement algorithm.
    """

    v = _as_device_tensor(vertices, device)
    f = _as_device_tensor(faces, v.device).long()
    areas, normals, _ = face_geometry(v, f, device=v.device)
    weighted = normals * areas.unsqueeze(-1)
    if v.ndim == 2:
        out_n = torch.zeros_like(v)
        out_a = torch.zeros((v.shape[0],), dtype=v.dtype, device=v.device)
        out_n.index_add_(0, f.reshape(-1), weighted.repeat_interleave(3, 0).reshape(-1, 3))
        out_a.index_add_(0, f.reshape(-1), areas.repeat_interleave(3) / 3)
    else:
        if f.ndim == 2:
            f = f.unsqueeze(0).expand(v.shape[0], -1, -1)
        b, nv = v.shape[0], v.shape[1]
        out_n = torch.zeros_like(v)
        out_a = torch.zeros((b, nv), dtype=v.dtype, device=v.device)
        out_n.scatter_add_(1, f.reshape(b, -1, 1).expand(-1, -1, 3), weighted.unsqueeze(2).expand(-1, -1, 3, -1).reshape(b, -1, 3))
        out_a.scatter_add_(1, f.reshape(b, -1), areas.unsqueeze(-1).expand(-1, -1, 3).reshape(b, -1) / 3)
    normals = out_n / torch.linalg.vector_norm(out_n, dim=-1, keepdim=True).clamp_min(torch.finfo(v.dtype).eps)
    return out_a, normals


def edge_lengths(vertices: torch.Tensor, faces: torch.Tensor, *, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return the three edge lengths for every triangle in millimetres."""

    v = _as_device_tensor(vertices, device)
    f = _as_device_tensor(faces, v.device).long()
    if v.ndim == 2:
        tri = v[f]
    else:
        if f.ndim == 2:
            tri = v[:, f]
        else:
            batch, n_faces = f.shape[:2]
            tri = v.gather(
                1,
                f.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, 3),
            ).reshape(batch, n_faces, 3, 3)
    edge_vectors = torch.stack(
        (
            tri[..., 1, :] - tri[..., 0, :],
            tri[..., 2, :] - tri[..., 1, :],
            tri[..., 0, :] - tri[..., 2, :],
        ),
        dim=-2,
    )
    return torch.linalg.vector_norm(edge_vectors, dim=-1)
