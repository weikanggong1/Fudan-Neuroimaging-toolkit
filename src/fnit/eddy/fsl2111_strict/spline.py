from __future__ import annotations

import math
import torch
import torch.nn.functional as F

_CUBIC_POLE = math.sqrt(3.0) - 2.0


def _periodic_prefilter_axis(x: torch.Tensor, axis: int, precision: float = 1e-8) -> torch.Tensor:
    """Port of Splinterpolator<T>::SplineColumn::Deconv(order=3, Periodic)."""
    orig_dtype = x.dtype
    x = x.movedim(axis, -1).to(torch.float64).clone()
    nsz = x.shape[-1]
    if nsz < 2:
        return x.to(orig_dtype).movedim(-1, axis)
    z = _CUBIC_POLE
    n = int(math.log(precision) / math.log(abs(z)) + 1.5)
    n = min(n, nsz)
    # Initial forward value uses the tail of the *original* column.
    iv = x[..., 0].clone()
    zp = z
    for i in range(1, n):
        iv = iv + zp * x[..., nsz - i]
        zp *= z
    x[..., 0] = iv
    for i in range(1, nsz):
        x[..., i] = x[..., i] + z * x[..., i - 1]
    # Periodic backward initialisation uses forward-swept values.
    iv = z * x[..., -1]
    z2i = z * z
    for i in range(1, n):
        iv = iv + z2i * x[..., i - 1]
        z2i *= z
    iv = iv / (z2i - 1.0)
    x[..., -1] = iv
    for i in range(nsz - 2, -1, -1):
        x[..., i] = z * (x[..., i + 1] - x[..., i])
    x = x * 6.0
    return x.to(orig_dtype).movedim(-1, axis)


def fsl_cubic_coefficients(volume: torch.Tensor, precision: float = 1e-8) -> torch.Tensor:
    """Cubic B-spline coefficients with FSL Splinterpolator periodic edges.

    Input can be (..., X, Y, Z). Coefficients are computed along the last
    three spatial dimensions and remain on the same device.
    """
    c = volume
    for axis in (-3, -2, -1):
        c = _periodic_prefilter_axis(c, axis, precision)
    return c


def _start_indices(x: torch.Tensor) -> torch.Tensor:
    # Exact order-3 branch from Splinterpolator::get_start_indicies.
    ix = torch.trunc(x + 0.5).to(torch.long)
    return torch.where(ix.to(x.dtype) < x, ix - 1, ix - 2)


def _cubic_weight(d: torch.Tensor) -> torch.Tensor:
    a = d.abs()
    out = torch.zeros_like(a)
    m = a < 1
    out[m] = (2.0 / 3.0) + 0.5 * a[m] * a[m] * (a[m] - 2.0)
    m2 = (a >= 1) & (a < 2)
    q = 2.0 - a[m2]
    out[m2] = (1.0 / 6.0) * q * q * q
    return out


def _periodic_index(idx: torch.Tensor, n: int) -> torch.Tensor:
    return torch.remainder(idx, n)


def sample_cubic_periodic(coeff: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """FSL-like cubic periodic interpolation on GPU.

    coeff: BxXxYxZ or XxYxZ prefiltered coefficients.
    coordinates: Bx3xXo xYo xZo, in source voxel coordinates.
    """
    if coeff.ndim == 3:
        coeff = coeff[None]
    if coordinates.ndim == 4:
        coordinates = coordinates[None]
    if coeff.shape[0] == 1 and coordinates.shape[0] > 1:
        coeff = coeff.expand(coordinates.shape[0], -1, -1, -1)
    if coeff.shape[0] != coordinates.shape[0]:
        raise ValueError("batch mismatch")
    B, X, Y, Z = coeff.shape
    ox, oy, oz = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    sx, sy, sz = _start_indices(ox), _start_indices(oy), _start_indices(oz)
    out = torch.zeros_like(ox)
    # Flatten spatial storage in C/FSL order x-fastest: tensor memory here z-fastest,
    # so direct advanced indexing is safer and preserves logical x/y/z indices.
    bidx = torch.arange(B, device=coeff.device).view(B, *([1] * (ox.ndim - 1)))
    for ix in range(4):
        xi = sx + ix; wx = _cubic_weight(ox - xi.to(ox.dtype)); xw = _periodic_index(xi, X)
        for iy in range(4):
            yi = sy + iy; wy = _cubic_weight(oy - yi.to(oy.dtype)); yw = _periodic_index(yi, Y)
            wxy = wx * wy
            for iz in range(4):
                zi = sz + iz; wz = _cubic_weight(oz - zi.to(oz.dtype)); zw = _periodic_index(zi, Z)
                out = out + coeff[bidx, xw, yw, zw] * wxy * wz
    return out


def sample_cubic_periodic_fast(coeff: torch.Tensor, coordinates: torch.Tensor,
                               boundary: str = 'periodic') -> torch.Tensor:
    """Evaluate the cubic spline with eight trilinear GPU fetches."""
    if coeff.ndim == 3:
        coeff = coeff[None]
    if coordinates.ndim == 4:
        coordinates = coordinates[None]
    if coeff.shape[0] == 1 and coordinates.shape[0] > 1:
        coeff = coeff.expand(coordinates.shape[0], -1, -1, -1)
    if coeff.shape[0] != coordinates.shape[0]:
        raise ValueError("batch mismatch")
    _, X, Y, Z = coeff.shape
    positions=[]; sums=[]
    for axis,n in enumerate((X,Y,Z)):
        x=coordinates[:,axis]
        start=_start_indices(x)
        weights=[_cubic_weight(x-(start+i)) for i in range(4)]
        a=weights[0]+weights[1]; b=weights[2]+weights[3]
        positions.append((start+weights[1]/a.clamp_min(1e-20),
                          start+2+weights[3]/b.clamp_min(1e-20)))
        sums.append((a,b))
    if boundary not in ('periodic','mirror'):
        raise ValueError('boundary must be periodic or mirror')
    padded=F.pad(coeff[:,None],(2,2,2,2,2,2),mode='reflect' if boundary=='mirror' else 'circular')
    grids=[]; products=[]
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                xyz=[positions[0][ix],positions[1][iy],positions[2][iz]]
                folded=[]
                for d,n in enumerate((X,Y,Z)):
                    pos=torch.remainder(xyz[d],2*n-2 if boundary=='mirror' else n)
                    if boundary=='mirror': pos=torch.minimum(pos,2*n-2-pos)
                    folded.append(2*(pos+2)/(n+3)-1)
                normalized=folded
                grids.append(torch.stack((normalized[2],normalized[1],normalized[0]),-1))
                products.append(sums[0][ix]*sums[1][iy]*sums[2][iz])
    width=coordinates.shape[-1]
    sampled=F.grid_sample(padded,torch.cat(grids,dim=3),mode='bilinear',
                          padding_mode='border',align_corners=True)[:,0]
    return sum(part*weight for part,weight in zip(sampled.split(width,dim=-1),products))


def valid_mask(coordinates: torch.Tensor, shape, ignore_axis: int | None = None) -> torch.Tensor:
    if coordinates.ndim == 4:
        coordinates = coordinates[None]
    m = torch.ones_like(coordinates[:, 0], dtype=torch.bool)
    for a, n in enumerate(shape):
        if a == ignore_axis:
            continue
        m &= (coordinates[:, a] > 0) & (coordinates[:, a] < n - 1)
    return m
