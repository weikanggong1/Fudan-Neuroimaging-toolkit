"""BET2 球面演化与 ``bet -R`` 脑中心更新的 PyTorch 移植。

算法依据 FSL bet2 2111.9、meshclass 2111.0、avwutils 2209.8 和
newimage 2203.11；授权与上游源码见仓库 THIRD_PARTY_NOTICES.md。
"""

from __future__ import annotations

import math
import struct

import torch
import torch.nn.functional as F


def _float32(value: float) -> float:
    return struct.unpack('f', struct.pack('f', value))[0]


def mean_bzero(dwi: torch.Tensor, bvalues: torch.Tensor, *,
               bzero_threshold: float = 50.0) -> torch.Tensor:
    """从校正后 DWI 计算 mean b0，匹配 ``dwiextract -bzero | mrmath mean -axis 3``。

    ``dwi`` 为浮点张量 ``[X,Y,Z,N]``，``bvalues`` 为各帧 b 值 ``[N]``，
    单位 s/mm²；选取 ``bvalues < bzero_threshold`` 的帧。先转换为 float32，
    按 float64 累加并除以帧数，返回同设备 float32 ``[X,Y,Z]`` 张量。
    不修改输入，也不做掩膜或图像重排。
    """
    if dwi.ndim != 4 or not dwi.is_floating_point():
        raise ValueError('dwi must be a floating-point [X,Y,Z,N] image')
    bvalues = torch.as_tensor(bvalues, device=dwi.device)
    if bvalues.ndim != 1 or bvalues.numel() != dwi.shape[3]:
        raise ValueError('bvalues must contain one value per DWI volume')
    if not math.isfinite(bzero_threshold) or bzero_threshold <= 0:
        raise ValueError('bzero_threshold must be finite and positive')
    selected = bvalues < bzero_threshold
    if not bool(selected.any()):
        raise ValueError('no b=0 volumes selected')
    return dwi.to(torch.float32)[..., selected].mean(dim=-1, dtype=torch.float64).to(torch.float32)


def mrtrix_roundtrip_voxel_size(voxel_size: tuple[float, float, float]
                                ) -> tuple[float, float, float]:
    """还原 ``mrconvert`` 经 MIF 写入再转 NIfTI 时的三轴体素尺寸，单位 mm。

    ``voxel_size`` 是输入 DWI NIfTI header 的 ``get_zooms()[:3]``。
    MIF ``vox:`` 按六位有效数字写出，NIfTI header 再保存为 float32。
    返回的三元组传给 :func:`bet_mask`。不改变图像数据或仿射矩阵。
    """
    if len(voxel_size) != 3 or any(not math.isfinite(v) or v <= 0 for v in voxel_size):
        raise ValueError('voxel_size must contain three finite positive values')
    return tuple(_float32(float(f'{float(v):.6g}')) for v in voxel_size)


def _icosphere(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 BET2 五级二十面体，含每点邻接点和邻接三角形。"""
    tau, one = 0.8506508084, 0.5257311121
    points = [
        (tau, one, 0), (-tau, one, 0), (-tau, -one, 0), (tau, -one, 0),
        (one, 0, tau), (one, 0, -tau), (-one, 0, -tau), (-one, 0, tau),
        (0, tau, one), (0, -tau, one), (0, -tau, -one), (0, tau, -one),
    ]
    faces = [
        (4, 8, 7), (4, 7, 9), (5, 6, 11), (5, 10, 6),
        (0, 4, 3), (0, 3, 5), (2, 7, 1), (2, 1, 6),
        (8, 0, 11), (8, 11, 1), (9, 10, 3), (9, 2, 10),
        (8, 4, 0), (11, 0, 5), (4, 9, 3), (5, 3, 10),
        (7, 8, 1), (6, 1, 11), (7, 2, 9), (6, 10, 2),
    ]
    faces = [(a, c, b) for a, b, c in faces]  # Mesh::make_mesh_from_icosa::swap
    for _ in range(4):
        mid: dict[tuple[int, int], int] = {}
        next_faces: list[tuple[int, int, int]] = []

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in mid:
                mid[key] = len(points)
                points.append(tuple((points[a][k] + points[b][k]) / 2 for k in range(3)))
            return mid[key]

        for v0, v1, v2 in faces:
            p0 = midpoint(v1, v2)
            p1 = midpoint(v0, v2)
            p2 = midpoint(v0, v1)
            next_faces.extend(((p2, p0, p1), (p1, v0, p2), (p0, v2, p1), (p2, v1, p0)))
        faces = next_faces
        points = [tuple(p[k] / math.sqrt(sum(x * x for x in p)) for k in range(3)) for p in points]

    neighbours: list[list[int]] = [[] for _ in points]
    incident: list[list[int]] = [[] for _ in points]
    for fi, face in enumerate(faces):
        for i, a in enumerate(face):
            incident[a].append(fi)
            for b in (face[(i + 1) % 3], face[(i + 2) % 3]):
                if b in neighbours[a]:
                    neighbours[a].remove(b)
                neighbours[a].append(b)
    assert len(points) == 2562 and len(faces) == 5120
    assert all(len(ns) in (5, 6) for ns in neighbours)
    def padded(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(map(len, rows))
        idx = torch.tensor([r + [r[-1]] * (width - len(r)) for r in rows], device=device)
        valid = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device, dtype=torch.float64)
        return idx, valid
    ni, nv = padded(neighbours)
    ti, tv = padded(incident)
    return (torch.tensor(points, device=device, dtype=torch.float64),
            torch.tensor(faces, device=device, dtype=torch.long),
            (ni, nv), (ti, tv))


def _robust_limits(image: torch.Tensor) -> tuple[float, float]:
    """FSL newimage ``find_thresholds`` 的 1000-bin 2%/98% 规则。"""
    values = image.reshape(-1).to(torch.float64)
    lo, hi = float(values.min()), float(values.max())
    orig_lo, orig_hi = lo, hi
    t2 = t98 = 0.0
    bottom, top = 0, 999
    for iteration in range(1, 11):
        if iteration > 1:
            bottom, top = max(bottom - 1, 0), min(top + 1, 999)
            width = _float32(hi - lo)
            new_lo = _float32(lo + bottom / 1000 * width)
            hi = _float32(lo + (top + 1) / 1000 * width)
            lo = new_lo
        if iteration == 10 or lo == hi:
            lo, hi = orig_lo, orig_hi
        width = _float32(hi - lo)
        if width <= 0:
            return lo, hi
        fa = 1000.0 / width
        fb = -1000.0 * lo / width
        bins = torch.clamp((values * fa + fb).to(torch.long), 0, 999)
        hist = torch.bincount(bins, minlength=1000).cpu().tolist()
        count, first, last = len(values), 0, 999
        if iteration == 10:
            count -= hist[first] + hist[last]
            first, last = first + 1, last - 1
        if count < 0:
            return lo, lo
        bottom, seen = first, 0
        while seen < count // 50:
            seen += hist[bottom]
            bottom += 1
        bottom -= 1
        top, seen = last, 0
        while seen < count // 50:
            seen += hist[top]
            top -= 1
        top += 1
        step = width / 1000.0
        t2 = _float32(lo + _float32(bottom * step))
        t98 = _float32(lo + _float32((top + 1) * step))
        if iteration == 10 or _float32(t98 - t2) >= width / 10:
            break
    return t2, t98


def _initial_parameters(image: torch.Tensor, voxel_size: torch.Tensor,
                        center_vox: torch.Tensor | None
                        ) -> tuple[torch.Tensor, float, float, float, float]:
    t2, t98 = _robust_limits(image)
    t = t2 + 0.1 * (t98 - t2)
    size = torch.tensor(image.shape, device=image.device, dtype=torch.float64)
    count = int((image > t).sum())
    radius = (0.75 * count * float(voxel_size.prod()) / math.pi) ** (1 / 3)
    for axis in range(3):
        if float(size[axis] * voxel_size[axis]) < radius:
            radius = math.sqrt(count * float(voxel_size.prod()) / (math.pi * float(size[axis] * voxel_size[axis])))
    coords = torch.meshgrid(*(torch.arange(n, device=image.device, dtype=torch.float64)
                              for n in image.shape), indexing='ij')
    if center_vox is None:
        weights = (image.to(torch.float64) - t2).clamp(min=0, max=t98 - t2)
        weights = torch.where(image.to(torch.float64) > t, weights, 0)
        total = weights.sum()
        center = torch.stack([(weights * c * voxel_size[k]).sum() / total for k, c in enumerate(coords)])
    else:
        center = center_vox.to(device=image.device, dtype=torch.float32).to(torch.float64) * voxel_size  # bet2 -c uses float options
    metric = sum(((coords[axis] * voxel_size[axis] - center[axis]) ** 2 for axis in range(3)))
    within = (metric < radius * radius) & (image > t2) & (image < t98)
    intensities = image[within].to(torch.float64)
    if intensities.numel():
        # BET2 uses floor(n/2)-1 (zero-based) for both odd and even n.
        tm = float(intensities.kthvalue(max(1, intensities.numel() // 2)).values)
    else:
        tm = t98
    return center, radius, t2, t, tm


def _sample_nearest(image: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    ix = xyz.to(torch.long)  # C++ int cast truncates toward zero.
    shape = image.shape
    ix = torch.stack([ix[:, k].clamp(0, shape[k] - 1) for k in range(3)], -1)
    return image[ix[:, 0], ix[:, 1], ix[:, 2]]


def _evolve(image: torch.Tensor, voxel_size: torch.Tensor, center_vox: torch.Tensor | None,
            fractional_threshold: float, vertical_gradient: float,
            mesh: tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
            *, iterations: int = 1000) -> torch.Tensor:
    unit, faces, (ni, nv), (ti, tv) = mesh
    fractional_threshold = _float32(fractional_threshold)
    vertical_gradient = _float32(vertical_gradient)
    center, radius, t2, t, tm = _initial_parameters(image, voxel_size, center_vox)
    v0 = center + unit * (radius / 2)
    v = v0.clone()
    vsize = torch.tensor(image.shape, device=image.device, dtype=torch.float64)
    scale = torch.where(vsize * voxel_size > radius, voxel_size, 2 * radius)
    dscale = min(float(scale.min()), 1.0)
    normal_cut = fractional_threshold ** 0.275
    e = (1 / 3.33 + 1 / 10) / 2
    f = 6 / (1 / 3.33 - 1 / 10)
    l = 0.0
    for iteration in range(iterations):
        if iteration == 50 or iteration % 100 == 0:
            dist = torch.linalg.vector_norm(v[ni] - v[:, None, :], dim=-1)
            l = float(((dist * nv).sum(1) / nv.sum(1)).mean())
        tri = v[faces]
        triangle_normal = torch.cross(tri[:, 2] - tri[:, 0], tri[:, 1] - tri[:, 0], dim=-1)
        n = (triangle_normal[ti] * tv[..., None]).sum(1)
        n = F.normalize(n, dim=-1)
        dv = (v[ni] * nv[..., None]).sum(1) / nv.sum(1)[:, None] - v
        dot = (dv * n).sum(-1)
        sn = n * dot[:, None]
        f2 = (1 + torch.tanh(f * (2 * dot.abs() / (l * l) - e))) / 2
        u1 = (dv - sn) * 0.5
        u2 = sn * f2[:, None]

        local_t = (normal_cut + vertical_gradient * (v[:, 2] - center[2]) / radius).clamp(0, 1)
        sampling = (v - n) / scale + 0.5
        in1 = ((sampling.to(torch.long) >= 0) & (sampling.to(torch.long) < vsize)).all(-1)
        sample2 = sampling - (7 - 1) * n / scale
        in2 = ((sample2.to(torch.long) >= 0) & (sample2.to(torch.long) < vsize)).all(-1)
        imin = torch.full((len(v),), tm, device=image.device, dtype=torch.float64)
        imax = torch.full((len(v),), t, device=image.device, dtype=torch.float64)
        i1 = _sample_nearest(image, sampling)
        imin = torch.minimum(imin, i1)
        imax = torch.maximum(imax, i1)
        imin = torch.minimum(imin, _sample_nearest(image, sample2))
        delta = n / scale * dscale
        pos = sampling
        step = 2.0
        while step < 7.0:
            pos = pos - delta
            val = _sample_nearest(image, pos)
            imin = torch.minimum(imin, val)
            if step < 3.0:
                imax = torch.maximum(imax, val)
            step += dscale
        imin = torch.maximum(imin, torch.as_tensor(t2, device=image.device))
        imax = torch.minimum(imax, torch.as_tensor(tm, device=image.device))
        tl = (imax - t2) * local_t + t2
        f3 = torch.where(imax - t2 > 0, 2 * (imin - tl) / (imax - t2).clamp_min(1e-30), 2 * (imin - tl))
        f3 = torch.where(in1 & in2, f3, 0)
        v = v + u1 + u2 + n * (f3 * 0.05 * l)[:, None]
    return v


def _fill_mesh(points: torch.Tensor, faces: torch.Tensor, voxel_size: torch.Tensor,
               shape: tuple[int, int, int]) -> torch.Tensor:
    """BET2 的三角面采样及从网格重心进行 6 邻接填充。"""
    device = points.device
    tri = points[faces]
    increment = float(voxel_size.min()) / 2
    edge = tri[:, 0] - tri[:, 1]
    edge_length = torch.linalg.vector_norm(edge, dim=-1)
    edge_steps = torch.floor(edge_length / increment).to(torch.long)
    max_edge = int(edge_steps.max()) + 1
    j = torch.arange(max_edge, device=device, dtype=torch.float64) * increment
    p = tri[:, 1, None, :] + j[None, :, None] * (edge / edge_length[:, None])[:, None, :]
    valid_p = j[None, :] <= edge_length[:, None]
    span = p - tri[:, 2, None, :]
    span_length = torch.linalg.vector_norm(span, dim=-1)
    max_span = int(torch.floor(span_length.max() / increment)) + 1
    i = torch.arange(max_span, device=device, dtype=torch.float64) * increment
    xyz = tri[:, 2, None, None, :] + i[None, None, :, None] * span[:, :, None, :] / span_length.clamp_min(1e-30)[:, :, None, None]
    valid = valid_p[:, :, None] & (i[None, None, :] <= span_length[:, :, None])
    vox = torch.floor(xyz[valid] / voxel_size + 0.5).to(torch.long)
    for axis in range(3):
        vox[:, axis].clamp_(0, shape[axis] - 1)
    barrier = torch.zeros(shape, device=device, dtype=torch.bool)
    barrier[vox[:, 0], vox[:, 1], vox[:, 2]] = True
    start = torch.floor(points.mean(0) / voxel_size).to(torch.long)
    filled = torch.zeros_like(barrier)
    filled[start[0], start[1], start[2]] = True
    while True:
        moved = filled.clone()
        moved[1:] |= filled[:-1]
        moved[:-1] |= filled[1:]
        moved[:, 1:] |= filled[:, :-1]
        moved[:, :-1] |= filled[:, 1:]
        moved[:, :, 1:] |= filled[:, :, :-1]
        moved[:, :, :-1] |= filled[:, :, 1:]
        moved &= ~barrier
        if bool(torch.equal(moved, filled)):
            break
        filled = moved
    return filled | barrier


def bet_mask(mean_b0: torch.Tensor, voxel_size: tuple[float, float, float],
             *, fractional_threshold: float = 0.2, vertical_gradient: float = -0.05,
             robust_center: bool = True) -> torch.Tensor:
    """从三维 mean b0 生成 BET2 风格二值脑掩膜 ``[X,Y,Z]``。

    ``mean_b0`` 是 FSL radiological 体素顺序的非负/实值 b0 图。
    ``voxel_size`` 是三轴物理体素尺寸，单位 mm。输出为同设备 bool 张量。
    ``fractional_threshold``、``vertical_gradient`` 对应 BET ``-f``、``-g``；
    ``robust_center=True`` 对应 ``-R``。仅采用 FP32 图像与 FP64 网格，不使用半精度。
    """
    if mean_b0.ndim != 3 or not mean_b0.is_floating_point():
        raise ValueError('mean_b0 must be a floating-point [X,Y,Z] image')
    if not 0 <= fractional_threshold <= 1 or not -1 <= vertical_gradient <= 1:
        raise ValueError('BET threshold parameters are out of range')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    image = mean_b0.to(dtype=torch.float32)
    resolution = torch.tensor(voxel_size, device=image.device, dtype=torch.float64)
    mesh = _icosphere(image.device)
    center_vox = None
    result = None
    for run in range(11 if robust_center else 1):
        points = _evolve(image, resolution, center_vox, fractional_threshold, vertical_gradient, mesh)
        result = _fill_mesh(points, mesh[1], resolution, tuple(image.shape))
        if not robust_center or run == 10:
            break
        selected = result & (image > 0.001)
        weights = torch.where(selected, image, 0).to(torch.float64)
        coords = torch.meshgrid(*(torch.arange(n, device=image.device, dtype=torch.float64)
                                  for n in image.shape), indexing='ij')
        next_center = torch.stack([(c * weights).sum() / weights.sum() for c in coords])
        next_center = torch.round(next_center * 1_000_000) / 1_000_000  # fslstats -C: fixed precision 6
        if center_vox is not None and float(((next_center - center_vox) ** 2).sum() / 100) < 1:
            break
        center_vox = next_center
    assert result is not None
    return result
