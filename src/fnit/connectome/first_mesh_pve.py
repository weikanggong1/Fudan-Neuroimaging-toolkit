"""用 PyTorch 将 FSL FIRST 三角网格栅格化到三维体素网格。

边界相交测试和每体素 1000 点分类对应 MRtrix3 eeab681 的
``src/surface/algo/mesh2image.cpp``。
"""

from pathlib import Path

import torch
import torch.nn.functional as F


def read_first_vtk(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 ASCII FIRST VTK；返回 ``[N,3]`` 顶点和 ``[M,3]`` 三角面。"""
    words = Path(path).read_text().split()
    if words[:2] != ["#", "vtk"] or "BINARY" in words:
        raise ValueError("expected an ASCII VTK polygon mesh")
    p = words.index("POINTS")
    n = int(words[p + 1])
    vertices = torch.tensor([float(v) for v in words[p + 3 : p + 3 + 3 * n]], dtype=torch.float64).reshape(n, 3)
    q = words.index("POLYGONS", p + 3 + 3 * n)
    m = int(words[q + 1])
    face_words = [int(v) for v in words[q + 3 : q + 3 + 4 * m]]
    faces = torch.tensor(face_words, dtype=torch.int64).reshape(m, 4)
    if not bool((faces[:, 0] == 3).all()):
        raise ValueError("FIRST mesh must contain only triangles")
    return vertices, faces[:, 1:].contiguous()


def first_vertices_to_voxel(
    vertices: torch.Tensor,
    template_affine: torch.Tensor,
    template_shape: tuple[int, int, int],
) -> torch.Tensor:
    """把原始 FIRST 毫米坐标变换为目标 NIfTI 体素坐标。

    对应固定版本 ``meshconvert -transform first2real`` 和后续 real2voxel。
    ``template_affine`` 是目标体素到扫描仪 RAS 的 ``[4,4]`` 矩阵。
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3 or template_affine.shape != (4, 4):
        raise ValueError("vertices must be [N,3] and template_affine [4,4]")
    affine = template_affine.to(device=vertices.device, dtype=torch.float64)
    shape = torch.tensor(template_shape, device=vertices.device, dtype=torch.float64)
    axis = affine[:3, :3].abs().argmax(0)
    if len(torch.unique(axis)) != 3:
        raise ValueError("template affine must have three distinct dominant scanner axes")
    order = torch.argsort(axis)
    columns = affine[:3, :3][:, order]
    sizes = shape[order]
    # 标准轴均指向扫描仪坐标的正方向。
    signs = torch.stack([torch.sign(columns[i, i]) for i in range(3)])
    origin = affine[:3, 3] + (columns * torch.where(signs < 0, sizes - 1, 0)[None]).sum(1)
    canonical = columns * signs[None]
    spacing = torch.linalg.vector_norm(canonical, dim=0)
    first = vertices.to(torch.float64) / spacing
    first = first.clone()
    first[:, 0] = sizes[0] - 1 - first[:, 0]
    scanner = first @ canonical.T + origin
    return (scanner - affine[:3, 3]) @ torch.linalg.inv(affine[:3, :3]).T


def _edge_voxels(vertices: torch.Tensor, faces: torch.Tensor, shape: tuple[int, int, int]):
    """用分离轴定理把每个边界体素映射到相交三角面。"""
    tri = vertices[faces]
    normal = F.normalize(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], dim=-1), dim=-1)
    low = tri.round().amin(dim=1).to(torch.int64).clamp_min(0)
    high = tri.round().amax(dim=1).to(torch.int64)
    high = torch.minimum(high, torch.tensor(shape, device=vertices.device) - 1)
    roi_low = (low.amin(dim=0) - 2).clamp_min(0)
    roi_high = torch.minimum(high.amax(dim=0) + 2, torch.tensor(shape, device=vertices.device) - 1)
    axes = [torch.arange(int(roi_low[i]), int(roi_high[i]) + 1, device=vertices.device) for i in range(3)]
    voxels = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).to(torch.float64)
    mapping: dict[int, list[int]] = {}
    basis = torch.eye(3, dtype=torch.float64, device=vertices.device)
    for start in range(0, len(faces), 64):
        end = min(start + 64, len(faces))
        t, n = tri[start:end], normal[start:end]
        mask = ((voxels[None] >= low[start:end, None]) & (voxels[None] <= high[start:end, None])).all(-1)
        edges = torch.roll(t, -1, dims=1) - t
        test_axes = [n] + [torch.cross(basis[i].expand_as(edges[:, j]), edges[:, j], dim=-1) for i in range(3) for j in range(3)]
        for axis in test_axes:
            proj = (t * axis[:, None]).sum(-1)
            vproj = voxels @ axis.T
            radius = 0.5 * axis.abs().sum(-1)
            mask &= ((vproj + radius >= proj.amin(-1)) & (vproj - radius <= proj.amax(-1))).T
        face_ids, voxel_ids = mask.nonzero(as_tuple=True)
        for face_id, voxel_id in zip(face_ids.tolist(), voxel_ids.tolist()):
            mapping.setdefault(voxel_id, []).append(start + face_id)
    return voxels, mapping, tri, normal, roi_low


def mesh_to_pve(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """将体素坐标下的三角网格转为 ``float32 [X,Y,Z]`` PVE。

    ``vertices`` 是 ``float64 [N,3]`` 体素坐标；``faces`` 是 ``int64
    [M,3]`` 顶点索引；``shape`` 是输出网格尺寸。边界体素以 0.001
    为采样分辨率，远离边界的体素取 0 或 1。
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("vertices and faces must have shape [N,3] and [M,3]")
    device = vertices.device
    vertices = vertices.to(torch.float64)
    faces = faces.to(device=device, dtype=torch.int64)
    voxels, mapping, tri, normal, roi_low = _edge_voxels(vertices, faces, shape)
    roi_high = voxels.amax(0).to(torch.int64)
    roi_shape = tuple((roi_high - roi_low + 1).tolist())
    barrier = torch.zeros(roi_shape, dtype=torch.bool, device=device)
    edge_ids = list(mapping)
    edge_vox = voxels[edge_ids].to(torch.int64)
    local = edge_vox - roi_low
    barrier[local[:, 0], local[:, 1], local[:, 2]] = True

    # 以三角面相交体素为屏障，对外部区域进行六邻域填充。
    exterior = torch.zeros_like(barrier)
    exterior[0] = exterior[-1] = True
    exterior[:, 0] = exterior[:, -1] = True
    exterior[:, :, 0] = exterior[:, :, -1] = True
    exterior &= ~barrier
    while True:
        expanded = exterior.clone()
        for axis in range(3):
            expanded |= torch.roll(exterior, 1, axis) | torch.roll(exterior, -1, axis)
        expanded &= ~barrier
        if torch.equal(expanded, exterior):
            break
        exterior = expanded
    roi = (~barrier & ~exterior).to(torch.float32)

    offsets = torch.stack(torch.meshgrid(*[torch.arange(10, dtype=torch.float64, device=device) for _ in range(3)], indexing="ij"), -1).reshape(-1, 3).add(0.5).div(10).add(-0.5)
    max_faces = max(len(v) for v in mapping.values())
    face_ids = torch.full((len(edge_ids), max_faces), -1, dtype=torch.int64, device=device)
    for row, ids in enumerate(mapping.values()):
        face_ids[row, : len(ids)] = torch.tensor(ids, device=device)
    fractions = []
    for start in range(0, len(edge_ids), 32):
        stop = min(start + 32, len(edge_ids))
        ids = face_ids[start:stop]
        valid = ids >= 0
        t = tri[ids.clamp_min(0)]
        n = normal[ids.clamp_min(0)]
        points = voxels[edge_ids[start:stop], None, None] + offsets[None, :, None]
        centre = (t[:, :, 0] + t[:, :, 1] + t[:, :, 2]) * (1.0 / 3.0)
        diff = points - centre[:, None]
        distance = (diff * n[:, None]).sum(-1)
        projected = points - distance[..., None] * n[:, None]
        edge_distance = []
        for k in range(3):
            a, b = (k + 1) % 3, (k + 2) % 3
            edge_normal = F.normalize(torch.cross(t[:, :, a] - t[:, :, b], n, dim=-1), dim=-1)
            edge_distance.append(((projected - t[:, None, :, b]) * edge_normal[:, None]).sum(-1))
        minimum = torch.stack(edge_distance).amin(0)
        projected_inside = (minimum > 0) & valid[:, None]
        best_in = torch.where(projected_inside, distance.abs(), torch.inf).argmin(-1, keepdim=True)
        best_out = torch.where(valid[:, None], minimum, -torch.inf).argmax(-1, keepdim=True)
        best = torch.where(projected_inside.any(-1, keepdim=True), best_in, best_out)
        inside = distance.gather(-1, best).squeeze(-1) <= 0
        fractions.append(inside.to(torch.float32).mean(1))
    roi[local[:, 0], local[:, 1], local[:, 2]] = torch.cat(fractions)
    output = torch.zeros(shape, device=device, dtype=torch.float32)
    a, b, c = (int(v) for v in roi_low)
    output[a : a + roi_shape[0], b : b + roi_shape[1], c : c + roi_shape[2]] = roi
    return output


def first_vtk_to_pve(
    vtk_path: str | Path,
    template_affine: torch.Tensor,
    template_shape: tuple[int, int, int],
    device: torch.device | str,
) -> torch.Tensor:
    """从原始 FIRST VTK 输出目标 NIfTI 网格上的 ``float32 [X,Y,Z]`` PVE。

    ``vtk_path`` 指向 FSL FIRST 的 ``*_first.vtk``；``template_affine``
    是输出体素到扫描仪 RAS 的 ``[4,4]`` 矩阵；``template_shape`` 是三维
    输出尺寸；``device`` 指定 PyTorch CPU/GPU 设备。运行时不调用原软件。
    """
    vertices, faces = read_first_vtk(vtk_path)
    affine = torch.as_tensor(template_affine, dtype=torch.float64, device=device)
    vertices = first_vertices_to_voxel(vertices.to(device), affine, template_shape)
    if bool(torch.linalg.det(affine[:3, :3]) < 0):
        faces = faces[:, [0, 2, 1]]
    return mesh_to_pve(vertices, faces.to(device), template_shape)
