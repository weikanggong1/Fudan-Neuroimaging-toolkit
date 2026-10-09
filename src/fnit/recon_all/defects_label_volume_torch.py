"""用 PyTorch 投射固定 recon-all 的拓扑缺陷；不运行外部程序。"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import struct
import time

import nibabel as nib
import numpy as np
import torch

from .place_surface_geometry import surface_ras_to_voxel


def _unit(vector: torch.Tensor) -> torch.Tensor:
    # 固定源码 mrisNormalize 的 float32、逐项求和顺序；不使用 eps 截断。
    length = torch.sqrt((vector[:, 0] * vector[:, 0] + vector[:, 1] * vector[:, 1])
                        + vector[:, 2] * vector[:, 2])
    return vector / torch.where(length > 0, length, torch.ones_like(length))[:, None]


def project_defects(vertices: torch.Tensor, faces: torch.Tensor, defects: torch.Tensor,
                    template: torch.Tensor, ras_to_voxel: torch.Tensor, *,
                    offset: int, voxel_size_x: float, merge: bool = False,
                    cortex: torch.Tensor | None = None) -> torch.Tensor:
    """返回模板网格的 int32 缺陷分割，不修改输入。

    vertices 为 (V,3) float32 surface RAS/mm；faces 为 (F,3) int64 有序面；
    defects 为 (V,) 非负整数缺陷号；template 为 (X,Y,Z)；ras_to_voxel 为
    (4,4) float32 的 surface RAS→模板体素变换。所有输入须同设备。
    offset 为半球标签偏移（1000/2000）；voxel_size_x 为模板 x 体素大小/mm；
    merge=False 清零模板，True 保留旧标签；cortex 为 (V,) bool，可为 None。
    面心两侧投射距离为 voxel_size_x/5，不沿顶点法向或修改拓扑。
    重叠处保留原遍历中最后一个顶点的标签，不能用最大标签代替。
    返回 (X,Y,Z) int32；非法输入抛 ValueError，CUDA错误原样传播。
    """
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype != torch.float32
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != torch.int64
            or defects.shape != vertices.shape[:1] or defects.is_floating_point()
            or template.ndim != 3 or ras_to_voxel.shape != (4, 4)
            or ras_to_voxel.dtype != torch.float32):
        raise ValueError("invalid surface, labels, template or transform shape/dtype")
    inputs = (faces, defects, template, ras_to_voxel)
    if any(value.device != vertices.device for value in inputs):
        raise ValueError("all tensors must be on the same device")
    if cortex is not None and (cortex.shape != defects.shape or cortex.dtype != torch.bool
                               or cortex.device != vertices.device):
        raise ValueError("cortex must be a matching bool tensor on the same device")
    if (offset < 0 or not np.isfinite(voxel_size_x) or voxel_size_x <= 0
            or not bool(torch.isfinite(vertices).all())
            or not bool(torch.isfinite(ras_to_voxel).all())
            or bool((defects < 0).any())):
        raise ValueError("nonfinite geometry, negative label/offset or invalid spacing")
    if faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError("face vertex index out of range")
    labels = torch.where(cortex, defects, 0) if cortex is not None else defects
    if len(labels) and int(labels.max()) + offset > np.iinfo(np.int32).max:
        raise ValueError("defect label exceeds int32 range")
    result = (template.to(torch.int32).clone() if merge else
              torch.zeros_like(template, dtype=torch.int32)).contiguous()
    # 面投射点与关联顶点无关。同一面的最后非零顶点优先，等价于源码顶点循环。
    owner = torch.where(labels[faces] != 0, faces, -1).amax(dim=1)
    selected = owner >= 0
    owner, selected_faces = owner[selected], faces[selected]
    if not len(owner):
        return result
    tri = vertices[selected_faces]
    normal = _unit(torch.cross(_unit(tri[:, 0] - tri[:, 2]),
                               _unit(tri[:, 1] - tri[:, 0]), dim=1))
    # 源码面心以 double 求和；再转 float32 进入 MATRIX_REAL 乘法。
    centroid = (tri[:, 0].double() + tri[:, 1].double() + tri[:, 2].double()) / 3.0
    points = torch.stack((centroid - (voxel_size_x / 5.0) * normal.double(),
                          centroid + (voxel_size_x / 5.0) * normal.double()), dim=1).float()
    coordinates = []
    for row in range(3):
        value = points[..., 0] * ras_to_voxel[row, 0]
        value = value + points[..., 1] * ras_to_voxel[row, 1]
        value = value + points[..., 2] * ras_to_voxel[row, 2]
        coordinates.append(value + ras_to_voxel[row, 3])
    xyz = torch.stack(coordinates, dim=-1)
    # FreeSurfer nint：正负半整数均远离0，不能用 torch.round 的偶数舍入。
    ijk = torch.where(xyz >= 0, torch.floor(xyz + .5), torch.ceil(xyz - .5)).long()
    shape = torch.tensor(template.shape, device=vertices.device)
    valid = ((ijk >= 0) & (ijk < shape)).all(dim=-1)
    ijk = ijk[valid]
    owners = owner[:, None].expand(-1, 2)[valid]
    flat = (ijk[:, 0] * template.shape[1] + ijk[:, 1]) * template.shape[2] + ijk[:, 2]
    # 整数 max 归约是确定的：选择遍历最后写入者，而不是atomic label写入。
    winners = torch.full((template.numel(),), -1, dtype=torch.int64, device=vertices.device)
    winners.scatter_reduce_(0, flat, owners, reduce="amax", include_self=True)
    occupied = winners >= 0
    result.reshape(-1)[occupied] = (labels[winners[occupied]] + offset).to(torch.int32)
    return result


def _ctab(payload: bytes, start: int):
    """读取已有的 TAG_OLD_COLORTABLE v2，返回终点和颜色表；不猜测格式。"""
    position = start
    def integer():
        nonlocal position
        value = struct.unpack_from(">i", payload, position)[0]
        position += 4
        return value
    if integer() != 1 or integer() != -2:
        raise ValueError("expected MGH color table version 2")
    capacity = integer()
    length = integer()
    if length < 0 or length > len(payload) - position:
        raise ValueError("invalid color table filename")
    position += length
    count, entries = integer(), {}
    if count < 0 or count > capacity:
        raise ValueError("invalid color table entry count")
    for _ in range(count):
        index, length = integer(), integer()
        if length < 1 or length > len(payload) - position:
            raise ValueError("invalid color table name")
        name = payload[position:position + length].rstrip(b"\0").decode("utf-8")
        position += length
        entries[index] = (name, integer(), integer(), integer(), integer())
    return position, entries


def _footer_and_colors(raw: bytes, end: int):
    footer = raw[end:]
    if len(footer) < 20:
        raise ValueError("truncated MGH scan parameters")
    kept, entries, cursor = bytearray(footer[:20]), {}, 20
    while cursor < len(footer):
        if len(footer) - cursor < 4:
            raise ValueError("truncated MGH tag")
        tag = struct.unpack_from(">i", footer, cursor)[0]
        if tag == 1:
            stop, entries = _ctab(footer, cursor)
        elif tag == 30:  # TAG_OLD_MGH_XFORM: int32 length includes an old extra byte
            length = struct.unpack_from(">i", footer, cursor + 4)[0] - 1
            stop = cursor + 8 + length
        elif tag == 2:  # TAG_OLD_USEREALRAS
            stop = cursor + 8
        else:
            length = struct.unpack_from(">q", footer, cursor + 4)[0]
            stop = cursor + 12 + length
        if stop <= cursor or stop > len(footer):
            raise ValueError("invalid MGH tag length")
        if tag != 1:
            kept.extend(footer[cursor:stop])
        cursor = stop
    return bytes(kept), entries


def _color_tag(entries: dict) -> bytes:
    name = b"none\0"
    payload = bytearray(struct.pack(">iiii", 1, -2, max(entries) + 1, len(name)))
    payload.extend(name)
    payload.extend(struct.pack(">i", len(entries)))
    for index, (label, r, g, b, transparency) in sorted(entries.items()):
        encoded = label.encode() + b"\0"
        payload.extend(struct.pack(">ii", index, len(encoded)))
        payload.extend(encoded)
        payload.extend(struct.pack(">iiii", r, g, b, transparency))
    return bytes(payload)


def defects_to_volume(surface_file: str | Path, defect_file: str | Path,
                      template_file: str | Path, output_file: str | Path, *,
                      offset: int, merge: bool = False, cortex_file: str | Path | None = None,
                      device: str = "cuda:0") -> dict:
    """读取有序表面/缺陷MGH及模板，写 int32 MGH/MGZ，返回实际设备与秒数。

    参数空间/单位与 project_defects 一致；defect_file 为 (V,1,1) 顶点序列，
    cortex_file 是同一顶点顺序的 ASCII label。defect_file 也支持无扩展名的
    FreeSurfer morph 顶点图。允许模板与输出同路径；先读取再写。
    保留模板 RAS、scan参数和其他标签；写出嵌入颜色表（名字/标签号不变，
    颜色确定性生成，不复现上游随机调色板）。merge 时保留已有颜色/名字。
    不接受缺失/非法几何或不匹配的顶点；失败抛异常，不回退 CPU。
    """
    tick = time.perf_counter()
    image = nib.load(str(template_file))
    if not isinstance(image, nib.MGHImage) or len(image.shape) != 3:
        raise ValueError("template must be a 3D MGH/MGZ volume")
    vertices, faces, metadata = nib.freesurfer.io.read_geometry(str(surface_file), read_metadata=True)
    if not metadata:
        raise ValueError("surface volume geometry is required")
    # 固定拓扑程序可以写无扩展名morph或MGH；nib.load依赖扩展名。
    overlay_payload = Path(defect_file).read_bytes()
    if overlay_payload[:2] == b"\x1f\x8b":
        overlay_payload = gzip.decompress(overlay_payload)
    if overlay_payload[:4] == b"\0\0\0\1":
        overlay = nib.MGHImage.from_bytes(overlay_payload)
        if overlay.shape != (len(vertices), 1, 1):
            raise ValueError("defect overlay does not match surface vertex order")
        labels = np.asarray(overlay.dataobj).reshape(-1)
    else:
        labels = nib.freesurfer.io.read_morph_data(str(defect_file))
        if labels.shape != (len(vertices),):
            raise ValueError("defect overlay does not match surface vertex order")
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all() or (labels < 0).any():
        raise ValueError("defect labels must be finite nonnegative integers")
    mask = None
    if cortex_file is not None:
        ids = nib.freesurfer.io.read_label(str(cortex_file))
        if (ids < 0).any() or (ids >= len(vertices)).any():
            raise ValueError("cortex label index out of range")
        mask = np.zeros(len(vertices), dtype=bool)
        mask[ids] = True
    matrix = surface_ras_to_voxel(image.header, metadata)
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    result = project_defects(
        torch.tensor(vertices.astype(np.float32), device=target),
        torch.tensor(faces.astype(np.int64), device=target),
        torch.tensor(labels.astype(np.int64), device=target),
        torch.tensor(np.asarray(image.dataobj).astype(np.int32), device=target),
        torch.tensor(matrix, device=target), offset=offset,
        voxel_size_x=float(image.header.get_zooms()[0]), merge=merge,
        cortex=None if mask is None else torch.tensor(mask, device=target),
    ).cpu().numpy()
    source = Path(template_file)
    raw = gzip.decompress(source.read_bytes()) if source.suffix == ".mgz" else source.read_bytes()
    footer, old_colors = _footer_and_colors(raw, 284 + int(np.prod(image.shape)) * image.get_data_dtype().itemsize)
    maximum = int(labels[mask].max(initial=0) if mask is not None else labels.max(initial=0))
    colors = {}
    for index in range(max(maximum + offset + 2, max(old_colors, default=-1) + 1)):
        rgb = (index * 23197) & 0xFFFFFF
        colors[index] = (f"cluster{index}", rgb & 255, (rgb >> 8) & 255, (rgb >> 16) & 255, 0)
    colors[0] = ("unknown", 0, 0, 0, 255)
    for index in range(offset, maximum + offset + 1):
        colors[index] = (f"Defect-{index:03d}", *colors[index][1:])
    if merge:
        colors.update(old_colors)
    header = image.header.copy()
    header.set_data_dtype(np.int32)
    # nibabel 的 MGHHeader.binaryblock 只含90字节字段；数据起点仍为284。
    # 保留原模板的头部填充，避免二次 merge 将体素尾部误读成标签。
    header_bytes = header.binaryblock + raw[len(header.binaryblock):header.get_data_offset()]
    data = header_bytes + np.asarray(result, dtype=">i4").tobytes(order="F") + footer + _color_tag(colors)
    output = Path(output_file)
    if output.suffix not in (".mgh", ".mgz"):
        raise ValueError("output must be MGH/MGZ")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(data, mtime=0) if output.suffix == ".mgz" else data)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    return {"device": str(target), "seconds": time.perf_counter() - tick,
            "output": str(output), "offset": offset, "merge": merge,
            "palette": "FNIT deterministic colors; semantic labels retained"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--defects", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--cortex")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    defects_to_volume(surface_file=args.surface, defect_file=args.defects,
                      template_file=args.template, output_file=args.output,
                      offset=args.offset, merge=args.merge, cortex_file=args.cortex,
                      device=args.device)


if __name__ == "__main__":
    main()
