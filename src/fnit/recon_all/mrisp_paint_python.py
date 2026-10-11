"""球面图谱到顶点图的自有接口，复用已验证的参数化采样和有序平均。

固定 recon-all 配方为 ``mrisp_paint -a 5 atlas#6 sphere.reg avg_curv``。
frame 是从0开始的TIFF帧号，averages 是完整一阶邻接平均轮数。
本模块不调用原生程序，不实现该命令的 variance/coords/normalize 分支。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import nibabel.freesurfer.io as fsio
import numpy as np
import tifffile
import torch
from numba import njit

from .mris_register_atlas import sample_atlas_on_canonical_sphere
from .place_surface_smoothing import average_marked_values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@njit(cache=True, fastmath=False)
def _sample_source_cpu(xyz: np.ndarray, atlas: np.ndarray) -> np.ndarray:
    """CPU使用源标量libm/float32写回顺序；GPU复用成熟Torch采样器。"""
    low = np.asarray([10000., 10000., 10000.], np.float64)
    high = -low.copy()
    for point in xyz:
        for axis in range(3):
            low[axis] = min(low[axis], np.float64(point[axis]))
            high[axis] = max(high[axis], np.float64(point[axis]))
    center = (low + high) / 2.
    total = 0.
    for point in xyz:
        x, y, z = np.float64(point[0]) - center[0], np.float64(point[1]) - center[1], np.float64(point[2]) - center[2]
        total += math.sqrt(x*x + y*y + z*z)
    radius = np.float32(total / len(xyz))
    output = np.empty(len(xyz), np.float32)
    vdim, udim = atlas.shape
    for vertex in range(len(xyz)):
        x, y, z = xyz[vertex]
        theta = np.float32(np.arctan2(np.float32(y / radius), np.float32(x / radius)))
        if theta < 0:
            theta = np.float32(2 * math.pi + np.float64(theta))
        d = np.float32(np.float32(radius * radius) - np.float32(z * z))
        phi = np.float32(np.arctan2(np.sqrt(max(d, np.float32(0))), z))
        uf = np.float32(np.float64(np.float32(np.float32(udim) * phi)) / math.pi)
        vf = np.float32(np.float64(np.float32(np.float32(vdim) * theta)) / (2 * math.pi))
        u0, u1, v0, v1 = math.floor(uf), math.ceil(uf), math.floor(vf), math.ceil(vf)
        du, dv = np.float32(uf - np.float32(u0)), np.float32(vf - np.float32(v0))
        u0, u1 = abs(u0), abs(u1)
        if u0 >= udim:
            u0 = udim - (u0 - udim + 1)
        if u1 >= udim:
            u1 = udim - (u1 - udim + 1)
        v0, v1 = v0 % vdim, v1 % vdim
        one_u, one_v = np.float32(1 - du), np.float32(1 - dv)
        first = np.float32(np.float32(du * dv) * atlas[v1, u1])
        second = np.float32(np.float32(one_u * dv) * atlas[v1, u0])
        third = np.float32(np.float32(one_u * one_v) * atlas[v0, u0])
        fourth = np.float32(np.float32(du * one_v) * atlas[v0, u1])
        output[vertex] = np.float32(np.float32(np.float32(first + second) + third) + fourth)
    return output


def paint_atlas_values(*, vertices: np.ndarray, faces: np.ndarray,
                      atlas_frame: np.ndarray, averages: int = 5,
                      device: str = "cpu") -> np.ndarray:
    """返回与有序网格对应的(N,)float32图谱值，不写文件。

    vertices 为(N,3)有限surface RAS/mm球面坐标；faces 为(F,3)整数索引；
    atlas_frame 为(azimuth,polar)float32图谱。averages 默认5、须为非负
    整数；device 默认cpu，明确CUDA不回退，不改变全局TF32或开启半精度。
    按源规则仅在bbox中心离原点超过0.1mm时平移临时球面；输出不改变
    原网格。采样与平均复用成熟函数，错误形状/索引/非有限值抛ValueError。
    对应无normalize/coords/variance的mrisp_paint内部计算。
    """
    xyz = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces)
    frame = np.asarray(atlas_frame, dtype=np.float32)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0
            or triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.issubdtype(triangles.dtype, np.integer)
            or frame.ndim != 2 or min(frame.shape) < 2):
        raise ValueError("expected nonempty vertices(N,3), faces(F,3), atlas_frame(V,U)")
    if not np.isfinite(xyz).all() or not np.isfinite(frame).all():
        raise ValueError("surface and atlas values must be finite")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(xyz)):
        raise ValueError("face index outside vertices")
    if not isinstance(averages, (int, np.integer)) or isinstance(averages, bool) or averages < 0:
        raise ValueError("averages must be a nonnegative integer")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"} or target.type == "cuda" and target.index is None:
        raise ValueError("device must be cpu or explicit cuda:N")
    center = np.float32(np.float32(xyz.min(axis=0) + xyz.max(axis=0)) / np.float32(2))
    squared = np.float32(np.float32(center[0] * center[0] + center[1] * center[1])
                         + center[2] * center[2])
    centered = np.float32(xyz - center) if np.sqrt(squared) > .1 else xyz
    sampled = (_sample_source_cpu(np.ascontiguousarray(centered), np.ascontiguousarray(frame))
               if target.type == "cpu" else sample_atlas_on_canonical_sphere(
                   torch.as_tensor(centered, device=target), torch.as_tensor(frame, device=target)).cpu().numpy())
    return average_marked_values(
        values=sampled, marked=np.ones(len(xyz), dtype=bool),
        ripped=np.zeros(len(xyz), dtype=bool), faces=triangles, iterations=int(averages),
        device=str(target))


def run_paint_atlas(*, atlas_path: str | Path, sphere_path: str | Path,
                    output_path: str | Path, frame: int = 6, averages: int = 5,
                    device: str = "cpu", report_path: str | Path | None = None) -> dict:
    """读取TIFF固定图谱及三角球面，写(N,)float32 morph并返回计时/哈希。

    三个路径必填；frame 默认6且从0开始；averages 默认5；device 默认cpu。
    TIFF整型像素按float32位型解释，浮点TIFF直接转换float32。输入几何
    为surface RAS/mm，输出按相同顶点顺序，无空间变换或新标签。报告含
    路径、输入/输出SHA、顶点/面数、dtype、参数、读/计算/写/总秒数及
    本模块源码SHA。report_path 默认None；给出则写JSON。文件不存在、
    TIFF帧越界、无效图谱或计算失败抛异常，不调用原生或复制参考输出。
    """
    started = time.perf_counter()
    atlas_path, sphere_path, output_path = map(Path, (atlas_path, sphere_path, output_path))
    if not isinstance(frame, (int, np.integer)) or isinstance(frame, bool) or frame < 0:
        raise ValueError("frame must be a nonnegative zero-based integer")
    atlas = tifffile.imread(atlas_path)
    if atlas.ndim == 2:
        atlas = atlas[None]
    if atlas.ndim != 3 or frame >= len(atlas):
        raise ValueError("atlas frame is outside TIFF image pages")
    image = atlas[int(frame)]
    if image.dtype.kind in "iu":
        if image.dtype.itemsize != 4:
            raise ValueError("integer atlas TIFF must store float32 bits in 32-bit pixels")
        image = image.view(np.dtype(image.dtype.byteorder + "f4"))
    vertices, faces = fsio.read_geometry(str(sphere_path))
    loaded = time.perf_counter()
    values = paint_atlas_values(vertices=vertices, faces=faces, atlas_frame=image,
                                averages=averages, device=device)
    computed = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fsio.write_morph_data(str(output_path), values, fnum=len(faces))
    written = time.perf_counter()
    report = {"scope": "complete_fixed_paint_file_API_not_recon_all",
              "atlas_path": str(atlas_path), "sphere_path": str(sphere_path),
              "output_path": str(output_path), "frame_zero_based": int(frame),
              "averages": int(averages), "device": device,
              "sampling_backend": "source_numba" if torch.device(device).type == "cpu" else "torch",
              "vertices": len(vertices), "faces": len(faces), "dtype": "float32",
              "vertex_order": "same_as_input_sphere", "input_coordinate_units": "surface_RAS_mm",
              "input_sha256": {"atlas": _sha256(atlas_path), "sphere": _sha256(sphere_path)},
              "output_sha256": _sha256(output_path), "source_sha256": _sha256(Path(__file__)),
              "read_seconds": loaded - started, "compute_seconds": computed - loaded,
              "write_seconds": written - computed,
              "total_seconds_including_io": written - started,
              "native_program_called": False}
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-path", type=Path, required=True)
    parser.add_argument("--sphere-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=6)
    parser.add_argument("--averages", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)
    report = run_paint_atlas(atlas_path=args.atlas_path, sphere_path=args.sphere_path,
                            output_path=args.output_path, frame=args.frame,
                            averages=args.averages, device=args.device, report_path=args.report_path)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
