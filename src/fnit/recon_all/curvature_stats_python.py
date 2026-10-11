"""自有smoothwm离散八图和完整curv.stats；不调用FreeSurfer程序。

复用有序离散Torch几何、四张派生图与原网格邻接；统计按原顶点顺序以
Numba累计。支持标准recon-all -m --writeCurvatureFiles -G -F smoothwm
以及原始纹理的平滑、label/高低通过滤；不把连续二次曲面拟合代替离散曲率。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
from numba import njit
import torch

from .curvature_stats_torch import curvature_derivatives_tensor
from .discrete_curvature_torch import DiscreteCurvatureTopology
from .inflate_python import distances
from .inflate_topology import _one_ring
from .place_surface_normals import ordered_face_csr

MAP_ORDER = ("K", "H", "K1", "K2", "S", "C", "BE", "FI")


@njit(cache=True)
def _ordered_vertex_area(face_area, offsets, face_ids):
    result = np.zeros(len(offsets) - 1, np.float32)
    for vertex in range(len(result)):
        area = np.float32(0)
        for index in range(offsets[vertex], offsets[vertex + 1]):
            area = np.float32(area + face_area[face_ids[index]])
        result[vertex] = np.float32(np.float64(area) / 3.0)
    return result


@njit(cache=True)
def _geometry_summary(face_area, edge_distances, neighbors, degree, ripped):
    total_area = 0.0
    for area in face_area:
        total_area += np.float64(area)
    total_area_f32 = np.float32(total_area)
    total = total_sq = count = 0.0
    for vertex in range(len(degree)):
        if ripped[vertex]:
            continue
        for column in range(degree[vertex]):
            if ripped[neighbors[vertex, column]]:
                continue
            value = np.float64(edge_distances[vertex, column])
            total += value
            total_sq += value * value
            count += 1.0
    mean = total / count if count else 0.0
    std = np.sqrt(count * (total_sq / count - mean * mean) / (count - 1)) if count > 1 else 0.0
    return total_area_f32, np.float32(total_area_f32 / np.float32(len(degree))), mean, std


@njit(cache=True)
def _ordered_statistics(values, areas, ripped):
    """保留F64均值/积分、F32乘法/面积累计及首个极值索引的源规则。"""
    total = total_sq = count = 0.0
    minimum = maximum = values[0]
    minimum_vertex = maximum_vertex = 0
    integrals = np.zeros(4, np.float64)
    counts = np.zeros(4, np.int64)
    counted_area = np.zeros(4, np.float32)
    for vertex in range(len(values)):
        if ripped[vertex]:
            continue
        value = values[vertex]
        d = np.float64(value)
        total += d
        total_sq += d * d
        count += 1.0
        if value < minimum:
            minimum, minimum_vertex = value, vertex
        if value > maximum:
            maximum, maximum_vertex = value, vertex
        for group in range(4):
            if group == 2 and value < 0 or group == 3 and value >= 0:
                continue
            term = value if group == 0 else np.float32(abs(value))
            integrals[group] += np.float64(np.float32(term * areas[vertex]))
            counted_area[group] = np.float32(counted_area[group] + areas[vertex])
            counts[group] += 1
    result = np.zeros((4, 5), np.float32)
    for group in range(4):
        result[group, 0] = np.float32(integrals[group])
        result[group, 1] = np.float32(counts[group])
        result[group, 2] = counted_area[group]
        if counts[group] > 1:
            result[group, 3] = np.float32(integrals[group] / np.float64(counts[group]))
            result[group, 4] = np.float32(integrals[group] / np.float64(counted_area[group]))
    mean = total / count if count else 0.0
    variance = total_sq / count - mean * mean if count else 0.0
    return mean, np.sqrt(variance), minimum, maximum, minimum_vertex, maximum_vertex, int(count), result


@njit(cache=True)
def _average_texture(values, neighbors, degree, ripped, iterations):
    current = values.copy()
    for _ in range(iterations):
        updated = current.copy()
        for vertex in range(len(current)):
            if ripped[vertex]:
                continue
            total = np.float32(current[vertex])
            count = 1
            for column in range(degree[vertex]):
                other = neighbors[vertex, column]
                if not ripped[other]:
                    total = np.float32(total + current[other])
                    count += 1
            updated[vertex] = np.float32(total / np.float32(count))
        current = updated
    return current


def _statistics(values, areas, ripped):
    mean, std, minimum, maximum, mini, maxi, count, integrals = _ordered_statistics(values, areas, ripped)
    return {"mean": float(mean), "std": float(std), "min": float(minimum), "max": float(maximum),
            "min_vertex": int(mini), "max_vertex": int(maxi), "count": int(count),
            "integrals": integrals.tolist()}


def _f32_percentage(first, second):
    return float(np.float32(np.float32(100) * np.float32(first) / np.float32(second))) if second else float("nan")


def _format_section(name, texture_name, hemisphere, summary, geometry, vertex_count, *, roi):
    units = "mm^-1" if name in ("H", "k1", "k2") else ("" if name == "Raw" else "mm^-2")
    title = f"\n{name} <mean> +- <std> (using '{hemisphere}.{texture_name}'):"
    lines = [f"{title:<50} {summary['mean']:12.5f} +- {summary['std']:2.4f} {units}\n"]
    lines += [f"{name:>10}{' Min:':<40}{summary['min']:12.5f} at vertex {summary['min_vertex']:<8d}\n",
              f"{name:>10}{' Max:':<40}{summary['max']:12.5f} at vertex {summary['max_vertex']:<8d}\n"]
    line = lambda label, value: f"{name:>10}{label:<40}{value}\n"
    region = "ROI Surface" if roi else "Whole Surface"
    lines += [line(" Curvature Calculation Type:", f"{'discrete':>12}"),
              line(f" Mean Vertex Separation ({region}):", f"{geometry['mean_vertex_dist']:12.5f} +- {geometry['std_vertex_dist']:2.5f} mm"),
              line(" Total Surface Area:", f"{geometry['total_area']:12.5f} mm^2"),
              line(" Total Number of Vertices:", f"{vertex_count:05d}".rjust(12)),
              line(" Average Vertex Area (Whole Surface):", f"{geometry['average_vertex_area']:12.5f} mm^2")]
    groups = ("Natural", "Rectified", "Positive", "Negative")
    rows = summary["integrals"]
    if roi:
        roi_area, roi_count = rows[0][2], int(rows[0][1])
        lines += [line(" ROI Surface Area:", f"{roi_area:12.5f} mm^2"),
                  line(" ROI Number of Vertices:", f"{roi_count:05d}".rjust(12)),
                  line(" ROI Surface Area Percentage:", f"{_f32_percentage(roi_area, geometry['total_area']):12.2f}%"),
                  line(" ROI Surface Vertex Percentage:", f"{_f32_percentage(roi_count, vertex_count):12.2f}%"),
                  line(" Average Vertex Area (ROI Surface):", f"{float(np.float32(roi_area / roi_count)) if roi_count else float('nan'):12.5f} mm^2")]
    for group, row in zip(groups, rows):
        lines.append(line(f" {group} Surface Integral:", f"{row[0]:12.5f}"))
    for group, row in zip(groups, rows):
        count = int(row[1])
        lines.append(line(f" Mean {group} Surface Integral:", f"{row[3]:12.5f} across {count} ({_f32_percentage(count, vertex_count):05.2f}%) vertices"))
    for group, row in zip(groups, rows):
        lines.append(line(f" AreaNorm {group} Surface Integral:", f"{row[4]:12.5f} across {row[2]:f} ({_f32_percentage(row[2], geometry['total_area']):05.2f}%) mm^2"))
    return "".join(lines)


def write_curvature_stats(*, subject_dir: str | Path, hemisphere: str,
                          stats_file: str | Path | None = None, output_prefix: str | Path | None = None,
                          surface_name: str = "smoothwm", curvature_names: tuple[str, ...] = ("curv", "sulc"),
                          device: str = "cuda:0", angle_backend: str = "source_libm", averages: int = 0,
                          label_file: str | Path | None = None, low_pass: float | None = None,
                          high_pass: float | None = None, overwrite: bool = False) -> dict:
    """自产smoothwm/curv/sulc→完整curv.stats与同序K/H/K1/K2/S/C/BE/FI。

    subject_dir含surf/H.{surface_name,curv,sulc}；三角surface RAS/mm，morph
    FP32(N,)。hemisphere必填lh/rh。stats_file默认stats/H.curv.stats；前缀
    默认surf/H.surface_name，八图为PREFIX.NAME.crv，面数写入morph头。
    device默认显式cuda:0，可cpu；angle_backend默认source_libm为源acosf的
    自有Numba步骤，其余几何/派生计算在目标Torch设备。torch为原设备acos。
    averages默认0，仅平滑原始纹理；label_file是同序label可选掩膜，高低通
    默认关闭，以abs(curv)<low_pass、abs(curv)>=high_pass叠加持久冻结标记。
    overwrite默认False保护已有文件。返回outputs、各纹理完整summary、几何
    与分项/total秒；计时包括检查/读取/JIT/传输/写出，CUDA显式同步。
    缺文件、参数/形状/有限性/闭合拓扑/设备错误或输出覆盖输入直接抛异常；
    不写占位统计、不读参考、不调用外部程序、不设置TF32/半精度/分配策略。
    当前完整入口对应标准离散profile；连续拟合、缩放/归一化及直方图扩展未提供。
    """
    start = perf_counter()
    subject = Path(subject_dir)
    if hemisphere not in ("lh", "rh") or not surface_name or "/" in surface_name:
        raise ValueError("hemisphere must be lh/rh and surface_name must be a simple name")
    if isinstance(averages, bool) or not isinstance(averages, int) or averages < 0:
        raise ValueError("averages must be a nonnegative integer")
    if not curvature_names or any(not name or "/" in name for name in curvature_names):
        raise ValueError("curvature_names must contain simple morph filenames")
    for threshold in (low_pass, high_pass):
        if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
            raise ValueError("filter thresholds must be finite nonnegative values")
    if angle_backend not in ("torch", "source_libm"):
        raise ValueError("angle_backend must be torch or source_libm")
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda") or target_device.type == "cuda" and target_device.index is None:
        raise ValueError("device must be cpu or an explicit cuda index")
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    surface = subject / "surf" / f"{hemisphere}.{surface_name}"
    sources = [surface] + [subject / "surf" / f"{hemisphere}.{name}" for name in curvature_names]
    destination = Path(stats_file) if stats_file is not None else subject / "stats" / f"{hemisphere}.curv.stats"
    prefix = Path(output_prefix) if output_prefix is not None else surface
    paths = {name: Path(str(prefix) + f".{name}.crv") for name in MAP_ORDER}
    all_outputs = [destination, *paths.values()]
    input_paths = sources + ([Path(label_file)] if label_file is not None else [])
    if len({p.resolve() for p in all_outputs}) != len(all_outputs) or any(p.resolve() in {s.resolve() for s in input_paths} for p in all_outputs):
        raise ValueError("output paths must be distinct and must not overwrite inputs")
    if not overwrite and any(p.exists() for p in all_outputs):
        raise FileExistsError("curvature output already exists; use a new directory or explicit overwrite")
    xyz, faces = fsio.read_geometry(str(surface))
    xyz = np.asarray(xyz, np.float32)
    textures = [np.asarray(fsio.read_morph_data(str(p)), np.float32) for p in sources[1:]]
    if any(a.shape != (len(xyz),) or not np.isfinite(a).all() for a in textures):
        raise ValueError("all source textures must be finite with the surface vertex count")
    ripped = np.zeros(len(xyz), np.bool_)
    if label_file is not None:
        label = np.asarray(fsio.read_label(str(label_file)), np.int64)
        if label.ndim != 1 or np.any(label < 0) or np.any(label >= len(xyz)):
            raise ValueError("label vertex indices do not match the surface")
        ripped[:] = True
        ripped[label] = False
    if bool(ripped.all()):
        raise ValueError("label must retain at least one surface vertex")
    read_seconds = perf_counter() - start
    tick = perf_counter()
    offsets, face_ids, corners = ordered_face_csr(faces, len(xyz))
    neighbors, degree = _one_ring(np.asarray(faces, np.int32), face_ids, corners, offsets)
    topology = DiscreteCurvatureTopology(faces=faces, nvertices=len(xyz), device=str(target_device))
    maps = topology.evaluate(vertices=torch.as_tensor(xyz, device=target_device), angle_backend=angle_backend)
    maps.update(curvature_derivatives_tensor(maps["K1"], maps["K2"]))
    arrays = {name: np.ascontiguousarray(maps[name].cpu().numpy()) for name in MAP_ORDER}
    face_area = np.ascontiguousarray(maps["face_area"].cpu().numpy())
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("nonfinite principal or derivative curvature")
    vertex_area = _ordered_vertex_area(face_area, offsets, face_ids)
    filtered = low_pass is not None or high_pass is not None
    reordered_vertex_area = None
    if filtered:
        ring_ids = topology.face_ids.cpu().numpy()
        ring_degree = np.diff(offsets)
        packed_ids = ring_ids[np.arange(ring_ids.shape[1])[None, :] < ring_degree[:, None]]
        reordered_vertex_area = _ordered_vertex_area(face_area, offsets, packed_ids)
    edge_distances = distances(xyz, neighbors, degree)
    def current_geometry():
        total_area, avg_area, mean_dist, std_dist = _geometry_summary(
            face_area, edge_distances, neighbors, degree, ripped)
        return {"total_area": float(total_area), "average_vertex_area": float(avg_area),
                "mean_vertex_dist": float(mean_dist), "std_vertex_dist": float(std_dist)}
    geometry = current_geometry()
    principal_seconds = perf_counter() - tick
    tick = perf_counter()
    text, summaries, means = [], {}, []
    roi = label_file is not None or low_pass is not None or high_pass is not None
    for name, texture in zip(curvature_names, textures):
        values = _average_texture(texture, neighbors, degree, ripped, averages)
        if low_pass is not None:
            ripped |= np.abs(values) >= low_pass
        if high_pass is not None:
            ripped |= np.abs(values) < high_pass
        if bool(ripped.all()):
            raise ValueError("persistent filtering removed every vertex")
        if filtered:
            geometry = current_geometry()
        summary = _statistics(values, vertex_area, ripped)
        summaries[name] = summary
        means.append(summary["mean"])
        text.append(_format_section("Raw", name, hemisphere, summary, geometry, len(xyz), roi=roi))
    for name in MAP_ORDER:
        # 原生MRISuse*只更新未ripped点；被排除点保留上一个纹理，写图也保留。
        original_map = arrays[name]
        current = values.copy()
        current[~ripped] = original_map[~ripped]
        arrays[name] = values = current
        # 原生命令实际使用MRISminMaxCurvaturesSearch扫描当前curv纹理。
        summary = _statistics(current, vertex_area, ripped)
        summaries[name] = summary
        means.append(summary["mean"])
        text.append(_format_section(name.lower() if name.startswith("K") and len(name) == 2 else name,
                                    surface_name, hemisphere, summary, geometry, len(xyz), roi=roi))
        if name == "K" and filtered:
            # 源离散KH先重排顶点面环；K积分后filter路径才重算metric properties。
            # 因而K用旧面序面积，H及后续统计用重排面序F32累计面积。
            vertex_area = reordered_vertex_area
    indices = {"FI": float(np.float32(np.float64(np.float32(summaries["FI"]["integrals"][0][0] / 4)) / math.pi)),
               "ICIp": float(np.float32(np.float64(np.float32(summaries["K"]["integrals"][2][0] / 4)) / math.pi)),
               "ICIn": float(np.float32(np.float64(np.float32(summaries["K"]["integrals"][3][0] / 4)) / math.pi)),
               "ICIt": float(np.float32(np.float64(np.float32(summaries["K"]["integrals"][0][0] / 4)) / math.pi))}
    text.append(f"\n{'curv -- calculation type:':<55}{'discrete':>10}\n")
    for label, value in (("curv -- Folding Index (FI):", indices["FI"]),
                         ("curv -- Intrinsic Curvature Index - positive (ICIp):", indices["ICIp"]),
                         ("curv -- Intrinsic Curvature Index - negative (ICIn):", indices["ICIn"]),
                         ("curv -- Intrinsic Curvature Index - natural  (ICIt):", indices["ICIt"])):
        text.append(f"{label:<55}{value:10.5f}\n")
    mean_total = sum(means) / len(means)
    mean_std = math.sqrt(sum(v * v for v in means) / len(means) - mean_total * mean_total)
    text.append(f"\nMean across {len(means)} curvatures: {mean_total:8.4e} +- {mean_std:8.4e}\n\n\n")
    stats_seconds = perf_counter() - tick
    tick = perf_counter()
    for path in all_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        fsio.write_morph_data(str(path), arrays[name], fnum=len(faces))
    destination.write_text("".join(text))
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    return {"outputs": {"stats": str(destination), **{k: str(v) for k, v in paths.items()}},
            "vertices": len(xyz), "faces": len(faces), "device": str(target_device), "angle_backend": angle_backend,
            "averages": averages, "geometry": geometry, "summaries": summaries, "indices": indices,
            "active_vertices": int((~ripped).sum()), "delta_violations": int(maps["delta_negative"].sum().item()),
            "seconds": {"read_and_checks": read_seconds, "topology_principal_geometry_jit_transfers": principal_seconds,
                        "full_ordered_stats_including_jit": stats_seconds, "write_and_sync": perf_counter() - tick,
                        "total_including_io": perf_counter() - start},
            "implementation": "own Torch discrete geometry/maps + NumPy/Numba source-order full statistics; no external commands"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--stats-file", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--angle-backend", choices=("source_libm", "torch"), default="source_libm")
    parser.add_argument("--averages", type=int, default=0)
    parser.add_argument("--label-file", type=Path)
    parser.add_argument("--low-pass", type=float)
    parser.add_argument("--high-pass", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = write_curvature_stats(subject_dir=args.subject_dir, hemisphere=args.hemisphere,
        stats_file=args.stats_file, output_prefix=args.output_prefix, device=args.device,
        angle_backend=args.angle_backend, averages=args.averages, label_file=args.label_file,
        low_pass=args.low_pass, high_pass=args.high_pass, overwrite=args.overwrite)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
