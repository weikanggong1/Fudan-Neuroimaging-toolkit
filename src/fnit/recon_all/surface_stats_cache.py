"""按表面文件版本缓存多图谱统计，保持 no-th3 与 TH3 定义独立。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import torch


_EXCLUDED = {"corpuscallosum", "unknown", "Unknown", "Medial_wall"}


def _file_version(path: str | Path) -> tuple:
    """以绝对路径、大小及纳秒修改/状态时间标识本次统计使用的文件版本。"""
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _device(device: str | torch.device) -> torch.device:
    """解析目标设备；裸 cuda 显式绑定当前设备，不查询其他 GPU。"""
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


@dataclass
class _Geometry:
    version: tuple
    vertices: np.ndarray
    faces: np.ndarray
    xyz: torch.Tensor
    triangles: torch.Tensor
    derived: dict = field(default_factory=dict)


class SurfaceStatsCache:
    """一次被试/半球统计期间的显式缓存，不跨运行保留 CUDA 张量。

    device 默认 cuda:0。几何采用 surface RAS/mm；面积 mm²、厚度 mm、
    no-th3 体积 mm³、主曲率 mm⁻¹。每个文件按完整路径与 stat 版本区分，
    white.preaparc、最终 white、pial 不共享坐标或曲率结果。
    文件必须在统计期间保持不变；不支持并发写同一表面或跨设备共享。
    使用 with 或 clear() 释放持有的张量；不主动 empty_cache() 或同步。
    文件缺失、顶点/有序面不一致及目标设备不一致会抛异常。
    counters 记录实际读取/计算次数，用于验证复用，而非整例性能声明。
    完整参数、输出和具名示例见 docs/recon_all/SURFACE_STATS_CACHE.md。
    """

    def __init__(self, *, device: str = "cuda:0") -> None:
        self.device = _device(device)
        self._geometries: dict[tuple, _Geometry] = {}
        self._arrays: dict[tuple, np.ndarray] = {}
        self._tensors: dict[tuple, torch.Tensor] = {}
        self._annotations: dict[tuple, tuple] = {}
        self._groupings: dict[tuple, tuple] = {}
        self._volumes: dict[tuple, torch.Tensor] = {}
        self._curvature: dict[tuple, tuple[np.ndarray, ...]] = {}
        self.header_totals: dict[tuple, tuple] = {}
        self.counters: Counter = Counter()

    def __enter__(self) -> SurfaceStatsCache:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.clear()

    def clear(self) -> None:
        """释放所有缓存引用；保留 counters，便于退出上下文后读取计算次数。"""
        for value in (self._geometries, self._arrays, self._tensors,
                      self._annotations, self._groupings, self._volumes,
                      self._curvature, self.header_totals):
            value.clear()

    def check_device(self, device: str) -> None:
        """禁止将同一缓存传给另一目标 GPU/CPU，失败时抛 ValueError。"""
        if _device(device) != self.device:
            raise ValueError(f"Stats cache device {self.device} differs from {device}")

    def geometry(self, path: str | Path) -> _Geometry:
        """读取并缓存 (N,3) 坐标/(F,3) 有序三角面与对应设备张量。"""
        version = _file_version(path)
        if version not in self._geometries:
            xyz, faces = fsio.read_geometry(str(path))
            if not np.isfinite(xyz).all() or faces.ndim != 2 or faces.shape[1] != 3 \
                    or (faces.size and (faces.min() < 0 or faces.max() >= len(xyz))):
                raise ValueError(f"Invalid finite triangular surface: {path}")
            self._geometries[version] = _Geometry(
                version, xyz, faces,
                torch.as_tensor(np.asarray(xyz, dtype=np.float32), device=self.device),
                torch.as_tensor(np.asarray(faces, dtype=np.int64), device=self.device))
            self.counters["geometry_reads"] += 1
        return self._geometries[version]

    def morph(self, path: str | Path) -> np.ndarray:
        """缓存 morph 文件，返回保持文件 dtype 与顶点顺序的 (N,) NumPy 数组。"""
        key = "morph", _file_version(path)
        if key not in self._arrays:
            self._arrays[key] = fsio.read_morph_data(str(path))
            self.counters["morph_reads"] += 1
        return self._arrays[key]

    def morph_tensor(self, path: str | Path, dtype: torch.dtype) -> torch.Tensor:
        """缓存指定 dtype 的 morph 张量；只做 float32/float64 明确转换。"""
        key = "morph", _file_version(path), dtype
        if key not in self._tensors:
            self._tensors[key] = torch.as_tensor(
                np.asarray(self.morph(path), dtype=(np.float32 if dtype == torch.float32
                                                    else np.float64)),
                dtype=dtype, device=self.device)
        return self._tensors[key]

    def annotation(self, path: str | Path) -> tuple:
        """缓存 .annot 的 (labels, color_table, names)，不改变标签索引语义。"""
        version = _file_version(path)
        if version not in self._annotations:
            self._annotations[version] = fsio.read_annot(str(path))
            self.counters["annotation_reads"] += 1
        return self._annotations[version]

    def cortex(self, path: str | Path | None, count: int) -> np.ndarray:
        """返回 (N,) bool 皮层选择；None 表示全部顶点，越界 label 抛 ValueError。"""
        key = "cortex", None if path is None else _file_version(path), count
        if key not in self._arrays:
            selected = np.ones(count, dtype=bool) if path is None else np.zeros(count, dtype=bool)
            if path is not None:
                vertices = fsio.read_label(str(path))
                if vertices.size and (vertices.min() < 0 or vertices.max() >= count):
                    raise ValueError(f"Cortex label contains out-of-range vertex: {path}")
                selected[vertices] = True
                self.counters["cortex_reads"] += 1
            self._arrays[key] = selected
        return self._arrays[key]

    @torch.inference_mode()
    def face_area(self, geometry: _Geometry) -> torch.Tensor:
        """缓存逐三角面 float32 面积，(F,)，单位 mm²，保留原叉乘公式。"""
        if "face_area" not in geometry.derived:
            v0, v1, v2 = (geometry.xyz[geometry.triangles[:, i]] for i in range(3))
            geometry.derived["face_area"] = torch.linalg.vector_norm(
                torch.cross(v1 - v0, v2 - v0, dim=1), dim=1).mul_(0.5)
            self.counters["face_area_computations"] += 1
        return geometry.derived["face_area"]

    @torch.inference_mode()
    def vertex_area(self, geometry: _Geometry) -> torch.Tensor:
        """缓存原 vertex_area 的 float32 累加结果，(N,)，单位 mm²。"""
        if "vertex_area" not in geometry.derived:
            # 保留 0.5/3 的单次乘法；先乘 0.5 再除 3 会改变 float32 舍入。
            v0, v1, v2 = (geometry.xyz[geometry.triangles[:, i]] for i in range(3))
            share = torch.linalg.vector_norm(
                torch.cross(v1 - v0, v2 - v0, dim=1), dim=1).mul_(0.5 / 3.0)
            area = torch.zeros(len(geometry.xyz), dtype=torch.float32, device=self.device)
            for corner in range(3):
                area.index_add_(0, geometry.triangles[:, corner], share)
            geometry.derived["vertex_area"] = area.to(torch.float64)
            self.counters["vertex_area_computations"] += 1
        return geometry.derived["vertex_area"]

    @torch.inference_mode()
    def principal(self, geometry: _Geometry) -> tuple[np.ndarray, np.ndarray]:
        """缓存法线、两跳邻接及主曲率；返回两个 (N,) float32/mm⁻¹ 数组。"""
        from .surface_curvature_gpu import _neighbours
        from .surface_roi_curvature_gpu import _principal_curvatures_tensor
        from .surface_thickness_gpu import _normals

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        if "principal" not in geometry.derived:
            if "normal" not in geometry.derived:
                geometry.derived["normal"] = _normals(geometry.xyz, geometry.triangles)
            normal = geometry.derived["normal"]
            _, _, indices, mask = _neighbours(geometry.faces, len(geometry.xyz))
            geometry.derived["two_hop_neighbours"] = torch.as_tensor(indices, device=self.device)
            geometry.derived["two_hop_mask"] = torch.as_tensor(mask, device=self.device)
            principal = _principal_curvatures_tensor(
                geometry.xyz, normal, geometry.derived["two_hop_neighbours"],
                geometry.derived["two_hop_mask"])
            values = principal.cpu().numpy()
            geometry.derived["principal"] = values[:, 0], values[:, 1]
            self.counters["principal_curvature_computations"] += 1
        return geometry.derived["principal"]

    @torch.inference_mode()
    def no_th3_volume(self, white: str | Path, pial: str | Path,
                      thickness: str | Path) -> torch.Tensor:
        """缓存 -no-th3 顶点基础量，(N,) float64/mm³，不能替代 TH3 顶点图。

        每面贡献 = mean(thickness[face]) * (Awhite + Apial) / 6；
        每个角累加一次。要求 white/pial 顶点数与有序三角面完全相同。
        """
        w, p = self.geometry(white), self.geometry(pial)
        if len(w.xyz) != len(p.xyz) or not np.array_equal(w.faces, p.faces):
            raise ValueError("White and pial surfaces must have identical ordered topology")
        key = w.version, p.version, _file_version(thickness)
        if key not in self._volumes:
            thick = self.morph_tensor(thickness, torch.float32)
            if len(thick) != len(w.xyz):
                raise ValueError("Surface and thickness vertex counts differ")
            mean_thick = thick[w.triangles].to(torch.float64).mean(1)
            share = mean_thick * (self.face_area(w).to(torch.float64) +
                                  self.face_area(p).to(torch.float64)) / 6.0
            volume = torch.zeros(len(w.xyz), device=self.device, dtype=torch.float64)
            for corner in range(3):
                volume.index_add_(0, w.triangles[:, corner], share)
            self._volumes[key] = volume
            self.counters["no_th3_volume_computations"] += 1
        return self._volumes[key]

    def _groups(self, annotation: str | Path, count: int) -> tuple:
        """CPU 确定每脑区的静态切片，保持各区顶点原有顺序后一次上传索引。"""
        key = _file_version(annotation), count
        if key not in self._groupings:
            labels, _, names = self.annotation(annotation)
            if len(labels) != count:
                raise ValueError("Surface and annotation vertex counts differ")
            groups, indices, offset = [], [], 0
            for index, raw_name in enumerate(names):
                name = raw_name.decode()
                if name in _EXCLUDED:
                    continue
                selected = np.flatnonzero(labels == index)
                if len(selected):
                    groups.append((name, len(selected), slice(offset, offset + len(selected))))
                    indices.append(selected)
                    offset += len(selected)
            ordered = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
            self._groupings[key] = groups, torch.as_tensor(ordered, device=self.device)
        return self._groupings[key]

    @torch.inference_mode()
    def roi_base(self, surface: str | Path, annotation: str | Path,
                 thickness: str | Path, *, white: str | Path | None = None,
                 pial: str | Path | None = None) -> tuple[dict, dict]:
        """一次回传脑区面积/厚度及可选 no-th3 体积摘要，避免逐区 GPU 同步。

        返回 ({name: (NumVert, SurfArea, ThickAvg, ThickStd)}, {name: GrayVol})。
        厚度 std 为总体标准差。white/pial 须同时给出，否则体积字典为空。
        不按 cortex.label 删除基本统计顶点，与原 -no-th3 行定义保持一致。
        """
        geometry = self.geometry(surface)
        values = self.morph_tensor(thickness, torch.float64)
        if len(values) != len(geometry.xyz):
            raise ValueError("Surface and thickness vertex counts differ")
        if (white is None) != (pial is None):
            raise ValueError("white and pial must both be supplied for no-th3 volume")
        groups, indices = self._groups(annotation, len(values))
        if not groups:
            return {}, {}
        grouped_area = self.vertex_area(geometry)[indices]
        grouped_thickness = values[indices]
        grouped_volume = (None if white is None else
                          self.no_th3_volume(white, pial, thickness)[indices])
        summaries = []
        for _, _, selected in groups:
            thick = grouped_thickness[selected]
            fields = [grouped_area[selected].sum(), thick.mean(), thick.std(unbiased=False)]
            if grouped_volume is not None:
                fields.append(grouped_volume[selected].sum())
            summaries.append(torch.stack(fields))
        # 唯一一次摘要 CPU 传输；切片边界来自 CPU，未执行 GPU bool()/float()。
        numeric = torch.stack(summaries).cpu().numpy()
        basic, volume = {}, {}
        for (name, count, _), row in zip(groups, numeric):
            basic[name] = count, float(row[0]), float(row[1]), float(row[2])
            if grouped_volume is not None:
                volume[name] = float(row[3])
        self.counters["roi_summary_transfers"] += 1
        return basic, volume

    def curvature_columns(self, surface: str | Path, area_map: str | Path,
                          annotation: str | Path,
                          cortex_label: str | Path | None) -> dict:
        """使用缓存主曲率/面积基础量按 CPU 标签汇总原四列，不改变求和顺序。"""
        geometry = self.geometry(surface)
        key = geometry.version, _file_version(area_map)
        if key not in self._curvature:
            area = self.morph(area_map).astype(np.float64)
            if len(area) != len(geometry.xyz):
                raise ValueError("Surface and area map vertex counts differ")
            k1, k2 = (k.astype(np.float64) for k in self.principal(geometry))
            self._curvature[key] = (
                np.abs((k1 + k2) / 2) * area,
                np.abs(k1 * k2) * area,
                area * np.abs(k1) * (np.abs(k1) - np.abs(k2)) / (4 * np.pi),
                area * np.maximum(k1 * k2, 0) / (4 * np.pi))
        mean, gauss, fold, intrinsic = self._curvature[key]
        labels, _, names = self.annotation(annotation)
        if len(labels) != len(geometry.xyz):
            raise ValueError("Surface and annotation vertex counts differ")
        cortex = self.cortex(cortex_label, len(labels))
        result = {}
        for index, raw_name in enumerate(names):
            name = raw_name.decode()
            selected = (labels == index) & cortex
            if not selected.any() or name in _EXCLUDED:
                continue
            result[name] = (float(mean[selected].sum() / selected.sum()),
                            float(gauss[selected].sum() / selected.sum()),
                            float(fold[selected].sum()), float(intrinsic[selected].sum()))
        return result

    @torch.inference_mode()
    def roi_volumes(self, white: str | Path, pial: str | Path,
                    thickness: str | Path, annotation: str | Path) -> dict:
        """只汇总 no-th3 脑区 mm³，一次回传结果，不计算无关的面积/曲率列。"""
        volume = self.no_th3_volume(white, pial, thickness)
        groups, indices = self._groups(annotation, len(volume))
        if not groups:
            return {}
        grouped = volume[indices]
        values = torch.stack([grouped[selected].sum()
                              for _, _, selected in groups]).cpu().numpy()
        self.counters["roi_summary_transfers"] += 1
        return {name: float(value) for (name, _, _), value in zip(groups, values)}
