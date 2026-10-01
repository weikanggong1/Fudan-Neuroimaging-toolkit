"""Format the ten FreeSurfer cortical parcellation table columns."""

from pathlib import Path

from .surface_roi_curvature_gpu import curvature_columns
from .surface_stats_cache import SurfaceStatsCache


def anatomical_stats_rows(white: str | Path, pial: str | Path,
                          surface: str | Path, area_map: str | Path,
                          thickness: str | Path, annotation: str | Path,
                          cortex_label: str | Path | None, *, device: str = "cuda:0",
                          cache: SurfaceStatsCache | None = None) -> list[str]:
    """返回 mris_anatomical_stats -no-th3 格式的十列脑区行（list[str]）。

    white/pial/surface 是 surface RAS/mm 网格；surface 与 area_map(mm²)
    对应，thickness 为同序 mm morph，annotation 为 .annot，cortex_label
    为曲率选择标签或 None。device 默认 cuda:0；cache 默认临时缓存，
    多图谱应传同设备缓存以复用同一表面版本的基础量。缺失文件、拓扑、
    顶点数、脑区名或设备不一致抛异常。全局表头由调用方另外计算。
    """
    if cache is None:
        with SurfaceStatsCache(device=device) as temporary:
            return anatomical_stats_rows(
                white=white, pial=pial, surface=surface, area_map=area_map,
                thickness=thickness, annotation=annotation,
                cortex_label=cortex_label, device=device, cache=temporary)
    cache.check_device(device)
    white, pial, surface = Path(white), Path(pial), Path(surface)
    area_map, thickness = Path(area_map), Path(thickness)
    annotation = Path(annotation)
    cortex_label = Path(cortex_label) if cortex_label is not None else None
    basic, volumes = cache.roi_base(surface, annotation, thickness,
                                   white=white, pial=pial)
    curvature = curvature_columns(surface, area_map, annotation, cortex_label,
                                  device=device, cache=cache)
    if set(basic) != set(volumes) or set(basic) != set(curvature):
        raise ValueError("Area, volume and curvature ROI names differ")
    lines = []
    for name, (count, area, mean, std) in basic.items():
        cm, cg, fold, intrinsic = curvature[name]
        lines.append(f"{name:<40}  {count:5d}  {area:5.0f}  {volumes[name]:5.0f}"
                     f"  {mean:5.3f} {std:5.3f}  {cm:8.3f}  {cg:8.3f}"
                     f"  {fold:7.0f}  {intrinsic:6.1f}")
    return lines
