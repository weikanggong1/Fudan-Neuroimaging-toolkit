"""Global numeric headers for FreeSurfer cortical parcellation tables."""

from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from .estimated_tiv import estimate_tiv


def read_brain_volume_stats(path: str | Path) -> dict[str, float]:
    """Read the 16 cached measures written by ``ComputeBrainVolumeStats``."""
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 5 and fields[0].startswith("# Measure "):
            result[fields[1]] = float(fields[3])
    required = ("BrainSegVol", "BrainSegVolNotVent", "SupraTentorialVol",
                "SupraTentorialVolNotVent", "SubCortGrayVol", "lhCortexVol",
                "rhCortexVol", "CortexVol", "TotalGrayVol",
                "lhCerebralWhiteMatterVol", "rhCerebralWhiteMatterVol",
                "CerebralWhiteMatterVol", "MaskVol", "SupraTentorialVolNotVentVox",
                "BrainSegVolNotVentSurf", "VentricleChoroidVol")
    if any(name not in result for name in required):
        raise ValueError("Incomplete brainvol.stats cache")
    return result


def anatomical_stats_global_lines(area_map: str | Path, thickness: str | Path,
                                  annotation: str | Path, cortex_label: str | Path | None,
                                  brainvol_stats: str | Path | dict[str, float],
                                  talairach_xfm: str | Path,
                                  *, surface: str = "white",
                                  voxel_volume: float = 1.0,
                                  cache=None) -> list[str]:
    """返回 mris_anatomical_stats 的 # Measure 表头（list[str]）。

    area_map/thickness 为同序 mm²/mm morph；annotation 为 .annot；
    cortex_label 为选择顶点的 .label 或 None；brainvol_stats 为 mm³
    字典或 brainvol.stats 路径；talairach_xfm 用于 eTIV。surface 默认
    white，voxel_volume 默认 1.0 mm³，cache 默认不缓存。面积/厚度全局
    累加保留原逐顶点 float32 顺序；多图谱仅在有效顶点完全相同时复用。
    顶点数不符、文件或必需体积字段缺失会抛异常。无需 CUDA 同步。
    """
    from .surface_stats_cache import _file_version

    area = fsio.read_morph_data(str(area_map)) if cache is None else cache.morph(area_map)
    thick = fsio.read_morph_data(str(thickness)) if cache is None else cache.morph(thickness)
    labels, _, _ = (fsio.read_annot(str(annotation)) if cache is None
                    else cache.annotation(annotation))
    if cache is None:
        cortex = np.ones(len(area), dtype=bool) if cortex_label is None else np.zeros(len(area), dtype=bool)
        if cortex_label is not None:
            cortex[fsio.read_label(str(cortex_label))] = True
    else:
        cortex = cache.cortex(cortex_label, len(area)).copy()
    if not (len(area) == len(thick) == len(labels)):
        raise ValueError("Area, thickness and annotation vertex counts differ")
    cortex &= labels >= 0
    key = ("global", _file_version(area_map), _file_version(thickness),
           cortex_label is None, cortex.tobytes()) if cache is not None else None
    if cache is not None and key in cache.header_totals:
        count, area_total, mean_thickness = cache.header_totals[key]
    else:
        count = int(cortex.sum())
        area_total = (np.float32(np.sum(area.astype(np.float64)))
                      if cortex_label is None else np.float32(0))
        thickness_total = np.float32(0)
        if cortex_label is not None:
            for vertex in np.flatnonzero(cortex):
                area_total = np.float32(area_total + area[vertex])
                thickness_total = np.float32(thickness_total + thick[vertex])
        mean_thickness = np.float32(thickness_total / count) if count else np.float32(0)
        if cache is not None:
            cache.header_totals[key] = count, area_total, mean_thickness
            cache.counters["global_total_computations"] += 1
    surface_name = {"white": "WhiteSurfArea", "pial": "PialSurfArea"}.get(surface,
                                                                          "SurfArea")
    surface_description = {"white": "White Surface Total Area",
                           "pial": "Pial Surface Total Area"}.get(surface,
                                                                     "Surface Total Area")
    volume = (brainvol_stats if isinstance(brainvol_stats, dict)
              else read_brain_volume_stats(brainvol_stats))
    values = (("BrainSeg", "BrainSegVol", "Brain Segmentation Volume"),
              ("BrainSegNotVent", "BrainSegVolNotVent", "Brain Segmentation Volume Without Ventricles"),
              ("BrainSegNotVentSurf", "BrainSegVolNotVentSurf", "Brain Segmentation Volume Without Ventricles from Surf"),
              ("Cortex", "CortexVol", "Total cortical gray matter volume"),
              ("SupraTentorial", "SupraTentorialVol", "Supratentorial volume"),
              ("SupraTentorialNotVent", "SupraTentorialVolNotVent", "Supratentorial volume"))
    result = [
        f"# Measure Cortex, NumVert, Number of Vertices, {count}, unitless",
        f"# Measure Cortex, {surface_name}, {surface_description}, {area_total:g}, mm^2",
    ]
    if cortex_label is not None:
        result.append(f"# Measure Cortex, MeanThickness, Mean Thickness, {mean_thickness:g}, mm")
    etiv_key = ("etiv", _file_version(talairach_xfm)) if cache is not None else None
    if cache is None:
        etiv = estimate_tiv(talairach_xfm)
    elif etiv_key in cache.header_totals:
        etiv = cache.header_totals[etiv_key]
    else:
        etiv = estimate_tiv(talairach_xfm)
        cache.header_totals[etiv_key] = etiv
        cache.counters["etiv_computations"] += 1
    result.extend([
        ("# BrainVolStatsFixed-NotNeeded because voxelvolume=1mm3"
         if abs(voxel_volume - 1) <= .01 else
         "# BrainVolStatsFixed see surfer.nmr.mgh.harvard.edu/fswiki/BrainVolStatsFixed"),
        *(f"# Measure {group}, {key}, {description}, {volume[key]:.6f}, mm^3"
          for group, key, description in values),
        ("# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, "
         f"{etiv:.6f}, mm^3"),
    ])
    return result
