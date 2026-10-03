"""Compute FreeSurfer surface ROI area, thickness and gray volume.

Curvature and folding columns are not implemented here.
"""

from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import torch



@torch.inference_mode()
def vertex_th3_volume(white: str | Path, pial: str | Path,
                      cortex_label: str | Path, *, device: str = "cuda:0") -> np.ndarray:
    """Divide each white/pial triangular prism into the official three tetrahedra."""
    wxyz, faces = fsio.read_geometry(str(white))
    pxyz, pfaces = fsio.read_geometry(str(pial))
    if len(wxyz) != len(pxyz) or not np.array_equal(faces, pfaces):
        raise ValueError("White and pial surfaces must have identical topology")
    w = torch.as_tensor(np.asarray(wxyz, dtype=np.float32), device=device)
    p = torch.as_tensor(np.asarray(pxyz, dtype=np.float32), device=device)
    tri = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device)
    a, b, c = (tri[:, i] for i in range(3))
    bw, cw, aw = w[b] - p[a], w[c] - p[a], w[a] - p[a]
    bp, cp = p[b] - p[a], p[c] - p[a]
    t1 = (aw * torch.cross(bw, cw, dim=1)).sum(1).abs()
    t2 = (bp * torch.cross(cp, bw, dim=1)).sum(1).abs()
    t3 = (cp * torch.cross(cw, bw, dim=1)).sum(1).abs()
    share = (t1 + t2 + t3).div_(18.0)
    result = torch.zeros(len(w), device=device, dtype=torch.float32)
    for corner in range(3):
        result.index_add_(0, tri[:, corner], share)
    cortex = np.zeros(len(wxyz), dtype=bool)
    cortex[fsio.read_label(str(cortex_label))] = True
    result *= torch.as_tensor(cortex, device=device)
    return result.cpu().numpy()


def vertex_volume_map(white: str | Path, pial: str | Path,
                      cortex_label: str | Path, output: str | Path,
                      *, device: str = "cuda:0") -> None:
    """Write the TH3 morphometry map used by `mris_convert --volume`."""
    fsio.write_morph_data(str(output), vertex_th3_volume(white, pial,
                                                         cortex_label, device=device))


@torch.inference_mode()
def roi_area_thickness(surface: str | Path, annotation: str | Path,
                       thickness: str | Path, *, device: str = "cuda:0",
                       cache=None) -> dict:
    """按 .annot 返回 {脑区: (顶点数, 面积 mm², 平均厚度 mm, 厚度总体 std mm)}。

    surface 是 surface RAS/mm 三角网格；thickness 是同序 (N,) morph；
    device 默认 cuda:0，cache 可复用同设备 SurfaceStatsCache。
    缺失文件、顶点数或设备不一致会抛异常。CUDA 摘要一次回传，
    保持各区顶点原有顺序、float64 归约及官方逐面 float32 面积分摊。
    对应 mris_anatomical_stats -no-th3 的四列，不包含全局表头。
    """
    from .surface_stats_cache import SurfaceStatsCache

    if cache is None:
        with SurfaceStatsCache(device=device) as temporary:
            return temporary.roi_base(surface, annotation, thickness)[0]
    cache.check_device(device)
    return cache.roi_base(surface, annotation, thickness)[0]


@torch.inference_mode()
def roi_gray_volume(white: str | Path, pial: str | Path,
                    thickness: str | Path, annotation: str | Path,
                    *, device: str = "cuda:0", cache=None) -> dict[str, float]:
    """按注释返回 mris_anatomical_stats -no-th3 脑区体积，单位 mm³。

    white/pial 为 surface RAS/mm 网格且有序面必须完全相同；thickness
    是同序 mm 顶点图，annotation 是 .annot。device 默认 cuda:0；
    cache 可复用同设备 SurfaceStatsCache。基础量按平均三角面厚度及
    white/pial 面积计算，未使用 TH3 四面体顶点图。输入/设备不一致
    或文件缺失会抛异常；空/排除脑区不出现在返回字典中。
    """
    from .surface_stats_cache import SurfaceStatsCache

    if cache is None:
        with SurfaceStatsCache(device=device) as temporary:
            return temporary.roi_volumes(white, pial, thickness, annotation)
    cache.check_device(device)
    return cache.roi_volumes(white, pial, thickness, annotation)
