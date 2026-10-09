"""最终white独立rip-surface的具名BG分支；复用已有精确最近邻采样。"""
from __future__ import annotations

import numpy as np

from .place_surface_rip import _nearest


def final_white_rip_flags(
    vertices: np.ndarray, normals: np.ndarray,
    segmentation: np.ndarray, sras2vox: np.ndarray,
    annotation: np.ndarray, color_table: np.ndarray, names: list,
    *, hemisphere: str,
) -> np.ndarray:
    """为固定rip-surface计算最终white的新增冻结顶点，不返回目标强度。

    vertices/normals为(N,3)float32，surface RAS/mm；segmentation为
    conform网格的3D整数标签，sras2vox为(4,4)surface RAS→体素变换。
    annotation为nibabel orig_ids=True的(N,)编码，color_table为(K,5)，
    names为K个名称。hemisphere必须lh/rh。输出(N,)int32冻结标记。
    独立rip-surface从零标记；--rip-label已经关闭RipMidline，故仅执行
    具名BG区域及247冻结。外部surface的label标记应在调用方合并。
    非法形状/半球抛ValueError；本实验接口尚须完整原生同输入验收。
    属于mris_place_surface --white --rip-surf --aparc内部步骤，无独立CLI。
    """
    xyz = np.asarray(vertices, dtype=np.float32)
    normal = np.asarray(normals, dtype=np.float32)
    annot = np.asarray(annotation, dtype=np.int64)
    table = np.asarray(color_table)
    if hemisphere not in ("lh", "rh"):
        raise ValueError("hemisphere must be lh or rh")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or normal.shape != xyz.shape:
        raise ValueError("vertices/normals must have matching (N,3) shape")
    if annot.shape != (len(xyz),) or table.ndim != 2 or table.shape[1] != 5 or len(names) != len(table):
        raise ValueError("annotation/color_table/names dimensions differ")
    seg = np.asarray(segmentation, dtype=np.int32)
    affine = np.asarray(sras2vox, dtype=np.float32)
    if seg.ndim != 3 or affine.shape != (4,4):
        raise ValueError("segmentation/transform must have 3D/(4,4) shapes")
    if not np.isfinite(xyz).all() or not np.isfinite(normal).all() or not np.isfinite(affine).all():
        raise ValueError("geometry and transform must be finite")
    codes = {(n.decode("utf8") if isinstance(n, bytes) else str(n)): int(row[4])
             for n, row in zip(names, table)}
    result = np.zeros(len(xyz), dtype=np.int32)
    allowed = np.isin(annot, [codes[n] for n in
        ("medialorbitofrontal", "rostralanteriorcingulate", "insula", "lateralorbitofrontal") if n in codes])
    for vertex in np.flatnonzero(result == 0):
        x, y, z = xyz[vertex]
        nx, ny, nz = normal[vertex]
        for step in range(9):
            distance = -2. + .5*step
            label = _nearest(seg, affine, float(x)+distance*float(nx),
                             float(y)+distance*float(ny), float(z)+distance*float(nz))
            if label == 247 or (allowed[vertex] and label in (11,12,26,50,51,58,138,139)):
                result[vertex] = 1
                break
    return result.astype(np.int32)
