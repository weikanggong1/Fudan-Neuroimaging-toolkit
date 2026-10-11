"""拓扑候选的源顺序 fitness 合成与排名，不执行 GA 或改变网格。

复用调用者计算的 MRI、unMRI、法向与主曲率似然。TP 标量先按源
float32 字段写回，再按固定 d932 源的 double 顺序加权合成；不调用
原版程序、不启用 CUDA 或修改精度。对应规则见 TOPOLOGY_PATCH_FITNESS.md。
"""

from __future__ import annotations

import numpy as np


_SOURCE_FLOAT_EPSILON = float(np.finfo(np.float32).eps)


def _finite_scalar(value, name: str, *, tp_field: bool = False) -> float:
    data = np.asarray(value)
    if data.ndim != 0 or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise ValueError(f"{name} must be a finite real scalar")
    if not np.isfinite(data):
        raise ValueError(f"{name} must be a finite real scalar")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.float32(data) if tp_field else np.float64(data)
    if not np.isfinite(result):
        raise ValueError(f"{name} is outside the source scalar range")
    return float(result)


def compose_patch_fitness(face_ll: float, vertex_ll: float, normal_dot_ll: float,
                          quadratic_curvature_ll: float, volume_unmri_ll: float,
                          valid_faces: bool, *, mri_weight: float = 1.0,
                          volume_unmri_weight: float = 10.0,
                          normal_dot_weight: float = 1.0,
                          quadratic_curvature_weight: float = 1.0,
                          defect_volume_shape: tuple[int, int, int] | None = None,
                          return_diagnostics: bool = False) -> float | dict:
    """五个完整子项→源有序 double fitness；可返回透明诊断字段。

    face_ll/vertex_ll 是 MRI 子函数返回的源 TP 字段；normal_dot_ll、
    quadratic_curvature_ll、volume_unmri_ll 分别是 curv_ll/qcurv_ll/unmri_ll。
    五项均为无单位 log-P 标量，先转换 float32 再拓宽为 double；unMRI
    输入必须已经完成源 float32 白灰相加，不能传未归一化体积总和。
    valid_faces 是原 mrisCheckDefectFaces 是否非负的 bool，不能用自相交
    QC 替代；False 在合成后扣 10,000,000。四个 *_weight 为实际源全局
    double 权重，默认 -ga 配方 1/10/1/1，允许负值；绝对值小于
    FLT_EPSILON 的项不执行。defect_volume_shape 是原局部 MRI 的 XYZ
    三个正整数大小；任一轴 <=5 时当次 unMRI 权重置0，其他项不变。
    None 表示调用者已确认体积项允许执行，不从标量猜测尺寸。
    return_diagnostics 默认 False，返回 Python float；True 返回 dict：
    fitness、五个 float32 写回后的 tp_fields、源顺序四项 terms、
    requested_weights/effective_weights、volume_term_enabled 和 face_penalty。
    输入不修改，不读写文件，不生成几何或 TP 面积加权诊断字段。
    非实数/非有限或超 float32 字段范围、非法 shape/bool 抛 ValueError；
    合成溢出抛 FloatingPointError。候选可有负面积或中间相交；此函数
    不替代最终连通性、流形、自相交或球面翻折检查。它是
    mris_fix_topology -ga 的内部步骤，没有独立原软件 CLI。
    """
    if not isinstance(valid_faces, (bool, np.bool_)) or not isinstance(return_diagnostics, (bool, np.bool_)):
        raise ValueError("valid_faces and return_diagnostics must be bool")
    fields = {name: _finite_scalar(value, name, tp_field=True) for name, value in (
        ("face_ll", face_ll), ("vertex_ll", vertex_ll), ("curv_ll", normal_dot_ll),
        ("qcurv_ll", quadratic_curvature_ll), ("unmri_ll", volume_unmri_ll))}
    weights = {name: _finite_scalar(value, name) for name, value in (
        ("l_mri", mri_weight), ("l_unmri", volume_unmri_weight),
        ("l_qcurv", quadratic_curvature_weight), ("l_curv", normal_dot_weight))}
    effective = weights.copy()
    if defect_volume_shape is not None:
        shape = np.asarray(defect_volume_shape)
        if (shape.shape != (3,) or not np.issubdtype(shape.dtype, np.integer)
                or np.any(shape <= 0)):
            raise ValueError("defect_volume_shape requires three positive integer XYZ sizes")
        if np.any(shape <= 5):
            effective["l_unmri"] = 0.0
    source_values = (fields["vertex_ll"] + fields["face_ll"], fields["unmri_ll"],
                     fields["qcurv_ll"], fields["curv_ll"])
    terms = {}
    score = 0.0
    for name, value in zip(("l_mri", "l_unmri", "l_qcurv", "l_curv"), source_values):
        term = effective[name] * value if abs(effective[name]) >= _SOURCE_FLOAT_EPSILON else 0.0
        terms[name] = term
        score += term
    if not valid_faces:
        score -= 10_000_000.0
    if not np.isfinite(score):
        raise FloatingPointError("nonfinite topology candidate fitness")
    if return_diagnostics:
        return {"fitness": score, "tp_fields": fields, "terms": terms,
                "requested_weights": weights, "effective_weights": effective,
                "volume_term_enabled": abs(effective["l_unmri"]) >= _SOURCE_FLOAT_EPSILON,
                "face_penalty": 0.0 if valid_faces else -10_000_000.0}
    return score


def rank_patch_fitness(fitness: np.ndarray) -> np.ndarray:
    """(P,) finite double fitness→源 defectPatchRank 的 (P,) int32 排名。

    逐对按 double 差的 abs < FLT_EPSILON 判同分，先出现的候选优先；
    更高 fitness 更优。输入转 float64、不修改或排序原数组，无随机性。
    空输入返回空数组；非一维、非有限值抛 ValueError。没有原版独立 CLI。
    """
    values = np.asarray(fitness, np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("finite one-dimensional fitness vector required")
    ranks = np.zeros(len(values), np.int32)
    for index, value in enumerate(values):
        for other, peer in enumerate(values):
            if other == index:
                continue
            if abs(peer - value) < _SOURCE_FLOAT_EPSILON:
                if index > other:
                    ranks[index] += 1
            elif peer > value:
                ranks[index] += 1
    return ranks


def invalid_modified_edges(topology_edges: np.ndarray, faces: np.ndarray,
                           modified_vertices: np.ndarray) -> int:
    """Count first-patch edges that would trigger ``mrisCheckDefectFaces``."""
    triangles = np.asarray(faces, np.int32)
    edge_rows = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                triangles[:, [2, 0]]), axis=0)
    edge_rows.sort(axis=1)
    unique, counts = np.unique(edge_rows, axis=0, return_counts=True)
    incidence = {tuple(edge): int(count) for edge, count in zip(unique, counts)}
    marked = set(np.asarray(modified_vertices, np.int32).tolist())
    checked = {tuple(sorted((int(a), int(b)))) for a, b in topology_edges
               if int(a) in marked or int(b) in marked}
    return sum(incidence.get(edge, 0) != 2 for edge in checked)
