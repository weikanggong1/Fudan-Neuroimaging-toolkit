"""候选匹配前后坐标生成完整未丢弃缺陷顶点位移；独立NumPy实现。"""
from __future__ import annotations
import numpy as np
from .topology_angle_pseudonormals import _indices


def candidate_displacement(*, original_vertices, prematch_positions_mm,
        translated_defect_vertices, discarded_defect_vertices) -> dict:
    """完整未丢弃缺陷域的 orig 与 fx/fy/fz → 源 F32 位移。

    original_vertices、prematch_positions_mm 均有限float32(N,3)，当前
    orig和匹配前保存的fx/fy/fz，采用相同surface RAS/mm、顶点顺序。
    translated_defect_vertices 是整数(K,)原缺陷到当前候选网格映射，
    不限于retessellation选中的inside；discarded_defect_vertices为源
    status==DISCARD的bool(K,)。全部参数必填，无设备/线程/精度默认。
    返回 vertex_indices int64(A,)、displacement_mm float32(A,)，其中
    A为全部未丢弃条目，保持原序及重复映射；F32平方、左结合相加、
    sqrtF32。不修改输入，也不伪造未触及顶点的curvbak；控制器显式
    将结果写回自身完整状态。discarded条目保留调用方旧状态。
    这是mris_fix_topology内部computeDisplacement，无独立官方CLI，
    不执行smooth/match/fitness/GA，不读取文件或原程序、不改TF32。
    非法类型/形状/映射抛ValueError，位移溢出抛FloatingPointError。
    """
    current, before = np.asarray(original_vertices), np.asarray(prematch_positions_mm)
    if (current.dtype != np.float32 or before.dtype != np.float32
            or current.ndim != 2 or current.shape[1] != 3 or before.shape != current.shape
            or not np.isfinite(current).all() or not np.isfinite(before).all()):
        raise ValueError('finite same-order float32 orig/prematch positions (N,3) required')
    mapped = _indices(translated_defect_vertices, len(current), 'translated_defect_vertices')
    discarded = np.asarray(discarded_defect_vertices)
    if discarded.dtype != np.bool_ or discarded.shape != mapped.shape:
        raise ValueError('source discarded_defect_vertices bool(K,) required')
    active = mapped[~discarded].copy()
    with np.errstate(over='ignore', invalid='ignore'):
        delta = np.subtract(before[active], current[active], dtype=np.float32)
        square = np.multiply(delta, delta, dtype=np.float32)
        total = np.add(np.add(square[:,0], square[:,1], dtype=np.float32), square[:,2], dtype=np.float32)
        displacement = np.sqrt(total, dtype=np.float32)
    if not np.isfinite(displacement).all():
        raise FloatingPointError('nonfinite source candidate displacement')
    return dict(vertex_indices=active, displacement_mm=displacement)
