"""准备后的同一候选→五个自产似然与源 double fitness。

复用现有 MRI、法向、主曲率、完整有向距离及 unMRI 聚合函数。
校准 PDF、候选主曲率和局部 MRI 必须由调用者先完成；此接口不将
冻结的参考字段冒充这些上游实现，不执行 GA，也不运行原版程序。
"""
from __future__ import annotations

from collections.abc import Mapping
import time

import numpy as np

from .topology_defect_normal_likelihood import (
    patch_curvature_likelihood, patch_normal_dot_likelihood,
)
from .topology_defect_signed_distance import defect_signed_distance
from .topology_defect_volume_likelihood import defect_volume_likelihood
from .topology_fitness_search import compose_patch_fitness
from .topology_mri_likelihood import patch_mri_likelihood


def _matching(first, second, field):
    if not np.array_equal(np.asarray(first), np.asarray(second)):
        raise ValueError(f"same candidate version required for {field}")


def _inputs(mri_inputs, normal_inputs, curvature_inputs, distance_inputs, volume_inputs):
    """检查共享输入版本；具体算子继续执行完整类型/CSR/PDF校验。"""
    named = dict(mri_inputs=mri_inputs, normal_inputs=normal_inputs,
                 curvature_inputs=curvature_inputs, distance_inputs=distance_inputs,
                 volume_inputs=volume_inputs)
    if any(not isinstance(value, Mapping) for value in named.values()):
        raise ValueError("five explicit prepared-input mappings required")
    if any(str(key).startswith("reference_") or key == "signed_distance_mm"
           for values in named.values() for key in values):
        raise ValueError("reference outputs and caller distance are not stage inputs")
    for field in ("original_vertices", "vertex_normals", "vertex_areas_mm2", "patch_vertices"):
        _matching(mri_inputs[field], normal_inputs[field], field)
    for field in ("vertex_areas_mm2", "patch_vertices"):
        _matching(mri_inputs[field], curvature_inputs[field], field)
    for field in ("original_vertices", "faces", "face_normals"):
        _matching(mri_inputs[field], distance_inputs[field], field)
    for field in ("volume_origin_mm", "volume_index_scale_per_mm"):
        _matching(distance_inputs[field], volume_inputs[field], field)
    shape = np.asarray(volume_inputs["defect_intensity_volume"]).shape
    _matching(distance_inputs["volume_shape"], shape, "local volume shape")
    count = len(np.asarray(mri_inputs["original_vertices"]))
    if (np.asarray(volume_inputs["initial_vertex_curvbak"]).shape != (count,)
            or np.asarray(curvature_inputs["principal_curvature_k1_mm_inv"]).shape != (count,)
            or np.asarray(curvature_inputs["principal_curvature_k2_mm_inv"]).shape != (count,)):
        raise ValueError("full same-order candidate vertex fields required")
    if np.float32(volume_inputs["distance_unset_sentinel_mm"]) != np.float32(1000):
        raise ValueError("source signed-distance sentinel NPY=1000 required")
    return shape


def evaluate_prepared_patch_fitness(*, mri_inputs, normal_inputs, curvature_inputs,
                                    distance_inputs, volume_inputs, valid_faces,
                                    mri_weight=1.0, volume_unmri_weight=10.0,
                                    normal_dot_weight=1.0, quadratic_curvature_weight=1.0,
                                    distance_evaluator=None) -> dict:
    """同一版本的准备输入→自产五项、距离/反馈和源有序 double fitness。

    五个 *_inputs 为对应现有函数的具名输入 Mapping，字段及数组格式
    见 TOPOLOGY_PATCH_FITNESS_ASSEMBLY.md。mri/normal/distance 共用同序
    surface RAS/mm 原坐标、面和当前无量纲法向；面积 mm²、主曲率 mm⁻¹。
    normal/curvature 共用原有序 patch 索引。curvature_inputs 的 k1/k2、
    全局 priors 与 volume_inputs 的局部 MRI/log-P 均是明确上游输入，
    本接口不生成或验证它们的上游来源。不得把参考输出当生产输入。
    volume_inputs 排除 signed_distance_mm：本次从几何完整生成距离，
    再送入聚合。初始 curvbak 和 fx/fy/fz 必须保持原控制器状态。
    valid_faces 为原 mrisCheckDefectFaces 是否非负的 bool，不替代QC。
    四个权重是实际源全局 double，默认1/10/1/1；负值合法。按源
    abs<FLT_EPSILON 跳过对应算子，局部任一轴<=5只跳过本次 unMRI。
    返回 fitness/composition、components、自产 distance 和独立 curvbak
    副本、seconds、executed_components；调用者显式决定反馈写回。
    distance_evaluator默认为None，调用成熟CPU距离；显式callable接收
    完整distance_inputs并返回同结构完整距离场，不改变跳过或算子顺序。
    不修改输入、全局线程/精度或控制器状态，不读写文件。缺字段抛
    KeyError，跨版本几何/非法输入抛 ValueError，算子非有限核心输出
    抛 FloatingPointError；原面积加权 NaN 诊断原样保留。不执行
    computeDefectSecondFundamentalForm/MRI准备、retessellation或完整GA。
    因此它是准备后评分阶段，不是原始T1→完整fitness/拓扑修复接口。
    """
    started = time.perf_counter()
    if distance_evaluator is not None and not callable(distance_evaluator):
        raise ValueError("distance_evaluator must be an explicit callable or None")
    shape = _inputs(mri_inputs, normal_inputs, curvature_inputs, distance_inputs, volume_inputs)
    weights = dict(mri_weight=mri_weight, volume_unmri_weight=volume_unmri_weight,
                   normal_dot_weight=normal_dot_weight,
                   quadratic_curvature_weight=quadratic_curvature_weight,
                   defect_volume_shape=shape)
    empty = compose_patch_fitness(0., 0., 0., 0., 0., valid_faces,
                                 **weights, return_diagnostics=True)
    effective = empty["effective_weights"]
    epsilon = float(np.finfo(np.float32).eps)
    enabled = {key: abs(value) >= epsilon for key, value in effective.items()}
    fields = dict(face_ll=0., vertex_ll=0., normal_dot_ll=0.,
                  quadratic_curvature_ll=0., volume_unmri_ll=0.)
    components, seconds, executed = {}, {}, []
    distance, curvbak = None, None
    if enabled["l_mri"]:
        tick = time.perf_counter()
        components["mri"] = patch_mri_likelihood(**mri_inputs)
        seconds["mri_complete_api"] = time.perf_counter() - tick
        fields.update(face_ll=components["mri"]["face_ll"], vertex_ll=components["mri"]["vertex_ll"])
        executed.append("mri")
    if enabled["l_unmri"]:
        tick = time.perf_counter()
        distance = (defect_signed_distance if distance_evaluator is None
                    else distance_evaluator)(**distance_inputs)
        seconds["signed_distance_complete_api"] = time.perf_counter() - tick
        tick = time.perf_counter()
        components["unmri"] = defect_volume_likelihood(
            **volume_inputs, signed_distance_mm=distance["distance_mm"])
        seconds["unmri_complete_api"] = time.perf_counter() - tick
        fields["volume_unmri_ll"] = components["unmri"]["unmri_ll"]
        curvbak = components["unmri"]["vertex_curvbak"]
        executed.extend(("signed_distance", "unmri"))
    if enabled["l_qcurv"]:
        tick = time.perf_counter()
        components["curvature"] = patch_curvature_likelihood(**curvature_inputs)
        seconds["curvature_complete_api"] = time.perf_counter() - tick
        fields["quadratic_curvature_ll"] = components["curvature"]["qcurv_ll"]
        executed.append("curvature")
    if enabled["l_curv"]:
        tick = time.perf_counter()
        components["normal"] = patch_normal_dot_likelihood(**normal_inputs)
        seconds["normal_complete_api"] = time.perf_counter() - tick
        fields["normal_dot_ll"] = components["normal"]["curv_ll"]
        executed.append("normal")
    composition = compose_patch_fitness(**fields, valid_faces=valid_faces,
                                       **weights, return_diagnostics=True)
    seconds["prepared_five_component_api_including_validation"] = time.perf_counter() - started
    return {"fitness": composition["fitness"], "composition": composition,
            "components": components, "distance": distance, "vertex_curvbak": curvbak,
            "seconds": seconds, "executed_components": executed,
            "scope": "prepared five-likelihood stage; no upstream curvature/MRI preparation or complete GA"}
