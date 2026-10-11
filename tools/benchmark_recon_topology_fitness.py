"""固定真实候选 TP/权重的完整 fitness 标量回归，仅用于隔离 benchmark。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from fnit.recon_all.topology_fitness_search import compose_patch_fitness, rank_patch_fitness


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inputs(reference):
    if reference.get("schema") != "fnit-topology-fitness-reference-v1":
        raise ValueError("complete same-call topology fitness reference required")
    if reference.get("scope") != "first complete mrisComputeDefectLogLikelihood call":
        raise ValueError("initial statistics or unMRI-only capture is not a full fitness reference")
    if reference.get("source_float_epsilon") != float(np.finfo(np.float32).eps):
        raise ValueError("reference translation-unit FLT_EPSILON differs")
    if reference.get("tp_field_bytes") != 4 or reference.get("global_weight_bytes") != 8:
        raise ValueError("source TP float32 and global weight double types required")
    fields, weights = reference["tp_fields"], reference["requested_weights"]
    return dict(face_ll=fields["face_ll"], vertex_ll=fields["vertex_ll"],
        normal_dot_ll=fields["curv_ll"], quadratic_curvature_ll=fields["qcurv_ll"],
        volume_unmri_ll=fields["unmri_ll"], valid_faces=reference["face_check"] >= 0,
        mri_weight=weights["l_mri"], volume_unmri_weight=weights["l_unmri"],
        normal_dot_weight=weights["l_curv"], quadratic_curvature_weight=weights["l_qcurv"],
        defect_volume_shape=reference["defect_volume_shape"])


def validate(*, reference_json: Path, repeated_reference_json: Path, output_json: Path,
             repeat_calls: int = 10000) -> dict:
    """完整真实源同一调用标量→复现/重复性、CPU API耗时及输入哈希。

    两个 JSON 是 benchmark 观察器派生的 schema 固定标量记录，必须各自
    绑定捕获文件和参考程序 SHA；不得用初始统计或不同候选子项替代。
    输出为 JSON 报告，不导出影像/原代码，也不改变任何生产输入。
    计时包括每次 scalar API 的验证、float32写回和double合成；不包含
    MRI/曲率/unMRI几何计算或原软件完整时间，不能称 GA 或整例提速。
    非法格式/参考变化/未知类型抛 ValueError；逐double字节不符保留
    failed 报告后抛 AssertionError；已有输出抛 FileExistsError。
    """
    if output_json.exists():
        raise FileExistsError("new isolated fitness report required")
    if repeat_calls < 2:
        raise ValueError("repeat_calls must be at least two")
    raw = reference_json.read_bytes()
    reference = json.loads(raw)
    repeated = json.loads(repeated_reference_json.read_text())
    arguments = _inputs(reference)
    if _inputs(repeated) != arguments or repeated["fitness"] != reference["fitness"]:
        raise ValueError("same input repeated source reference fields/fitness changed")
    for name in ("source_commit", "source_defect_sha256", "observer_program_sha256",
                 "input_sha256", "tp_field_bytes", "global_weight_bytes", "source_float_epsilon"):
        if reference[name] != repeated[name]:
            raise ValueError(f"source reference repeat binding differs: {name}")
    first = compose_patch_fitness(**arguments, return_diagnostics=True)
    values = np.empty(repeat_calls, np.float64)
    started = time.perf_counter()
    for index in range(repeat_calls):
        values[index] = compose_patch_fitness(**arguments)
    seconds = time.perf_counter() - started
    expected = np.float64(reference["fitness"])
    exact = values.tobytes() == np.full(repeat_calls, expected, np.float64).tobytes()
    result = dict(status="complete_same_input_fitness_composition" if exact else "failed_fitness_composition",
        scope="frozen complete candidate TP scalar composition only; no independently generated component or GA claim",
        reference_json_sha256=_sha(reference_json), repeated_reference_json_sha256=_sha(repeated_reference_json),
        references={"source_commit": reference["source_commit"],
                    "source_defect_sha256": reference["source_defect_sha256"],
                    "observer_program_sha256": reference["observer_program_sha256"],
                    "input_sha256": reference["input_sha256"],
                    "captures_sha256": [reference["capture_sha256"], repeated["capture_sha256"]]},
        source_repeatability={"semantic_tp_inputs_equal": True, "fitness_double_equal": True,
                              "capture_bytes_equal": reference["capture_sha256"] == repeated["capture_sha256"]},
        candidate=first, source_fitness=float(expected),
        absolute_error=abs(first["fitness"]-float(expected)),
        candidate_reference_double_bytes_equal=exact, candidate_repeated_double_bytes_equal=bool(np.all(values==values[0])),
        seconds=seconds, repeat_calls=repeat_calls, seconds_per_complete_scalar_api=seconds/repeat_calls,
        timing_boundary="scalar validation/casts/composition only; excludes all geometry/MRI/volume/GA and file I/O",
        source_sha256={"topology_fitness_search.py":_sha(Path(__import__(compose_patch_fitness.__module__,fromlist=['']).__file__)),
                       "benchmark_recon_topology_fitness.py":_sha(__file__)},
        environment={"python":platform.python_version(),"numpy":np.__version__,"host":platform.node(),
                     "gpu_used":False,"tf32_changed":False,"half_precision_used":False},
        execution_complete=True, strict_composition_reproduction=exact,
        complete_ga_validated=False, mesh_quality="not_assessed_here", overall_metric_equivalence="not_assessed")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    if not exact:
        raise AssertionError("fitness scalar differs from complete fixed-source same-input reference")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json",type=Path,required=True)
    parser.add_argument("--repeated-reference-json",type=Path,required=True)
    parser.add_argument("--output-json",type=Path,required=True)
    parser.add_argument("--repeat-calls",type=int,default=10000)
    result=validate(**vars(parser.parse_args()))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
