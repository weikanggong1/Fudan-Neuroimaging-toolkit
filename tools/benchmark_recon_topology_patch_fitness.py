"""准备后完整五项评分回归：自产距离，不消费参考标量或反馈。"""
from __future__ import annotations

import time
_PROCESS_ENTRY = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource

import nibabel as nib
import numba
import numpy as np
import torch

from benchmark_recon_topology_distance_fields import load_distance_capture
from benchmark_recon_topology_mri_likelihood import load_statistics_capture
from benchmark_recon_topology_unmri_fields import load_volume_capture
from benchmark_recon_topology_volume_likelihood import arguments as volume_arguments
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
from fnit.recon_all.topology_patch_fitness import evaluate_prepared_patch_fitness
from fnit.recon_all.thread_budget import thread_budget


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(candidate, reference, *, dtype=None):
    a, b = np.asarray(candidate, dtype=dtype), np.asarray(reference, dtype=dtype)
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(shape_equal=a.shape == b.shape, dtype_equal=a.dtype == b.dtype,
                    candidate_shape=list(a.shape), reference_shape=list(b.shape),
                    candidate_dtype=str(a.dtype), reference_dtype=str(b.dtype))
    # 真正的零面积 NaN 诊断也比较其 IEEE bits，不能填0。
    differing = np.frombuffer(a.tobytes(), np.uint8) != np.frombuffer(b.tobytes(), np.uint8)
    byte_equal = not np.any(differing)
    same = (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else a == b
    finite = np.isfinite(a) & np.isfinite(b)
    error = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
    return dict(shape_equal=True, dtype_equal=True, byte_equal=bool(byte_equal),
                differing_elements=int(np.count_nonzero(~same)),
                differing_bytes=int(np.count_nonzero(differing)),
                max_abs=float(error.max(initial=0)),
                p99_abs=float(np.percentile(error, 99)) if error.size else 0.,
                nonfinite_elements=int(a.size - np.count_nonzero(finite)),
                different_zero_sign_elements=(int(np.count_nonzero((a == 0) & (b == 0)
                    & (np.signbit(a) != np.signbit(b)))) if a.dtype.kind == "f" else 0))


def regression_gates(*, comparisons, repeat, reference_repeat_exact, initial_feedback_unchanged,
                     source_context_equal):
    """已有严格子项门与准备矩阵字节诊断分列，保留±0的原byte结果。

    五项/fitness/distance/feedback/诊断位模式仍须原字节完全相同；MRI
    sampling变换沿已有MRI子函数检查同dtype、同shape、全部有限且逐值
    相同。正负零不是非零容差，也不修改矩阵数据或原报告。
    """
    matrix = comparisons["ras_to_voxel"]
    numeric_matrix_same = (matrix.get("shape_equal") is True and matrix.get("dtype_equal") is True
        and matrix.get("differing_elements") == 0 and matrix.get("nonfinite_elements") == 0)
    stage_output_exact = all(row.get("byte_equal", False) for key, row in comparisons.items()
                             if key != "ras_to_voxel")
    candidate_repeat_exact = all(row.get("byte_equal", False) for row in repeat.values())
    all_inputs_byte_exact = all(comparisons[key].get("byte_equal",False)
                               for key in ("prior_joint_pdf", "ras_to_voxel"))
    passed = bool(stage_output_exact and candidate_repeat_exact and numeric_matrix_same
                  and reference_repeat_exact and initial_feedback_unchanged and source_context_equal)
    return dict(scoring_execution_gate_passed=passed, strict_stage_output_bytes=stage_output_exact,
        strict_candidate_repeat_bytes=candidate_repeat_exact,
        strict_preparation_input_bytes=all_inputs_byte_exact,
        overall_internal_byte_reproduction=bool(passed and all_inputs_byte_exact),
        preparation_input_values_equal=numeric_matrix_same and comparisons["prior_joint_pdf"].get("byte_equal",False))


def prepared_arguments(*, statistics, volume_fields, distance_fields, brain_image, joint_pdf):
    """完整同输入检查点→已过滤的五个 kwargs，无参考值/距离/反馈字段。

    此工具只用于隔离诊断。k1/k2、校准hist/PDF、局部强度与log-P是
    明确的准备后输入；不把它们的冻结来源说成上游已自产。MRI到体素
    矩阵由原uint8主MRI header生成；joint_pdf复用先前FNIT自产校准。
    """
    vertex, face = statistics["vertex_fields"], statistics["face_fields"]
    matrix = source_topology_ras_to_voxel(volume_header=brain_image.header)
    mri = dict(original_vertices=vertex[:, :3], faces=statistics["faces"],
        vertex_normals=vertex[:, 6:9], face_normals=face[:, :3], vertex_areas_mm2=vertex[:, 9],
        face_areas_mm2=face[:, 3], patch_vertices=statistics["selected_vertices"],
        patch_faces=statistics["selected_faces"], volume=np.asarray(brain_image.dataobj),
        ras_to_voxel=matrix, gray_white_pdf=joint_pdf)

    def histogram(label):
        return dict(bins=statistics[label + "_bins"], counts=statistics[label + "_counts"],
                    bin_size=statistics[label + "_bin_size"][0])

    normal = dict(original_vertices=vertex[:, :3], vertex_normals=vertex[:, 6:9],
        vertex_areas_mm2=vertex[:, 9], patch_vertices=statistics["selected_vertices"],
        neighbor_offsets=statistics["neighbor_offsets"], ordered_neighbors=statistics["neighbors"],
        normal_dot_histogram=histogram("dot"))
    curvature = dict(principal_curvature_k1_mm_inv=vertex[:, 10],
        principal_curvature_k2_mm_inv=vertex[:, 11], vertex_areas_mm2=vertex[:, 9],
        patch_vertices=statistics["selected_vertices"], k1_histogram=histogram("k1"),
        k2_histogram=histogram("k2"), joint_curvature_pdf=statistics["k_pdf"],
        joint_origin_mm_inv=statistics["k_pdf_geometry"][:2],
        joint_spacing_mm_inv=statistics["k_pdf_geometry"][2:])
    if "algorithm_volume_geometry" not in distance_fields:
        raise ValueError("actual mri_defect algorithm geometry requires FNITDST2")
    geo = distance_fields["algorithm_volume_geometry"]
    distance = dict(original_vertices=distance_fields["vertex_fields"][:, :3],
        faces=distance_fields["faces"], face_normals=distance_fields["face_normals"],
        incident_offsets=distance_fields["incident_offsets"], incident_faces=distance_fields["incident_faces"],
        incident_corners=distance_fields["incident_corners"],
        volume_shape=tuple(map(int, distance_fields["algorithm_volume_dimensions"][:3])),
        volume_origin_mm=geo[:3], volume_index_scale_per_mm=geo[3:6])
    # 仅复用输入转换；明确移除该旧单算子诊断的参考距离。
    volume = volume_arguments(volume_fields)
    del volume["signed_distance_mm"]
    for key in ["faces", "patch_vertices", "patch_faces"]:
        actual = mri[key]
        if not np.array_equal(actual, volume_fields[key]):
            raise ValueError(f"statistics/volume candidate mismatch: {key}")
    if not np.array_equal(vertex[:, :3], volume_fields["vertex_fields"][:, :3]):
        raise ValueError("statistics/volume original coordinates mismatch")
    return dict(mri_inputs=mri, normal_inputs=normal, curvature_inputs=curvature,
                distance_inputs=distance, volume_inputs=volume)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_candidate(result, output, stem):
    # 候选落盘后才允许下面的参考比较；反馈是新的数组，不回灌原capture。
    scores = np.array([result["composition"]["tp_fields"][key]
        for key in ("face_ll", "vertex_ll", "curv_ll", "qcurv_ll", "unmri_ll")], np.float32)
    np.savez_compressed(output / (stem + ".npz"), core_fields=scores,
        fitness=np.array(result["fitness"], np.float64),
        distance_mm=result["distance"]["distance_mm"],
        vertex_curvbak=result["vertex_curvbak"])


def source_modules():
    from fnit.recon_all import (topology_patch_fitness, topology_fitness_search,
        topology_mri_likelihood, topology_edge_mri_score, topology_defect_normal_likelihood,
        topology_defect_volume_likelihood, topology_defect_signed_distance,
        topology_angle_pseudonormals, place_surface_border, thread_budget as budget_module)
    return {Path(module.__file__).stem: sha(module.__file__) for module in (
        topology_patch_fitness, topology_fitness_search, topology_mri_likelihood,
        topology_edge_mri_score, topology_defect_normal_likelihood,
        topology_defect_volume_likelihood, topology_defect_signed_distance,
        topology_angle_pseudonormals, place_surface_border, budget_module)}


def fitness_reference(path):
    """复用已验证的完整candidate schema；运行时间不参与重复性相等比较。"""
    value = json.loads(Path(path).read_text())
    if (value.get("schema") != "fnit-topology-fitness-reference-v1"
            or value.get("scope") != "first complete mrisComputeDefectLogLikelihood call"
            or value.get("tp_field_bytes") != 4 or value.get("global_weight_bytes") != 8
            or value.get("source_float_epsilon") != float(np.finfo(np.float32).eps)):
        raise ValueError("bound complete-candidate source float32 TP/double reference required")
    stable_keys = ("tp_fields", "tp_field_uint32_bits", "requested_weights", "captured_effective_weights",
        "defect_volume_shape", "face_check", "fitness", "source_commit", "source_defect_sha256",
        "observer_program_sha256", "input_sha256", "capture_sha256", "same_candidate_captures_sha256")
    return value, {key:value[key] for key in stable_keys}


def run(*, statistics_capture, volume_capture, distance_capture, brain, fnit_joint_pdf,
        reference_fitness, reference_repeat, source_manifest, output, threads=4,
        mri_weight=1., volume_unmri_weight=10., normal_dot_weight=1.,
        quadratic_curvature_weight=1., valid_faces=True):
    """同candidate准备后五项→完整两次API、IEEE回归与源码/资源收据。

    全部路径属于隔离benchmark。output必须是空的新目录；threads为
    CPU预算，默认4，原语无GPU/半精度。明确fourweights/facecheck调用
    参数而不从参考JSON产生控制分支。brain为同源uint8MRI，fnit_joint_pdf
    为先前FNIT自产灰白PDF；其余准备字段是冻结阶段输入。参考JSON
    只在候选写出后读取。返回JSON，写两个私有候选NPZ、report；上游
    candidateSFF和局部MRI准备缺口单列，不宣称原MRI→全GA或等效。
    """
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError("fresh independent prepared-fitness output required")
    if threads < 1:
        raise ValueError("positive explicit CPU thread budget required")
    statistics = load_statistics_capture(capture_path=statistics_capture)
    volume_fields = load_volume_capture(capture_path=volume_capture)
    distance_fields = load_distance_capture(capture_path=distance_capture)
    brain_image = nib.load(str(brain))
    with np.load(fnit_joint_pdf, allow_pickle=False) as archive:
        joint_pdf = archive["gray_white_pdf"].copy()
    args = prepared_arguments(statistics=statistics, volume_fields=volume_fields,
        distance_fields=distance_fields, brain_image=brain_image, joint_pdf=joint_pdf)
    args.update(valid_faces=valid_faces, mri_weight=mri_weight,
        volume_unmri_weight=volume_unmri_weight, normal_dot_weight=normal_dot_weight,
        quadratic_curvature_weight=quadratic_curvature_weight)
    loaded = time.perf_counter()
    output.mkdir(parents=True)
    before_feedback = args["volume_inputs"]["initial_vertex_curvbak"].copy()
    before_cuda = torch.cuda.is_initialized()
    with thread_budget(threads=threads) as budget:
        first = evaluate_prepared_patch_fitness(**args)
        second = evaluate_prepared_patch_fitness(**args)
        tick = time.perf_counter()
        save_candidate(first, output, "candidate")
        save_candidate(second, output, "repeat")
        candidate_io_seconds = time.perf_counter() - tick
    # 与源 outputs 的比较发生在独立候选完整落盘之后。
    reference, reference_stable = fitness_reference(reference_fitness)
    repeated_reference, repeat_stable = fitness_reference(reference_repeat)
    # 观察器 schema由其元信息固定，不猜字段顺序。
    names = ("face_ll", "vertex_ll", "curv_ll", "qcurv_ll", "unmri_ll")
    core_reference = {name: reference["tp_fields"][name] for name in names}
    comparisons = {name: difference(first["composition"]["tp_fields"][name], core_reference[name], dtype=np.float32)
        for name in names}
    comparisons["fitness"] = difference(first["fitness"], reference["fitness"], dtype=np.float64)
    comparisons["distance"] = difference(first["distance"]["distance_mm"], distance_fields["distance_after"])
    comparisons["feedback"] = difference(first["vertex_curvbak"], volume_fields["reference_curvbak"])
    for key, candidate_bits in dict(
        fll=first["components"]["mri"]["area_weighted_face_bits"],
        vll=first["components"]["mri"]["area_weighted_vertex_bits"],
        cll=first["components"]["normal"]["cll_float32_bits"],
        qcll=first["components"]["curvature"]["qcll_float32_bits"]).items():
        comparisons[key + "_uint32_bits"] = difference(candidate_bits,
            reference["tp_field_uint32_bits"][key], dtype=np.uint32)
    comparisons["prior_joint_pdf"] = difference(joint_pdf, statistics["gray_white_pdf"])
    comparisons["ras_to_voxel"] = difference(args["mri_inputs"]["ras_to_voxel"], statistics["matrix"])
    context_equal = (reference["captured_effective_weights"] == first["composition"]["effective_weights"]
        and reference["requested_weights"] == first["composition"]["requested_weights"]
        and reference["defect_volume_shape"] == list(args["distance_inputs"]["volume_shape"])
        and (reference["face_check"] >= 0) == valid_faces)
    for index, key in enumerate(("white_ll", "gray_ll", "white_mean_intensity", "gray_mean_intensity",
                               "max_distance_mm", "unmri_ll", "nwhite", "ngray")):
        comparisons[key] = difference(first["components"]["unmri"][key], volume_fields["reference_results"][index], dtype=np.float32)
    repeat = {name: difference(first["composition"]["tp_fields"][name], second["composition"]["tp_fields"][name], dtype=np.float32)
        for name in names}
    repeat.update(fitness=difference(first["fitness"], second["fitness"], dtype=np.float64),
        distance=difference(first["distance"]["distance_mm"], second["distance"]["distance_mm"]),
        feedback=difference(first["vertex_curvbak"], second["vertex_curvbak"]))
    reference_repeat_exact = reference_stable == repeat_stable
    untouched = np.array_equal(before_feedback, args["volume_inputs"]["initial_vertex_curvbak"])
    gates = regression_gates(comparisons=comparisons,repeat=repeat,
        reference_repeat_exact=reference_repeat_exact,initial_feedback_unchanged=untouched,
        source_context_equal=context_equal)
    passed = gates["scoring_execution_gate_passed"]
    summary_components = {name: {key: value for key, value in values.items() if not isinstance(value, np.ndarray)}
                          for name, values in first["components"].items()}
    report = dict(status="complete_prepared_five_component_stage", **gates,
        execution_completed=True, output_complete=True,
        scope="frozen prepared same-input stage; no candidate SFF/local MRI regeneration/full GA",
        overall_equivalence="not_assessed", final_mesh_quality="not_assessed",
        code_manifest_sha256=sha(source_manifest), module_sha256=source_modules(), benchmark_sha256=sha(__file__),
        input_sha256={key:sha(value) for key,value in dict(statistics_capture=statistics_capture,
            volume_capture=volume_capture, distance_capture=distance_capture, brain=brain,
            fnit_initial_joint_pdf=fnit_joint_pdf).items()},
        reference_json_sha256=[sha(reference_fitness),sha(reference_repeat)],
        reference_reproducibility=dict(two_same_input_full_candidate_repeats=reference_repeat_exact,
            captured_fields_only=True, source_program_version_from_reference_binding=True),
        source_reference=dict(source_commit=reference["source_commit"],
            source_defect_sha256=reference["source_defect_sha256"],
            observer_program_sha256=reference["observer_program_sha256"],
            input_sha256=reference["input_sha256"], captures_sha256=reference["same_candidate_captures_sha256"],
            source_context_equal=context_equal,
            observer_full_process_seconds=[reference["reference_process_seconds"],repeated_reference["reference_process_seconds"]],
            timing_boundary=reference["reference_time_scope"]),
        prepared_field_origins=dict(principal_curvature="frozen output of upstream candidate SFF, not regenerated",
            local_intensity_and_white_gray_log_p="frozen prepared stage inputs, not regenerated",
            normal_and_curvature_calibration="frozen upstream calibration",
            gray_white_joint_pdf="prior FNIT initial calibration output",
            signed_distance="entire candidate mesh recomputed by FNIT",
            face_check="explicit caller source-face validity flag, no inferred/reference scoring branch"),
        candidate=dict(composition=first["composition"], components=summary_components,
            executed_components=first["executed_components"]), comparisons=comparisons, candidate_repeat=repeat,
        initial_feedback_unchanged=untouched, vertices=len(args["mri_inputs"]["original_vertices"]),
        faces=len(args["mri_inputs"]["faces"]), selected_vertices=len(args["mri_inputs"]["patch_vertices"]),
        selected_faces=len(args["mri_inputs"]["patch_faces"]),
        volume_shape=list(args["distance_inputs"]["volume_shape"]),
        prior_joint_pdf=difference(joint_pdf,statistics["gray_white_pdf"]),
        matrix=difference(args["mri_inputs"]["ras_to_voxel"],statistics["matrix"]),
        seconds=dict(input_validation_load=loaded-started, first_complete_api=first["seconds"],
            repeated_complete_api=second["seconds"], candidate_npz_io=candidate_io_seconds,
            script_until_report_serialization=time.perf_counter()-_PROCESS_ENTRY),
        threads=budget, cpu_affinity=sorted(os.sched_getaffinity(0)), host_role="CFFF_CPU_validation",
        actual_hostname=platform.node(), python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
        torch=torch.__version__, cuda_initialization_unchanged=before_cuda == torch.cuda.is_initialized(),
        cuda_initialized_before=before_cuda, gpu_memory="not sampled; CUDA not used for this CPU stage",
        precision=dict(numba_fastmath=False, source_tp="float32", fitness="float64",
            autocast=False, torch_cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
        process_rss_high_water_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
        candidate_sha256={name:sha(output/(name+".npz")) for name in ("candidate","repeat")},
        known_gaps=["candidate computeDefectSecondFundamentalForm", "local defect MRI/log-P generation",
                    "full GA ordering/retessellation/controller and final mesh QC"],
        timing_boundary="CPU API validation+all own components; import/load/write separately, no independent reference component timing")
    (output / "report.private.json").write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+"\n")
    if not passed:
        raise RuntimeError("prepared five-component source/repeat regression failed; retained report")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("statistics-capture", "volume-capture", "distance-capture", "brain", "fnit-joint-pdf",
                 "reference-fitness", "reference-repeat", "source-manifest", "output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--threads",type=int,default=4)
    for name,value in (("mri-weight",1.),("volume-unmri-weight",10.),("normal-dot-weight",1.),("quadratic-curvature-weight",1.)):
        parser.add_argument("--"+name,type=float,default=value)
    parser.add_argument("--invalid-faces",action="store_true")
    options = vars(parser.parse_args())
    options["valid_faces"] = not options.pop("invalid_faces")
    run(**options)


if __name__ == "__main__":
    main()
