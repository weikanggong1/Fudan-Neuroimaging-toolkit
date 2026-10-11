"""复用既有同次源观测的完整 SFF2 输入；仅核 K1/K2 核心，不核 SFF1 priors。"""
from __future__ import annotations
import time
_ENTRY = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import numpy as np

from benchmark_recon_topology_mri_likelihood import load_statistics_capture
from benchmark_recon_topology_calibration_fields import load_calibration_capture
from benchmark_recon_topology_global_sff import _compare
from fnit.recon_all.topology_global_sff import global_second_fundamental_form
from fnit.recon_all.thread_budget import thread_budget


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actual_inputs(*, statistics, calibration):
    """同原域 FNITFIT1 几何/法向/rip + 后续 FNITHST1 未变有序 vtotal。

    所有字段来自独立冻结同输入阶段诊断；不是生产的上游生成器。
    源 SFF2 与 HST1 入口之间未改邻域；必须同 N、有序面、一圈前缀
    和全 rip。initial_k1/k2 独立置 0，不读取捕获的输出 K。若任何
    rip/无邻域保留分支消费初始值，则完整 K 核门不作通过声明。
    输入 dict 由已有完整 parser 提供；非法相位、域、CSR 或前缀报
    ValueError。当前 XYZ 和 actual normals 的单位为 mm 与无量纲。
    """
    vertex = statistics['vertex_fields']
    top, flags = calibration['topology'], calibration['flags']
    count = len(vertex)
    if (int(calibration['header'][6]) != 0 or len(top) != count
            or not np.array_equal(statistics['faces'], calibration['faces'])
            or not np.array_equal(vertex[:, 15], flags[:, 1])
            or not np.array_equal(vertex[:, :3], calibration['vertex_fields'][:, :3])):
        raise ValueError('same unchanged original-domain phase0 geometry/faces/rip required')
    # vnum is retained as the prefix of the actual, captured vtotal list.
    one_offsets, one = statistics['neighbor_offsets'], statistics['neighbors']
    full_offsets, full = calibration['neighbor_offsets'], calibration['neighbors']
    if (not np.array_equal(np.diff(one_offsets), top[:, 0])
            or not np.array_equal(np.diff(full_offsets), top[:, 1])
            or np.any(top[:, 2] != 2)):
        raise ValueError('original actual nsize2/vtotal CSR required')
    for vertex_id in range(count):
        prefix = full[full_offsets[vertex_id]:full_offsets[vertex_id] + top[vertex_id, 0]]
        if not np.array_equal(prefix, one[one_offsets[vertex_id]:one_offsets[vertex_id + 1]]):
            raise ValueError('original ordered one-ring prefix changed')
    args = dict(current_vertices=vertex[:, 3:6], vertex_normals=vertex[:, 6:9],
                neighbor_offsets=full_offsets, neighbors=full,
                vertex_rip=flags[:, 1], initial_k1=np.zeros(count, np.float32),
                initial_k2=np.zeros(count, np.float32))
    audit = dict(vertices=count, faces=len(statistics['faces']),
        original_neighbors=len(one), actual_vtotal_neighbors=len(full),
        same_ordered_faces=True, same_ordered_one_ring_prefix=True,
        same_original_xyz=True, same_rip=True, actual_nsize2_all=True,
        rip_vertices=int(np.count_nonzero(flags[:, 1])),
        degree_zero_vertices=int(np.count_nonzero(top[:, 1] == 0)),
        current_equals_smoothed_original=np.array_equal(vertex[:, :3], vertex[:, 3:6]),
        reference_output_k_used_as_input=False,
        initial_k='independent zero; consumption checked by fit_kind',
        normal_input_phase='actual SFF2 normals after preceding metric; unchanged by SFF/MRI likelihood',
        actual_vtotal_phase='first defect calibration entrance after SFF2; source has no intervening topology change')
    return args, audit


def run(*, statistics, statistics_repeat, calibration, calibration_repeat,
        source_manifest, output, threads):
    """冻结全 N 同输入 SFF2 核与重复；包含加载/JIT/结果写出，不运行源程序。

    statistics/statistics_repeat 是同固定源两次 FNITFIT1；calibration 两
    路径为对应 phase0 FNITHST1。source_manifest 记录本次实际源码哈希；
    output 必须新目录；threads≥1，Numba/OpenMP 由成熟作用域限制。
    输出完整 candidate.private.npz、report.json；K 两数组逐值/逐位，
    输入域覆盖、rip/degree0 分支、自身/参考重复分别报告。未捕获主方向、
    H/K、摘要明确 not_assessed；不是原始 MRI→SFF1/PDF 或完整 GA。
    """
    if output.exists(): raise FileExistsError('fresh SFF2 core output required')
    if threads < 1: raise ValueError('positive total thread budget required')
    started = time.perf_counter()
    paths = dict(statistics=statistics, statistics_repeat=statistics_repeat,
        calibration=calibration, calibration_repeat=calibration_repeat)
    hashes = {name: sha(path) for name, path in paths.items()}
    captured = load_statistics_capture(capture_path=statistics)
    full = load_calibration_capture(capture_path=calibration)
    args, audit = actual_inputs(statistics=captured, calibration=full)
    loaded = time.perf_counter() - started
    with thread_budget(threads=threads) as budget:
        first = global_second_fundamental_form(**args)
        second = global_second_fundamental_form(**args)
    output.mkdir(parents=True)
    arrays = {key: value for key, value in first.items() if isinstance(value, np.ndarray)}
    writing = time.perf_counter()
    np.savez_compressed(output / 'candidate.private.npz', **arrays)
    writing = time.perf_counter() - writing
    # Only now consume the reference post-SFF2 K values for comparison.
    reference = dict(k1=captured['vertex_fields'][:, 10], k2=captured['vertex_fields'][:, 11])
    comparisons = _compare(arrays, reference)
    repeated = _compare(arrays, {key: second[key] for key in arrays})
    branch_count = {str(kind): int(np.count_nonzero(first['fit_kind'] == kind))
                    for kind in (-1, 0, 1, 2, 3)}
    preserved = int(np.count_nonzero(first['fit_kind'] <= 0))
    audit.update(fit_kind_counts=branch_count, retained_initial_k_vertices=preserved,
        initial_k_consumed=bool(preserved))
    all_covered = preserved == 0 and len(first['k1']) == audit['vertices']
    reference_repeated = (hashes['statistics'] == hashes['statistics_repeat']
                          and hashes['calibration'] == hashes['calibration_repeat'])
    # Re-open the reference repeat to check the same input domain too.
    args2, audit2 = actual_inputs(
        statistics=load_statistics_capture(capture_path=statistics_repeat),
        calibration=load_calibration_capture(capture_path=calibration_repeat))
    unchanged_inputs = all(np.array_equal(args[name], args2[name]) for name in args)
    strict = all_covered and reference_repeated and unchanged_inputs and all(
        item['byte_equal'] for item in list(comparisons.values()) + list(repeated.values()))
    from fnit.recon_all import topology_global_sff, topology_candidate_sff, topology_vnl_svd
    report = dict(schema_version=1, execution_complete=True, scope='frozen same-input SFF2 K1/K2 core only',
        source_manifest_sha256=sha(source_manifest), capture_sha256=hashes,
        candidate_sha256=sha(output/'candidate.private.npz'),
        executed_module_sha256={name: sha(module.__file__) for name, module in [
            ('topology_global_sff', topology_global_sff), ('topology_candidate_sff', topology_candidate_sff),
            ('topology_vnl_svd', topology_vnl_svd)]}, benchmark_sha256=sha(__file__),
        input_audit=audit, repeated_domain_audit_equal=audit2['vertices'] == audit['vertices'],
        reference_repeat_bytes_exact=reference_repeated, reference_repeat_input_values_exact=unchanged_inputs,
        comparisons=comparisons, self_repeat=repeated, strict_sff2_core_reproduction=bool(strict),
        full_global_sff_state_reproduction='not_assessed', sff1_priors='not_assessed',
        not_assessed=['reference tangent_e1/e2', 'H and Gaussian K fields', 'surface summaries',
                      'raw MRI to SFF1/PDF', 'global dot prior', 'full GA', 'whole recon-all', 'mesh quality'],
        seconds=dict(loading=loaded, first_api=first['seconds'], repeat_api=second['seconds'],
            candidate_writing=writing, script_before_report=time.perf_counter()-_ENTRY),
        environment=dict(python=platform.python_version(), numpy=np.__version__,
            affinity=sorted(os.sched_getaffinity(0)), thread_budget=budget),
        precision=dict(ordered_float32=True, eispack_float64=True, fastmath=False,
            cuda_executed=False, tf32_unchanged=True, gpu_memory='not_assessed'),
        performance_ratio='not_assessed; source observer captures many phases',
        overall_equivalence='not_assessed', production_default_changed=False)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('statistics', 'statistics-repeat', 'calibration', 'calibration-repeat', 'source-manifest', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    return 0 if run(**vars(args))['strict_sff2_core_reproduction'] else 1


if __name__ == '__main__': raise SystemExit(main())
