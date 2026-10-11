"""自产 raw→SFF1→base 的 after-only 曲率核验；参考不进入生产计算。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np

from benchmark_recon_topology_candidate_state import (
    _compare_value, load_state_capture, load_state_checkpoint)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_subset(*, candidate, candidate_maps, reference, reference_selection,
                   reference_metadata):
    """已落盘自产base与DVS record→同索引门及K1/K2，永不生成输入状态。

    candidate/reference是显式state字典：original/current_vertices为
    float32(N,3) surface RAS/mm，faces为int32容量数组(F,3)，nfaces为
    logical面数，k1/k2为float32(N)，单位mm⁻¹。candidate_maps必有整数
    source_order/vertex_translation(N)，分别为自产新→原及原→新全排列。
    reference_selection的vertex_translation是原DVS实际保存的映射前缀；
    不根据参考坐标寻找或调整映射。reference_metadata.phase必须0，
    即首次candidate前record，不能将restored/其他candidate当作初态。

    无默认阈值：完整original/current坐标、有序logical面及映射前缀
    逐字节相同后才返回K比较；门失败时K明确not_assessed。返回dict仅
    包含聚合差异、gate及边界，不含原影像/网格数组。未捕获H/K/E、
    PDF及全GA不评估。非法类型、容量、映射或相位ValueError；输入
    不修改，不调用原程序、不更改线程/CUDA/TF32或生产默认。
    """
    if reference_metadata.get('phase') != 0:
        raise ValueError('DVS first record phase0 required, not restored/candidate state')
    n = len(candidate['original_vertices'])
    if n == 0 or len(reference['original_vertices']) != n:
        raise ValueError('same nonempty complete base vertex domain required')
    for state in (candidate, reference):
        for key in ('original_vertices', 'current_vertices'):
            value = np.asarray(state[key])
            if value.shape != (n, 3) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('finite float32(N,3) complete base geometry required')
        for key in ('k1', 'k2'):
            value = np.asarray(state[key])
            if value.shape != (n,) or value.dtype != np.float32:
                raise ValueError('float32(N) complete base K fields required')
        faces = np.asarray(state['faces'])
        logical = int(state['nfaces'])
        if (faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != np.int32
                or logical < 0 or logical > len(faces)
                or (faces[:logical].size and (faces[:logical].min() < 0 or faces[:logical].max() >= n))):
            raise ValueError('valid ordered logical int32 base faces required')
    order = np.asarray(candidate_maps['source_order'])
    translation = np.asarray(candidate_maps['vertex_translation'])
    for value in (order, translation):
        if (value.shape != (n,) or value.dtype.kind not in 'iu'
                or not np.array_equal(np.sort(value), np.arange(n))):
            raise ValueError('own source order and translation must be complete permutations')
    if not np.array_equal(translation[order], np.arange(n)):
        raise ValueError('own source order and translation must be inverse, never inferred from reference')
    captured_map = np.asarray(reference_selection['vertex_translation'])
    if (captured_map.ndim != 1 or captured_map.dtype.kind not in 'iu'
            or len(captured_map) > n or captured_map.size == 0
            or captured_map.min() < 0 or captured_map.max() >= n):
        raise ValueError('actual DVS record mapping prefix required')
    mapping = _compare_value(np.asarray(translation[:len(captured_map)], np.int32),
                             np.asarray(captured_map, np.int32))
    geometry = {key: _compare_value(candidate[key], reference[key])
                for key in ('original_vertices', 'current_vertices')}
    geometry['ordered_logical_faces'] = _compare_value(
        candidate['faces'][:int(candidate['nfaces'])],
        reference['faces'][:int(reference['nfaces'])])
    same_index = mapping['byte_equal'] and all(row['byte_equal'] for row in geometry.values())
    curvature = ({key: _compare_value(candidate[key], reference[key]) for key in ('k1', 'k2')}
                 if same_index else 'not_assessed_same_index_domain_failed')
    return dict(same_index_domain_strict=same_index, geometry=geometry,
        mapping_prefix=mapping, own_full_permutation_valid=True,
        mapping_reference_entries=len(captured_map), vertices=n,
        logical_faces_candidate=int(candidate['nfaces']),
        logical_faces_reference=int(reference['nfaces']), curvature=curvature,
        strict_sff1_base_k_reproduction=(same_index and all(row['byte_equal'] for row in curvature.values())),
        reference_used_to_generate_or_adjust_state=False,
        full_global_sff1_state='not_assessed', curvature_PDF='not_assessed',
        H_K_E_reference='not_captured', overall_metric_equivalence='not_assessed',
        complete_GA=False, mesh_quality='not_assessed')


def run(*, candidate_state, candidate_maps, candidate_report, reference_state,
        reference_repeat, reference_receipt, source_manifest, output):
    """完整路径均必填，新输出目录；仅比较已有独立生产产物，不重生base。"""
    if output.exists():
        raise FileExistsError('fresh SFF1 base subset diagnostic output required')
    started = time.perf_counter()
    provenance = json.loads(candidate_report.read_text())
    if (provenance.get('reference_state_consumed') is not False
            or provenance.get('all_computed_state_self_generated') is not True):
        raise ValueError('explicit own producer no-reference provenance required')
    for key, path in (('base_state.private.npz', candidate_state), ('base_maps.private.npz', candidate_maps)):
        if provenance['checkpoint_sha256'][key] != sha(path):
            raise ValueError('candidate checkpoint no longer matches own completed producer receipt')
    candidate = load_state_checkpoint(candidate_state)
    with np.load(candidate_maps, allow_pickle=False) as fields:
        maps = {key: fields[key].copy() for key in ('source_order', 'vertex_translation')}
    loading_candidate = time.perf_counter() - started
    # Reference is opened only after independently generated state/provenance are frozen and loaded.
    receipt = json.loads(reference_receipt.read_text())
    references = receipt['references']
    if len(references) != 2 or any(row['exit_code'] != 0 for row in references):
        raise ValueError('two completed independent source reference runs required')
    declared_inputs = dict(nofix='surf/lh.orig.nofix', qsphere='surf/lh.qsphere.nofix', brain='mri/brain.mgz')
    for key, source_key in declared_inputs.items():
        if any(provenance['input_sha256'][key] != row['source_input_sha256'][source_key] for row in references):
            raise ValueError('own raw-stage input differs from reference source input: ' + key)
    for path, row in zip((reference_state, reference_repeat), references):
        if sha(path) != row['files_sha256']['state.record.bin']:
            raise ValueError('reference record capture does not match actual source receipt')
    ref, selection, metadata = load_state_capture(reference_state)
    repeated_ref, repeated_selection, repeated_metadata = load_state_capture(reference_repeat)
    first = compare_subset(candidate=candidate, candidate_maps=maps, reference=ref,
        reference_selection=selection, reference_metadata=metadata)
    second = compare_subset(candidate=candidate, candidate_maps=maps, reference=repeated_ref,
        reference_selection=repeated_selection, reference_metadata=repeated_metadata)
    reference_repeat_exact = reference_state.read_bytes() == reference_repeat.read_bytes()
    strict = first['strict_sff1_base_k_reproduction'] and second['strict_sff1_base_k_reproduction'] and reference_repeat_exact
    report = dict(status='complete_sff1_base_subset_comparison', execution_complete=True,
        strict_sff1_base_k_reproduction=strict, comparisons=first,
        reference_repeat_exact=reference_repeat_exact, second_reference_gate=second['strict_sff1_base_k_reproduction'],
        scope='independent raw nofix -> SFF1 -> base; source DVS after-only K subset, not full GA',
        producer_input_sha256={key: provenance['input_sha256'][key] for key in declared_inputs},
        producer_module_sha256=provenance['actual_module_sha256'],
        producer_source_manifest_sha256=provenance['source_manifest_sha256'],
        producer_api_timing=provenance['timing'],
        source_reference_program_sha256=[row['program_sha256'] for row in references],
        comparator_input_sha256={key: sha(path) for key, path in dict(candidate_state=candidate_state,
            candidate_maps=candidate_maps, candidate_report=candidate_report, reference_state=reference_state,
            reference_repeat=reference_repeat, reference_receipt=reference_receipt, source_manifest=source_manifest).items()},
        script_sha256=sha(__file__), timing=dict(candidate_loading_seconds=loading_candidate,
            full_comparison_before_report_write_seconds=time.perf_counter()-started),
        environment=dict(python=platform.python_version(), numpy=np.__version__, affinity=sorted(os.sched_getaffinity(0))),
        reference_and_candidate_cross_environment=True, performance_ratio='not_assessed',
        GPU_used=False, GPU_memory_bytes=None, OS_isolation='not_verified',
        complete_GA=False, mesh_quality='not_assessed', overall_metric_equivalence='not_assessed',
        production_default_changed=False)
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-state', 'candidate-maps', 'candidate-report', 'reference-state',
                 'reference-repeat', 'reference-receipt', 'source-manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    report = run(**vars(parser.parse_args()))
    print(json.dumps({key: report[key] for key in ('status', 'strict_sff1_base_k_reproduction')}))
    return 0 if report['strict_sff1_base_k_reproduction'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
