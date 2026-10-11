"""完整首缺陷GA面缓存的指标白名单导出；不传原影像/网格或私有日志。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_recon_topology_genetic_optimization import export_metrics, sha
from export_recon_gca_receipts import numeric_leaves


def export(*, input_run, baseline_run, frozen_source, output_directory):
    """四个具名必填目录→基础完整GA指标＋显式面缓存调用指标。

    前三个为已完成实际新/旧运行与本次不可变源码，最后是不存在的
    新目录。先验证实际face-cache与继承完成回执/源码一致，再复用
    export_metrics完整80字段/496评分/旧报告身份门。输出无MRI/NPZ/
    私有路径/命令，只含原数值及安全SHA；全部值和类型逐项保留。
    缺完成门、串错源码/报告或数值改变时抛错，不输出成功报告。
    """
    input_run, frozen_source = Path(input_run), Path(frozen_source)
    output_directory = Path(output_directory)
    candidate = input_run / 'candidate'
    path = candidate / 'face_cache.json'
    face = json.loads(path.read_text())
    complete = json.loads((candidate / 'FACE_CACHE_COMPLETE.json').read_text())
    if (not face['strict_complete_pair_passed']
            or face['status'] != 'complete_default_first_GA_face_cache'
            or complete['status'] != face['status']
            or complete['face_cache_report_sha256'] != sha(path)
            or complete['inherited_complete_sha256'] != sha(candidate / 'OPTIMIZATION_COMPLETE.json')
            or face['inherited_complete_sha256'] != complete['inherited_complete_sha256']
            or face['optimization_report_sha256'] != sha(candidate / 'optimization.json')):
        raise ValueError('same complete face-cache and inherited full GA receipts required')
    overlay_path = frozen_source / 'FACE_CACHE_OVERLAY_SHA.json'
    overlay = json.loads(overlay_path.read_text())
    if (face['helper_sha256'] != overlay['payload']['src/fnit/recon_all/topology_face_detection_cache.py']['sha256']
            or face['actual_tool_sha256'] != overlay['payload']['tools/benchmark_recon_topology_genetic_face_cache.py']['sha256']):
        raise ValueError('same actually measured face-cache module and adapter SHA required')
    selected = {name: face[name] for name in ('status', 'strict_complete_pair_passed',
        'face_detection_cache', 'preparations', 'detector_calls', 'context_preparations',
        'detector_call_count', 'generation_trace_count', 'last_zero_based_generation_index',
        'search_evaluations', 'random_draws', 'complete_wrapper_seconds', 'actual_tool_sha256',
        'delegate_tool_sha256', 'helper_sha256', 'optimization_report_sha256',
        'inherited_complete_sha256', 'production_default_changed', 'whole_GA',
        'whole_recon_all', 'overall_metric_equivalence')}
    selected['root_stage_freeze_manifest_sha256'] = overlay['root_face_detection_freeze_manifest_sha256']
    selected['dynamic_module_sha256'] = overlay['payload']['src/fnit/recon_all/topology_retessellation_dynamic.py']['sha256']
    data = (json.dumps(selected, indent=2, allow_nan=False)+'\n').encode()
    before, after = numeric_leaves(selected), numeric_leaves(json.loads(data))
    if [(type(v), v) for v in before] != [(type(v), v) for v in after]:
        raise RuntimeError('numeric value/type changed in face-cache whitelist')
    base = export_metrics(input_run=input_run, baseline_run=baseline_run,
        frozen_source=frozen_source, output_directory=output_directory)
    result_path = output_directory / 'face_cache.json'
    result_path.write_bytes(data)
    receipt = dict(scope='completed full default first defect only; numeric/algorithm SHA whitelist',
        actual_face_report_sha256=sha(path), actual_face_complete_sha256=sha(candidate / 'FACE_CACHE_COMPLETE.json'),
        actual_overlay_sha256=sha(overlay_path), exporter_sha256=sha(__file__),
        base_export_receipt_sha256=sha(output_directory / 'EXPORT_RECEIPT.json'),
        numeric_boolean_null_leaves_retained=len(before), numeric_values_and_types_preserved=True,
        payload=dict(face_cache=dict(sha256=sha(result_path), size=len(data))))
    (output_directory / 'FACE_CACHE_EXPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-run', 'baseline-run', 'frozen-source', 'output-directory'):
        parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(export(**vars(parser.parse_args())),indent=2))


if __name__ == '__main__':
    main()
