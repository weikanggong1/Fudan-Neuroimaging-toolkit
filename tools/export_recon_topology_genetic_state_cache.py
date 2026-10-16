"""完整首缺陷状态缓存的安全指标导出；原始日志/影像留在服务器。"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

from export_recon_gca_receipts import numeric_leaves
import export_recon_topology_genetic_optimization as complete_export


def export_metrics(*, input_run, baseline_run, frozen_source, output_directory):
    """四个必填Path→完成门绑定的公开指标，不修改任何算法产物。

    输入为本轮完整run、已完成baseline、原冻结source及不存在的公开
    输出目录。继承成熟完整495/80字段/三NPZ导出，只新增明确状态
    上下文计数/耗时/源码SHA。私有metadata镜像只供旧导出接口绑定，
    SOURCE_MANIFEST和protocol原bytes保持；不改变冻结source或原
    报告。数字/类型往返保持，公开禁止地址、账户、命令和服务器路径。
    无完整EXIT0和497装配门时失败，不将running输出当成功benchmark。
    """
    input_run, baseline_run, frozen_source, output_directory = map(Path,
        (input_run, baseline_run, frozen_source, output_directory))
    if output_directory.exists():
        raise FileExistsError('fresh public state-cache metrics directory required')
    state_path = input_run/'candidate/state_cache.json'
    state = json.loads(state_path.read_text())
    complete_path = input_run/'candidate/STATE_CACHE_COMPLETE.json'
    complete = json.loads(complete_path.read_text())
    exited = json.loads((input_run/'EXIT.json').read_text())
    inherited = input_run/'candidate/OPTIMIZATION_COMPLETE.json'
    if (exited['exit_code'] != 0 or not exited['state_cache_receipt_present']
            or not state['strict_complete_pair_passed'] or not state['state_cache']
            or complete['state_cache_report_sha256'] != complete_export.sha(state_path)
            or complete['inherited_complete_sha256'] != complete_export.sha(inherited)
            or state['inherited_complete_sha256'] != complete_export.sha(inherited)
            or state['context_preparations'] != 1
            or state['materialization_count'] != state['search_evaluations']+2
            or state['final_score_calls'] != 1 or state['final_write_calls'] != 1):
        raise ValueError('actual complete first-GA state-cache gate required')
    scientific = json.loads((input_run/'candidate/optimization.json').read_text())
    manifest = json.loads((frozen_source/'SOURCE_MANIFEST.private.json').read_text())
    resource_path = input_run/'resource_before.private.json'
    baseline_resource_path = baseline_run/'resource_before.private.json'
    resource = json.loads(resource_path.read_text())
    baseline_resource = json.loads(baseline_resource_path.read_text())
    same_host = bool(resource.get('hostname') and baseline_resource.get('hostname')
        and resource['hostname'] == baseline_resource['hostname'])
    if not same_host:
        raise ValueError('same actual host receipt required for this scientific pairing')
    if (state['optimization_report_sha256'] != complete_export.sha(input_run/'candidate/optimization.json')
            or scientific['source_manifest_sha256'] != complete_export.sha(frozen_source/'SOURCE_MANIFEST.private.json')):
        raise ValueError('same runtime source and inherited science report required')
    # 旧完整导出器需要OVERLAY_SHA。另建事后metadata镜像，不回填原freeze。
    # 六原成熟模块和delegate的SHA从实际冻结文件验证，额外cache/tool另报。
    binding = output_directory.parent/('.'+output_directory.name+'-binding-private')
    binding.mkdir(parents=True, exist_ok=False)
    for filename in ('SOURCE_MANIFEST.private.json','protocol.private.json'):
        shutil.copyfile(frozen_source/filename,binding/filename)
    names=('place_surface_intersection_marking','smooth_surface_python','topology_candidate_cycle',
           'topology_candidate_geometry','topology_defect_smooth','topology_defect_mri_match')
    payload={}
    for name in names:
        relative='src/fnit/recon_all/'+name+'.py'
        expected=manifest['expected_module_sha256']['fnit.recon_all.'+name]
        if (complete_export.sha(frozen_source/relative)!=expected
                or scientific['loaded_module_sha256']['fnit.recon_all.'+name]!=expected):
            raise ValueError('actual complete mature module SHA differs: '+name)
        payload[relative]={'sha256':expected}
    delegate='tools/benchmark_recon_topology_genetic_optimization.py'
    payload[delegate]={'sha256':complete_export.sha(frozen_source/delegate)}
    (binding/'OVERLAY_SHA.json').write_text(json.dumps(dict(payload=payload,
        kind='post_execution_binding_metadata_only_no_algorithm_mutation'),indent=2)+'\n')
    receipt = complete_export.export_metrics(input_run=input_run,baseline_run=baseline_run,
        frozen_source=binding,output_directory=output_directory)
    kernel='fnit.recon_all.topology_candidate_state_cache'
    extra=dict(state, cache_kernel_sha256=scientific['loaded_module_sha256'][kernel],
        actual_cycle_sha256=scientific['loaded_module_sha256']['fnit.recon_all.topology_candidate_cycle'],
        state_cache_report_sha256=complete_export.sha(state_path),
        state_cache_complete_sha256=complete_export.sha(complete_path),
        source_manifest_sha256=scientific['source_manifest_sha256'],
        completed_baseline_source_manifest_sha256=manifest['state_cache_ancestry']['completed_baseline_manifest_sha256'],
        minimal_own_transfer_manifest_sha256=manifest['state_cache_ancestry']['transfer_manifest_sha256'],
        actual_host_identity_equal=same_host,
        host_receipt_sha256=complete_export.sha(resource_path),
        baseline_host_receipt_sha256=complete_export.sha(baseline_resource_path),
        software_versions_observed_after_execution={
            'python':sys.version.split()[0],
            **{name:importlib.metadata.version(name)
               for name in ('numpy','numba','torch','nibabel','scipy')}},
        software_version_scope='same declared runtime queried by exporter after execution; not an in-loop precision observation',
        exporter_sha256=complete_export.sha(__file__),
        metadata_mirror='created after execution for compatible export only; original frozen bytes unchanged')
    raw=(json.dumps(extra,indent=2,allow_nan=False)+'\n').encode()
    if [(type(v),v)for v in numeric_leaves(extra)] != [(type(v),v)for v in numeric_leaves(json.loads(raw))]:
        raise RuntimeError('state-cache metric value/type changed in public JSON')
    for forbidden in ('/cpfs','/data/shared/','/home/','FS_LICENSE','PRIVATE KEY'):
        if forbidden in raw.decode():
            raise ValueError('private metadata cannot be published')
    import re
    if re.search(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])',raw.decode()):
        raise ValueError('network addresses cannot be published')
    (output_directory/'state_cache.json').write_bytes(raw)
    import hashlib
    augmented=dict(inherited_exporter_sha256=receipt['exporter_sha256'],
        own_exporter_sha256=complete_export.sha(__file__),
        inherited_export_receipt_sha256=complete_export.sha(output_directory/'EXPORT_RECEIPT.json'),
        safe_additional_payload={'state_cache.json':dict(size=len(raw),sha256=hashlib.sha256(raw).hexdigest())},
        numeric_values_and_types_preserved=True, original_source_and_reports_unchanged=True)
    (output_directory/'STATE_EXPORT_RECEIPT.json').write_text(json.dumps(augmented,indent=2)+'\n')
    return augmented


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-run','baseline-run','frozen-source','output-directory'):
        parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(export_metrics(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':main()
