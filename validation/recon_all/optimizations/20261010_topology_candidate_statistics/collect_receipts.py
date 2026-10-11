"""服务器就地去敏，导出真实三次候选统计回归的收据、复现脚本和图。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def leaves(value, path=()):
    if isinstance(value, dict):
        return [leaf for key,item in value.items() for leaf in leaves(item,path+(key,))]
    if isinstance(value, list):
        return [leaf for key,item in enumerate(value) for leaf in leaves(item,path+(key,))]
    return [(path,value)] if value is None or isinstance(value,(bool,int,float)) else []


PUBLIC_STAGE_FIELDS = (
    'status', 'strict_statistics_passed', 'candidate_repeat_exact',
    'native_captured_fields_repeat_exact', 'comparisons', 'candidate_repeat',
    'native_repeat', 'full_vertices', 'complete_defect_entries', 'discarded_entries',
    'updated_entries', 'complete_edges', 'seconds', 'module_sha256',
    'benchmark_sha256', 'source_inputs_sha256', 'candidate_npz_sha256',
    'figure_sha256', 'cpu_threads', 'cpu_affinity', 'python', 'numpy',
    'gpu_used', 'process_gpu_memory_bytes', 'native_component_seconds',
    'original_component_speedup', 'scope', 'overall_equivalence',
)


def collect(*, fnit_root, output):
    """正式exit0 → 公开指标及图，私有日志/路径/工作区清单留服务器。

    fnit_root为已授权服务器根，output必须不存在。公开字段使用固定
    白名单，含完整比较、重复性、耗时、线程、代码/输入/程序哈希；不含
    主机名、observer全文、日志或source文件清单。逐已选字段验证数值、
    bool与null原样保留，PNG绑定实际结果SHA。失败或hash不符抛异常。
    """
    if output.exists():
        raise FileExistsError('new public output required')
    task='topology_candidate_statistics_20261010_v1'
    final=fnit_root/'runs'/task
    complete=json.loads((final/'COMPLETE.private.json').read_text())
    if ((final/'driver.exitcode').read_text().strip()!='0'
            or not complete['strict_three_call_gate_passed']
            or len(complete['reports'])!=3):
        raise RuntimeError('complete three-call real regression required')
    gates=['strict_statistics_passed','candidate_repeat_exact','native_captured_fields_repeat_exact']
    if not all(all(report[key] for key in gates) for report in complete['reports']):
        raise RuntimeError('all complete statistics/repeat gates required')
    output.mkdir(parents=True)
    entries=[]
    public_calls=[]
    for call,report in enumerate(complete['reports']):
        selected={key:report[key] for key in PUBLIC_STAGE_FIELDS}
        for key in PUBLIC_STAGE_FIELDS:
            assert leaves(report[key])==leaves(selected[key]),'selected metric changed'
        selected.update({key:report[key] for key in [
            'reference_call_index','maximum_use_count_before',
            'maximum_use_count_after','previously_used_entries_before']})
        raw=(final/f'regression{call}/report.private.json').read_bytes()
        published=(json.dumps(selected,indent=2,allow_nan=False)+'\n').encode()
        name=f'call{call}/metrics.json';p=output/name;p.parent.mkdir(parents=True);p.write_bytes(published)
        entries.append(dict(path=name,raw_sha256=sha(raw),published_sha256=sha(published),
                            raw_bytes=len(raw),published_bytes=len(published),
                            selected_numeric_boolean_null_leaves_preserved=True))
        public_calls.append(selected)
        image=(final/f'regression{call}/statistics_error.png').read_bytes()
        if not image.startswith(b'\x89PNG\r\n\x1a\n') or sha(image)!=report['figure_sha256']:
            raise ValueError('actual figure hash differs')
        name=f'call{call}/statistics_error.png';(output/name).write_bytes(image)
        entries.append(dict(path=name,raw_sha256=sha(image),published_sha256=sha(image),
                            raw_bytes=len(image),published_bytes=len(image),
                            selected_numeric_boolean_null_leaves_preserved=True))
    selected={key:complete[key] for key in [
        'status','actual_snapshot_git_head','source_manifest_sha256','contracts',
        'reference_build_sha256','cuda_context_initialized_before',
        'cuda_context_initialized_after','precision_before','precision_after',
        'driver_seconds_after_imports_including_contracts_and_three_regression_io',
        'scope','overall_equivalence','strict_three_call_gate_passed']}
    selected['reference_build']={key:value for key,value in complete['reference_build'].items()
                                 if key.endswith('_sha256') or key in ['scope','status','seconds','format','public_build_field_values_exact']}
    selected['calls']=public_calls
    selected['excluded_private_metadata']='host names, paths, logs, observer receipt contents and workspace file list retained in private validation directory'
    raw=(final/'COMPLETE.private.json').read_bytes()
    published=(json.dumps(selected,indent=2,allow_nan=False)+'\n').encode()
    (output/'complete_metrics.json').write_bytes(published)
    entries.append(dict(path='complete_metrics.json',raw_sha256=sha(raw),published_sha256=sha(published),
                        raw_bytes=len(raw),published_bytes=len(published),
                        selected_numeric_boolean_null_leaves_preserved=True))
    manifest=dict(scope='complete metric fields for three frozen candidate statistics calls; not GA or whole reconstruction',
                  reports=entries)
    (output/'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fnit-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps({'reports':len(collect(fnit_root=args.fnit_root,output=args.output)['reports'])}))
