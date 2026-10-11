"""已完成六评分显式缓存配对的数值白名单；原MRI/状态/日志留服务器。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from export_recon_gca_receipts import numeric_leaves


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_bytes(value):
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    before=numeric_leaves(value);after=numeric_leaves(json.loads(data))
    if [(type(x),x)for x in before]!=[(type(x),x)for x in after]:
        raise RuntimeError('whitelist numeric value/type changed')
    return data,len(before)


def export(*,input_run,frozen_source,output_directory):
    """三项具名Path→完整退出/源码/双方产物门后的安全数值JSON。

    input_run为已完成pair父目录，frozen_source为实际不可变源码，
    output_directory必须不存在。检查EXIT0、父/两child的COMPLETE
    与report/checkpoint SHA、两个实际新核心源码SHA、六评分及全文
    数值/控制比较。只输出数值指标、程序/输入/产物SHA、无私有路径
    的状态和政策。每个选中原数字/布尔/null的值和类型逐项保留。
    不导出MRI、NPZ、命令、源码清单、原日志或地址；不运行GA。
    """
    run,source,out=map(Path,(input_run,frozen_source,output_directory))
    exit_record=json.loads((run/'EXIT.json').read_text())
    process=json.loads((run/'process.private.json').read_text())
    contracts=json.loads((run/'contracts.private.json').read_text())
    report_path=run/'pair/pair.private.json'
    report=json.loads(report_path.read_text())
    complete=json.loads((run/'pair/COMPLETE.json').read_text())
    manifest=json.loads((source/'SOURCE_MANIFEST.private.json').read_text())
    source_sha=sha(source/'SOURCE_MANIFEST.private.json')
    for name,record in manifest['payload'].items():
        path=source/name
        if path.stat().st_size!=record['size']or sha(path)!=record['sha256']:
            raise ValueError('actual frozen source payload changed')
    if (out.exists()or exit_record['exit_code']!=0 or process['exit_code']!=0
            or contracts['exit_code']!=0 or not exit_record['pair_complete']
            or report['status']!='passed_six_score_explicit_face_cache_pair'
            or complete['status']!=report['status']or complete['report_sha256']!=sha(report_path)
            or report['source_manifest_sha256']!=source_sha
            or contracts['source_manifest_sha256']!=source_sha
            or process['source_manifest_sha256']!=source_sha):
        raise ValueError('same complete explicit production stage receipts required')
    if ([row['label']for row in report['records']]!=['disabled','enabled']or
            [row['report']['face_detection_cache']for row in report['records']]!=[False,True]):
        raise ValueError('same disabled/enabled stage order required')
    names=('topology_genetic_search','topology_genetic_repair')
    current={name:sha(source/'src/fnit/recon_all'/str(name+'.py'))for name in names}
    for row in report['records']:
        path=run/'pair'/row['label'];value=json.loads((path/'report.private.json').read_text())
        receipt=json.loads((path/'COMPLETE.json').read_text())
        if (row['report']!=value or value['scores']!=6 or value['search_evaluations']!=5
                or receipt['status']!=value['status']or receipt['report_sha256']!=sha(path/'report.private.json')
                or value['source_manifest_sha256']!=source_sha):
            raise ValueError('same actual six-score child reports required')
        for name,digest in value['checkpoint_sha256'].items():
            if sha(path/name)!=digest:raise ValueError('actual complete child output SHA changed')
        for name,digest in current.items():
            if value['actual_loaded_module_sha256']['fnit.recon_all.'+name]!=digest:
                raise ValueError('actual new production source SHA differs')
    if (not all(report['controls'].values())or
            not all(v['all_fields_byte_equal']and v['complete_file_sha_equal']for v in report['comparisons'].values())):
        raise ValueError('complete numeric/control pair failed')
    selected={key:report[key]for key in ('status','source_manifest_sha256','comparisons','controls',
        'compared_numeric_fields','complete_pair_wall_seconds','production_default_changed',
        'complete_default_GA','whole_recon_all','final_mesh_quality','overall_metric_equivalence',
        'algorithm_reference_inputs')}
    selected['records']=[dict(label=row['label'],fresh_cli_seconds=row['fresh_cli_seconds'],
        **{key:row['report'][key]for key in ('status','face_detection_cache','input_sha256','policy','scores',
        'search_evaluations','stages','nested_search_seconds','random_draws','random_sequence_sha256',
        'cpu_affinity','target_cuda','allocated_bytes','reserved_bytes','self_peak_rss_kib',
        'checkpoint_sha256','complete_worker_seconds_before_report','explicit_production_search_parameter',
        'global_function_replacement')})for row in report['records']]
    selected['source_sha256']=dict(current,
        face_detection_helper=manifest['expected_module_sha256']['fnit.recon_all.topology_face_detection_cache'],
        stage_tool=sha(source/'tools/benchmark_recon_topology_genetic_face_cache_wiring.py'),
        focused_test=sha(source/'tests/recon_all/test_topology_genetic_face_cache_wiring.py'))
    selected['execution']={key:process[key]for key in ('exit_code','fresh_pair_cli_seconds','child_peak_rss_kib',
        'sampler_interval_seconds','cpu_affinity','threads','target_cuda','allocated_bytes','reserved_bytes')}
    selected['focused_contracts']={key:contracts[key]for key in ('exit_code','wall_seconds','focused_contract_count')}
    data,count=safe_bytes(selected)
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    (out/'metrics.json').write_bytes(data)
    receipt=dict(status='complete_safe_metric_whitelist',actual_pair_report_sha256=sha(report_path),
        actual_complete_sha256=sha(run/'pair/COMPLETE.json'),actual_process_sha256=sha(run/'process.private.json'),
        exporter_sha256=sha(__file__),numeric_boolean_null_leaves_preserved=count,
        numeric_values_and_types_preserved=True,source_manifest_sha256=source_sha,
        payload=dict(metrics=dict(sha256=sha(out/'metrics.json'),size=len(data))))
    (out/'EXPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-run','frozen-source','output-directory'):
        parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(export(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':
    main()
