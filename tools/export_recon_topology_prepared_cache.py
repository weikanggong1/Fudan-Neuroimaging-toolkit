"""不可变拓扑基底缓存：只公开指标和哈希白名单，保留原始私有收据。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from export_recon_gca_receipts import numeric_leaves


def export_metrics(*, input_run, reference_receipt, output_directory):
    """完整ABBA目录/参考收据→新公开指标目录；全部参数必填。

    不读取MRI/NPZ，不发布日志、命令、主机、私有路径或全源码清单。
    标量时间、完整差异指标、输入/程序哈希和实际范围逐项保留；解析
    后数值的类型和值须不变。已有输出FileExistsError，不改变原记录。
    """
    input_run,output_directory=Path(input_run),Path(output_directory)
    if output_directory.exists():raise FileExistsError(output_directory)
    source=input_run/'candidate/cache.report.private.json';raw=source.read_bytes();report=json.loads(raw)
    process_raw=(input_run/'process.private.json').read_bytes();process=json.loads(process_raw)
    ref_raw=Path(reference_receipt).read_bytes();ref=json.loads(ref_raw)
    metrics={key:report[key] for key in ('status','scope','declared_case','threads','target_cuda',
        'cache_preparation_seconds','immutable_base_state_sha256','cache_is_independent_immutable_snapshot',
        'cache_identity_and_metadata_validation','records','ordinary_warm_api_median_seconds',
        'prepared_warm_api_median_seconds','warm_complete_candidate_speedup','full_diagnostic_before_report_seconds',
        'existing_post_reference_gate','reference_after_used_as_input','initial_inputs_sha256',
        'comparison_reference_sha256','complete_base_producer','complete_ga','final_mesh_quality','overall_metric_equivalence')}
    metrics['execution']={key:process[key] for key in ('exit_code','complete_fresh_cli_seconds','cpu_affinity','threads',
        'target_cuda','gpu_memory','child_peak_rss_kib')}
    metrics['binding']=dict(candidate_code_sha256=report['source_sha256'],reference_program_sha256=[row['program_sha256'] for row in ref['references']],
        source_report_sha256=hashlib.sha256(raw).hexdigest(),source_process_sha256=hashlib.sha256(process_raw).hexdigest(),
        source_reference_receipt_sha256=hashlib.sha256(ref_raw).hexdigest(),
        existing_frozen_base_receipt_sha256=process['code_receipt']['existing_frozen_base_receipt_sha256'])
    encoded=(json.dumps(metrics,indent=2,allow_nan=False)+'\n').encode()
    a,b=numeric_leaves(metrics),numeric_leaves(json.loads(encoded))
    if [(type(x),x) for x in a]!=[(type(x),x) for x in b]:raise RuntimeError('public export changed numeric type/value')
    output_directory.mkdir(parents=True);(output_directory/'report.json').write_bytes(encoded)
    manifest=dict(scope='metrics and SHA whitelist only; no original MRI NPZ native program private logs or source-file list',
        report_sha256=hashlib.sha256(encoded).hexdigest(),numeric_leaves_retained=len(a),numeric_values_and_types_preserved=True,
        source_report_sha256=hashlib.sha256(raw).hexdigest())
    (output_directory/'EXPORT_RECEIPT.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-run','reference-receipt','output-directory'):parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(export_metrics(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':main()
