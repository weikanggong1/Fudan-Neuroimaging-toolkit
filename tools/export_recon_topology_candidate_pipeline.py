"""候选链公开指标白名单导出，私有日志/路径/完整源码清单原样留在服务器。"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from export_recon_gca_receipts import numeric_leaves


def export_metrics(*, input_run, reference_receipt, output_directory):
    """输入独占运行目录与参考收据→新公开JSON，数值/类型不变。

    input_run含candidate/report.private.json、process.private.json及可选
    profile.metrics.json；reference_receipt仅读取program SHA与repeat bool。
    全部参数为必填路径；output_directory必须不存在。只拷贝列举的
    标量/差异指标/计时/输入哈希，排除command、host、日志、路径和全
    文件清单。不读取或发布MRI/NPZ。保留原JSON SHA；通过序列化前后
    数值/布尔/null类型和值逐项检查，失败抛异常且不改原收据。
    """
    input_run,output_directory=Path(input_run),Path(output_directory)
    if output_directory.exists():raise FileExistsError(output_directory)
    path=input_run/'candidate/report.private.json';raw=path.read_bytes();report=json.loads(raw)
    process_raw=(input_run/'process.private.json').read_bytes();process=json.loads(process_raw)
    reference_raw=Path(reference_receipt).read_bytes();reference=json.loads(reference_raw)
    names=['status','scope','initial_inputs_sha256','comparison_reference_sha256','loading_seconds',
        'raw_calibration_seconds','raw_calibration_subtimers','results','checks','repeat','threads',
        'source_calloc_face_area','frozen_initial_attribute_names','frozen_global_priors',
        'reference_after_consumed_to_generate','complete_base_producer','complete_ga','final_mesh_quality',
        'overall_metric_equivalence','pipeline_wall_before_report_seconds']
    metrics={key:report[key] for key in names}
    process_names=['exit_code','complete_fresh_cli_seconds','cpu_affinity','threads','target_cuda','gpu_memory','child_peak_rss_kib']
    metrics['execution']={key:process[key] for key in process_names}
    metrics['binding']=dict(base_git_commit=process['code_receipt']['base_git_commit'],
        base_archive_sha256=process['code_receipt']['base_archive_sha256'],
        pipeline_module_sha256=process['code_receipt']['overlay_sha256']['src/fnit/recon_all/topology_candidate_pipeline.py'],
        benchmark_tool_sha256=process['code_receipt']['overlay_sha256']['tools/benchmark_recon_topology_candidate_pipeline.py'],
        reference_program_sha256=[row['program_sha256'] for row in reference['references']],
        source_report_sha256=hashlib.sha256(raw).hexdigest(),source_process_receipt_sha256=hashlib.sha256(process_raw).hexdigest(),
        source_reference_receipt_sha256=hashlib.sha256(reference_raw).hexdigest())
    metrics['declared_case']='public ds000114 sub-07 ses-test LH; FNIT 79a41cdd upstream'
    metrics['validation_layer']='frozen initial attributes/global priors; continuous one-candidate algorithm; no original-T1 whole run'
    profile_path=input_run/'candidate/profile.metrics.json'
    if profile_path.exists():
        metrics['profile']=json.loads(profile_path.read_text())
    if process['exit_code'] and all(
        value['byte_equal'] for check in report['checks'] for key,value in check.items() if key!='sampling'):
        metrics['failure_classification']='checker_schema_error: v1 compared source normals columns 3:6 with fx/fy/fz; correct columns 6:9'
    encoded=(json.dumps(metrics,indent=2,allow_nan=False)+'\n').encode();parsed=json.loads(encoded)
    before,after=numeric_leaves(metrics),numeric_leaves(parsed)
    if [(type(x),x) for x in before]!=[(type(x),x) for x in after]:raise RuntimeError('export changed numeric value/type')
    output_directory.mkdir(parents=True);(output_directory/'report.json').write_bytes(encoded)
    manifest=dict(scope='explicit public metrics whitelist only; original receipts remain private',numeric_leaves_retained=len(before),
        numeric_values_and_types_preserved=True,report_sha256=hashlib.sha256(encoded).hexdigest(),
        source_report_sha256=hashlib.sha256(raw).hexdigest())
    (output_directory/'EXPORT_RECEIPT.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input-run','reference-receipt','output-directory']:p.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(export_metrics(**vars(p.parse_args())),indent=2))


if __name__=='__main__':main()
