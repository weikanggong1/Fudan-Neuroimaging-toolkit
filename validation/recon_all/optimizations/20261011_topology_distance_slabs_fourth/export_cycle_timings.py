"""严格候选配对完成后，另导出固定九项 API 子时钟；不重跑候选。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from diagnose_distance_slabs import ROLES, RUNS, sha, write
from export_distance_slabs import digest, number

CYCLE_FIELDS=('candidate_complete','statistics_update','selective_full_restore','complete_cycle')
CANDIDATE_FIELDS=('geometry_complete_api','displacement_complete_api','sff_complete_api',
                  'complete_prepared_five_api','complete_candidate_api_including_validation')


def export(*,input_run: Path,output_directory: Path):
    """私有完整配对目录→有限数字/SHA报告和绑定收据。

    input_run为diagnose_distance_slabs已完成的私有新目录，须包含父报告、
    COMPLETE/EXIT及五个实际worker报告；output_directory须不存在。
    返回固定公开dict，只含预先列出的cycle/candidate秒数、受控组/重复名、
    源码/协议/报告SHA，不选择路径/命令/日志/任意文本或影像。九项时钟
    是嵌套API，不与CLI重复求和。失败抛ValueError/FileExistsError；不执行
    MRI、不改原报告/线程/精度。原软件内部步骤无独立CLI。
    """
    root,out=Path(input_run),Path(output_directory)
    if out.exists():raise FileExistsError('fresh typed cycle output required')
    report=json.loads((root/'report.private.json').read_text())
    complete=json.loads((root/'COMPLETE.json').read_text())
    if (report['status']!='complete_fixed_fourth_distance_backend_pair'
            or not report['all_complete_fields_and_controls_exact']
            or complete['report_sha256']!=sha(root/'report.private.json')
            or json.loads((root/'EXIT.json').read_text())!={'exit_code':0,'strict_pair_passed':True}
            or report['source_rng_draws']!=0 or report['production_default_changed']
            or not report['global_function_replacement']
            or report['whole_search'] or report['whole_recon_all']):
        raise ValueError('complete strict diagnostic pair required')
    if [r['worker']['role']for r in report['workers']]!=list(ROLES):
        raise ValueError('fixed completed worker set required')
    public=dict(scope='fixed fourth first candidate; nested API timings only',
        all_complete_fields_and_controls_exact=True,whole_recon_all=False,
        source_manifest_sha256=digest(report['source_manifest_sha256']),
        protocol_sha256=digest(report['protocol_sha256']),
        private_pair_report_sha256=sha(root/'report.private.json'),workers=[])
    for item,role in zip(report['workers'],ROLES):
        folder=root/role
        worker=json.loads((folder/'worker.private.json').read_text())
        inherited=json.loads((folder/'candidate/report.private.json').read_text())
        if (worker!=item['worker'] or inherited!=item['inherited']
                or worker['inherited_report_sha256']!=sha(folder/'candidate/report.private.json')
                or [r['run']for r in inherited['records']]!=list(RUNS)):
            raise ValueError('actual complete timed reports must remain bound')
        row=dict(role=role,inherited_report_sha256=sha(folder/'candidate/report.private.json'),records=[])
        for record,run in zip(inherited['records'],RUNS):
            a,b=record['cycle_seconds'],record['candidate_seconds']
            if set(a)!=set(CYCLE_FIELDS) or set(b)!=set(CANDIDATE_FIELDS):
                raise ValueError('predeclared complete API timer fields required')
            row['records'].append(dict(run=run,
                cycle_seconds={name:number(a[name])for name in CYCLE_FIELDS},
                candidate_seconds={name:number(b[name])for name in CANDIDATE_FIELDS}))
        public['workers'].append(row)
    out.mkdir(parents=True,mode=0o700)
    write(out/'CYCLE_TIMERS.json',public)
    write(out/'CYCLE_TIMERS_EXPORT_RECEIPT.json',dict(
        exporter_sha256=sha(__file__),source_pair_report_sha256=sha(root/'report.private.json'),
        public_metrics_sha256=sha(out/'CYCLE_TIMERS.json'),worker_count=5,
        repeated_records=15,timers_per_record=9,
        paths_commands_logs_mri_selected=False,arbitrary_text_selected=False))
    return public


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    arguments=parser.parse_args()
    export(input_run=arguments.input_run,output_directory=arguments.output_directory)


if __name__=='__main__':main()
