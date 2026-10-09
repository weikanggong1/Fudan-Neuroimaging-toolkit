"""用标准库从两例完成收据重建交叉归因表，不改变真实数字或门槛。"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def summarize_cross_results(*, reports_directory: Path, output_directory: Path) -> None:
    """读取sub06/sub07.report.json与.controls.json，写三份CSV和summary.json。

    reports_directory是公开脱敏收据目录，output_directory须不存在。LTA
    元素无量纲/图谱体素；MRI差为存储强度单位，Dice无单位，wall为秒。
    不完整报告、缺指标或已有输出拒绝；不做整体等效或速度通过判断。
    """
    if output_directory.exists():raise FileExistsError(output_directory)
    rows=[];frames=[];masked=[];summary={}
    for subject in ('sub06','sub07'):
        report=json.loads((reports_directory/(subject+'.report.json')).read_text())
        controls=json.loads((reports_directory/(subject+'.controls.json')).read_text())
        if report['status']!='complete' or controls['status']!='diagnostic_complete':raise ValueError('two completed diagnostics required')
        summary[subject]={'scope':report['scope'],'code_version':report['code_version'],
            'original_T1_sha256':controls['original_T1_sha256'],
            'native_binary_sha256':report['native_binary_sha256'],
            'diagnostic_wall_seconds':report['diagnostic_wall_seconds'],
            'mask_predicates':report['mask_predicates'],'source_sha256':report['source_sha256'],
            'registration':{},'torch_vs_original':report['torch_vs_original']}
        for backend,groups in report['backends'].items():
            summary[subject]['registration'][backend]={}
            for group,item in groups.items():
                summary[subject]['registration'][backend][group]={
                    'register_api_wall_seconds':item['register_api_wall_seconds_including_process_exit'],
                    'matrix_vs_11':item['matrix_vs_11'], 'lta_matrix':item['lta_matrix']}
                for scope in ('same-input-norm','fixed-11-lta-norm','lta-only-norm'):
                    if scope+'_vs_11' not in item:continue
                    for volume,comparison in item[scope+'_vs_11'].items():
                        values=comparison['voxels']
                        rows.append({'subject':subject,'backend':backend,'group':group,'scope':scope,'volume':volume,
                            'different_elements':values['elements']-values['exact'],'elements':values['elements'],
                            'max_abs_error':values['max_abs'],'p99_abs_error':values['p99_abs'],
                            'dtype_reference':comparison['dtype'][0],'dtype_candidate':comparison['dtype'][1],
                            'affine_pass':comparison['affine']['pass'],'header_exact':comparison['header_exact']})
        for backend,groups in controls['controls'].items():
            for group,scopes in groups.items():
                for scope,passes in scopes.items():
                    for item in passes:
                        frames.append({'subject':subject,'backend':backend,'group':group,'scope':scope,
                            **{name:value for name,value in item.items() if name!='per_label_dice'},
                            'per_label_dice_json':json.dumps(item['per_label_dice'],sort_keys=True,separators=(',',':'))})
        for stage,groups in controls['effective_masked_inputs'].items():
            for group,item in groups.items():masked.append({'subject':subject,'stage':stage,'group':group,**item})
    output_directory.mkdir(parents=True)
    for filename,values in (('norm_comparisons.csv',rows),('control_frames.csv',frames),('masked_input_comparisons.csv',masked)):
        with (output_directory/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
    (output_directory/'summary.json').write_text(json.dumps({'status':'diagnostic_complete',
        'overall_metric_equivalence':'not_assessed','subjects':summary},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-directory',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    summarize_cross_results(**vars(parser.parse_args()))
