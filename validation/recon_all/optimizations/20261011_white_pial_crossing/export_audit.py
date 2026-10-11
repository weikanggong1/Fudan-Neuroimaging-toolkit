"""独立裁剪审查的固定schema数值导出，拒绝未知字段和私有文本。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def sha(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('SHA-256 required')
    return value


def numeric(value,integer=False):
    if type(value) not in ((int,) if integer else (int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('typed nonnegative finite number required')
    return value


def exact(value,keys):
    if not isinstance(value,dict) or set(value)!=set(keys):raise ValueError('unexpected schema fields')


def export(*, report_path: Path, timer_path: Path, output: Path):
    """从本审查的实际原始JSON导出数值/哈希，output必须不存在。

    两输入为只读审查与冷CLI计时报告；输出含全部计数、精确分类、
    输入/程序/原报告SHA和CPU RSS/墙钟，无路径、坐标或主机文本。
    原始失败可以按固定枚举导出；类型/枚举/计数不符先失败，不写文件。
    返回dict，不执行任何MRI/优化器/GPU或原软件程序。
    """
    if output.exists():raise FileExistsError(output)
    raw=report_path.read_bytes();d=json.loads(raw)
    ints={'vertices','faces','global_proper_pairs','global_proper_sample_requested',
          'global_proper_sample_actual','affected_faces','independent_local_bbox_pairs',
          'audited_pairs','mismatch_pairs','peak_process_cpu_rss_bytes'}
    floats={'independent_exact_seconds','wall_seconds_before_report_write'}
    hashes={'script_sha256','helper_sha256','terminal_sha256','mismatch_pair_set_sha256'}
    bools={'inputs_unchanged','complete_local_proper_set_equal','complete_local_contact_set_equal',
           'gpu_computation','original_stage_program_called'}
    other={'schema_version','execution_status','scope','input_sha256','exact_classes',
           'gpu_memory_bytes','overall_metric_equivalence'}
    exact(d,ints|floats|hashes|bools|other)
    if type(d['schema_version']) is not int or d['schema_version']!=1:raise ValueError('schema version required')
    if d['scope']!='frozen own pial; deterministic global sample plus complete affected-face bbox cross-product':raise ValueError('unknown audit scope')
    for key in bools:
        if type(d[key]) is not bool:raise ValueError('typed boolean required')
    if d['inputs_unchanged'] is not True or d['gpu_computation'] is not False or d['original_stage_program_called'] is not False:
        raise ValueError('unexpected inputs or runtime route')
    if d['gpu_memory_bytes'] is not None or d['overall_metric_equivalence']!='not_assessed':raise ValueError('GPU/whole not assessed')
    values={k:numeric(d[k],True) for k in ints}|{k:numeric(d[k]) for k in floats}|{k:sha(d[k]) for k in hashes}
    exact(d['exact_classes'],{'none','proper','edge_point_contact','coplanar_contact'})
    classes={k:numeric(v,True) for k,v in d['exact_classes'].items()}
    if sum(classes.values())!=values['audited_pairs']:raise ValueError('class count mismatch')
    if values['global_proper_sample_actual']>values['global_proper_sample_requested'] or values['audited_pairs']<values['independent_local_bbox_pairs']:
        raise ValueError('sample scope mismatch')
    success=(d['mismatch_pairs']==0 and d['complete_local_proper_set_equal'] and d['complete_local_contact_set_equal'])
    status='pass_independent_exact_audit' if success else 'failed_independent_exact_audit'
    if d['execution_status']!=status:raise ValueError('status contradicts numerical gates')
    exact(d['input_sha256'],{'surf/lh.white','surf/autodet.gw.stats.lh.dat','label/lh.cortex.label',
           'label/lh.cortex+hipamyg.label','mri/brain.finalsurfs.mgz','mri/wm.mgz','mri/aseg.presurf.mgz'})
    inputs={k:sha(v) for k,v in d['input_sha256'].items()}
    timer_raw=timer_path.read_bytes();t=json.loads(timer_raw)
    exact(t,{'schema_version','scope','command_sha256','timer_script_sha256','gpu_memory_bytes',
       'simultaneous_parent_child_peak_cpu_memory_bytes','execution_status','child_exit_code','cold_cli_wall_seconds','max_child_cpu_rss_bytes'})
    if type(t['schema_version']) is not int or t['schema_version']!=1 or t['scope']!='single frozen Python benchmark cold CLI; not end-to-end recon-all':
        raise ValueError('unknown timer schema/scope')
    expected_exit=0 if success else 1
    if type(t['child_exit_code']) is not int or t['child_exit_code']!=expected_exit or t['execution_status']!=('complete' if success else 'failed'):
        raise ValueError('timer status contradicts audit gate')
    if t['gpu_memory_bytes'] is not None or t['simultaneous_parent_child_peak_cpu_memory_bytes'] is not None:
        raise ValueError('unmeasured resources must stay null')
    result={'schema_version':1,'execution_status':status,'scope':d['scope'],
            'raw_report_sha256':hashlib.sha256(raw).hexdigest(),'input_sha256':inputs,
            **values,'exact_classes':classes,**{k:d[k] for k in bools},
            'gpu_memory_bytes':None,'overall_metric_equivalence':'not_assessed',
            'cold_cli':{'raw_timer_sha256':hashlib.sha256(timer_raw).hexdigest(),
                'child_exit_code':expected_exit,'cold_cli_wall_seconds':numeric(t['cold_cli_wall_seconds']),
                'max_child_cpu_rss_bytes':numeric(t['max_child_cpu_rss_bytes'],True),
                'command_sha256':sha(t['command_sha256']),'timer_script_sha256':sha(t['timer_script_sha256']),
                'gpu_memory_bytes':None,'simultaneous_parent_child_peak_cpu_memory_bytes':None}}
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('report-path','timer-path','output'):parser.add_argument('--'+name,type=Path,required=True)
    r=export(**vars(parser.parse_args()));print(json.dumps({'status':r['execution_status'],'raw_report_sha256':r['raw_report_sha256']}))
