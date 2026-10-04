"""已完成FNIT与official单对诊断；默认baseline，启动与精度候选各用独立角色。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from evaluated_role_bindings import evaluation_spec, verify_startup_binding, verify_admitted_candidate_binding


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true',
                        help='仅完整核验执行绑定，输出必须为新目录，不执行数值比较')
    args = parser.parse_args()
    c = read(args.config)
    evaluation = evaluation_spec(c)
    evaluated_role, prefix = evaluation['role'], evaluation['prefix']
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '4'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONPATH'] = str(Path(c['code_root'])/'src')
    sys.path.insert(0, os.environ['PYTHONPATH'])
    spec = importlib.util.spec_from_file_location('existing_whole', c['whole_driver'])
    w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    out = Path(c['output'])
    if args.verify_only and out.exists():
        raise FileExistsError('verify-only requires a new output directory; original results must remain unchanged')
    out.mkdir(parents=True, exist_ok=not args.verify_only)
    checkpoint = out/'checkpoint.json'
    state = read(checkpoint) if checkpoint.exists() else {'schema':'fnit-recon-pair-evaluation-v1',
        'case':c['case'], 'pair':prefix, 'reference_role':'official', 'evaluated_role':evaluated_role,
        'candidate_status':evaluation['candidate_status'], 'overall_metric_equivalence':'not_assessed',
        'config_sha256':w.digest(args.config), 'script_sha256':w.digest(Path(__file__)), 'phases':{},
        'comparison_time_scope':'comparison and figures only; excluded from whole execution timing'}
    if evaluated_role != 'baseline':
        helper_sha = w.digest(Path(__file__).with_name('evaluated_role_bindings.py'))
        if checkpoint.exists() and state.get('role_helper_sha256') != helper_sha:
            raise ValueError('cannot resume checkpoint with changed role/binding helper')
        state['role_helper_sha256'] = helper_sha
    if state['config_sha256'] != w.digest(args.config) or state['script_sha256'] != w.digest(Path(__file__)):
        raise ValueError('cannot resume checkpoint with changed config/script')
    def phase(name, operation):
        if (state['phases'].get(name,{}).get('status') == 'complete'
                and not (name == 'verify_binding' and evaluated_role != 'baseline')): return
        state.update(status='running', current_phase=name, pid=os.getpid(), host=socket.gethostname())
        state['phases'][name]={'status':'waiting_for_lock'}; w.write(checkpoint,state)
        wait=time.perf_counter()
        with w.common_lock(Path(c['lock'])):
            acquired=time.perf_counter()
            state['phases'][name].update(status='running',lock_wait_seconds=acquired-wait)
            w.write(checkpoint,state)
            try:
                operation()
                state['phases'][name].update(status='complete',seconds=time.perf_counter()-acquired)
            except Exception as error:
                state['phases'][name].update(status='failed',seconds=time.perf_counter()-acquired,error=repr(error))
                state.update(status='failed');w.write(checkpoint,state);raise
            w.write(checkpoint,state)
    b=read(c[evaluation['config_key']]); o=read(c['official_config'])
    baseline, official = Path(b['output']), Path(o['output'])
    ref, got=official,baseline
    scripts=Path(c['scripts_dir'])
    tools={name:w.tool(name,scripts) for name in ('compare_complete_subject','compare_surface_chain','compare_region_stats')}
    surface=tools['compare_surface_chain']; region=tools['compare_region_stats']
    commands=[]
    def verify():
        cohort=read(c['cohort_manifest']); case=next(r for r in cohort['cases'] if r['id']==c['case'])
        input_sha256=w.digest(Path(b['input']))
        if b['input'] != o['input'] or input_sha256 != case['sha256'] or input_sha256 != o['input_sha256']:
            raise ValueError('input SHA differs')
        bindings={}
        for role,config in ((evaluated_role,b),('official',o)):
            d=Path(config['diagnostic_root']); completion=read(d/'completion.json'); launch=read(d/'launch.json')
            if completion['execution_status']!='complete' or completion['exit_code']!=0: raise ValueError(role+' incomplete')
            if launch['host'] != socket.gethostname() or config['threads']!=4 or config['gpu_uuid']!=c['gpu_uuid']:
                raise ValueError('same host/GPU/thread configuration differs')
            if launch['config_sha256']!=w.digest(Path(c[evaluation['config_key'] if role==evaluated_role else 'official_config'])): raise ValueError('config/launch differs')
            bindings[role]={'subject':config['output'],'completion':completion,'completion_sha256':w.digest(d/'completion.json'),
                'launch_sha256':w.digest(d/'launch.json'),'config_sha256':w.digest(Path(c[evaluation['config_key'] if role==evaluated_role else 'official_config']))}
        bc=bindings[evaluated_role]['completion']; oc=bindings['official']['completion']
        if b['code_commit']!=c[evaluation['commit_key']] or bc['code_commit']!=b['code_commit'] or bc['source_archive_sha256']!=b['source_archive_sha256']:
            raise ValueError(evaluated_role+' source binding differs')
        if w.digest(Path(c['source_archive']))!=b['source_archive_sha256']: raise ValueError('archive differs')
        launch=read(Path(b['diagnostic_root'])/'launch.json')
        if w.digest(Path(b['code_root'])/'src/fnit/recon_all/native_free.py')!=launch['candidate_native_free_sha256']:
            raise ValueError(evaluated_role+' native_free differs')
        if bc['output_validation']['expected']!=138 or bc['output_validation']['present']!=138: raise ValueError(evaluated_role+' outputs incomplete')
        if oc['code_version']!='FreeSurfer 8.2.0 d932c45': raise ValueError('official version differs')
        pm=Path(o['program_manifest']); manifest=read(pm)
        if w.digest(pm)!=oc['program_manifest_sha256']: raise ValueError('official program manifest differs')
        for path,program_sha256 in manifest['programs'].items():
            if w.digest(Path(path))!=program_sha256: raise ValueError('official program changed: '+path)
        for path,row in manifest['resources'].items():
            if w.digest(Path(path))!=row['sha256']: raise ValueError('official resource changed: '+path)
        for name,key in (('recon-all.log','log_sha256'),('recon-all.done','done_sha256')):
            if w.digest(official/'scripts'/name)!=oc[key]: raise ValueError('official '+name+' differs')
        if evaluated_role == 'baseline':
            resources=read(c['baseline_resources'])
            if resources['mismatches'] or resources['code_commit']!=b['code_commit']: raise ValueError('baseline resource freeze failed')
            for category in ('weights','assets','binaries'):
                for name,row in resources[category].items():
                    if w.digest(Path(row['resolved_path']))!=row['sha256']: raise ValueError('baseline resource changed: '+name)
        elif evaluated_role == 'startup_only_candidate':
            bindings['startup_resource_verification'] = verify_startup_binding(c, b)
        else:
            bindings['precision_resource_verification'] = verify_admitted_candidate_binding(c, b)
        bindings.update(input_sha256=input_sha256,cohort_manifest_sha256=w.digest(Path(c['cohort_manifest'])),
            **{evaluated_role+'_resources_sha256':w.digest(Path(c[evaluation['resources_key']]))},
            official_program_manifest_sha256=w.digest(pm),
            comparison_driver_sha256=w.digest(Path(c['whole_driver'])), comparator_sha256={p.name:w.digest(p) for p in scripts.glob('*.py')},
            same_host='gpucw1',total_threads=4,declared_gpu_uuid=c['gpu_uuid'],
            hardware_scope='same launch host and declared GPU UUID; complete monitor receipts retained separately',
            **{evaluated_role+'_code_commit':b['code_commit']})
        w.write(out/'execution_binding.json',bindings)
    # 显式只初始化一次；后续分段共享总线程4。
    w.configure_runtime()
    phase('verify_binding',verify)
    if args.verify_only:
        state.update(status='binding_verified_only', current_phase=None, verify_only=True,
                     numerical_comparison_executed=False,
                     verification_scope='execution/source/resource binding only; no 138-item or numerical comparison')
        w.write(checkpoint,state)
        return
    def strict_geometry():
        strict=tools['compare_complete_subject'].compare(ref,got)
        if strict['checked']!=138:raise ValueError('strict item count differs')
        w.write(out/(f'strict_{prefix}.json'),strict)
        w.write(out/(f'geometry_{prefix}.json'),w.geometry(ref,got,surface))
    phase('strict_geometry',strict_geometry)
    def command(script,extra):
        w.execute([c['python'],str(scripts/(script+'.py')),*extra],out/'commands.log',commands)
    common=['--reference',str(ref),'--candidate',str(got),'--code-commit',b['code_commit']]
    phase('region_stats',lambda:command('compare_region_stats',common+['--output',str(out/(f'region_{prefix}.json'))]))
    phase('label_dice',lambda:command('compare_parcellation_dice',common+['--label-table',c['label_table'],'--output',str(out/(f'dice_{prefix}.json'))]))
    def all_regions():
        report={}
        for atlas in ('aparc','aparc.DKTatlas','aparc.a2009s','aparc.pial'):
            a,z={},{}
            for hemi in ('lh','rh'):
                pa,pz=[root/'stats'/f'{hemi}.{atlas}.stats' for root in (ref,got)]
                ar,_=region._rows(pa);zr,_=region._rows(pz)
                a.update({hemi+'/'+k:v for k,v in ar.items()});z.update({hemi+'/'+k:v for k,v in zr.items()})
            report[atlas]={field:region._metric(a,z,field) for field in ('SurfArea','GrayVol','ThickAvg','MeanCurv')}
        w.write(out/(f'all_regions_{prefix}.json'),report)
    phase('all_partition_stats',all_regions)
    def local():
        gate=read(out/(f'geometry_{prefix}.json'))
        w.write(out/(f'local_{prefix}.json'),w.local_differences(ref,got,gate))
    phase('local_anomalies',local)
    def no_th3():
        derived={str(root):w.no_th3_subject(root) for root in (ref,got)}
        w.write(out/'no_th3_inputs.json',derived)
        report={atlas:region._metric(derived[str(ref)]['atlases'][atlas],derived[str(got)]['atlases'][atlas],'GrayVolNoTH3') for atlas in derived[str(ref)]['atlases']}
        w.write(out/(f'no_th3_{prefix}.json'),{'unit':'mm3','atlases':report,'scope':'same established no-th3 on each version own outputs',
            'reference_stats_modes':w.stats_modes(ref),evaluated_role+'_stats_modes':w.stats_modes(got)})
    phase('no_th3',no_th3)
    for role,root in ((evaluated_role,got),('official',ref)):
        phase('quality_'+role,lambda role=role,root=root:command('benchmark_surface_quality_extended',
            ['--subject',str(root),'--output',str(out/('quality_'+role)),'--code-version',b['code_commit'] if role==evaluated_role else 'FreeSurfer 8.2.0 d932c45',
             '--source-kind','fnit' if role==evaluated_role else 'official','--threads','4','--cross-timeout-seconds','180','--max-bbox-pairs','20000000']))
    phase('figures',lambda:command('plot_recon_all_comparison',common+['--region-report',str(out/(f'region_{prefix}.json')),
        '--dice-report',str(out/(f'dice_{prefix}.json')),'--output-dir',str(out/'figures')]))
    # 长距离逐阶段完成并持久写报告，每阶段单独取得并释放共用锁。
    for stage in surface.STAGES:
        def distances(stage=stage):
            import nibabel.freesurfer.io as fs
            import numpy as np
            surface.STAGES=(stage,)
            result=surface.compare(ref,got)
            gate=read(out/(f'geometry_{prefix}.json'))
            for hemi in ('lh','rh'):
                row=result['stages'][hemi][stage]
                g=gate['surfaces'].get(hemi+'.'+stage,{})
                if not g.get('vertex_correspondence'):row['indexed_vertex_distance']=None
                if not g.get('surface_ras_mm_comparable'):
                    row.update(distance_status='not_assessed_invalid_space');continue
                if stage in ('white','pial') and 'candidate_to_reference_triangle' not in row:
                    (a,fa),(z,fz)=[fs.read_geometry(str(root/'surf'/f'{hemi}.{stage}')) for root in (ref,got)]
                    row['candidate_to_reference_triangle']=surface._summary(surface._point_to_mesh(z,a,fa))
                    row['reference_to_candidate_triangle']=surface._summary(surface._point_to_mesh(a,z,fz))
                row['scope']='vertex sampled bidirectional triangle distances; not continuous Hausdorff'
            w.write(out/f'surface_{stage}_{prefix}.json',result)
        phase('surface_'+stage,distances)
    state.update(status='complete',current_phase=None)
    state['strict_138']={k:read(out/(f'strict_{prefix}.json'))[k] for k in ('checked','passed')}
    w.write(checkpoint,state)


if __name__=='__main__':
    main()
