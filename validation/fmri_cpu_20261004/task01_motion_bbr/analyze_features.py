"""Check every frozen functional job and compare full real outputs outside timing."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,sys
from pathlib import Path
from benchmark_baseline import digest

def configuration(job):
    command=job['command'];result=[];i=2
    ignore={'--source','--source-revision','--output-dir','--case-json','--cpu-list','--cpu-lock','--backend'}
    while i<len(command):
        if command[i] in ignore:i+=2;continue
        if command[i]=='--native-trace':i+=1;continue
        result.append(command[i]);i+=1
    return tuple(result)

def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--controller',type=Path,required=True)
    parser.add_argument('--analysis-helpers',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text());state=json.loads(args.controller.read_text())
    assert state['status']=='completed' and len(state['jobs'])==31
    assert all(row['exit_code']==0 for row in state['jobs'])
    assert state['manifest_sha256']==digest(args.manifest)
    args.output.mkdir(parents=True,exist_ok=False)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='';os.sched_setaffinity(0,{manifest['cpu_group'][0]})
    sys.path.insert(0,str(args.analysis_helpers))
    from analyze_complete import publish,compare_prefixes,trace_proof,array_precision
    rows=[];native=[];jobs=manifest['jobs'];reports={}
    publish(args.output/'status.safe.json',{'status':'waiting_for_cpu_lease','pid':os.getpid()})
    with Path(manifest['cpu_lock']).open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        import numpy as np
        inputs=json.loads(args.inputs.read_text())
        for job in jobs:
            name=job['name'];report=json.loads((Path(job['output_dir'])/'report.safe.json').read_text());reports[name]=report
            recorded=next(row for row in state['jobs'] if row['name']==name)
            assert recorded['command_sha256']==hashlib.sha256(json.dumps(job['command'],separators=(',',':')).encode()).hexdigest(),name
            assert report['adapter_sha256']==job['adapter_sha256']==digest(job['command'][1]),name
            assert report['cpu_threads']==job['threads'] and report['cpu_affinity']==[int(v) for v in job['cpu_list'].split(',')],name
            if job['backend']=='official':
                prefix=Path(job['output_dir'])/'repeat_0/result'
                proof=trace_proof(prefix.parent/'native_exec.private.trace',report['native_exit_code'],job['command'][job['command'].index('--source')+1])
                native.append({'job':name,'evidence':proof});publish(args.output/'native_exit.safe.json',native)
                assert proof['accepted_official_payload_and_complete_chain'],name
            else:
                assert report['status']=='complete',name
                assert report.get('source_unchanged',True) and report.get('input_unchanged',True),name
                if job['source_kind']=='candidate_v4':
                    source=Path(job['command'][job['command'].index('--source')+1])
                    for key,expected in manifest['candidate_source_sha256'].items():
                        relative=key.removeprefix('src/')
                        assert digest(source/relative)==expected and report['source_sha256'][relative]==expected,name
        for job in jobs:
            if job['source_kind']!='candidate_v4':continue
            candidates=[other for other in jobs if other['source_kind']=='baseline' and configuration(other)==configuration(job)]
            report=reports[job['name']]
            for reference in candidates:
                old=reports[reference['name']];official=reference['backend']=='official'
                assert report['input_sha256']==old['input_sha256']
                if job['function'] in ('cli_contract','wrapper_parity'):
                    field='complete_normal_outputs' if job['function']=='cli_contract' else 'logical_output_sha256'
                    equality={'complete_logical_outputs':report['details'][field]==old['details'][field]};precision=None
                else:
                    prefix=Path(job['output_dir'])/'repeat_0/result';expected=Path(reference['output_dir'])/'repeat_0/result'
                    if '--estimate-only' in job['command']:
                        precision={key:array_precision(np.load(str(prefix)+suffix),np.load(str(expected)+suffix))
                                   for key,suffix in (('matrices','.matrices.npy'),('parameters','.parameters.npy'))}
                        equality={key:value['all_values_bit_equal'] for key,value in precision.items()}
                        equality['estimate_only_no_image']=not Path(str(prefix)+'.nii.gz').exists()
                    else:
                        mask=inputs[job['case']].get('brain_mask') if job['function']=='mcflirt' else None
                        precision=compare_prefixes(prefix,expected,job['function'],mask)
                        equality={'matrices':precision['matrices']['all_values_bit_equal'],
                                  'image_values':precision['images']['whole_grid']['all_values_bit_equal'],
                                  'binary_header':precision['images']['header_binary_equal']}
                        if job['function']=='mcflirt':equality['parameters']=precision['parameters']['all_values_bit_equal']
                    if not official:
                        for key in ('cost_evaluations','boundary_points','phase_cost_evaluations','initial_cost','final_cost'):
                            if key in report:equality[key]=report[key]==old[key]
                rows.append({'job':job['name'],'reference_job':reference['name'],'function':job['function'],'threads':job['threads'],
                             'comparison':'candidate_vs_official' if official else 'candidate_vs_frozen_baseline',
                             'precision':precision,'equality':equality,'candidate_application_seconds':report['measured_application_seconds'],
                             'reference_application_seconds':old['measured_application_seconds'],
                             'timing_scope':'Full real functional calls; matched fresh-process default speed acceptance is maintained separately.'})
                publish(args.output/'precision.safe.json',rows)
        oldnew=[row for row in rows if row['comparison']=='candidate_vs_frozen_baseline']
        official=[row for row in rows if row['comparison']=='candidate_vs_official']
        accepted=len(oldnew)==11 and len(official)==6 and len(native)==6 and all(all(row['equality'].values()) for row in oldnew)
        result={'status':'completed' if accepted else 'failed','accepted':accepted,'jobs':31,'full_oldnew_comparisons':len(oldnew),
                'full_official_comparisons':len(official),'native_exec_chains':len(native),'full_real_frames':True,
                'scope':'Official precision metrics are reported without declaring bitwise equality; all frozen baseline comparisons must be exact.'}
        publish(args.output/'status.safe.json',result)
        if not accepted:raise RuntimeError('Functional gate failed; inspect preserved full precision records')
    print(json.dumps(result))
if __name__=='__main__':main()
