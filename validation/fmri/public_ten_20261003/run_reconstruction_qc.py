"""等待本轮10例两条整链完成后，在CPU节点逐例运行独立4线程重建比较。

只接受显式指定正式candidate根与fresh reference attempt报告，不复用旧subject。
一次仅一例。失败/超预算报告保留；额外QC不计入生产whole benchmark。
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
    return d.hexdigest()


def save(path,r):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--lut',type=Path)
    p.add_argument('--reference-report-name',default='report.public.json')
    p.add_argument('--helpers',type=Path,help='explicit frozen independent helper source directory; defaults to output-root/code for old archived runs')
    cases=['CON01','CON03','CON04','CON05','CON06','CON07','CON08','CON09','CON10','CON11']
    p.add_argument('--subjects',nargs='+',choices=cases,default=cases);p.add_argument('--poll-seconds',type=float,default=60)
    args=p.parse_args()
    if len(set(args.subjects))!=len(args.subjects) or args.poll_seconds<=0:p.error('unique subjects and positive poll interval required')
    output=args.output_root.resolve();candidate_root=args.candidate_root.resolve();reference_base=(args.root/'reference_fmriprep25_2_4_v1').resolve();raw_root=(args.root/'raw').resolve()
    if any(output.is_relative_to(protected) or protected.is_relative_to(output) for protected in (candidate_root,reference_base,raw_root,args.source.resolve())):raise ValueError('posthoc output must be separate from MRI inputs, frozen source and production outputs')
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['NUMBA_CACHE_DIR']=str(output/'numba_cache');os.environ['CUDA_VISIBLE_DEVICES']='';output.mkdir(exist_ok=True);helpers=args.helpers.resolve() if args.helpers is not None else output/'code';wrapper=helpers/'compare_reconstruction.py'
    reference_root=reference_base/'cases';source_files={p.relative_to(args.source).as_posix():sha(p) for p in sorted((args.source/'src/fnit').rglob('*.py'))}
    if not source_files:raise ValueError('frozen FNIT source is absent')
    source_fingerprint=hashlib.sha256(json.dumps(source_files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    roots_fingerprint=hashlib.sha256(json.dumps([str(candidate_root),str(reference_base),str(raw_root),args.reference_report_name],separators=(',',':')).encode()).hexdigest()
    helper_hashes={name:sha(helpers/name) for name in ('compare_reconstruction.py','compare_surface_chain.py','compare_region_stats.py','benchmark_surface_quality_extended.py','verify_provided_transform.py')}
    helper_fingerprint=hashlib.sha256(json.dumps(helper_hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    state={'status':'waiting','cases':{},'launcher_sha256':sha(__file__),'candidate_cohort':candidate_root.name,'source_fingerprint':source_fingerprint,'binding_roots_fingerprint':roots_fingerprint,'helpers_fingerprint':helper_fingerprint,'lut_sha256':sha(args.lut) if args.lut is not None else None,'scope':'independent posthoc, excluded from production whole-time'}
    old=output/'cohort.public.json'
    if old.exists():
        previous=json.loads(old.read_text())
        if any(previous.get(k)!=state[k] for k in ('launcher_sha256','candidate_cohort','source_fingerprint','binding_roots_fingerprint','helpers_fingerprint','lut_sha256')):raise ValueError('existing posthoc cohort belongs to a different launcher/candidate/source/binding/helper/LUT')
    manifest=json.loads((args.root/'raw/public_manifest.json').read_text());source_raw={}
    for x in manifest['subjects']:
        case=x.get('subject') or x.get('subject_id') or x.get('case_id')
        source_raw[case]=x
    pending=list(args.subjects)
    while pending:
        made_progress=False
        for case in list(pending):
            target=output/case;report=target/'report.public.json'
            if target.is_dir() and not report.is_file():
                process_file=target/'process.private.json'
                process=json.loads(process_file.read_text()) if process_file.exists() else {}
                proc_cmd=Path('/proc')/str(process.get('pid','missing'))/'cmdline'
                if proc_cmd.exists() and str(report).encode() in proc_cmd.read_bytes():continue
                result={'status':'failed_interrupted_before_report','process_exit_code':None,'process_wall_seconds':None,'report_sha256':None}
                save(target/'completion.public.json',result);state['cases'][case]=result;pending.remove(case);made_progress=True;continue
            if report.is_file():
                prior=json.loads(report.read_text());status=prior.get('status')
                if status!='running':state['cases'][case]={'status':status,'report_sha256':sha(report)};pending.remove(case);made_progress=True;continue
                if (target/'completion.public.json').is_file():state['cases'][case]=json.loads((target/'completion.public.json').read_text());pending.remove(case);made_progress=True;continue
                process_file=target/'process.private.json'
                process=json.loads(process_file.read_text()) if process_file.exists() else {}
                proc_cmd=Path('/proc')/str(process.get('pid','missing'))/'cmdline'
                if proc_cmd.exists() and str(report).encode() in proc_cmd.read_bytes():continue
                prior.update(status='failed_interrupted_posthoc',reason='running report has no live bound worker; preserved without overwrite')
                save(report,prior);result={'status':prior['status'],'process_exit_code':None,'process_wall_seconds':None,'report_sha256':sha(report)}
                save(target/'completion.public.json',result);state['cases'][case]=result;pending.remove(case);made_progress=True;continue
            candidate=candidate_root/case/'report/report.public.json'
            attempts=sorted((reference_root/f'sub-{case}').glob('attempt-*/'+args.reference_report_name))
            refs=[x for x in attempts if json.loads(x.read_text()).get('status')=='complete']
            if len(refs)>1:state['cases'][case]={'status':'failed_multiple_completed_reference_attempts'};pending.remove(case);made_progress=True;continue
            if not candidate.is_file() or not refs:continue
            cand=json.loads(candidate.read_text())
            if cand.get('status')!='complete':continue
            reference=refs[0];rr=json.loads(reference.read_text())
            if rr.get('source_unchanged_during_run') is not True:continue
            candidate_subject=candidate_root/case/'reconstruction/subject'
            reference_subject=reference.parent/'derivatives/sourcedata/freesurfer'/f'sub-{case}_ses-preop'
            raw=raw_root/source_raw[case]['T1w']['relative_path']
            if not (candidate_subject/'mri/orig.mgz').is_file() or not (reference_subject/'mri/orig.mgz').is_file():raise FileNotFoundError('completed run lacks its exact current subject orig')
            if sha(raw)!=source_raw[case]['T1w']['sha256']:raise ValueError(f'{case} manifest raw SHA mismatch')
            for role,r in [('candidate',cand),('reference',rr)]:
                if r.get('input_sha256',{}).get('t1w')!=sha(raw):raise ValueError(f'{case} {role} raw binding mismatch')
            target.mkdir(exist_ok=False)
            bindings={'case_id':case,'candidate_report':str(candidate),'reference_report':str(reference),
                      'candidate_subject':str(candidate_subject),'reference_subject':str(reference_subject),'raw_t1w':str(raw),
                      'reference_recon_all_path_correction':'original frozen reference optional recon_all omitted session; independently bind exact current fresh sub-CONxx_ses-preop',
                      'input_sha256':{k:sha(v) for k,v in [('candidate_report',candidate),('reference_report',reference),('raw_t1w',raw)]}}
            save(target/'files.private.json',bindings)
            cmd=[sys.executable,str(wrapper),'--case-id',case,'--reference',str(reference_subject),'--candidate',str(candidate_subject),
                 '--reference-report',str(reference),'--candidate-report',str(candidate),'--raw-t1w',str(raw),'--source',str(args.source),
                 '--helpers',str(helpers),'--threads','4','--quality-timeout','180','--pair-budget','20000000','--output',str(report)]
            if args.lut is not None:cmd+=['--lut',str(args.lut)]
            state['cases'][case]={'status':'running'};state['status']='running';save(output/'cohort.public.json',state)
            started=time.perf_counter()
            with (target/'process.private.log').open('w') as log:
                run=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                save(target/'process.private.json',{'pid':run.pid,'command':cmd,'start_unix_seconds':time.time()})
                run.wait()
            final=json.loads(report.read_text()) if report.is_file() else {}
            result={'status':final.get('status') if run.returncode==0 else 'failed_posthoc','process_exit_code':run.returncode,
                    'process_wall_seconds':time.perf_counter()-started,'report_sha256':sha(report) if report.exists() else None}
            save(target/'completion.public.json',result);state['cases'][case]=result;pending.remove(case);made_progress=True;save(output/'cohort.public.json',state)
        state['pending_cases']=list(pending);save(output/'cohort.public.json',state)
        if pending and not made_progress:time.sleep(args.poll_seconds)
    state['status']='measured' if all(x['status']=='measured' for x in state['cases'].values()) else 'partially_measured_or_failed';state['launcher_sha256_after']=sha(__file__);save(output/'cohort.public.json',state)


if __name__=='__main__':main()
