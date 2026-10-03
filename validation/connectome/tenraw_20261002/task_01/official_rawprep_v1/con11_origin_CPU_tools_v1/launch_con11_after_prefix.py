#!/usr/bin/env python3
"""Launch one fresh official CON11 CPU chain only after explicit origin and first-nine resource guards."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone

ORIGIN_SHA='9291567c0645c4c5f8c1cf1047d48f80f954c90984f77c7fc6f84d8532cee5d4'
REFERENCE_SHA='6d58dcc69c98e7772525075c49e5364f23869644c686ddec106bd5965f2d123e'
SUBJECTS=['CON01','CON03']+[f'CON{i:02d}' for i in range(4,11)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    q=p.with_suffix('.partial');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def children(pid):
    result=set();parents={}
    for f in Path('/proc').glob('[0-9]*/stat'):
        try:parents[int(f.parent.name)]=int(f.read_text().rsplit(')',1)[1].split()[1])
        except (OSError,ValueError,IndexError):pass
    changed=True
    while changed:
        found={child for child,parent in parents.items() if parent==pid or parent in result}-result;result|=found;changed=bool(found)
    return sorted(result)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('origin','origin-verified','manifest','raw-root','baseline-source','prefix-root','reference-tool','output-root'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prefix-controller-pid',type=int,required=True);p.add_argument('--aligned-controller-pid',type=int,required=True);a=p.parse_args()
    if a.output_root.exists():raise ValueError('fresh CON11 namespace required')
    if sha(a.origin)!=ORIGIN_SHA or sha(a.reference_tool)!=REFERENCE_SHA:raise ValueError('explicit origin/reference source differs')
    receipt=json.loads(a.origin_verified.read_text());origin=json.loads(a.origin.read_text())
    if receipt['state']!='actual_CON11_origin_verified_metadata_only' or receipt['root_origin_sha256']!=ORIGIN_SHA or receipt['canonical_manifest_sha256']!=sha(a.manifest):raise ValueError('actual v2 origin receipt missing')
    for item in ('packing','acqparams','config','driver_lineage_receipt','baseline_source_receipt'):
        if sha(origin[item]['path'])!=origin[item]['sha256']:raise ValueError('immutable actual origin changed')
    state_path=a.output_root.with_name(a.output_root.name+'_activation.json')
    status={'state':'waiting_nine_actual_verified_CPU_cases_and_no_own_native_children','origin_SHA256':ORIGIN_SHA,'verified_origin_SHA256':sha(a.origin_verified),'tool_SHA256':sha(Path(__file__)),'GPU':False,'CPU_threads':8,'CPU_concurrency':1,'own_legacy_controllers_not_signaled':True}
    while True:
        verified={};valid=True
        for s in SUBJECTS:
            d=a.prefix_root/f'sub-{s}';vp=d/'completed_contract_verified.json';rp=d/'report.json'
            if not vp.exists():valid=False;continue
            v=json.loads(vp.read_text());r=json.loads(rp.read_text())
            if not r['completed'] or v['report_sha256']!=sha(rp) or v['EDDY_solver']!='cpu' or v['GPU_UUID'] is not None:raise ValueError('prefix completed contract changed')
            stage=next(c for c in r['commands'] if c['stage']=='official_EDDY_CPU')
            if stage['returncode']!=0:raise ValueError('prefix native CPU solver failed')
            verified[s]={'contract':str(vp),'SHA256':sha(vp),'report_SHA256':sha(rp)}
        live={}
        for pid,needle in [(a.prefix_controller_pid,'run_cpu_budget_reference.py'),(a.aligned_controller_pid,'run_official_rawprep.py')]:
            proc=Path('/proc')/str(pid)
            if proc.exists():
                cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
                if needle not in cmd or '/fnit_connectome_tenraw_20261002/task_01/' not in cmd:raise ValueError('legacy PID identity changed')
                live[str(pid)]={'command':cmd,'start_ticks':proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],'descendants':children(pid)}
        status.update(observed_UTC=datetime.now(timezone.utc).isoformat(),verified_prefix_cases=verified,legacy_controller_identity=live);save(state_path,status)
        if valid and len(verified)==9 and all(not item['descendants'] for item in live.values()):break
        time.sleep(15)
    command=[sys.executable,str(a.reference_tool),'--manifest',str(a.manifest),'--raw-root',str(a.raw_root),'--fnit-source',str(a.baseline_source),'--output-root',str(a.output_root),'--subjects','CON11','--workers','1','--phase','all','--solver','cpu','--fnit-selection-root',str(Path(origin['actual_fnit_case_directory']).parent)]
    log_path=a.output_root.with_name(a.output_root.name+'.log')
    with log_path.open('w') as log:
        child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status.update(state='fresh_CON11_official_CPU_chain_running',PID=child.pid,command=command,log=str(log_path),reference_SHA256=sha(a.reference_tool),execution_kind='fresh_official_CPU_rawprep',source_CPU_stage_lineage='not_applicable_not_fabricated');save(state_path,status);code=child.wait()
    report=a.output_root/'sub-CON11/report.json';completed=report.exists() and json.loads(report.read_text()).get('completed') is True
    status.update(state='fresh_CON11_CPU_chain_completed' if code==0 and completed else 'fresh_CON11_CPU_chain_failed_preserved',returncode=code,completed=completed,observed_UTC=datetime.now(timezone.utc).isoformat());save(state_path,status)
if __name__=='__main__':main()
