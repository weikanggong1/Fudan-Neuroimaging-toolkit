"""Strict metadata orchestration around frozen model_case; no new scientific worker."""
import argparse,copy,datetime,fcntl,importlib.util,json,os,socket,time
from pathlib import Path
import nibabel as nib
import numpy as np
from validate_CON11_packing_route_v1 import sha,validate_actual_CON11_packing_route

WORKER_SHA='616b3f01197447da8255c20592165f066f03bc20b48765b9612ea6ada29d11c1'
CONFIG_SHA='7ec2008b29c077797c259cad1245c8d2f9999c86af621d0821c9bc7d52335627'
ROUTE_SHA='c48f34b9ec30b8e8379ea8eda3b82bacef8208c342457a765cf8cbf1a2c0fc15'
LINEAGE_SHA='10179d3126ca32dd4513ea45602d9b9b27320de96dc9efe1b67337a644a9b81a'
ORIGIN_SHA='1ca5d7cfb588b52566fe5dc3638a1cb33d1acf0e0120cfeda5502abd1715a2d3'
REFERENCE_SHA='6d58dcc69c98e7772525075c49e5364f23869644c686ddec106bd5965f2d123e'
GATE_SHA='3241d9287d013ca237a540f33b58f0faa46d51e17bde73b4f010f1c874cc4538'
PREFIX=['CON01','CON03']+[f'CON{i:02d}' for i in range(4,11)]
def read(p):return json.loads(Path(p).read_text())
def bound(p):return {'path':str(p),'sha256':sha(p)}
def save(p,d,exclusive=False):
    p=Path(p)
    if exclusive:
        with p.open('x') as f:f.write(json.dumps(d,indent=2,allow_nan=False)+'\n')
        p.chmod(0o444)
    else:
        tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load_worker(root):
    own=root/'task_02';wp=own/'official_modeling_cohort_cpu_v3.py';cp=own/'official_modeling_CPU_budget_raw10_v2.config.json'
    if sha(Path(__file__).with_name('validate_CON11_packing_route_v1.py'))!='debcce670fe87ce84f174e299377eba9e18b937872ee7b9ef23eeaf183ecec71':raise ValueError('Frozen packing guard changed')
    if sha(wp)!=WORKER_SHA or sha(cp)!=CONFIG_SHA:raise ValueError('Frozen worker/config changed')
    spec=importlib.util.spec_from_file_location('frozen_official_modeling',wp);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    return worker,read(cp),wp,cp

def validate_verified_records(report,completed,verified,report_path,case_root,route):
    if not report.get('completed') or report.get('state')!='completed':raise ValueError('Fresh report not completed')
    if report.get('subject')!='CON11' or route.get('case_id')!='sub-CON11' or route.get('subject')!='CON11':raise ValueError('CON11 identity differs')
    if report.get('source_CPU_stage_lineage'):raise ValueError('Fresh CPU cannot carry restored lineage')
    for key in ('completed','subject','EDDY_solver','GPU_UUID','GP_seed','raw_sha256','output_sha256','data','rotated_bvecs','bvals','mask','topup_prefix','ref_scan_no'):
        if completed.get(key)!=verified.get(key):raise ValueError('Verified completed fields differ: '+key)
    if verified.get('completed') is not True or verified.get('EDDY_solver')!='cpu' or verified.get('GPU_UUID','absent') is not None or verified.get('GP_seed')!=12345 or verified.get('ref_scan_no')!=0:raise ValueError('CPU scientific labels differ')
    if verified.get('report')!=str(report_path) or verified.get('report_sha256')!=sha(report_path) or route.get('report')!=str(report_path) or route.get('report_SHA256')!=sha(report_path):raise ValueError('Actual verified report binding differs')
    if verified.get('raw_sha256')!=report['input_sha256'] or verified.get('output_sha256')!=report['output_sha256'] or route.get('output_sha256')!=report['output_sha256']:raise ValueError('Report/raw/output digest records differ')
    if verified.get('execution_kind')!='fresh_official_CPU_rawprep' or route.get('execution_kind')!='fresh_official_CPU_rawprep':raise ValueError('Fresh execution identity absent')
    selection=verified.get('actual_selection_origin',{})
    if selection.get('receipt_SHA256')!=ORIGIN_SHA:raise ValueError('Fresh verified origin receipt differs')
    for key,relative in {'data':'eddy/data.nii.gz','mask':'mask/nodif_brain_mask.nii.gz','rotated_bvecs':'eddy/data.eddy_rotated_bvecs','bvals':'raw/AP.bval','topup_prefix':'topup/fieldmap_out'}.items():
        if verified.get(key)!=str(case_root/relative) or route.get(key)!=verified[key]:raise ValueError('Fresh case own path differs: '+key)
    for relative,digest in report['output_sha256'].items():
        path=case_root/relative
        if not path.resolve().is_relative_to(case_root.resolve()) or sha(path)!=digest:raise ValueError('Own official output SHA/path differs')
    for path,digest in report['EDDY_inputs_sha256'].items():
        lexical=Path(path);target=lexical.resolve()
        raw_link=lexical in {case_root/'raw/AP.nii.gz',case_root/'raw/AP.bval',case_root/'raw/AP.bvec'}
        canonical_target=raw_link and report['input_sha256'].get(str(target))==digest
        if not lexical.is_relative_to(case_root) or (not target.is_relative_to(case_root.resolve()) and not canonical_target) or sha(path)!=digest:raise ValueError('Own EDDY input/canonical raw link differs')
    return True

def ready(root,worker,config):
    routes_path=root/'task_01/explicit_actual_CPU_case_routes_v1.json'
    if not routes_path.is_file():return None,'waiting_explicit_route'
    routes=read(routes_path);route=routes.get('cases',{}).get('sub-CON11')
    if not route or route.get('state')!='actual_verified_CPU_contract_ready':return None,'waiting_actual_verified_CON11_CPU_contract'
    case_root=root/'task_01/official_CON11_CPU_fresh_origin_v1/sub-CON11';rp=case_root/'report.json';vp=case_root/'completed_contract_verified.json';sp=case_root/'completed_contract.json'
    if route.get('official_case_directory')!=str(case_root) or route.get('verified_contract_expected_path')!=str(vp):raise ValueError('Explicit CON11 route changed own root')
    if not all(p.is_file() for p in [rp,vp,sp]):return None,'waiting_actual_completed_sidecars'
    if sha(vp)!=route.get('verified_contract_SHA256'):raise ValueError('Verified route SHA changed')
    r=read(rp);v=read(vp);s=read(sp);validate_verified_records(r,s,v,rp,case_root,route)
    route_root=root/'root_CON11_packing_task02_route_v1';packing_route=route_root/'actual_packing_route.json';lineage=route_root/'expected_baseline_lineage.json'
    if sha(packing_route)!=ROUTE_SHA or sha(lineage)!=LINEAGE_SHA:raise ValueError('Frozen root packing/lineage changed')
    proof=validate_actual_CON11_packing_route(read(packing_route),config,read(lineage))
    pr=read(packing_route);ap_dwi=Path(pr['canonical_inputs']['AP']['dwi']['path']);ap_bval=Path(pr['canonical_inputs']['AP']['bvals']['path']);ap_bvec=Path(str(ap_dwi).replace('.nii.gz','.bvec'))
    for raw_name,target in [('AP.nii.gz',ap_dwi),('AP.bval',ap_bval),('AP.bvec',ap_bvec)]:
        if (case_root/'raw'/raw_name).resolve()!=target.resolve():raise ValueError('Actual AP canonical raw link target differs')
    if proof is None:raise ValueError('Actual packing guard did not pass')
    selection=v['actual_selection_origin'];origin_path=Path(selection['receipt'])
    if sha(origin_path)!=ORIGIN_SHA or origin_path!=root/'task_01/CON11_origin_verified_v2.json':raise ValueError('Actual Task1 origin receipt differs')
    if route.get('actual_FNIT_case_directory')!=selection.get('actual_FNIT_case_directory') or proof['actual_packing']['path']!=str(Path(route['actual_FNIT_case_directory'])/'connectome/preproc/topup/B0_AP_PA.nii.gz'):raise ValueError('Baseline case origin differs')
    if r.get('selection',{}).get('ap_index')!=0 or r.get('selection',{}).get('pa_index')!=0:raise ValueError('Actual CON11 selection differs')
    ev=r.get('FNIT_selection_evidence',{})
    if ev.get('packed_pair')!=proof['actual_packing']['path'] or ev.get('packed_pair_sha256')!=proof['actual_packing']['sha256']:raise ValueError('Actual fresh selection evidence differs')
    own_pair=nib.load(case_root/'topup/B0_AP_PA.nii.gz');pair=nib.load(proof['actual_packing']['path'])
    if not np.array_equal(own_pair.affine,pair.affine) or not np.array_equal(np.asarray(own_pair.dataobj),np.asarray(pair.dataobj)):raise ValueError('Fresh official raw pair differs')
    activation_path=root/'task_01/official_CON11_CPU_fresh_origin_v1_activation.json';activation=read(activation_path)
    if activation.get('state')!='fresh_CON11_CPU_chain_completed' or activation.get('completed') is not True or activation.get('returncode')!=0:return None,'waiting_fresh_CPU_driver_exit'
    if activation.get('tool_SHA256')!=GATE_SHA or activation.get('reference_SHA256')!=REFERENCE_SHA or activation.get('GPU') is not False or activation.get('CPU_threads')!=8 or activation.get('CPU_concurrency')!=1:raise ValueError('Fresh activation scientific/source identity differs')
    gate_path=root/'task_01/con11_origin_CPU_tools_v1/launch_con11_after_prefix.py'
    if sha(gate_path)!=GATE_SHA:raise ValueError('Actual fresh launch gate source changed')
    command=activation['command'];reference_path=Path(command[1])
    if sha(reference_path)!=REFERENCE_SHA:raise ValueError('Original rawprep helper changed')
    for flag,value in {'--subjects':'CON11','--workers':'1','--phase':'all','--solver':'cpu','--output-root':str(case_root.parent),'--manifest':config['manifest'],'--raw-root':config['raw_root'],'--fnit-selection-root':str(Path(route['actual_FNIT_case_directory']).parent)}.items():
        if command.count(flag)!=1 or command[command.index(flag)+1]!=value:raise ValueError('Fresh activation command differs: '+flag)
    freeze_path=case_root.parent/'freeze.json';freeze=read(freeze_path)
    if freeze.get('subjects')!=['CON11'] or freeze.get('phase')!='all' or freeze.get('EDDY_solver')!='cpu' or freeze.get('GPU_UUID','absent') is not None or freeze.get('parent_CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fresh all-phase single CPU freeze differs')
    if freeze.get('manifest_sha256')!=config['manifest_sha256'] or freeze.get('raw_root')!=config['raw_root'] or freeze.get('CPU_threads_per_subject')!=8 or freeze.get('CPU_concurrency')!=1 or r['source_sha256']!=freeze['source_sha256']:raise ValueError('Fresh raw/source freeze differs')
    if freeze['source_sha256'].get(str(reference_path))!=REFERENCE_SHA:raise ValueError('Frozen reference helper SHA absent')
    for path,digest in freeze['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Fresh frozen producer/dependency source changed')
    for subject in PREFIX:
        cp=root/'task_02/official_modeling_CPU_budget_raw10_v1'/('sub-'+subject)/'consumer_contract.json'
        if not cp.exists() or read(cp).get('execution_completed') is not True:return None,'waiting_first9_modeling_complete_no_overlap'
    contract=worker.ready_contract(rp,config['forbidden_input_roots']);contract=worker.validate_raw_binding(contract,config)
    contract['completed_rawprep_sidecar'].update(verified_path=str(vp),verified_sha256=sha(vp))
    raw=nib.load(case_root/'raw/AP.nii.gz');dwi=nib.load(contract['files']['dwi']);mask=nib.load(contract['files']['eddy_mask']);bvals=np.loadtxt(contract['files']['bvals']).reshape(-1);bvecs=np.loadtxt(contract['files']['bvecs'])
    if dwi.shape!=raw.shape or dwi.shape[-1]!=102 or mask.shape!=raw.shape[:3] or bvals.shape!=(102,) or bvecs.shape!=(3,102) or not np.isfinite(bvals).all() or not np.isfinite(bvecs).all():raise ValueError('Fresh 102-volume/gradient format differs')
    if not np.allclose(dwi.affine,raw.affine,rtol=0,atol=1e-5) or not np.allclose(mask.affine,raw.affine,rtol=0,atol=1e-5):raise ValueError('Fresh native grid differs')
    bindings={'packing_route':bound(packing_route),'expected_baseline_lineage':bound(lineage),'producer_activation':bound(activation_path),'producer_freeze':bound(freeze_path),'upstream_report':bound(rp),'completed_contract':bound(sp),'verified_contract':bound(vp),'task01_origin':bound(origin_path),'reference_helper':bound(reference_path),'producer_gate':bound(gate_path),'actual_explicit_route_snapshot_sha256':sha(routes_path)}
    return {'contract':contract,'upstream_report':str(rp),'bindings':bindings,'packing_proof':proof,'actual_FNIT_case_directory':route['actual_FNIT_case_directory']},'ready_actual_verified_CPU'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--state-root',type=Path,required=True);p.add_argument('--poll-seconds',type=float,default=30);p.add_argument('--once',action='store_true');a=p.parse_args()
    a.state_root.mkdir(parents=True,exist_ok=False);lock=(a.state_root/'owner.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=a.root;worker,config,wp,cp=load_worker(root);output=root/'task_02/official_modeling_CON11_selected_recovery_v1'
    if output.exists():raise ValueError('Fresh subset modeling namespace required')
    save(a.state_root/'freeze.json',{'worker':bound(wp),'base_config':bound(cp),'wrapper':bound(Path(__file__)),'GPU':False,'CPU_threads':8,'CPU_concurrency':1,'output_root_planned':str(output),'subset_created':False},exclusive=True)
    while True:
        result,state=ready(root,worker,config);save(a.state_root/'status.json',{'state':state,'UTC':utc(),'PID':os.getpid(),'host':socket.gethostname(),'modeling_ready':result is not None,'subset_launched':False,'planned_output_root':str(output)})
        if result is not None:break
        if a.once:return
        time.sleep(a.poll_seconds)
    output.mkdir(exist_ok=False);subset=copy.deepcopy(config);subset['output_root']=str(output);subset['subjects']=[{'subject':'CON11','ready_contract':result['upstream_report'],'binding_ready':True,'expected_upstream_solver':'cpu','expected_selection':{'ap_index':0,'pa_index':0}}];subset['explicit_fresh_CON11_origin_bindings']=result['bindings'];subset['profile']['upstream_EDDY_solver']='official fresh all-phase FSL eddy_cpu, GPU_UUID=null; separate from restored first9 timing'
    save(output/'configuration.json',subset,exclusive=True);evidence={'scope':'actual packing plus independent verified fresh CPU origin','bindings':result['bindings'],'packing_proof':result['packing_proof'],'actual_FNIT_case_directory':result['actual_FNIT_case_directory'],'modeling_ready':True,'actual_dispatcher_status_path':str(a.state_root/'status.json')};save(output/'alignment_binding.json',evidence,exclusive=True);result['contract']['alignment_binding']={**bound(output/'alignment_binding.json'),'evidence':evidence}
    save(output/'freeze.json',{'worker':bound(wp),'base_config':bound(cp),'wrapper':bound(Path(__file__)),'configuration':bound(output/'configuration.json'),'actual_input_bindings':result['bindings'],'CPU_threads':8,'CPU_concurrency':1,'GPU':False},exclusive=True)
    save(a.state_root/'launch_once.json',{'UTC':utc(),'PID':os.getpid(),'actual_output_root':str(output),'worker':bound(wp),'config':bound(output/'configuration.json')},exclusive=True)
    save(a.state_root/'status.json',{'state':'CON11_subset_modeling_running','UTC':utc(),'PID':os.getpid(),'modeling_ready':True,'subset_launched':True,'actual_output_root':str(output)})
    started=time.perf_counter()
    try:
        report=worker.model_case(subset,'CON11',result['contract'],result['upstream_report']);consumer=output/'sub-CON11/consumer_contract.json';assert read(consumer)['execution_completed'] and report['execution_completed']
        save(a.state_root/'status.json',{'state':'CON11_subset_modeling_completed','UTC':utc(),'PID':os.getpid(),'subset_launched':True,'subset_completed':True,'consumer_contract':bound(consumer),'report':bound(output/'sub-CON11/report.json'),'model_case_wall_s':time.perf_counter()-started})
    except Exception as e:
        save(a.state_root/'status.json',{'state':'CON11_subset_modeling_failed_preserved','UTC':utc(),'PID':os.getpid(),'subset_launched':True,'subset_completed':False,'error':repr(e)});raise
if __name__=='__main__':main()
