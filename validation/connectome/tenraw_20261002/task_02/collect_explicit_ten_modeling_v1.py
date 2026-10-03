"""Read-only explicit nine-original plus one-fresh-CON11 consumer mapping."""
import argparse,datetime,json,os,socket,time
from pathlib import Path
import nibabel as nib
import numpy as np
from validate_CON11_packing_route_v1 import sha
from orchestrate_CON11_CPU_subset_v1 import WORKER_SHA,PREFIX,read,save,bound

def inspect_consumer(path,subject):
    c=read(path)
    if c.get('execution_completed') is not True or c.get('state')!='completed' or c.get('case_id')!='sub-'+subject or c.get('subject')!=subject or c.get('harness_sha256')!=WORKER_SHA or c.get('CPU_threads')!=8:raise ValueError('Actual modeling consumer identity/completion differs')
    rp=Path(c['modeling_report']);r=read(rp)
    if rp.parent!=path.parent or sha(rp)!=c['modeling_report_sha256'] or r.get('execution_completed') is not True or r.get('state')!='completed' or r.get('harness_sha256')!=WORKER_SHA:raise ValueError('Actual modeling report differs')
    if c['actual_upstream_EDDY']['solver']!='cpu' or c['actual_upstream_EDDY']['GPU_UUID'] is not None:raise ValueError('Consumer solver differs')
    binding=c.get('alignment_binding')
    if subject=='CON11':
        if not binding or sha(binding['path'])!=binding['sha256'] or c.get('restored_CPU_stage_lineage'):raise ValueError('Fresh CON11 alignment/source binding missing')
        if r['configuration']['output_root']!=str(path.parent.parent):raise ValueError('Fresh CON11 model namespace mismatch')
    grid=None;products={}
    for key,f in c['files'].items():
        p=Path(f['path'])
        if sha(p)!=f['sha256'] or p.stat().st_size!=f['size_bytes']:raise ValueError('Actual consumer product changed: '+key)
        item={'path':str(p),'sha256':f['sha256'],'size_bytes':p.stat().st_size,'readback':f['readback']}
        if str(p).endswith('.nii.gz'):
            im=nib.load(p)
            if im.shape!=tuple(f['grid']['shape']) or not np.allclose(im.affine,f['grid']['affine'],rtol=0,atol=1e-5):raise ValueError('Actual product geometry differs: '+key)
            item['shape']=list(im.shape);item['affine']=im.affine.tolist()
            if grid is None:grid=(im.shape[:3],im.affine)
            if im.shape[:3]!=grid[0] or not np.allclose(im.affine,grid[1],rtol=0,atol=1e-5):raise ValueError('Consumer native grids differ')
        products[key]=item
    return {'state':'actual_completed_modeling_consumer_verified','case_id':'sub-'+subject,'consumer_contract':bound(path),'modeling_report':bound(rp),'upstream_report':{'path':c['upstream_official_rawprep_report'],'sha256':c['upstream_official_rawprep_report_sha256']},'execution_kind':'fresh_CON11_CPU_rawprep_and_subset_modeling' if subject=='CON11' else 'restored_own_CPU_stages_plus_new_EDDY_and_original_modeling','actual_output_namespace':str(path.parent.parent),'actual_modeling_source_sha256':c['harness_sha256'],'CPU_threads':8,'GPU':False,'alignment_binding':binding,'files':products,'products_hash_and_geometry_verified_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_consumer_unchanged':True,'nonspatial_axis_note':'Original full consumer contract hash retained. Component-axis spacing including NaN is not rewritten or represented as spatial distance.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--poll-seconds',type=float,default=30);p.add_argument('--once',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    original=a.root/'task_02/official_modeling_CPU_budget_raw10_v1';fresh=a.root/'task_02/official_modeling_CON11_selected_recovery_v1';sources={s:original/('sub-'+s)/'consumer_contract.json' for s in PREFIX};sources['CON11']=fresh/'sub-CON11/consumer_contract.json';cache={}
    save(a.output/'freeze.json',{'collector':bound(Path(__file__)),'explicit_planned_case_consumers':{s:str(p) for s,p in sources.items()},'scope':'Explicit planned paths are not readiness. Nine original and one fresh CON11 namespace; no old CON11 path alias, no scientific computation.'},exclusive=True)
    while True:
        cases={}
        for s,p in sources.items():
            if not p.exists():cases[s]={'state':'waiting_actual_completed_modeling_consumer','planned_consumer_path':str(p)};continue
            if s in cache:
                if sha(p)!=cache[s]['consumer_contract']['sha256'] or sha(cache[s]['modeling_report']['path'])!=cache[s]['modeling_report']['sha256']:raise ValueError('Previously verified consumer/report changed')
            else:
                c=read(p)
                if c.get('execution_completed') is not True:cases[s]={'state':'waiting_actual_completed_modeling_consumer','planned_consumer_path':str(p)};continue
                cache[s]=inspect_consumer(p,s)
            cases[s]=cache[s]
        d={'schema_version':1,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':socket.gethostname(),'PID':os.getpid(),'state':'explicit_ten_actual_modeling_completed' if len(cache)==10 else 'waiting_actual_modeling_consumers','models_completed':len(cache),'actual_completed_case_map':{s:cache[s] for s in cache},'cases':cases,'all_ten_actual_modeling_completed':len(cache)==10,'original_global_cohort_CON11_not_relabelled':True,'collector_source':bound(Path(__file__))};save(a.output/'status.json',d)
        if len(cache)==10:save(a.output/'completed_case_map.json',d,exclusive=True);return
        if a.once:return
        time.sleep(a.poll_seconds)
if __name__=='__main__':main()
