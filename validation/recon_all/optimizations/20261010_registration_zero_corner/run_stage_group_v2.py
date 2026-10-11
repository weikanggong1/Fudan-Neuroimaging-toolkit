"""零角点和零面双修复：每一项独立进程、全新目录，保留失败收据。"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--device',required=True);a=p.parse_args();r=a.root
 task=r/'workspaces/recon_registration_zero_corner_20261010_v2';previous=r/'workspaces/recon_registration_zero_corner_20261010_v1'
 out=r/'runs/recon_registration_zero_corner_20261010_v2';out.mkdir(parents=True,exist_ok=False)
 atlas=r/'assets/recon_resources_20261009_v1/assets/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif'
 subjects={'v4':r/'runs/recon_python_profile_20261010_v4/sub06-candidate/subject','79':r/'runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1/sub06-candidate/subject'}
 cases=[('corner-checkpoint','corner'),('face-checkpoint','face'),('v4-new2','v4'),('79-new2','79')];rows=[]
 for name,kind in cases:
  env=dict(os.environ,PYTHONPATH=str(r/'workspaces/recon_python_profile_20261010_v4/tools/recon_python_execution_audit')+':'+str(task/'new2/src'),FNIT_RECON_AUDIT_DIR=str(out/name/'execution_audit'),NUMBA_CACHE_DIR=str(out/'numba_cache'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMBA_NUM_THREADS='4');env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('PYTORCH_NO_CUDA_MEMORY_CACHING',None)
  cmd=[sys.executable,'-X','faulthandler',str(task/'incoming/benchmark_registration_v2.py'),'--mode','checkpoint' if 'checkpoint' in name else 'full','--source',str(task/'new2'),'--output',str(out/name),'--device',a.device,'--threads','4']
  if 'checkpoint' in name:
   version='v1' if kind=='corner' else 'v2';baseline='old' if kind=='corner' else 'new'
   cmd+=['--checkpoint-function',kind,'--baseline-source',str(previous/baseline),'--checkpoint',str(r/f'runs/diagnose_registration_nonfinite_20261010_{version}/sub06-lh/first_nonfinite.npz')]
  else:cmd+=['--subject',str(subjects[kind]),'--atlas',str(atlas),'--hemi','lh']
  (out/name).mkdir();started=time.perf_counter()
  with (out/name/'console.log').open('w') as f:cp=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  rows.append({'name':name,'exit_code':cp.returncode,'full_cold_process_wall_seconds':time.perf_counter()-started,'command':cmd,'benchmark_script_sha256':sha(task/'incoming/benchmark_registration_v2.py')})
  (out/'driver.json').write_text(json.dumps({'status':'running','cases':rows},indent=2)+'\n');print(json.dumps(rows[-1]),flush=True)
 if rows[-1]['exit_code']==0:
  cp=subprocess.run([sys.executable,str(task/'incoming/compare_frozen_outputs.py'),'--control',str(r/'runs/recon_registration_zero_corner_20261010_v1/79-old'),'--candidate',str(out/'79-new2'),'--output',str(out/'79-comparison.json')]);rows.append({'name':'79-comparison','exit_code':cp.returncode,'comparison_script_sha256':sha(task/'incoming/compare_frozen_outputs.py')})
 status='complete' if all(x['exit_code']==0 for x in rows) else 'finished_with_stage_failures'
 (out/'driver.json').write_text(json.dumps({'status':status,'cases':rows},indent=2)+'\n');(out/'EXIT_CODE.txt').write_text('0\n' if status=='complete' else '1\n')
if __name__=='__main__':main()
