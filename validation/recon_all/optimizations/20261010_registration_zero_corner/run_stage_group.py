"""串行真实球面配准阶段驱动；新目录、全轮次，保留每个退出码。"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 task=r/'workspaces/recon_registration_zero_corner_20261010_v1'
 out=r/'runs/recon_registration_zero_corner_20261010_v1';out.mkdir(parents=True,exist_ok=False)
 atlas=r/'assets/recon_resources_20261009_v1/assets/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif'
 subjects={'v4':r/'runs/recon_python_profile_20261010_v4/sub06-candidate/subject',
           '79':r/'runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1/sub06-candidate/subject'}
 cases=[('checkpoint','new',None),('v4-new','new','v4'),('79-old','old','79'),('79-new','new','79')]
 rows=[]
 for name,kind,subject in cases:
  env=dict(os.environ,PYTHONPATH=str(r/'workspaces/recon_python_profile_20261010_v4/tools/recon_python_execution_audit')+':'+str(task/kind/'src'),
   FNIT_RECON_AUDIT_DIR=str(out/name/'execution_audit'),NUMBA_CACHE_DIR=str(out/'numba_cache'/kind),
   OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMBA_NUM_THREADS='4')
  env.pop('CUDA_VISIBLE_DEVICES',None)
  env.pop('PYTORCH_NO_CUDA_MEMORY_CACHING',None)
  cmd=[sys.executable,'-X','faulthandler',str(task/'incoming/benchmark_registration.py'),
       '--mode','checkpoint' if name=='checkpoint' else 'full','--source',str(task/kind),
       '--output',str(out/name),'--device','cuda:5','--threads','4']
  if name=='checkpoint':cmd+=['--baseline-source',str(task/'old'),'--checkpoint',str(r/'runs/diagnose_registration_nonfinite_20261010_v1/sub06-lh/first_nonfinite.npz')]
  else:cmd+=['--subject',str(subjects[subject]),'--atlas',str(atlas),'--hemi','lh']
  tick=time.perf_counter();(out/name).mkdir()
  with (out/name/'console.log').open('w') as log:
   result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  row={'name':name,'source':kind,'exit_code':result.returncode,'full_cold_process_wall_seconds':time.perf_counter()-tick,'command':cmd,
       'benchmark_script_sha256':sha(task/'incoming/benchmark_registration.py')}
  rows.append(row);(out/'driver.json').write_text(json.dumps({'status':'running','cases':rows},indent=2)+'\n')
  print(json.dumps(row),flush=True)
 (out/'driver.json').write_text(json.dumps({'status':'complete' if all(x['exit_code']==0 for x in rows) else 'finished_with_stage_failures','cases':rows},indent=2)+'\n')
 (out/'EXIT_CODE.txt').write_text('0\n' if all(x['exit_code']==0 for x in rows) else '1\n')
if __name__=='__main__':main()
