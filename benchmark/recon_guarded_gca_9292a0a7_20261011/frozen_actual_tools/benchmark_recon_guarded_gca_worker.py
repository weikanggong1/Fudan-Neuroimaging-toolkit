"""冻结 wheel 私有安装的两例同输入完整 GCA worker；隔离读取历史对照。

输入 root：既有 FNIT 索引根目录；workspace：仅含 wheel/探针/installed
的新私有源码目录；output：新的空运行目录。两例 nu/brainmask、GCA 以及
旧自产/原版 LTA 均按成功回执 SHA 核对；旧 LTA 仅在 worker 退出后读入。
每例新 exec CLI、独立 NumBa 缓存、4线程与CPU亲和性；原版禁令=1。
不启动原版参考、不运行原始T1前段，不修改共享环境/数学函数/旧产物。
"""
import argparse,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
WHEEL_SHA='3e5e2eb0332d328cd92a968a17ae8cc2c6d872a961d103bca4a9d0d65df39e51'
SOURCE_COMMIT='9292a0a7be63f6bc96f2dfa358261cb1587e0f23'
SOURCE_MANIFEST='81138d195eae18f341fb7a0485f67e3532fc103a65f9f522c7047549f7ead17a'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('root','workspace','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 wheel=a.workspace/'fudan_neuroimaging_toolkit-0.16.0-py3-none-any.whl';installed=a.workspace/'installed'
 assert wheel.stat().st_size==5309664 and sha(wheel)==WHEEL_SHA
 verified={}
 with zipfile.ZipFile(wheel) as z:
  for name in z.namelist():
   if name.startswith('fnit/') and name.endswith('.py'):
    digest=hashlib.sha256(z.read(name)).hexdigest();assert sha(installed/name)==digest,name;verified[name]=digest
 report={'scope':'two_same_input_guarded_complete_GCA_workers_only_not_raw_T1_or_whole_reconstruction','wheel_sha256':WHEEL_SHA,'wheel_bytes':5309664,'source_commit':SOURCE_COMMIT,'target_GPU_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'parent_CPU_affinity':sorted(os.sched_getaffinity(0)),'thread_environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')},'declared_coordinator_source_manifest_sha256':SOURCE_MANIFEST,'installed_Python_file_count_verified_against_wheel':len(verified),'installed_module_sha256':{k:v for k,v in verified.items() if k.startswith('fnit/recon_all/mri_em_register') or k.endswith(('gca_torch_worker.py','worker_program_guard.py','execution_recipe.py'))},'benchmark_source_sha256':sha(__file__),'probe_source_sha256':sha(a.workspace/'audit/sitecustomize.py'),'cold_definition':'fresh exec CUDA context and per-case empty NumBa cache; filesystem/device hardware cache not controlled','cold_CLI_timing':'spawn through exit including interpreter/import/audit/hash/load/transfer/JIT/stable and all EM/write; excludes lock wait, install and parent preflight/comparison','own_combined_GPU_memory_bytes':None,'GPU_20GB_gate':'not_assessed','GPU_attribution':'driver PID mapping unproved; allocator peaks are child-only, own simultaneous total unknown','overall_equivalence':'not_assessed','cases':{}}
 (a.output/'installation_verified.public.json').write_text(json.dumps(report,indent=2)+'\n')
 try:
  for case,label in (('sub06','public_A'),('sub07','public_B')):
   olddir=a.root/f'runs/recon_fnit3_20261011_v7/gca_{case}_complete';refdir=a.root/f'runs/recon_fnit3_20261011_v2/gca_{case}_native'
   old=json.loads((olddir/'metrics.public.json').read_text());reference=json.loads((refdir/'metrics.public.json').read_text())
   assert old['execution_complete'] and reference['execution_complete'] and old['input_sha256']==reference['input_sha256']
   inp=a.root/f'runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1/{case}-candidate/subject/mri';nu=inp/'nu.mgz';mask=inp/'brainmask.mgz';atlas=a.root/'assets/recon_resources_20261009_v1/assets/average/RB_all_2020-01-02.gca'
   hashes={'nu':sha(nu),'brainmask':sha(mask),'GCA':sha(atlas)};assert hashes==old['input_sha256']
   for directory,receipt in ((olddir,old),(refdir,reference)):assert sha(directory/'talairach.lta')==receipt['output_sha256']
   target=a.output/label;target.mkdir();env=os.environ.copy();env.update(PYTHONPATH=str(a.workspace/'audit')+os.pathsep+str(installed),FNIT_GCA_INSTALLED_TARGET=str(installed),FNIT_GCA_AUDIT_PATH=str(target/'runtime_audit.public.json'),NUMBA_CACHE_DIR=str(target/'numba_cache'),FNIT_RECON_FORBID_ORIGINAL_STAGES='1')
   command=[sys.executable,'-m','fnit.recon_all.gca_torch_worker','--nu',str(nu),'--mask',str(mask),'--atlas',str(atlas),'--output',str(target/'talairach.lta'),'--report',str(target/'worker.private.json'),'--device','cuda:0','--threads','4','--candidate-chunk','64','--inverse-backend','torch','--code-version',SOURCE_COMMIT]
   tick=time.perf_counter()
   with (target/'worker.stdout.private.log').open('w') as stream:result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
   seconds=time.perf_counter()-tick;(target/'worker.EXIT').write_text(str(result.returncode)+'\n');assert result.returncode==0
   worker=json.loads((target/'worker.private.json').read_text());audit=json.loads((target/'runtime_audit.public.json').read_text());assert audit['registration_entry']['guard_enabled'] and audit['registration_entry']['environment_guard']=='1' and audit['package_loaded_from_installed_target']
   assert all(row and not row['attempted_forbidden'] for row in audit['guard_records'])
   from fnit.recon_all.ca_normalize_python import read_voxel_lta
   matrix=read_voxel_lta(target/'talairach.lta');oldmatrix=read_voxel_lta(olddir/'talairach.lta');refmatrix=read_voxel_lta(refdir/'talairach.lta')
   import numpy as np
   protocol_keys=('histogram_preprocess','translation_history','linear_history','em_complete_protocol','em_cost','em_trials','linear_iterations')
   protocol_comparison={key:worker[key]==old['api'][key] for key in protocol_keys}
   bound_inputs={'nu':worker['inputs_sha256']['nu'],'brainmask':worker['inputs_sha256']['mask'],'GCA':worker['inputs_sha256']['atlas']};assert bound_inputs==hashes
   row={'input_sha256':hashes,'frozen_own_receipt_sha256':sha(olddir/'metrics.public.json'),'isolated_original_receipt_sha256':sha(refdir/'metrics.public.json'),'LTA_sha256':{'current':sha(target/'talairach.lta'),'frozen_own':sha(olddir/'talairach.lta'),'original':sha(refdir/'talairach.lta')},'strict_matrix_tolerance':0,'matrix':matrix.tolist(),'frozen_own_matrix_max_error':float(np.abs(matrix-oldmatrix).max()),'original_matrix_max_error':float(np.abs(matrix-refmatrix).max()),'complete_process_metrics_equal':protocol_comparison,'cold_CLI_seconds_including_exit':seconds,'old_Python_API_seconds_excluding_import':old['seconds_including_spawn_load_IO'],'old_original_CLI_seconds':reference['seconds_including_spawn_load_IO'],'worker_exit':result.returncode,'runtime_audit':audit,'worker_metrics':{k:v for k,v in worker.items() if k not in ('output','worker_pid','cuda_visible_devices')},'accepted_updates_limitation':'unchanged real inputs; both original successful cases had zero accepted stable/all EM updates; no perturbed inputs or whole-pipeline inference','execution_complete':True}
   assert row['frozen_own_matrix_max_error']==0 and row['original_matrix_max_error']==0 and all(protocol_comparison.values()),'same-input regression failed'
   report['cases'][label]=row;(a.output/f'{label}.public.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');print(label,'strict LTA and complete process metrics passed',seconds,flush=True)
  report['execution_complete']=True
 except BaseException as error:
  report.update(execution_complete=False,error_type=type(error).__name__)
  raise
 finally:(a.output/'guarded_gca.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
