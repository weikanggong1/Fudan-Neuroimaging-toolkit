from pathlib import Path
import hashlib,json,os,subprocess,sys,time,inspect
source=Path(sys.argv[1]); run=Path(sys.argv[2]); python=sys.executable
state={'status':'running','commit':'e34a1829145b34158f7b37b5d3046f8b02c03617','scope':'build wheel and install private target within existing declared Conda runtime; not fresh Conda or physical isolation','precision':'no inference performed; no GPU precision changed','stages':{}}
try:
 compiler=Path(python).parent/'x86_64-conda-linux-gnu-c++'; gcc=Path(python).parent/'x86_64-conda-linux-gnu-cc'
 env=dict(os.environ,CXX=str(compiler),CC=str(gcc));env.pop('PYTHONPATH',None)
 state['compiler']=str(compiler.resolve());state['source_pyproject_sha256']=hashlib.sha256((source/'pyproject.toml').read_bytes()).hexdigest()
 for name,cmd in [('wheel',[python,'-m','pip','wheel',str(source),'--no-deps','--no-build-isolation','--wheel-dir',str(run/'wheels')])]:
  start=time.perf_counter()
  with (run/(name+'.log')).open('w') as log: subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  state['stages'][name]={'wall_seconds':time.perf_counter()-start,'command':cmd}
 wheels=list((run/'wheels').glob('*.whl'));assert len(wheels)==1;wheel=wheels[0]
 state['wheel_sha256']=hashlib.sha256(wheel.read_bytes()).hexdigest();state['wheel_bytes']=wheel.stat().st_size
 target=run/'installed_target';cmd=[python,'-m','pip','install','--no-deps','--target',str(target),str(wheel)];start=time.perf_counter()
 with (run/'install.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 state['stages']['install']={'wall_seconds':time.perf_counter()-start,'command':cmd}
 modules=['recon_all/sphere_standard_run.py','recon_all/sphere_standard_finish.py','recon_all/mris_register_overlap.py','recon_all/mris_register_overlap_marked.py','recon_all/native_free.py','recon_all/batch.py','recon_all/mni_mesh_parallel.py','recon_all/mni_warp_inverse.py','recon_all/inflate_standard_run.py','recon_all/inflate_torch.py','recon_all/hemisphere_parallel.py','recon_all/normalization/pipeline.py','recon_all/normalization/aseg_pipeline.py']
 state['installed_modules_sha256']={}
 for module in modules:
  candidate=target/'fnit'/module; original=source/'src/fnit'/module
  assert candidate.read_bytes()==original.read_bytes(),module
  state['installed_modules_sha256'][module]=hashlib.sha256(candidate.read_bytes()).hexdigest()
 env['PYTHONPATH']=str(target);env['CUDA_VISIBLE_DEVICES']=''
 cmd=[python,'-m','fnit.recon_all.native_free','--help'];start=time.perf_counter()
 result=subprocess.run(cmd,cwd=run,env=env,capture_output=True,text=True,check=True);(run/'cli_help.txt').write_text(result.stdout+result.stderr)
 assert '--sphere-finish-backend' in result.stdout and '--mni-execution' in result.stdout and '--inflate-backend' in result.stdout and '--normalization-initial-bias-backend' in result.stdout
 state['stages']['installed_cli_help']={'wall_seconds':time.perf_counter()-start,'command':cmd}
 state['status']='passed'
except BaseException as e:
 state.update(status='failed',error=repr(e));raise
finally:
 (run/'report.json').write_text(json.dumps(state,indent=2)+'\n')
