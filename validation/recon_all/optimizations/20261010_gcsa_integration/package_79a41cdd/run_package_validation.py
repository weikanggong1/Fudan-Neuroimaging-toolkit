"""从冻结Git提交副本构建wheel并安装私有目标；只验证包装，不执行影像推理。

--source-root为含src/pyproject的冻结源码，--output-root为新空目录。
返回码0与report.json.status=passed表示包装通过；错误保留实际命令、
耗时与失败信息。不修改原冻结源码、当前Conda prefix、精度或GPU状态。
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,inspect
import argparse,shutil
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--output-root', type=Path, required=True)
args=parser.parse_args()
run=args.output_root.resolve();run.mkdir(mode=0o700,exist_ok=False)
source=run/'build_source';shutil.copytree(args.source_root,source,ignore=shutil.ignore_patterns('__pycache__','*.tar','source.json'))
python=sys.executable
state={'status':'running','commit':'79a41cdda2397525a8952dd5af303372b352916c','scope':'build wheel and install private target within existing declared Conda runtime; not fresh Conda or physical isolation','precision':'no inference performed; no GPU precision changed','stages':{}}
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
 modules=[str(path.relative_to(source/'src/fnit')) for path in sorted((source/'src/fnit').rglob('*.py'))]
 state['installed_modules_sha256']={}
 for module in modules:
  candidate=target/'fnit'/module; original=source/'src/fnit'/module
  assert candidate.read_bytes()==original.read_bytes(),module
  state['installed_modules_sha256'][module]=hashlib.sha256(candidate.read_bytes()).hexdigest()
 env['PYTHONPATH']=str(target);env['CUDA_VISIBLE_DEVICES']=''
 cmd=[python,'-m','fnit.recon_all.native_free','--help'];start=time.perf_counter()
 result=subprocess.run(cmd,cwd=run,env=env,capture_output=True,text=True,check=True);(run/'cli_help.txt').write_text(result.stdout+result.stderr)
 assert '--annotation-gibbs-backend' in result.stdout and '--remesh-scalar-storage' in result.stdout and '--sphere-finish-backend' in result.stdout and '--mni-execution' in result.stdout and '--inflate-backend' in result.stdout and '--normalization-initial-bias-backend' in result.stdout
 state['stages']['installed_cli_help']={'wall_seconds':time.perf_counter()-start,'command':cmd}
 state['packaging_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 state['status']='passed'
except BaseException as e:
 state.update(status='failed',error=repr(e));raise
finally:
 (run/'report.json').write_text(json.dumps(state,indent=2)+'\n')
