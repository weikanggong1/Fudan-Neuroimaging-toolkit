import sys,json,subprocess,time,hashlib
from pathlib import Path
target=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(target))
import torch
def precision():
    return dict(cuda_initialized=torch.cuda.is_initialized(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
before=precision()
import fnit.recon_all.native_free as pipeline
import fnit.recon_all.topology_genetic_files as ga
from fnit.recon_all.python_profile import python_profile_options
assert Path(pipeline.__file__).resolve().is_relative_to(target)
assert Path(ga.__file__).resolve().is_relative_to(target)
profile=python_profile_options(device='cuda:0',threads=4,topology_backend='python-ga')
assert profile['topology_backend']=='python-ga' and len(profile)==21 and 'native' not in profile.values()
after=precision()
assert before==after
cli=[]
for module,required in [('fnit.recon_all.native_free',['python-ga','--topology-backend','--backend']),
                       ('fnit.recon_all.topology_genetic_files',['--final-orig','--remesh-scalar-storage','--threads'])]:
    tick=time.perf_counter()
    command=[sys.executable,'-c',"import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module(sys.argv.pop(1),run_name='__main__')",str(target),module,'--help']
    completed=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
    assert completed.returncode==0 and all(option in completed.stdout for option in required)
    cli.append(dict(module=module,exit_code=completed.returncode,
        fresh_cli_seconds_including_imports=time.perf_counter()-tick,
        output_sha256=hashlib.sha256(completed.stdout.encode()).hexdigest(),verified_options=required))
report=dict(status='passed',python_version=sys.version.split()[0],torch_version=torch.__version__,
    selected_profile=profile,precision_and_cuda_at_entry=before,
    precision_and_cuda_after_import_and_profile_resolution=after,cli=cli)
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n')
