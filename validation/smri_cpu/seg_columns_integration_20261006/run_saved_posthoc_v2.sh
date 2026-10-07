set -eu
/cwStorage/home/gongwk/Notebook_code/FNIT/envs/default/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,os,subprocess
r=Path('/cwStorage/home/gongwk/Notebook_code/FNIT');w=r/'workspaces/smri_cpu_20261004/remaining_20261006/seg-columns-integration-v1';run=r/'runs/smri_cpu_20261004/remaining_20261006/seg-columns-integration-v1';post=run/'posthoc_v2'
def ident(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
expected={'score_saved_whole_v2.py':'ecb4e08b0ef9d42cdd22c02797fc543f6dc1824476362cb470537777141c3ea6','png_encoder.py':'91923c06c85088c3ec162c0fff58590b029ce304b66928d98fb2abec1299cdd2'}
for name,sha in expected.items():assert ident(w/name)['sha256']==sha
plan=json.loads((w/'PLAN.json').read_text());paths=[]
for variant,folder in [('baseline',r/'repo/src/fnit/synthseg_parc'),('candidate',w/'source_candidate/fnit/synthseg_parc')]:
 for name,sha in plan['production_sources'][variant].items():assert ident(folder/name)['sha256']==sha;paths.append(folder/name)
for folder in (r/'repo/src',w/'source_candidate'):
 for name,sha in plan['common_support_sha256'].items():assert ident(folder/name)['sha256']==sha;paths.append(folder/name)
for group,arms in [('whole_cpu',('A1_baseline','B1_cold','B2_warm','A2_baseline')),('whole_gpu',('A_baseline','B_candidate'))]:
 assert json.loads((run/group/'QUEUE.json').read_text())['status']=='complete'
 paths.append(run/group/'QUEUE.json')
 for arm in arms:
  paths.extend(run/group/arm/name for name in ('WHOLE.json','segmentation.nii.gz','volumes.csv'))
paths.extend(run/name for name in ('CPU_EXIT.json','GPU_EXIT.json'));paths.extend(w/name for name in expected)
assert all(json.loads((run/name).read_text())['returncode']==0 for name in ('CPU_EXIT.json','GPU_EXIT.json'))
assert not post.exists();os.umask(0o077);post.mkdir(mode=0o700)
before={str(p.relative_to(r)):ident(p) for p in paths};(post/'PRESERVATION_BEFORE.json').write_text(json.dumps(before,indent=2)+'\n')
command=['/usr/bin/flock','-w','23000',str(r/'runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock'),'/usr/bin/taskset','-c','32,36,40,44,48,52,56,60','/usr/bin/timeout','--kill-after=30','180',str(r/'envs/default/bin/python'),str(w/'score_saved_whole_v2.py'),'--root',str(r),'--workspace',str(w),'--run',str(run),'--output',str(post/'POSTHOC.json')]
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8'}
for name in ('PYTHONPATH','LD_PRELOAD','LD_LIBRARY_PATH','OPENBLAS_CORETYPE'):env.pop(name,None)
supervisor="""import hashlib,json,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[3]);post=Path(sys.argv[2]).parent;t=time.monotonic();rc=subprocess.call(json.loads(sys.argv[1]));before=json.loads((post/'PRESERVATION_BEFORE.json').read_text());after={name:{'bytes':(r/name).stat().st_size,'sha256':hashlib.sha256((r/name).read_bytes()).hexdigest()} for name in before};same=after==before
(post/'PRESERVATION_AFTER.json').write_text(json.dumps(after,indent=2)+'\\n')
Path(sys.argv[2]).write_text(json.dumps({'schema':'fnit_columns_posthoc_exit/v2','returncode':rc,'outer_seconds':time.monotonic()-t,'new_inference_native_GPU_calls':0,'all_original_six_arms_sources_exits_unchanged':same})+'\\n');raise SystemExit(rc if rc else (0 if same else 91))
"""
with (post/'posthoc.log').open('x') as log:
 child=subprocess.Popen([str(r/'envs/default/bin/python'),'-c',supervisor,json.dumps(command),str(post/'EXIT.json'),str(r)],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
x={'schema':'fnit_columns_posthoc_dispatch/v2','controller_PID':child.pid,'scope':'Read-only saved output numerical scores and CC0 discrete RGB PNG; zero new inference/native/GPU','worker_sources':{name:ident(w/name) for name in expected},'worker_timeout_seconds':180,'model_native_or_GPU_calls':0,'local_PNG_contract_positive':4,'local_PNG_contract_negative':3,'preservation_file_count':len(before)}
(post/'DISPATCH.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
PY
