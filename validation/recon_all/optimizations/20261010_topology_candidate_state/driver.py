from pathlib import Path
import hashlib,json,time,subprocess,sys,os,platform
import numpy as np
from benchmark_recon_topology_candidate_state import load_state_capture,dump_state_checkpoint
w=Path(__file__).parent
root=w.parents[1]
d=root/'runs/topology_candidate_state_20261010_v1'
if d.exists():raise FileExistsError(d)
d.mkdir(parents=True)
manifest=json.loads((w/'SOURCE_MANIFEST.json').read_text())
for rel,sha in manifest['files'].items():
 if hashlib.sha256((w/rel).read_bytes()).hexdigest()!=sha:raise ValueError('source SHA '+rel)
reference=w/'captures'
raw=json.loads((reference/'receipt.private.json').read_text())
if (reference/'EXIT').read_text().strip()!='0' or raw['status']!='complete_full_restore_reference':raise ValueError('reference exit')
for receipt in raw['references']:
 for rel,sha in receipt['files_sha256'].items():
  if hashlib.sha256((reference/receipt['label']/rel).read_bytes()).hexdigest()!=sha:raise ValueError('capture SHA '+rel)
started=time.perf_counter();converted=d/'converted';converted.mkdir()
for repeat in [0,1]:
 for name in ['record','before_restore','restored']:
  state,selection,meta=load_state_capture(reference/f'repeat{repeat}'/f'state.{name}.bin')
  dump_state_checkpoint(converted/f'{repeat}-{name}.npz',state)
  if repeat==0 and name=='record':
   np.savez_compressed(converted/'selection.npz',**selection)
   (d/'record_metadata.private.json').write_text(json.dumps(meta,indent=2)+'\n')
convert_seconds=time.perf_counter()-started
cmd=[sys.executable,str(w/'tools/benchmark_recon_topology_candidate_state.py'),'--record-state',str(converted/'0-record.npz'),'--candidate-state',str(converted/'0-before_restore.npz'),'--reference-state',str(converted/'0-restored.npz'),'--reference-repeat',str(converted/'1-restored.npz'),'--selection',str(converted/'selection.npz'),'--source-manifest',str(w/'SOURCE_MANIFEST.json'),'--threads','4','--profile','chull','--output',str(d/'candidate')]
t=time.perf_counter()
with (d/'stdout.log').open('wb') as out,(d/'stderr.log').open('wb') as err:result=subprocess.run(cmd,stdout=out,stderr=err,check=False)
cli_seconds=time.perf_counter()-t
receipt=dict(status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,conversion_seconds=convert_seconds,fresh_cli_wall_seconds=cli_seconds,hostname=platform.node(),cpu_affinity=sorted(os.sched_getaffinity(0)),source_manifest=manifest,source_manifest_sha256=hashlib.sha256((w/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),reference_receipt_sha256=hashlib.sha256((reference/'receipt.private.json').read_bytes()).hexdigest(),reference_and_candidate_cross_host=True,speed_ratio_not_assessed=True,face_only_source='not_executed_in_first_candidate')
(d/'receipt.private.json').write_text(json.dumps(receipt,indent=2)+'\n');(d/'EXIT').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
