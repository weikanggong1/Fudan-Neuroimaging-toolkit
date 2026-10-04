"""Reuse owned-tree stage controller under an additional GPU1 advisory lock."""
import argparse,fcntl,json,pathlib,time
import stage_benchmark_controller as base

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=pathlib.Path);a=p.parse_args();raw=json.loads(a.config.read_text())
 if raw.get('benchmark_kind')!='metrics_roi_stage' or raw.get('minimum_free_bytes',0)<20_000_000_000:raise ValueError('explicit metrics+ROI stage and20GB admission required')
 if raw.get('lock')!='/tmp/fnit-shared-benchmark.lock' or raw.get('gpu_lock')!='/tmp/fnit-stage-gpu1-benchmark.lock':raise ValueError('both fixed shared/GPU1 locks required')
 original=base.validate_config
 def validated(value):
  value=dict(value);value['benchmark_kind']='annotation_stage';out=original(value);out['benchmark_kind']='metrics_roi_stage';return out
 base.validate_config=validated
 # Lock persists around base.run, including every descendant cleanup. base acquires
 # shared nonblocking and checks free>=20GB before/under it; releases shared if low.
 with pathlib.Path(raw['gpu_lock']).open('a+') as gpu_lock:
  deadline=time.monotonic()+raw.get('maximum_wait_seconds',3600)
  while True:
   try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:
    if time.monotonic()>deadline:raise TimeoutError('GPU1 advisory lock wait exceeded')
    time.sleep(1)
  try:return base.run(a.config)
  finally:fcntl.flock(gpu_lock,fcntl.LOCK_UN)
if __name__=='__main__':raise SystemExit(main())
