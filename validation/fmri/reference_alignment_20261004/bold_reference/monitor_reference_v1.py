from pathlib import Path
import datetime,hashlib,json,subprocess,time,os
r=Path('/cwStorage/home/gongwk/Notebook_code/FNIT/runs/fmri_reference_alignment_20261004/bold_reference_v1');state=r/'candidate.queue.private.json';samples=r/'candidate.gpu_observation.private.jsonl';rows=[]
while True:
 q=json.loads(state.read_text());pid=q.get('child_pid');case=q.get('current_case')
 t=datetime.datetime.now(datetime.timezone.utc).isoformat()
 gpu=subprocess.run(['nvidia-smi','--id=0','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
 row={'utc':t,'case':case,'own_pid':pid,'queue_status':q['status'],'gpu_csv':gpu.stdout.strip(),'process_csv':apps.stdout.strip(),'gpu_exit_code':gpu.returncode,'process_exit_code':apps.returncode}
 rows.append(row)
 with samples.open('a') as stream:stream.write(json.dumps(row)+'\n')
 if q['status'] in ('complete','failed'):break
 time.sleep(1)
summary={'status':'complete','scope':'independent 1-second nvidia-smi observation; sampled own-process peak, not continuous peak; shared GPU load recorded','observer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'samples':len(rows),'first_utc':rows[0]['utc'],'last_utc':rows[-1]['utc'],'cases':{}}
for case in ('CON01','CON06'):
 own=[];util=[];globalmem=[];uuids=set();errors=0;waiting=0
 for row in rows:
  if row['case']!=case:continue
  if row['gpu_exit_code'] or row['process_exit_code']:errors+=1;continue
  fields=[x.strip() for x in row['gpu_csv'].split(',')]
  if len(fields)!=3:errors+=1;continue
  uuid=fields[0];current=[]
  for line in row['process_csv'].splitlines():
   values=[x.strip() for x in line.split(',')]
   if len(values)==3 and values[0]==str(row['own_pid']) and values[1]==uuid and values[2].isdigit():current.append(int(values[2])*1024**2)
  if current:
   own.append(sum(current));util.append(int(fields[2]));globalmem.append(int(fields[1])*1024**2);uuids.add(uuid)
  else:waiting+=1
 summary['cases'][case]={'own_gpu_active_samples':len(own),'no_own_gpu_memory_samples':waiting,'observation_errors':errors,'sampled_own_peak_bytes':max(own) if own else None,'sampled_own_peak_decimal_GB':max(own)/1e9 if own else None,'under_20_decimal_GB_sampled':max(own)<=20_000_000_000 if own else None,'shared_gpu_utilization_mean_percent':sum(util)/len(util) if util else None,'shared_gpu_utilization_min_percent':min(util) if util else None,'shared_gpu_utilization_max_percent':max(util) if util else None,'shared_gpu_memory_used_max_bytes':max(globalmem) if globalmem else None,'actual_gpu_uuids':sorted(uuids)}
(r/'candidate.gpu_observation.public.json').write_text(json.dumps(summary,indent=2)+'\n')
