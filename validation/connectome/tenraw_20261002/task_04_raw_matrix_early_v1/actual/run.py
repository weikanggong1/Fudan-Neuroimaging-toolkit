import json,pathlib,hashlib,subprocess,time,datetime,os,sys
p=pathlib.Path(__file__).parent;c=json.loads((p/'config.json').read_text())
sha=lambda q:hashlib.sha256(pathlib.Path(q).read_bytes()).hexdigest()
assert not os.environ.get('CUDA_VISIBLE_DEVICES','')
for f,h in c['source_files'].items():assert sha(p/f)==h
summary={'scope':'read-only early same-raw final matrix envelope; no GPU or scientific solver','configuration_sha256':sha(p/'config.json'),'python':sys.executable,'python_binary_sha256':sha(sys.executable),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':[]}
for arm,base in c['arms'].items():
 for cid in c['cases']:
  dest=p/f'{arm}_{cid}';dest.mkdir(exist_ok=False)
  report=pathlib.Path(c['remote_root'])/base/arm/cid/'gpu_report.json'
  data=json.loads(report.read_text());(dest/'gpu_report.original.json').write_bytes(report.read_bytes())
  assert data['status']=='completed' and data['exit_code']==0
  eligibility=data.get('memory_budget',{})
  argv=[sys.executable,str(p/'tools/benchmark_connectome_raw_cohort_envelope.py'),'--official-root',c['official_view']+'/'+cid,'--fnit',str(report.parent/'connectome'),'--fnit-gpu-reports',str(report),'--raw-manifest',c['raw_manifest']['path'],'--raw-manifest-sha256',c['raw_manifest']['sha256'],'--case-id',cid,'--fnit-seeds','0','--output',str(dest/'envelope.json')]
  rec={'arm':arm,'case_id':cid,'argv':argv,'gpu_report':{'path':str(report),'sha256':sha(report)},'memory_budget':eligibility,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  t=time.monotonic();r=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);rec.update(returncode=r.returncode,wall_seconds=time.monotonic()-t,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  (dest/'stdout.log').write_text(r.stdout);(dest/'stderr.log').write_text(r.stderr)
  rec['logs']={n:{'path':str(dest/n),'sha256':sha(dest/n)} for n in ('stdout.log','stderr.log')}
  if r.returncode==0:
   d=json.loads((dest/'envelope.json').read_text());rec.update(matrix_envelope_status=d['matrix_envelope_status'],fnit_reproducibility_status=d['fnit_reproducibility_status'],population_envelope_status=d['population_envelope_status'],accepted=sum(v['accepted_count'] for x in d['profiles'].values() for v in x['ranges'].values()),total=sum(len(v['comparison_accepted']) for x in d['profiles'].values() for v in x['ranges'].values()),envelope_sha256=sha(dest/'envelope.json'))
  else:rec['guard_failure']=r.stderr[-5000:]
  assert sha(report)==rec['gpu_report']['sha256']
  (dest/'execution.json').write_text(json.dumps(rec,indent=2)+'\n');summary['runs'].append(rec);(p/'status.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:rec.get(k) for k in ('arm','case_id','returncode','wall_seconds','accepted','total','matrix_envelope_status','fnit_reproducibility_status','guard_failure')}),flush=True)
summary['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();summary['finished']=True;summary['all_sources_unchanged']=all(sha(p/f)==h for f,h in c['source_files'].items());(p/'status.json').write_text(json.dumps(summary,indent=2)+'\n')
