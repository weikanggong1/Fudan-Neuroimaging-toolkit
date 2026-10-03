"""Root on headcw only: persistent metadata wait, then original CPU readers."""
from pathlib import Path
import os,json,hashlib,subprocess,datetime
r=Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002')
chain=r/'formal_actual_mixed_final_CPU_tools_v1/run_connectome_actual_mixed_final_cpu.py'
common=chain.parent/'watch_connectome_actual_completion.py'
configuration=r/'formal_actual_mixed_waiter_tools_v1/configuration.json'
mixed=r/'formal_actual_mixed_waiter_tools_v1/tools/watch_connectome_actual_mixed_completion.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(chain)=='5b09fd00d8967a1ff39968acb03e62c19ffe1b5b833fa0e2ce33ffd1b7e28c09' and sha(common)=='6a9bb15609fd1d1fddf2931f6b598ff344c0e1306cefc75420ae90f1bd390cf9'
assert sha(configuration)=='d4e443a6045aee2a475ba5b56051145acb209cb6ba5ff7405877449bf70d2dbe' and sha(mixed)=='dfe59311f8015fc04e2286f7b72fd7df3690b832849f2d367fca269453eac21f'
c=json.loads(configuration.read_bytes())
preflight=r/'root_actual_mixed_waiter_preflight_v1/status.json';p=json.loads(preflight.read_bytes())
assert p['status'] in ('pending_actual_v4_driver','pending_actual_v4_completed_subset','pending_finalized_v4_origins','ready_actual_finalized_twenty_reports')
assert p['configuration_sha256']=='d4e443a6045aee2a475ba5b56051145acb209cb6ba5ff7405877449bf70d2dbe'
root=r/'root_actual_mixed_final_CPU_chain_v1';waiter=r/'root_actual_mixed_waiter_driver_v1';export=r/'root_actual_cohort_export_v3'
log=root.with_suffix('.log');receipt=root.with_suffix('.launch.json')
assert not any(path.exists() for path in (root,waiter,export,log,receipt,Path(c['comparison_report_dir']),Path(c['summary_report_dir'])))
argv=[c['CPU_python'],str(chain),'--configuration',str(configuration),'--mixed-tool',str(mixed),'--mixed-report-dir',str(waiter),'--report-dir',str(root),'--export-report-dir',str(export),'--representative-cases','sub-CON01','sub-CON09','--figure-atlas','fs-aparc']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
with log.open('xb') as stream:
 child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
 ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]
 proof={'PID':child.pid,'start_ticks':ticks,'argv':argv,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'chain_sha256':'5b09fd00d8967a1ff39968acb03e62c19ffe1b5b833fa0e2ce33ffd1b7e28c09','configuration_sha256':'d4e443a6045aee2a475ba5b56051145acb209cb6ba5ff7405877449bf70d2dbe','mixed_entry_sha256':'dfe59311f8015fc04e2286f7b72fd7df3690b832849f2d367fca269453eac21f','log':str(log),'scope':'six explicit eligible v3 cases plus four real normal completed v4 cases, conditional unchanged frozen CPU comparison/summary/export; no GPU/MRI work; original failed v3 controller/08/report chain retained','selected_attempts':c['selected_attempts'],'prior_v1':next(x for x in c['static_JSON_bindings'] if x['path'].endswith('/root_actual_cohort_comparison_v1/status.json')),'prior_v2':c['prior_v2'],'prior_v2_pairs':c['prior_v2_pairs'],'outputs':{'chain_status':str(root/'status.json'),'mixed_waiter_status':str(waiter/'status.json'),'comparison':c['comparison_report_dir'],'summary':c['summary_report_dir'],'export':str(export)}}
 receipt.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
