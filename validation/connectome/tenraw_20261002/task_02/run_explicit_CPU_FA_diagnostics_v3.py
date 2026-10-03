from pathlib import Path
import json,hashlib,os,subprocess,sys,socket,argparse
parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--case-map',required=True);parser.add_argument('--output',required=True);parser.add_argument('--subjects',nargs='+',required=True);args=parser.parse_args()
r=Path(args.root);own=r/'task_02';mapping=json.loads(Path(args.case_map).read_text());out=Path(args.output);out.mkdir(exist_ok=False)
source=r/'baseline_raw_compatible_v3/src/fnit/connectome/response.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(source)=='258938a5f20ee7a6efc2fc1c50af5c93412030a60b78036438e8473f072e3b91'
freeze={'actual_host':socket.gethostname(),'GPU':False,'threads':8,'FNIT_source':str(source),'source_sha256':sha(source),'diagnostic_tool_sha256':sha(own/'compare_tensor_cpu.py'),'scope':'real official corrected DWI same-input CPU FA diagnostic; no GPU/full raw benchmark claim','commands':[]}
for s in args.subjects:
 record=mapping['actual_completed_case_map'][s];cp=Path(record['consumer_contract']['path']);assert sha(cp)==record['consumer_contract']['sha256'];ref=cp.parent;c=json.loads(cp.read_text());assert c['execution_completed'] and c['state']=='completed'
 case=out/('sub-'+s);case.mkdir();config={'subject':s,'manifest':str(r/'task_01/final_manifest.json'),'dwi':c['files']['official_corrected_dwi']['path'],'gradient':c['files']['official_gradient_mrtrix']['path'],'brain_mask':c['files']['brain_mask']['path']};checkpoint=case/'same_input_config.json';checkpoint.write_text(json.dumps(config,indent=2)+'\n')
 cmd=[sys.executable,str(own/'compare_tensor_cpu.py'),'--checkpoint',str(checkpoint),'--baseline',str(source.parent),'--official',str(ref),'--output',str(case/'tensor_same_input')]
 freeze['commands'].append({'subject':s,'command':cmd,'consumer_contract_sha256':sha(ref/'consumer_contract.json')});(out/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8');result=subprocess.run(cmd,env=env,capture_output=True,text=True);(case/'diagnostic.stdout').write_text(result.stdout);(case/'diagnostic.stderr').write_text(result.stderr)
 if result.returncode:raise RuntimeError(f'{s} CPU diagnostic failed, inspect actual stderr')
 report=json.loads((case/'tensor_same_input/report.json').read_text());print(json.dumps({'subject':s,'fit_s':report['fit_s'],'fa':report['fa'],'direction_antipodal_degrees':report['direction_antipodal_degrees']}),flush=True)
