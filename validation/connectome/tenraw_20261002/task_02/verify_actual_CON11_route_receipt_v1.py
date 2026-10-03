"""Run existing read-only packing guard; freeze actual receipt, never start modeling."""
import argparse,datetime,importlib.util,json,socket
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=Path(a.root);own=r/'task_02';route_root=r/'root_CON11_packing_task02_route_v1';guard=own/'validate_CON11_packing_route_v1.py'
spec=importlib.util.spec_from_file_location('guard',guard);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
config_path=own/'official_modeling_CPU_budget_raw10_v2.config.json';source_path=own/'official_modeling_cohort_cpu_v3.py';assert g.sha(config_path)=='7ec2008b29c077797c259cad1245c8d2f9999c86af621d0821c9bc7d52335627';assert g.sha(source_path)=='616b3f01197447da8255c20592165f066f03bc20b48765b9612ea6ada29d11c1';assert g.sha(guard)=='debcce670fe87ce84f174e299377eba9e18b937872ee7b9ef23eeaf183ecec71'
route_path=route_root/'actual_packing_route.json';expected_path=route_root/'expected_baseline_lineage.json';route=json.loads(route_path.read_text());expected=json.loads(expected_path.read_text());config=json.loads(config_path.read_text())
proof=g.validate_actual_CON11_packing_route(route,config,expected);assert proof and proof['modeling_ready'] is False
old=Path(config['formal_FNIT_packing_root'])/'sub-CON11/connectome/preproc/topup/B0_AP_PA.nii.gz';assert not old.exists() and not old.is_symlink()
bound=lambda path:{'path':str(path),'sha256':g.sha(path)}
receipt={'state':'actual_CON11_packing_route_guard_passed_CPU_completion_pending','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'modeling_ready':False,'subset_modeling_launched':False,'proof':proof,'bindings':{k:bound(v) for k,v in {'route':route_path,'expected_baseline_lineage':expected_path,'frozen_modeling_config':config_path,'frozen_modeling_source':source_path,'packing_guard':guard,'receipt_tool':Path(__file__)}.items()},'old_expected_path':str(old),'old_expected_path_absent_and_not_symlink':True,'scope':'Actual full-voxel/raw SHA packing origin validation only; independent completed verified CPU rawprep still required. No modeling namespace created.'}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:f.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
out.chmod(0o444);print(json.dumps({'state':receipt['state'],'receipt':bound(out),'route':receipt['bindings']['route'],'expected_lineage':receipt['bindings']['expected_baseline_lineage'],'modeling_ready':False},indent=2))
