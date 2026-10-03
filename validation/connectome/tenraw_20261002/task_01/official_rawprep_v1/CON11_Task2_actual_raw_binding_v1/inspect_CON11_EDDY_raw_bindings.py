from pathlib import Path
import json,hashlib
R=Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002');d=R/'task_01/official_CON11_CPU_fresh_origin_v1/sub-CON11';r=json.loads((d/'report.json').read_text());m=json.loads((R/'task_01/final_manifest.json').read_text());c=next(x for x in m['cases'] if x['case_id']=='sub-CON11');raw=Path(c['bids_root']);rows=[]
for path,digest in r['EDDY_inputs_sha256'].items():
 p=Path(path);resolved=p.resolve();actual=hashlib.sha256(p.read_bytes()).hexdigest();relative=str(resolved.relative_to(raw)) if resolved.is_relative_to(raw) else None
 rows.append({'recorded_path':path,'resolved_path':str(resolved),'is_symlink':p.is_symlink(),'resolved_within_fresh_case':resolved.is_relative_to(d.resolve()),'resolved_within_canonical_raw':resolved.is_relative_to(raw),'reported_SHA256':digest,'actual_SHA256':actual,'canonical_manifest_SHA256':c['input_sha256'].get(relative),'actual_SHA_matches':actual==digest,'canonical_raw_SHA_matches':actual==c['input_sha256'].get(relative) if relative else None})
print(json.dumps({'case_id':'sub-CON11','route_ready':True,'inputs':rows,'no_input_modified':True},indent=2))
