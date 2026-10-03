"""Package completed reports and actual source identities; no scientific recomputation."""
from pathlib import Path
import csv
import hashlib
import io
import json
import tarfile

root = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002/task_04')
out = root.parent.parent/'fnit_connectome_final_raw_task04_20261003_v7'
final = json.loads((out/'final_report.json').read_text())
assert final['status'] == 'completed_twenty_actual_raw_matrix_comparisons'
assert len(final['runs']) == 20 and sum(run['total'] for run in final['runs']) == 4800
package = root/'final_raw_matrix_evidence_v7.tar.gz'
assert not package.exists()
rows = []
timing_rows = []
for run in final['runs']:
    directory = out/(run['arm']+'_'+run['case_id'])
    report = json.loads((directory/'envelope.json').read_text())
    assert hashlib.sha256((directory/'envelope.json').read_bytes()).hexdigest() == run['envelope_sha256']
    inputs = json.loads((directory/'actual_inputs.json').read_text())
    qualification = inputs['qualified_origin']
    official = json.loads(Path(inputs['official_manifest']['path']).read_text())
    timing_rows.append({'arm':run['arm'],'case_id':run['case_id'],
        'statistics_seconds':run['wall_seconds'],
        'FNIT_raw_dwi_cli_seconds':qualification['raw_dwi_cli_total_runtime_seconds'],
        'FNIT_gpu_command_seconds':qualification['gpu_command_wall_seconds'],
        'FNIT_worker_seconds':qualification['worker_wall_seconds'],
        'FNIT_driver_timing_json':json.dumps(qualification['driver_timing'],separators=(',',':')),
        'FNIT_saved_stages_json':json.dumps(qualification['stages'],separators=(',',':')),
        'FNIT_stage_qc_json':json.dumps(qualification['stage_qc'],separators=(',',':')),
        'FNIT_stage_note':qualification['stage_timing_note'],
        'official_five_repeat_tracking_downstream_seconds':official['total_wall_seconds'],
        'official_component_scope':official['timing_scope']})
    for name, profile in report['profiles'].items():
        for field, value in profile['ranges'].items():
            rows.append({'arm':run['arm'], 'case_id':run['case_id'], 'atlas':name, 'field':field,
                         'accepted':value['accepted_count'], 'total':len(value['comparison_accepted']),
                         'status':value['status'], 'details_json':json.dumps(value, separators=(',',':'))})
buffer = io.StringIO()
writer = csv.DictWriter(buffer,fieldnames=list(rows[0]))
writer.writeheader(); writer.writerows(rows)
(out/'matrix_field_summary.csv').write_text(buffer.getvalue())
buffer = io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=list(timing_rows[0]))
writer.writeheader(); writer.writerows(timing_rows)
(out/'timing_summary.csv').write_text(buffer.getvalue())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
atlases=sorted({row['atlas'] for row in rows})
labels=[run['arm']+' '+run['case_id'].replace('sub-','') for run in final['runs']]
values=np.array([[sum(row['accepted'] for row in rows if row['arm']==run['arm'] and row['case_id']==run['case_id'] and row['atlas']==atlas)/30*100
                 for atlas in atlases] for run in final['runs']])
fig,ax=plt.subplots(figsize=(12,9))
heat=ax.imshow(values,vmin=0,vmax=100,cmap='viridis',aspect='auto')
ax.set_xticks(range(8),atlases,rotation=35,ha='right');ax.set_yticks(range(20),labels)
for a in range(20):
    for b in range(8):
        ax.text(b,a,f'{values[a,b]:.0f}',ha='center',va='center',fontsize=8,color='white' if values[a,b]<50 else 'black')
ax.set_title('Actual raw10: six fields x five cross pairs per atlas\nAccepted decisions (%) within observed official envelope')
fig.colorbar(heat,ax=ax,label='Accepted decisions (%)')
fig.tight_layout();fig.savefig(out/'matrix_acceptance.png',dpi=180);plt.close(fig)
files = {p: 'actual/'+str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
for version in (2,3,4,5,6,7):
    for suffix in ('.launch.json','.log'):
        p = root/(f'final_raw_matrix_twenty_v{version}'+suffix)
        files[p] = 'launches/'+p.name
    directory = root/f'final_raw_matrix_tools_v{version}'
    for p in directory.glob('*.json'):
        files[p] = f'frozen_tools_v{version}/'+p.name
    for p in directory.glob('*.py'):
        files[p] = f'frozen_tools_v{version}/'+p.name
    if version < 7:
        directory = root/f'final_raw_matrix_twenty_v{version}'
        for p in directory.glob('*.json'):
            files[p] = f'failed_v{version}/'+p.name
mapping = root/'explicit_case_map_v2_completed_view/case_origin_binding.json'
files[mapping] = 'official_origin/case_origin_binding.json'
route = json.loads(mapping.read_text())
for key, identity in route['source_identity'].items():
    p=Path(identity['path'])
    files[p] = 'official_origin/sources/'+p.name
for case,item in route['cases'].items():
    p=Path(item['reference_manifest']['path'])
    files[p] = f'official_manifests/{case}.json'
for launch in route['launch_bindings'].values():
    p=Path(launch['configuration_identity']['path'])
    files[p] = 'official_launches/'+launch['configuration']['case_ids'][0]+'_'+p.name
index = {'scope':'lossless reports/configs/source/failures; scientific files remain at actual SHA-bound server paths', 'files':[]}
with tarfile.open(package,'x:gz') as archive:
    for p,name in sorted(files.items(), key=lambda pair:pair[1]):
        raw=p.read_bytes()
        index['files'].append({'archive_path':name,'actual_path':str(p),'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        archive.add(p,arcname=name,recursive=False)
    raw=(json.dumps(index,indent=2)+'\n').encode()
    info=tarfile.TarInfo('evidence_index.json'); info.size=len(raw)
    archive.addfile(info,io.BytesIO(raw))
receipt={'package':str(package),'sha256':hashlib.sha256(package.read_bytes()).hexdigest(),
         'size_bytes':package.stat().st_size,'file_count':len(files),'summary_rows':len(rows),
         'accepted':sum(run['accepted'] for run in final['runs']),'total':4800,
         'immutable_input_files':len(json.loads((out/'immutable_input_snapshot.json').read_text())),
         'FNIT_self':'not_assessed','population':'not_assessed',
         'all_scientific_match':final['full_ten_scientific_match']}
(root/'final_raw_matrix_evidence_v7.package.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
