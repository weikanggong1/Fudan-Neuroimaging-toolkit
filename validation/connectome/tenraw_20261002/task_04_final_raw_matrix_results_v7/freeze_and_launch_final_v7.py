import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

root = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002')
own = root / 'task_04'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old = own / 'final_raw_matrix_tools_v1/configuration.json'
assert digest(old) == 'bec71f47418e78ed1830e29da96eff66526d272e16c39f03b7f8cdc276d5b07e'
config = json.loads(old.read_text())
view = own / 'explicit_case_map_v2_completed_view'
binding = view / 'case_origin_binding.json'
routing = json.loads(binding.read_text())
assert routing['state'] == 'completed' and routing['execution_completed'] is True
assert len(routing['cases']) == 10
mapper = own / 'explicit_case_map_v1_tools/tools/reference/bind_connectome_raw_reference_case_map.py'
assert digest(mapper) == '1d7916dd621da99dbddb7619d201cec2332864811c8c0996fb7e1e3afb49acbc'
tool = own / 'final_raw_matrix_tools_v7/benchmark_connectome_final_raw_envelope.py'
assert digest(tool) == '9be317df1bc38fdaf993ccb16b29e52f902c681ecbf80a3d85c08521ef4ee235'
config['official_view'] = str(view)
config['official_origin_binding'] = {'path': str(binding), 'sha256': digest(binding)}
config['official_mapper'] = {'path': str(mapper), 'sha256': digest(mapper)}
config['source_files'] = [item for item in config['source_files'] if 'final_raw_matrix_tools_v1/' not in item['path']]
config['source_files'].append({'path': str(tool), 'sha256': digest(tool)})
mixed = json.loads(Path(config['mixed_configuration']['path']).read_text())
origins = json.loads(Path(mixed['GPU_origin_bindings']).read_text())
roots = set()
def walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('run_root', 'root', 'output_root') and isinstance(item, str) and item.startswith('/'):
                roots.add(item)
            walk(item)
    elif isinstance(value, list):
        for item in value:
            walk(item)
walk(origins)
for item in origins['bindings']:
    selected = Path(item['replacement']['root'])
    assert selected.is_dir() and selected.name == item['case_id'] and selected.parent.name == item['arm']
    roots.add(str(selected.parent.parent))
config['selected_protected_roots'] = sorted(roots)
prior_directories = {str(Path(mixed['prior_v2']['path']).parent), mixed['supported_v1_prior_directory']}
prior_summary = str(root / 'root_actual_cohort_summary_v2/summary.json')
config['historical_observation_JSONs'] = [{**item, 'role': 'historical_prior_comparison_observation'} for item in mixed['static_JSON_bindings'] if str(Path(item['path']).parent) in prior_directories or item['path'] == prior_summary]
assert len(config['historical_observation_JSONs']) == 7

path = tool.parent / 'configuration.json'
assert not path.exists()
path.write_text(json.dumps(config, indent=2) + '\n')
output = root.parent / 'fnit_connectome_final_raw_task04_20261003_v7'
log = own / 'final_raw_matrix_twenty_v7.log'
receipt = own / 'final_raw_matrix_twenty_v7.launch.json'
assert not any(p.exists() for p in (output, log, receipt))
assert socket.gethostname() == 'nodecw10'
argv = ['/cwStorage/home/gongwk/anaconda3/bin/python3.11', '-u', str(tool), '--config', str(path),
        '--config-sha256', digest(path), '--output-root', str(output)]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
with log.open('xb') as stream:
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=env)
record = {'PID': child.pid, 'start_ticks': Path(f'/proc/{child.pid}/stat').read_text().split()[21],
          'argv': argv, 'host': socket.gethostname(), 'configuration_sha256': digest(path),
          'tool_sha256': digest(tool), 'official_origin_binding': config['official_origin_binding'],
          'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'log': str(log), 'output_root': str(output), 'MRI_or_GPU_started': False,
          'scope': 'Actual twenty qualified saved FNIT arms versus original five official seeds; CPU statistics only'}
receipt.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
