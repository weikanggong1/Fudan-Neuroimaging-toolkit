"""Private controller: analyze only real completed producers, GPU hidden."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1')
frozen = root / 'formal_frozen_v1'
configuration = frozen / 'accuracy_configuration.json'
expected = 'f8eba1cbf3eab2baa549172b32be2c9d3b1014ad478f6e3702d7987378f52f8c'
if hashlib.sha256(configuration.read_bytes()).hexdigest() != expected:
    raise ValueError('actual frozen configuration changed')
config = json.loads(configuration.read_bytes())
job = root / 'root_matrix_analysis_v1'
job.mkdir(exist_ok=False)
tool = frozen / 'tools_source/tools/analyze_connectome_accuracy_cohort.py'
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(frozen / 'tools_source'), MPLBACKEND='Agg')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    environment[key] = '8'
report = {'status': 'waiting_for_completed_producers', 'pid': os.getpid(),
          'configuration_sha256': expected, 'analysis_tool_sha256': hashlib.sha256(tool.read_bytes()).hexdigest(),
          'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': {}}

def save():
    temporary = job / 'status.json.tmp'
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(job / 'status.json')

def analyze(name, cases):
    output = job / name
    command = [config['gpu_python'], str(tool), '--configuration', str(configuration),
               '--output-dir', str(output), '--case-id', *cases]
    started = time.perf_counter()
    report['status'] = 'running_cpu_analysis'
    report['current_analysis'] = name
    save()
    with (job / (name + '.log')).open('xb') as log:
        result = subprocess.run(command, cwd=frozen / 'tools_source', env=environment,
                                stdout=log, stderr=subprocess.STDOUT)
    actual = {'returncode': result.returncode, 'seconds': time.perf_counter() - started,
              'command': command, 'output_directory': str(output)}
    if (output / 'report.json').is_file():
        actual['report_sha256'] = hashlib.sha256((output / 'report.json').read_bytes()).hexdigest()
    return actual

save()
selected = [item['case_id'] for item in config['execution_order'] if item['version'] == 'candidate']
while True:
    state = json.loads((Path(config['run_root']) / 'status.json').read_bytes())
    for case in selected:
        if case in report['cases']:
            continue
        completed = state['cases'].get('candidate/' + case, {})
        if completed.get('status') == 'completed':
            report['cases'][case] = analyze(case, [case])
            save()
        elif completed.get('status') == 'failed':
            report['cases'][case] = {'status': 'producer_failed', 'actual_error': completed.get('error')}
            save()
    if state['status'] != 'running':
        failures = [case for case in selected if report['cases'].get(case, {}).get('returncode') != 0]
        if not failures:
            report['full_cohort'] = analyze('full_cohort', selected)
            report['status'] = ('analysis_completed' if report['full_cohort']['returncode'] == 0 else 'failed')
        else:
            report.update(status='failed', incomplete_cases=failures)
        report['actual_gpu_execution_status'] = state['status']
        report.pop('current_analysis', None)
        save()
        break
    report['status'] = 'waiting_for_completed_producers'
    report.pop('current_analysis', None)
    save()
    time.sleep(30)
