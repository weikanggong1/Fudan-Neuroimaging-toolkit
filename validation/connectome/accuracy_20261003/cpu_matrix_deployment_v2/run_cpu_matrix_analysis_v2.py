"""Private CPU deployment controller; scientific sources remain frozen."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time


ROOT = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1')
FROZEN = ROOT / 'formal_frozen_v1'
CONFIGURATION = FROZEN / 'accuracy_configuration.json'
TOOL = FROZEN / 'tools_source/tools/analyze_connectome_accuracy_cohort.py'
ANALYSIS_PYTHON = '/cwStorage/home/gongwk/anaconda3/bin/python3.11'
CONFIGURATION_SHA256 = 'f8eba1cbf3eab2baa549172b32be2c9d3b1014ad478f6e3702d7987378f52f8c'
TOOL_SHA256 = '02f5e22f91f423f17d8d0ee429e0a702c0ac6a64baa8e76b06238e3c297cf446'
JOB = ROOT / 'root_matrix_analysis_v2'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_binding():
    actual = {'configuration_sha256': sha256(CONFIGURATION), 'analysis_tool_sha256': sha256(TOOL)}
    if actual != {'configuration_sha256': CONFIGURATION_SHA256, 'analysis_tool_sha256': TOOL_SHA256}:
        raise ValueError('actual frozen configuration or scientific analysis source changed')
    return actual


frozen_binding()
if Path(sys.executable).resolve() != Path(ANALYSIS_PYTHON).resolve():
    raise ValueError('controller must use the verified existing analysis Python')
config = json.loads(CONFIGURATION.read_bytes())
JOB.mkdir(exist_ok=False)
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(FROZEN / 'tools_source'),
                   MPLBACKEND='Agg', MPLCONFIGDIR=str(JOB / 'matplotlib_config'))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    environment[key] = '8'
os.environ.update(environment)

# Import the existing reporting packages before starting any case.
import nibabel
import numpy
import scipy
import matplotlib

versions = {'executable': sys.executable, 'python': sys.version,
            'nibabel': nibabel.__version__, 'numpy': numpy.__version__,
            'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__}
v1_files = [ROOT / 'run_cpu_matrix_analysis.py', ROOT / 'root_matrix_analysis_v1/status.json',
            ROOT / 'root_matrix_analysis_v1/sub-CON01/report.json',
            ROOT / 'root_matrix_analysis_v1/sub-CON01.log', ROOT / 'cpu_matrix_controller_v1_stop_receipt.json']
old_failure = json.loads((ROOT / 'root_matrix_analysis_v1/sub-CON01/report.json').read_bytes())
provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'hostname': socket.gethostname(),
              'uid': os.getuid(), 'pid': os.getpid(), 'platform': platform.platform(),
              'command': [sys.executable, str(Path(__file__).resolve())], 'versions': versions,
              **frozen_binding(), 'controller_sha256': sha256(__file__),
              'configured_gpu_python_unchanged': config['gpu_python'],
              'analysis_environment': {key: environment[key] for key in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH',
                  'MPLBACKEND', 'MPLCONFIGDIR', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
              'v1_preserved_files': {str(path): sha256(path) for path in v1_files},
              'v1_failure': {'status': old_failure['status'], 'error': old_failure['error'],
                             'start_utc': old_failure['start_utc'], 'end_utc': old_failure['end_utc']},
              'scope': 'CPU read-only reporting; no scientific source/configuration/package/GPU changes'}
(JOB / 'deployment_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
report = {'status': 'waiting_for_completed_producers', 'pid': os.getpid(),
          **frozen_binding(), 'launcher_sha256': sha256(__file__), 'analysis_python': ANALYSIS_PYTHON,
          'deployment_provenance_sha256': sha256(JOB / 'deployment_provenance.json'), 'cases': {}}


def save():
    temporary = JOB / 'status.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(JOB / 'status.json')


def analyze(name, cases):
    before = frozen_binding()
    output = JOB / name
    command = [ANALYSIS_PYTHON, str(TOOL), '--configuration', str(CONFIGURATION),
               '--output-dir', str(output), '--case-id', *cases]
    started = time.perf_counter()
    report['status'] = 'running_cpu_analysis'
    report['current_analysis'] = name
    save()
    with (JOB / (name + '.log')).open('xb') as log:
        result = subprocess.run(command, cwd=FROZEN / 'tools_source', env=environment,
                                stdout=log, stderr=subprocess.STDOUT)
    actual = {'returncode': result.returncode, 'seconds': time.perf_counter() - started,
              'command': command, 'output_directory': str(output),
              'frozen_binding_before': before, 'frozen_binding_after': frozen_binding()}
    if (output / 'report.json').is_file():
        actual['report_sha256'] = sha256(output / 'report.json')
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
