"""Private CPU-only all-ten original-byte collection readiness watcher.

No MRI calculation. The existing collector supplies the scientific/provenance
gates. A genuine producer failure stops this watcher with no fabricated results.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1')
WATCH = ROOT / 'root_component_collection_all10_wait_v1'
CONFIG = ROOT / 'formal_frozen_v1/accuracy_configuration.json'
CONFIG_SHA = 'f8eba1cbf3eab2baa549172b32be2c9d3b1014ad478f6e3702d7987378f52f8c'
HELPER = ROOT / 'root_component_tools_v1/validation/connectome/accuracy_20261003/root/compare_raw_components.py'
HELPER_SHA = '08015c11c2662cb2f8115843414ed9148c1dadc60c175b66dcf20f51381c18ce'
CONTROL = HELPER.with_name('control_raw_components.py')
CONTROL_SHA = '9499c20c21b313a008e115b4b25df413638fcfd648d7d1ac14e6b3828d15b999'
COLLECTOR = ROOT / 'root_component_collection_tools_v1/validation/connectome/accuracy_20261003/root/collect_raw_component_evidence.py'
COLLECTOR_SHA = '35496ed07b66e269e97b50efaaaad4d2ce4081fe78ce25b61dbbe22dead64a99'
COLLECTED = ROOT / 'root_component_evidence_collection_all10_v1'
ANALYSIS = ROOT / 'root_component_analysis_v1'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

def utc():
    return datetime.now(timezone.utc).isoformat()

def record(p):
    p = Path(p)
    return {'path': str(p), 'sha256': sha(p), 'size_bytes': p.stat().st_size}

def save(x):
    t = WATCH / f'status.tmp-{os.getpid()}.json'
    t.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')
    os.replace(t, WATCH / 'status.json')

def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert socket.gethostname() == 'nodecw10'
    assert not (WATCH/'status.json').exists()
    assert not COLLECTED.exists()
    sources = [(CONFIG, CONFIG_SHA), (HELPER, HELPER_SHA), (CONTROL, CONTROL_SHA), (COLLECTOR, COLLECTOR_SHA),
               (Path(__file__), sha(__file__))]
    for p, s in sources:
        assert sha(p) == s, str(p)
    specification = importlib.util.spec_from_file_location('raw_components_readiness_control', CONTROL)
    control = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(control)
    config = json.loads(CONFIG.read_bytes())
    pairs = control.plan(config)
    assert len(pairs) == 12
    keys = [f'{v}/{c}' for v, c in pairs]
    config_record = record(CONFIG)
    status = {'schema_version': 1, 'status': 'waiting_for_actual_all12_and_all10_summary',
              'scientific_parity': 'not_assessed', 'hostname': socket.gethostname(), 'PID': os.getpid(),
              'UTC_started': utc(), 'poll_seconds': 45, 'environment': {k: os.environ.get(k) for k in
                  ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
              'watcher': record(__file__), 'configuration': config_record, 'helper': record(HELPER),
              'controller': record(CONTROL), 'collector': record(COLLECTOR), 'execution_keys': keys,
              'collection_output_dir': str(COLLECTED),
              'scope': 'CPU status reads and existing strict original-byte collector only; no MRI/GPU/official solver'}
    prior = None
    while True:
        for p, s in sources:
            assert sha(p) == s, str(p)
        producer = json.loads((Path(config['run_root'])/'status.json').read_bytes())
        analysis = json.loads((ANALYSIS/'status.json').read_bytes())
        control.check_producer_state(producer, config, config_record)
        assert control.same_file(analysis['configuration'], config_record)
        assert control.same_file(analysis['helper'], record(HELPER))
        assert control.same_file(analysis['controller'], record(CONTROL))
        readiness = {'raw_rows': {k: producer['cases'].get(k, {}).get('status') for k in keys},
                     'component_rows': {k: analysis['cases'].get(k, {}).get('status') for k in keys},
                     'all10_summary': analysis.get('all10_summary')}
        if readiness != prior:
            status['actual_readiness'] = readiness
            status['UTC_changed'] = utc()
            save(status)
            print(json.dumps({'UTC': status['UTC_changed'], 'raw_rows': readiness['raw_rows'],
                              'component_rows': readiness['component_rows'],
                              'all10_summary': readiness['all10_summary']}, allow_nan=False), flush=True)
            prior = readiness
        if any(v == 'failed' for v in readiness['raw_rows'].values()):
            status['status'] = 'actual_producer_failed_no_collection_or_metrics_created'
            status['UTC_stopped'] = utc()
            save(status)
            return
        if any(v == 'analysis_failed' for v in readiness['component_rows'].values()):
            status['status'] = 'actual_component_analysis_failed_no_collection_or_metrics_created'
            status['UTC_stopped'] = utc()
            save(status)
            return
        if all(v == 'completed' for v in readiness['raw_rows'].values()) and readiness['all10_summary'] is not None:
            # Bind the existing aggregate bytes, never invent a completed summary.
            control.bound(readiness['all10_summary'])
            command = [sys.executable, str(COLLECTOR), '--analysis-dir', str(ANALYSIS),
                       '--configuration', str(CONFIG), '--configuration-sha256', CONFIG_SHA,
                       '--helper-sha256', HELPER_SHA, '--output-dir', str(COLLECTED)]
            status['status'] = 'collecting_actual_completed_ten_candidates'
            status['command'] = command
            status['UTC_collection_started'] = utc()
            save(status)
            started = time.perf_counter()
            with (WATCH/'collector.stdout.log').open('xb') as out, (WATCH/'collector.stderr.log').open('xb') as err:
                result = subprocess.run(command, env=dict(os.environ), stdout=out, stderr=err)
            status['collector_returncode'] = result.returncode
            status['collector_process_wall_seconds'] = time.perf_counter()-started
            status['stdout'] = record(WATCH/'collector.stdout.log')
            status['stderr'] = record(WATCH/'collector.stderr.log')
            for p, s in sources:
                assert sha(p) == s, str(p)
            current = json.loads((Path(config['run_root'])/'status.json').read_bytes())
            assert all(current['cases'].get(k) == producer['cases'].get(k) for k in keys)
            if result.returncode == 0:
                manifest = COLLECTED/'collection.json'
                report = json.loads(manifest.read_bytes())
                expected_keys = sorted(k for k in keys if k.startswith('candidate/'))
                assert report['coverage'] == expected_keys and report['status'] == 'all_ten_actual_candidate_evidence_collected'
                assert report['scientific_parity'] == 'not_assessed'
                status['status'] = 'actual_all10_original_bytes_collected'
                status['collection'] = record(manifest)
                status['actual_all10_summary'] = readiness['all10_summary']
            else:
                status['status'] = 'collector_failed_preserved_no_metrics_fabricated'
            status['UTC_stopped'] = utc()
            save(status)
            return
        time.sleep(45)

if __name__ == '__main__':
    try:
        run()
    except Exception:
        traceback.print_exc()
        failure = {'schema_version': 1, 'status': 'watcher_failed_preserved', 'scientific_parity': 'not_assessed',
                   'UTC': utc(), 'hostname': socket.gethostname(), 'PID': os.getpid(), 'traceback': traceback.format_exc()}
        (WATCH/'watcher_failure.json').write_text(json.dumps(failure, indent=2, allow_nan=False)+'\n')
        raise
