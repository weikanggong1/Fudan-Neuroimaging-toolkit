"""A measurement-only adapter must not redirect the frozen worker's source."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_official_and_frozen_baseline_run_without_candidate_measurement(tmp_path):
    source = tmp_path / 'candidate'
    baseline = tmp_path / 'frozen'
    for root, version in ((source, 'candidate'), (baseline, 'baseline')):
        package = root / 'src/fnit'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text(f'__version__ = {version!r}\n')
    fixture = tmp_path / 'input.txt'
    fixture.write_text('1\n2\n3\n')
    adapter = source / 'adapter.py'
    adapter.write_text('''
from pathlib import Path
import shutil
import sys

def reference_command(case, output, resources):
    return [sys.executable, '-c', 'import shutil,sys;shutil.copyfile(sys.argv[1],sys.argv[2])',
            case['input'], str(output/'result.txt')]

def reference_outputs(case, output, resources):
    return {'result': output/'result.txt'}

def run_case(case, output, device):
    import fnit
    if fnit.__version__ != 'baseline':
        raise RuntimeError('Measurement adapter redirected the frozen source')
    shutil.copyfile(case['input'], output/'result.txt')
    return {'result': output/'result.txt'}
''')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'resources': {}, 'cases': [
        {'id': 'adapter_source_control', 'adapter': 'adapter.py', 'input': str(fixture)}]}))
    report = tmp_path / 'reports'
    script = Path(__file__).resolve().parents[2] / 'tools/benchmark_multimodal_cpu.py'
    subprocess.run([sys.executable, str(script), 'run', '--manifest', str(manifest),
                    '--candidate-root', str(source), '--baseline-root', str(baseline),
                    '--output-dir', str(report), '--cpuset', str(min(os.sched_getaffinity(0))),
                    '--threads', '1', '--backends', 'official,baseline',
                    '--single-observation', '--repetitions', '1', '--api-repetitions', '0'],
                   capture_output=True, check=True, timeout=60)
    suite = json.loads((report / 'suite.private.json').read_text())
    assert suite['status'] == 'executed_with_numeric_comparisons'
    record, = suite['records']
    assert set(record['accuracy']) == {'baseline_vs_official'}
    assert record['accuracy']['baseline_vs_official']['result']['max_absolute_error'] == 0
    worker, = record['worker_results']['baseline']
    assert Path(worker['fnit_source']).is_relative_to(baseline)
    assert record['full_process']['candidate'] == []
