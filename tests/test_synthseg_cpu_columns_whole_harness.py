"""Queue control only: mocked child receipts, no compiler/Torch/MRI execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


@pytest.mark.parametrize('failure', ['map_sha', 'compiler_count', 'worker_after_gate', None])
def test_first_mismatch_persisted_before_later_arm(tmp_path, monkeypatch, failure):
    repository = Path(__file__).resolve().parents[1]
    leaf = repository / 'validation/smri_cpu/seg_columns_integration_20261006'
    spec = importlib.util.spec_from_file_location('columns_whole_queue_contract', leaf / 'whole_queue.py')
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    root, run = tmp_path / 'root', tmp_path / 'run'
    root.mkdir()
    group = {'status':'prepared_requires_separate_approval',
             'arm_order':[['A1_baseline','baseline'],['B1_cold','candidate'],['B2_warm','candidate'],['A2_baseline','baseline']],
             'affinity':[0], 'common_lock':'common.lock', 'worker_timeout_seconds':1}
    plan = {'whole_CPU':group, 'validation_sources':{'whole_queue.py':queue.sha(leaf / 'whole_queue.py'),
                                                    'whole_worker.py':queue.sha(leaf / 'whole_worker.py')}}
    plan_path = tmp_path / 'PLAN.json'
    plan_path.write_text(json.dumps(plan))
    calls = []
    def mock_child(command, **kwargs):
        # Only small receipt/byte fixtures: no invocation of the actual worker.
        output = Path(command[command.index('--output') + 1])
        calls.append(output.name)
        output.mkdir(mode=0o700)
        if output.name == 'B1_cold':
            (run / 'fresh_whole_cache').mkdir(mode=0o700)
        (output / 'segmentation.nii.gz').write_bytes(b'changed' if failure == 'map_sha' and output.name == 'B1_cold' else b'same mock bytes')
        (output / 'volumes.csv').write_bytes(b'same mock CSV')
        report = {'valid_complete_arm':not(failure == 'worker_after_gate' and output.name == 'B1_cold'),
                  'compile_calls':int(output.name == 'B1_cold'),
                  'saved_outputs':{file:{'bytes':(output/file).stat().st_size,'sha256':queue.sha(output/file)}
                                   for file in ('segmentation.nii.gz','volumes.csv')}}
        if failure == 'compiler_count' and output.name == 'B1_cold':
            report['compile_calls'] = 0
        (output / 'WHOLE.json').write_text(json.dumps(report))
        return SimpleNamespace(wait=lambda timeout:0, pid=123456789)
    monkeypatch.setattr(queue.subprocess, 'Popen', mock_child)
    monkeypatch.setattr(sys, 'argv', ['whole_queue.py','--root',str(root),'--workspace',str(leaf),
        '--baseline',str(tmp_path/'baseline'),'--candidate',str(tmp_path/'candidate'),'--plan',str(plan_path),
        '--run',str(run),'--device','cpu','--approved-whole'])
    rc = queue.main()
    report = json.loads((run/'QUEUE.json').read_text())
    assert report['jobs'][0]['comparison_executed'] is False
    if failure is None:
        assert rc == 0 and report['status'] == 'complete' and len(calls) == 4
        assert all(job['comparison_executed'] and job['immediate_full_file_SHA_gate_passed'] for job in report['jobs'][1:])
    else:
        assert rc == 1 and calls == ['A1_baseline','B1_cold']
        assert report['status'] == 'failed_postcondition_stop_remaining'
        assert report['failed_arm'] == 'B1_cold' and len(report['jobs']) == 2
        assert report['jobs'][-1]['returncode'] == 0  # Worker exit and controller gate are distinct.
        assert report['jobs'][-1]['postcondition_error_type'] == 'AssertionError'
