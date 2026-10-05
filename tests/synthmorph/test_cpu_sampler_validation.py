"""Reject altered runtime records before transferring scientific precision gates."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def scorer(monkeypatch):
    directory = Path(__file__).parents[2] / 'validation/synthmorph/cpu_fixes_20261004'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('fnit_sampler_validation_test', directory / 'compare_sampler_full.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def complete_record(tmp_path):
    folder = tmp_path / 'complete-arm'
    folder.mkdir()
    job = {'id': folder.name, 'argv': ['python', 'worker.py'], 'scope': 'contract fixture'}
    record = {'status': 'complete', 'returncode': 0, 'hostname': 'nodecw7',
        'max_cpu_threads': 8, 'cpu_affinity': '2,6,10,14,18,22,26,30',
        'job': job, 'job_sha256': hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
        'environment': {**{name: '8' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS')}, 'CUDA_VISIBLE_DEVICES': ''},
        'argv': ['taskset', '-c', '2,6,10,14,18,22,26,30', '/usr/bin/time', '-v', '-o',
            str(folder / 'time.txt'), *job['argv']],
        'wall_seconds': 1., 'started_utc': 'fixture', 'finished_utc': 'fixture',
        'load_before': [0., 0., 0.], 'load_after': [0., 0., 0.],
        'maximum_sampled_tree_rss_bytes': 1024, 'maximum_sampled_tree_threads': 1,
        'resource_samples': 4}
    (folder / 'time.txt').write_text('User time (seconds): 1.\n')
    return folder, record


def test_selected_complete_record_is_accepted(scorer, complete_record):
    folder, record = complete_record
    (folder / 'record.json').write_text(json.dumps(record))
    _, actual, resources = scorer.selected_run(folder)
    assert actual['status'] == 'complete'
    assert resources['User time (seconds)'] == '1.'


@pytest.mark.parametrize('field,value', [
    ('status', 'running'), ('returncode', 1), ('hostname', 'other-host'),
    ('max_cpu_threads', 16), ('cpu_affinity', '0-7'), ('job_sha256', 'wrong')])
def test_altered_run_identity_is_rejected(scorer, complete_record, field, value):
    folder, record = complete_record
    record[field] = value
    (folder / 'record.json').write_text(json.dumps(record))
    with pytest.raises(RuntimeError):
        scorer.selected_run(folder)


@pytest.mark.parametrize('environment', ['OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'CUDA_VISIBLE_DEVICES'])
def test_actual_runtime_budget_is_rejected(scorer, complete_record, environment):
    folder, record = complete_record
    record['environment'][environment] = '16'
    (folder / 'record.json').write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='budget or same-node'):
        scorer.selected_run(folder)


def test_argv_does_not_trust_the_recorded_job_only(scorer, complete_record):
    folder, record = complete_record
    record['argv'][-1] = 'other-worker.py'
    (folder / 'record.json').write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='actual command differs'):
        scorer.selected_run(folder)


def test_modified_source_is_rejected_even_when_manifest_is_unchanged(scorer, tmp_path):
    source = tmp_path / 'freeze'
    source.mkdir()
    path = source / 'module.py'
    path.write_bytes(b'original')
    manifest = {'files': {'module.py': hashlib.sha256(path.read_bytes()).hexdigest()}}
    raw = json.dumps(manifest).encode()
    (source / 'source_manifest.json').write_bytes(raw)
    path.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='frozen source changed'):
        scorer.verified_source(source, hashlib.sha256(raw).hexdigest())
