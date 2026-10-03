"""Staged demonstration cache guards; these small fixtures are not MRI benchmarks."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def staged_runner():
    path = Path(__file__).with_name('run_surface_backend_demo.py')
    spec = importlib.util.spec_from_file_location('staged_backend_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(tmp_path, runner, monkeypatch, *, default_command=False):
    source = tmp_path / 'source.nii.gz'
    source.write_bytes(b'contract-source')
    old_bin = tmp_path / 'oldbin'
    new_bin = tmp_path / 'newbin'
    old_bin.mkdir()
    new_bin.mkdir()
    for directory in (old_bin, new_bin):
        for name in ('recon-all', 'mris_expand'):
            path = directory / name
            path.write_bytes(b'#!/bin/sh\nexit 0\n')
            path.chmod(0o700)
    command_alias = tmp_path / 'recon-all-selected'
    expand_alias = tmp_path / 'expand-selected'
    command_alias.symlink_to(old_bin / 'recon-all')
    expand_alias.symlink_to(old_bin / 'mris_expand')
    options = {'mris_expand_command': str(expand_alias)}
    if not default_command:
        options['command'] = str(command_alias)
    monkeypatch.setenv('PATH', str(old_bin))
    producer = {'adapter': 'fixed-source-bytes'}
    monkeypatch.setattr(runner.reconstruction_adapter, '_producer_fingerprint', lambda backend: producer)
    monkeypatch.setattr(runner.reconstruction_adapter, '_cached_report', lambda manifest, request, subject: True)
    root = tmp_path / 'owned'
    root.mkdir()
    manifest = {'owner': runner.reconstruction_adapter._OWNER, 'schema': 1,
        'status': 'complete', 'subject_dir': str(root / 'subject'),
        'request': {'backend': 'freesurfer', 'source_t1w': str(source),
            'source_sha256': runner.reconstruction_adapter._sha256(source),
            'device': 'cuda:0', 'options': options, 'producer': producer,
            'effective': {'command': str(old_bin / 'recon-all'),
                'command_sha256': runner.reconstruction_adapter._sha256(old_bin / 'recon-all'),
                'mris_expand': str(old_bin / 'mris_expand'),
                'mris_expand_sha256': runner.reconstruction_adapter._sha256(old_bin / 'mris_expand')}}}
    (root / runner.reconstruction_adapter._MANIFEST).write_text(json.dumps(manifest))
    config = {'recon_all_backend': 'freesurfer', 'recon_all_output_dir': str(root), 'device': 'cuda:0'}
    return config, source, options, command_alias, expand_alias, new_bin


def test_same_resolved_binary_and_request_passes(tmp_path, staged_runner, monkeypatch):
    config, source, options, *_ = configure(tmp_path, staged_runner, monkeypatch)
    assert staged_runner.verify_owned_cache(config, source, options)['status'] == 'complete'


@pytest.mark.parametrize('selection', ['command_symlink', 'middle_symlink', 'default_path'])
def test_selection_drift_rejected_even_if_old_binary_and_bytes_remain(tmp_path, staged_runner, monkeypatch, selection):
    config, source, options, command_alias, expand_alias, new_bin = configure(
        tmp_path, staged_runner, monkeypatch, default_command=selection == 'default_path')
    if selection == 'command_symlink':
        command_alias.unlink()
        command_alias.symlink_to(new_bin / 'recon-all')
    elif selection == 'middle_symlink':
        expand_alias.unlink()
        expand_alias.symlink_to(new_bin / 'mris_expand')
    else:
        monkeypatch.setenv('PATH', str(new_bin))
    with pytest.raises(ValueError, match='current resolved'):
        staged_runner.verify_owned_cache(config, source, options)


def test_requested_device_mismatch_rejected_before_cache_or_compute(tmp_path, staged_runner, monkeypatch):
    config, source, options, *_ = configure(tmp_path, staged_runner, monkeypatch)
    config['device'] = 'cpu'
    monkeypatch.setattr(staged_runner.reconstruction_adapter, '_cached_report',
                        lambda *args: pytest.fail('request mismatch must stop before closure acceptance'))
    with pytest.raises(ValueError, match='request differs'):
        staged_runner.verify_owned_cache(config, source, options)


def test_uuid_is_verified_for_current_process_not_unrelated_process(staged_runner,monkeypatch):
    monkeypatch.setattr(staged_runner.subprocess,'check_output',lambda *args,**kwargs:
        f'999999, GPU-unrelated\n{staged_runner.os.getpid()}, GPU-locked\n')
    assert staged_runner.cuda_context_uuid('GPU-locked')=='GPU-locked'


@pytest.mark.parametrize('rows',['wrong','multiple'])
def test_uuid_mismatch_or_multiple_cuda_devices_are_rejected(staged_runner,monkeypatch,rows):
    pid = staged_runner.os.getpid()
    output = f'{pid}, GPU-wrong\n'
    if rows=='multiple':
        output += f'{pid}, GPU-locked\n'
    monkeypatch.setattr(staged_runner.subprocess,'check_output',lambda *args,**kwargs:output)
    with pytest.raises(RuntimeError,match='actual CUDA context UUID differs'):
        staged_runner.cuda_context_uuid('GPU-locked')
