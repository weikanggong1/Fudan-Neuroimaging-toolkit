"""自有 worker 真实入口/exec 故障合同；不是影像精度或耗时 benchmark。"""
from contextlib import ExitStack
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from fnit.recon_all import gca_torch_worker as gca
from fnit.recon_all import wm_torch_worker as wm
from fnit.recon_all import talairach_synthmorph as talairach
from fnit.recon_all.execution_recipe import original_stage_guard
from fnit.recon_all.worker_program_guard import inherit_original_stage_guard

FLAG = 'FNIT_RECON_FORBID_ORIGINAL_STAGES'


@pytest.fixture
def probe(tmp_path):
    # 本地独立无影像测试脚本；名字模拟禁令命名规则，内容只写一个标记。
    executable = tmp_path / 'mri_fnit_contract_probe'
    executable.write_text('#!/bin/sh\nprintf executed > "$1"\n')
    executable.chmod(0o700)
    return executable, tmp_path / 'probe-executed'


@pytest.fixture(autouse=True)
def caller_state():
    environment = dict(os.environ)
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    threads = torch.get_num_threads()
    yield
    os.environ.clear()
    os.environ.update(environment)
    torch.set_num_threads(threads)
    (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
     torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = flags


def invoke(kind, root):
    if kind == 'gca':
        return gca.run_worker(nu_path=root/'nu.mgz', mask_path=root/'mask.mgz',
            atlas_path=root/'declared.gca', output_path=root/'result.lta',
            report_path=root/'gca.json', device='cuda:0', threads=2, code_version='contract')
    if kind == 'wm':
        return wm.run_worker(source_path=root/'nu.mgz', output_path=root/'wm.mgz',
            report_path=root/'wm.json', device='cuda:0', threads=2, code_version='contract')
    return talairach.register_talairach(root/'strip.mgz', root/'template.mgz',
        root/'weights.h5', root/'talairach.xfm', device='cpu', threads=2)


@pytest.mark.parametrize('kind', ['gca', 'wm', 'talairach'])
@pytest.mark.parametrize('flag', [None, '0', '1'])
def test_guard_is_installed_before_first_stage_and_restored(kind, flag, probe, tmp_path, monkeypatch):
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    before = dict(os.environ)
    precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    threads = torch.get_num_threads()
    executable, marker = probe

    def first_stage(*args, **kwargs):
        subprocess.run([str(executable), str(marker)], check=True)
        raise LookupError('stop before numerical computation')

    target = 'torch.set_num_threads' if kind == 'talairach' else 'torch.cuda.is_initialized'
    expected = RuntimeError if flag == '1' else LookupError
    with patch(target, side_effect=first_stage), pytest.raises(expected):
        invoke(kind, tmp_path)
    assert marker.exists() is (flag != '1')
    assert dict(os.environ) == before
    assert torch.get_num_threads() == threads
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == precision
    # 常驻 audit hook 在作用域外惰性；启用标志本身不等于活动 ContextVar。
    subprocess.run([str(executable), str(marker)], check=True)


def test_decorator_preserves_signature_result_and_numerical_policy(monkeypatch):
    def operation(*, value, record):
        record.append((torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32,
                       torch.backends.cudnn.allow_tf32, torch.cuda.is_initialized()))
        return value
    wrapped = inherit_original_stage_guard(operation)
    assert inspect.signature(wrapped) == inspect.signature(operation)
    value = object()
    record = []
    operation(value=value, record=record)
    monkeypatch.setenv(FLAG, '1')
    assert wrapped(value=value, record=record) is value
    assert record[0] == record[1]


@pytest.mark.parametrize('fail', [False, True])
def test_guard_restores_flag_changed_by_callee_on_success_or_error(fail, monkeypatch, probe):
    monkeypatch.setenv(FLAG, '1')
    @inherit_original_stage_guard
    def operation():
        os.environ[FLAG] = 'callee-change'
        if fail:
            raise LookupError('callee error')
        return 'same result'
    if fail:
        with pytest.raises(LookupError):
            operation()
    else:
        assert operation() == 'same result'
    assert os.environ[FLAG] == '1'
    executable, marker = probe
    subprocess.run([str(executable), str(marker)], check=True)
    assert marker.exists()


@pytest.mark.parametrize('kind', ['gca', 'wm', 'talairach'])
def test_nested_parent_guard_survives_child_scope_error(kind, probe, tmp_path):
    executable, marker = probe
    target = 'torch.set_num_threads' if kind == 'talairach' else 'torch.cuda.is_initialized'
    with original_stage_guard(enabled=True) as outer:
        with patch(target, side_effect=LookupError('stage failed')), pytest.raises(LookupError):
            invoke(kind, tmp_path)
        assert os.environ[FLAG] == '1'
        with pytest.raises(RuntimeError):
            subprocess.run([str(executable), str(marker)], check=True)
        assert outer['attempted_forbidden'] == [executable.name]
    assert not marker.exists()


@pytest.mark.parametrize('kind', ['gca', 'wm', 'talairach'])
@pytest.mark.parametrize('flag', [None, '1'])
def test_real_exec_inherits_flag_and_propagates_stage_failure(kind, flag, probe, tmp_path, monkeypatch):
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    executable, marker = probe
    receipt = tmp_path/'child-receipt.json'
    probe_script = Path(__file__).with_name('worker_guard_exec_probe.py')
    real_run = subprocess.run
    before = dict(os.environ)
    state = (torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32)

    def actual_child(command, **kwargs):
        if '-m' in command:
            index = command.index('-m')
            module, arguments = command[index+1], command[index+2:]
        else:
            # WM 的自有冻结路径 bootstrap：保留真实CLI参数与环境。
            module, arguments = command[4], command[5:]
        return real_run([sys.executable, str(probe_script), module,
                         str(executable), str(marker), str(receipt), *arguments],
                        **kwargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with pytest.raises(subprocess.CalledProcessError) as caught:
        if kind == 'gca':
            with patch.object(gca.subprocess, 'run', side_effect=actual_child):
                gca.run_isolated_registration(nu_path=tmp_path/'nu.mgz', mask_path=tmp_path/'mask.mgz',
                    atlas_path=tmp_path/'declared.gca', output_path=tmp_path/'result.lta',
                    report_path=tmp_path/'report.json', device='cuda:0', threads=2)
        elif kind == 'wm':
            with patch.object(wm.subprocess, 'run', side_effect=actual_child):
                wm.run_isolated_segmentation(source_path=tmp_path/'nu.mgz', output_path=tmp_path/'wm.mgz',
                    report_path=tmp_path/'report.json', device='cuda:0', threads=2)
        else:
            # 与 input_talairach_chain 的实际模块/参数调用一致，数值入口前注入故障。
            actual_child([sys.executable, '-m', 'fnit.recon_all.talairach_synthmorph',
                str(tmp_path/'strip.mgz'), str(tmp_path/'template.mgz'), '--weights', str(tmp_path/'weights.h5'),
                '--xfm', str(tmp_path/'talairach.xfm'), '--device', 'cuda:0', '--threads', '2'],
                env=dict(os.environ), check=True)
    assert caught.value.returncode == 19, caught.value.stderr
    child = json.loads(receipt.read_text())
    assert child['inherited_flag'] == flag
    assert child['error'] == ('RuntimeError' if flag == '1' else 'LookupError')
    assert child['probe_executed'] is (flag != '1')
    assert child['environment_restored'] and child['thread_count_restored']
    assert dict(os.environ) == before
    assert (torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32) == state
    assert not (tmp_path/'report.json').exists()


@pytest.mark.parametrize('kind', ['gca', 'wm'])
def test_late_stage_fault_restores_guard_and_gca_score_method(kind, probe, tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, '1')
    executable, marker = probe
    before = dict(os.environ)
    original_score = gca.GCASearchScorer.score_many
    def forbidden(*args, **kwargs):
        subprocess.run([str(executable), str(marker)], check=True)
    module = gca if kind == 'gca' else wm
    with ExitStack() as stack:
        stack.enter_context(patch('torch.cuda.is_initialized', return_value=False))
        for name in ('set_device', 'synchronize', 'reset_peak_memory_stats'):
            stack.enter_context(patch('torch.cuda.'+name))
        stack.enter_context(patch.object(module, 'configure_cuda_allocator', return_value={
            'torch_stats_known_valid': True, 'torch_stats_known_unavailable': False}))
        stack.enter_context(patch.object(module, '_sha256', return_value='contract-hash'))
        stage = 'register_t1' if kind == 'gca' else 'segment_white_matter_mgz'
        stack.enter_context(patch.object(module, stage, side_effect=forbidden))
        with pytest.raises(RuntimeError, match='forbids original stage'):
            invoke(kind, tmp_path)
    assert dict(os.environ) == before
    assert gca.GCASearchScorer.score_many is original_score
    assert not marker.exists()
    assert not (tmp_path/('result.lta' if kind == 'gca' else 'wm.mgz')).exists()
