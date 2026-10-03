"""标准库等待器合同；小 JSON fixture 只验证队列门禁，不是 MRI benchmark。"""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


def load_waiter():
    path = Path(__file__).with_name('wait_and_run_surface_backend_demos.py')
    spec = importlib.util.spec_from_file_location('backend_stdlib_waiter',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def complete_gate():
    module = load_waiter()
    queue = {'source_revision':module.REVISION,'status':'complete','cases':{
        subject:{'status':'complete','exit_code':0} for subject in module.SUBJECTS}}
    reports = {subject:{'status':'complete','source_revision':module.REVISION} for subject in module.SUBJECTS}
    return module,queue,reports


def gate(module,queue,reports,*,parent=False,children=False):
    return module.queue_gate(queue,reports,original_process_alive=parent,tracked_descendants_alive=children)[0]


def test_only_all_five_normal_complete_and_no_original_identity_are_eligible(complete_gate):
    module,queue,reports = complete_gate
    assert gate(module,queue,reports)=='eligible'
    assert gate(module,queue,reports,parent=True)=='waiting'
    assert gate(module,queue,reports,children=True)=='blocked'


@pytest.mark.parametrize('failure',['queue','entry','exit','report','revision','missing','extra'])
def test_an_empty_physical_lock_cannot_override_failed_or_partial_formal_contract(complete_gate,failure):
    module,queue,reports = complete_gate
    subject = module.SUBJECTS[-1]
    if failure=='queue':queue['status']='stopped_on_failure'
    elif failure=='entry':queue['cases'][subject]['status']='running'
    elif failure=='exit':queue['cases'][subject]['exit_code']=1
    elif failure=='report':reports[subject]['status']='failed'
    elif failure=='revision':reports[subject]['source_revision']='different-source'
    elif failure=='missing':reports.pop(subject)
    else:queue['cases']['CON01']={'status':'complete','exit_code':0}
    assert gate(module,queue,reports)=='blocked'


def test_parent_exit_with_running_summary_blocks_even_when_tracked_child_was_not_observed(complete_gate):
    module,queue,reports = complete_gate
    queue['status']='running'
    assert gate(module,queue,reports,parent=False)=='blocked'
    assert gate(module,queue,reports,parent=True)=='waiting'


def test_loading_waiter_does_not_import_scientific_packages():
    before = set(sys.modules)
    load_waiter()
    added = set(sys.modules)-before
    assert not any(name in added for name in ('torch','numpy','nibabel','fnit'))


def test_original_lock_is_opened_without_creating_or_replacing_it(tmp_path):
    module = load_waiter()
    path = tmp_path/'original.lock'
    path.write_text('preserved')
    info = path.stat()
    identity = {'device':info.st_dev,'inode':info.st_ino}
    descriptor = module.open_original_lock(path,identity)
    os.close(descriptor)
    assert path.read_text()=='preserved'
    assert path.stat().st_ino==identity['inode']
    replacement = tmp_path/'different.lock'
    replacement.write_text('other')
    with pytest.raises(ValueError,match='inode changed'):
        module.open_original_lock(replacement,identity)
    link = tmp_path/'linked.lock'
    link.symlink_to(path)
    with pytest.raises(OSError):
        module.open_original_lock(link,identity)
    with pytest.raises(FileNotFoundError):
        module.open_original_lock(tmp_path/'absent.lock',identity)
    assert not (tmp_path/'absent.lock').exists()


def test_native_selection_and_bytes_are_bound_without_importing_extensions(tmp_path,monkeypatch):
    module=load_waiter()
    source=tmp_path/'source'
    native=source/'src/fnit/msm/_fastpd_src'
    native.mkdir(parents=True)
    for name in ('FastPD.h','block.h','fastpd_module.cpp','fnit_fastpd_model_stub.h','graph.h'):
        (native/name).write_text(name)
    binary=native.parent/'_fastpd_native.fixture.so'
    binary.write_bytes(b'compiled fixture only, not executable MRI benchmark')
    monkeypatch.setattr(module,'FASTPD_BINARY_SHA256',hashlib.sha256(binary.read_bytes()).hexdigest())
    old=tmp_path/'old_wb';new=tmp_path/'new_wb'
    for path in (old,new):
        path.write_text('#!/bin/sh\nexit 0\n');path.chmod(0o700)
    alias=tmp_path/'selected_wb';alias.symlink_to(old)
    case=tmp_path/'case';case.mkdir()
    (case/'config.private.json').write_text(json.dumps({'wb_command':str(alias)}))
    records=module.surface_native_records(source,{'provided':case},'')
    assert len(records)==7
    assert records['workbench/provided']['resolved_path']==str(old)
    alias.unlink();alias.symlink_to(new)
    assert module.surface_native_records(source,{'provided':case},'')!=records
    binary.write_bytes(b'changed independently compiled module')
    with pytest.raises(ValueError,match='FastPD binary differs'):
        module.surface_native_records(source,{'provided':case},'')
