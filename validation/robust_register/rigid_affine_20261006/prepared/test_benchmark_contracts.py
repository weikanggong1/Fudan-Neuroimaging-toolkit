"""Finite writer/dispatch safeguards; no MRI/native/registration/solver."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE=Path(__file__).parent


def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+".py"))
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


def test_report_is_finite_complete_private_and_no_clobber(tmp_path):
    m=module("register_benchmark")
    path=tmp_path/"report.json"
    m.atomic_report(path,{"bbox_lower":[0,1,2],"shape":[3,4,5],"cost":1.25})
    assert json.loads(path.read_text())["shape"]==[3,4,5]
    assert path.stat().st_mode & 0o777 == 0o600
    before=path.read_bytes()
    with pytest.raises(FileExistsError):
        m.atomic_report(path,{"overwritten":True})
    assert path.read_bytes()==before
    assert not list(tmp_path.glob('*.writing.*'))


@pytest.mark.parametrize("bad", (float('nan'),float('inf')))
def test_nonfinite_report_leaves_no_partial_json(tmp_path,bad):
    m=module("register_benchmark");path=tmp_path/'report.json'
    with pytest.raises(ValueError):
        m.atomic_report(path,{"cost":bad})
    assert not path.exists() and not list(tmp_path.glob('*.writing.*'))


def test_changed_source_or_input_binding_is_rejected(tmp_path):
    m=module('register_benchmark');path=tmp_path/'source.py';path.write_text('fixed')
    plan={'bindings':{'source':{'path':str(path),'bytes':5,'sha256':m.digest(path)}}}
    assert m.check_bindings(plan)['source']['bytes']==5
    path.write_text('changed')
    with pytest.raises(ValueError,match='binding changed'):
        m.check_bindings(plan)


def test_clean_environment_keeps_cpu8_and_hides_gpu(monkeypatch):
    module('register_benchmark');m=module('run_benchmark')
    for name in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','OPENBLAS_CORETYPE',
                 'FS_SetVoxToRasXform_Change_VoxSize'):
        monkeypatch.setenv(name,'must_be_removed')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    result=m.clean_environment()
    assert result['CUDA_VISIBLE_DEVICES']=='' and result['PYTHONDONTWRITEBYTECODE']=='1'
    assert all(result[k]=='8' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    assert all(k not in result for k in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','OPENBLAS_CORETYPE',
                                       'FS_SetVoxToRasXform_Change_VoxSize'))
    assert __import__('os').environ['CUDA_VISIBLE_DEVICES']=='0'


def test_finished_child_never_killed_by_group(monkeypatch):
    module('register_benchmark');m=module('run_benchmark')
    class Finished:
        pid=1234
        def poll(self):return 0
    monkeypatch.setattr(m.os,'killpg',lambda *args:pytest.fail('must not kill a finished/reused PID'))
    m.stop_owned_group(Finished())
