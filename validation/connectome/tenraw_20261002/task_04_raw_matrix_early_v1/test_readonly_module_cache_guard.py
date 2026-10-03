"""CPU import-protocol fixtures for the actual preserved cache-guard failure."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

DRIVER=Path(__file__).resolve().parents[4]/'tools/benchmark_connectome_selected_raw_recovery.py'

def load_driver():
    specification=importlib.util.spec_from_file_location('curation_cache_guard_driver',DRIVER)
    module=importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def helper_row(directory,arm):
    directory.mkdir()
    names=('benchmark_connectome_raw_cohort','benchmark_connectome_raw_recovery',
           'benchmark_connectome_raw_rerun')
    if arm=='candidate': names+=('benchmark_connectome_staged_gpu',)
    files={}
    for name in names:
        path=directory/(name+'.py')
        path.write_text('protocol_fixture_only = True\n')
        files[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return {'arm':arm,'helpers':files}


def test_cached_candidate_helper_cannot_masquerade_as_original_baseline(tmp_path):
    driver=load_driver(); original=sys.path.copy()
    names=(*driver.HELPERS,'benchmark_connectome_staged_gpu')
    saved={name:sys.modules.pop(name) for name in names if name in sys.modules}
    try:
        candidate=helper_row(tmp_path/'candidate','candidate')
        baseline=helper_row(tmp_path/'baseline','baseline')
        driver.load_helpers(candidate)
        with pytest.raises(ValueError,match='imported helper differs from declared original helper'):
            driver.load_helpers(baseline)
    finally:
        sys.path[:]=original
        for name in names:sys.modules.pop(name,None)
        sys.modules.update(saved)


def test_original_baseline_guard_passes_in_fresh_process(tmp_path):
    baseline=helper_row(tmp_path/'baseline','baseline')
    import json
    declaration=tmp_path/'baseline.json';declaration.write_text(json.dumps(baseline))
    script="import importlib.util,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); s=importlib.util.spec_from_file_location('fixture_original',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); row=json.loads(pathlib.Path(sys.argv[2]).read_bytes()); actual=m.load_helpers(row); assert str(pathlib.Path(actual[0].__file__).parent)==str(pathlib.Path(row['helpers']['benchmark_connectome_raw_cohort']['path']).parent)"
    result=subprocess.run([sys.executable,'-c',script,str(DRIVER),str(declaration)],
                          text=True,capture_output=True)
    assert result.returncode==0,result.stderr


def test_guard_still_rejects_changed_declared_helper_bytes(tmp_path):
    driver=load_driver();baseline=helper_row(tmp_path/'baseline','baseline')
    path=Path(baseline['helpers']['benchmark_connectome_raw_cohort']['path'])
    path.write_text('protocol_fixture_changed = True\n')
    with pytest.raises(ValueError,match='original helper SHA or filename changed'):
        driver.load_helpers(baseline)
