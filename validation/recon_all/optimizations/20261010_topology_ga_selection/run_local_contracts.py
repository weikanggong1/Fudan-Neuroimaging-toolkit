"""GA父代选择的本地合同、只读源码锚点及版本报告。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest


def run(*, fixed_source, output):
    """可选固定源码、全新JSON路径 → 版本绑定的本地合同结果。

    fixed_source为固定d932 mrisurf_defect.cpp的可选Path；仅只读核SHA
    和7个选择规则锚点，不执行原版。output为不存在的JSON Path。
    返回成功exit0/失败exit1并写HEAD前后、代码SHA、测试数和耗时。
    输出不含服务器路径/账号或影像；这不是MRI benchmark、完整GA
    或显存验证。输出已存在抛FileExistsError，源码SHA/锚点不符抛
    ValueError；测试期间被测文件改动抛RuntimeError。
    """
    root=Path(__file__).resolve().parents[4]
    if output.exists():raise FileExistsError('new immutable report path required')
    sys.path.insert(0,str(root/'src'))
    head=lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    before=head()
    paths=['src/fnit/recon_all/topology_ga_selection.py',
        'src/fnit/recon_all/topology_fitness_search.py',
        'tests/recon_all/test_topology_ga_selection.py',
        'validation/recon_all/optimizations/20261010_topology_ga_selection/run_local_contracts.py']
    files={relative:dict(size=(root/relative).stat().st_size,
        sha256=hashlib.sha256((root/relative).read_bytes()).hexdigest()) for relative in paths}
    audit=None
    if fixed_source is not None:
        expected='55ace70ab09b1868a8cb152a9547879abdce790635b484bef1bf056986958187'
        source_bytes=fixed_source.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest()!=expected:raise ValueError('fixed reference SHA differs')
        text=source_bytes.decode()
        anchors=['nelite = nint(ELITISM_PCT * max_patches)',
            'nselected = nint(SELECTION_PCT * max_patches)',
            'nreplacements = nint(REPLACEMENT_PCT * max_patches)',
            'two_sigma_sq = (dps[ranks[0]].fitness - dps[ranks[nselected - 1]].fitness)',
            'nadd = nint(pfitness * ncrossovers)',
            'pfitness = exp(dp->fitness / two_sigma_sq) / fitness_norm',
            'p2 = selected[(int)randomNumber(0, ncrossovers - .001)]']
        for anchor in anchors:
            if text.count(anchor)!=1:raise ValueError('source selection anchor absent or duplicated')
        audit=dict(source_commit='d932c45b7941662ea380a05efef580568b98d41a',
            source_sha256=expected,unique_rules_checked=len(anchors),passed=True)
    suite=unittest.defaultTestLoader.discover(str(root/'tests/recon_all'),pattern='test_topology_ga_selection.py')
    start=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(suite)
    elapsed=time.perf_counter()-start
    for relative,record in files.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=record['sha256']:
            raise RuntimeError('tested source changed during contracts')
    report=dict(status='complete_local_interface_contracts_only',
        observed_utc=datetime.now(timezone.utc).isoformat(),git_head_before_tests=before,
        git_head_after_tests=head(),source_files=files,source_rule_audit=audit,
        python=platform.python_version(),tests_run=result.testsRun,
        tests_failed=len(result.failures)+len(result.errors),tests_skipped=len(result.skipped),
        contract_runner_seconds=elapsed,real_before_after='not_assessed',
        whole_ga='not_assessed',whole_recon_all='not_assessed',
        overall_equivalence='not_assessed',process_gpu_memory_bytes=None,
        original_program_runtime_dependency=False,production_default_changed=False,
        failed_preliminary_fixture='initial 10-patch fixture wrongly expected a tail draw; corrected test uses 13-patch case, algorithm unchanged')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-source',type=Path,default=None)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(**vars(parser.parse_args())))
