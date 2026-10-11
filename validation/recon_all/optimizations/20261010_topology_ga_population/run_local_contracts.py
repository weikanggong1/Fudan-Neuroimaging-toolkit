"""GA完整种群控制的本地合同及只读固定源码核对，不运行影像或原版。"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,platform,subprocess,sys,time,unittest
from pathlib import Path


def run(*,fixed_source,output):
    """可选固定d932源码Path、新JSON Path→exit0/1与版本绑定合同报告。

    fixed_source=None不作只读源码核对；提供时核SHA及种群/初始排序
    控制锚点，不执行或写原版。output必须不存在，保存HEAD前后、
    代码SHA、测试计数/耗时及未完成的真实/整例状态。无GPU/线程设置。
    已有输出FileExistsError，源锚点错误ValueError，被测文件改动
    RuntimeError；没有影像输入，不可用本地合同宣称真实MRI一致性。
    """
    root=Path(__file__).resolve().parents[4]
    if output.exists():raise FileExistsError('new immutable report path required')
    sys.path.insert(0,str(root/'src'))
    head=lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    before=head()
    paths=['src/fnit/recon_all/topology_ga_population.py',
        'src/fnit/recon_all/topology_ga_ordering.py','src/fnit/recon_all/topology_ga_selection.py',
        'src/fnit/recon_all/topology_fitness_search.py','src/fnit/recon_all/topology_vertex_elimination.py',
        'tests/recon_all/test_topology_ga_population.py',
        'validation/recon_all/optimizations/20261010_topology_ga_population/run_local_contracts.py']
    files={p:dict(size=(root/p).stat().st_size,sha256=hashlib.sha256((root/p).read_bytes()).hexdigest()) for p in paths}
    audit=None
    if fixed_source is not None:
        expected='55ace70ab09b1868a8cb152a9547879abdce790635b484bef1bf056986958187'
        data=fixed_source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('fixed source SHA differs')
        text=data.decode()
        start=text.index('static NOINLINE int mrisComputeOptimalRetessellation_wkr(')
        body='\n'.join(text[start:].splitlines()[:1400])
        anchors=['while (nunchanged < max_unchanged)', 'if (ngenerations == parms->niters)',
            'last_best = best_fitness;', 'mrisCopyDefectPatch(&dps[ranks[i]], dp);',
            'pfitness = exp(dp->fitness / two_sigma_sq) / fitness_norm;',
            'if (FEQUAL(last_best, best_fitness))', 'nunchanged -= NEXT;',
            'else if (ngenerations >= 10 && (ngenerations % 3 == 0))',
            'deleteWorstVertices(mris_corrected, &rp, defect, vertex_trans, 0.2, count)',
            'deleteWorstVertices(mris_corrected, &rp, defect, vertex_trans, 0.1, count)']
        for anchor in anchors:
            if body.count(anchor)!=1:raise ValueError('population source rule not unique')
        first=text.index('static void generateOrdering(DP *dp, SEGMENTATION *segmentation, int i)\n{')
        last=text.index('static MRIS *extractDefect',first);initial=text[first:last]
        initial_rules=['r = i % segmentation->nsegments + 1;',
            'for (m = 0; m < 11; m++)', 'r = nint(randomNumber(0.0, (double)nseg - 1));',
            'if (r != i + 1)']
        for anchor in initial_rules:
            if initial.count(anchor)!=1:raise ValueError('initial source rule not unique')
        audit=dict(source_commit='d932c45b7941662ea380a05efef580568b98d41a',
            source_sha256=expected,unique_rules_checked=len(anchors)+len(initial_rules),passed=True)
    suite=unittest.defaultTestLoader.discover(str(root/'tests/recon_all'),pattern='test_topology_ga_population.py')
    start=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(suite);elapsed=time.perf_counter()-start
    for p,record in files.items():
        if hashlib.sha256((root/p).read_bytes()).hexdigest()!=record['sha256']:raise RuntimeError('tested source changed')
    report=dict(status='complete_local_population_contracts_only',observed_utc=datetime.now(timezone.utc).isoformat(),
        git_head_before_tests=before,git_head_after_tests=head(),source_files=files,source_rule_audit=audit,
        python=platform.python_version(),tests_run=result.testsRun,tests_failed=len(result.failures)+len(result.errors),
        tests_skipped=len(result.skipped),contract_runner_seconds=elapsed,
        real_before_after='not_assessed',whole_ga='not_assessed',whole_recon_all='not_assessed',
        overall_equivalence='not_assessed',process_gpu_memory_bytes=None,
        production_default_changed=False,original_program_runtime_dependency=False,
        failed_preliminary_fixture='expected an even permutation after 11 source shuffles; fixed test expectation, algorithm unchanged')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-source',type=Path,default=None)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(**vars(parser.parse_args())))
