"""最终GA写回的状态合同及固定源控制顺序核对；不运行原版程序。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib, json, platform, subprocess, sys, time, unittest
from pathlib import Path


def run(*, fixed_source, output):
    """固定源Path或None、新JSON Path→版本绑定合同报告与exit0/1。

    提供fixed_source时核对固定SHA和final score/restore→几何→清理
    次序，只读源码，不上传/发布源文件。output须为未存在路径；记录
    HEAD前后、实际文件SHA、Python、测试计数/耗时。无影像输入，不
    宣称真实回归/完整GA/整例/等效。既有输出FileExistsError，固定源
    不同/关键顺序不同ValueError，被测文件变更RuntimeError。线程、
    精度和CUDA策略不更改；GPU显存未观测为None，不记0。
    """
    root=Path(__file__).resolve().parents[4]
    if output.exists():raise FileExistsError('immutable report output already exists')
    sys.path.insert(0,str(root/'src'))
    head=lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    before=head()
    paths=['src/fnit/recon_all/topology_ga_finalize.py',
        'src/fnit/recon_all/topology_candidate_cycle.py',
        'src/fnit/recon_all/topology_candidate_pipeline.py',
        'src/fnit/recon_all/topology_candidate_geometry.py',
        'src/fnit/recon_all/topology_candidate_state.py',
        'src/fnit/recon_all/topology_candidate_statistics.py',
        'tests/recon_all/test_topology_ga_finalize.py',
        'validation/recon_all/optimizations/20261010_topology_ga_finalize/run_local_contracts.py']
    files={p:dict(size=(root/p).stat().st_size,sha256=hashlib.sha256((root/p).read_bytes()).hexdigest())for p in paths}
    audit=None
    if fixed_source is not None:
        expected='55ace70ab09b1868a8cb152a9547879abdce790635b484bef1bf056986958187'
        data=fixed_source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('fixed source SHA differs')
        text=data.decode();start=text.index('  nkilled += nfinalvertices;')
        end=text.index('  /* free everything */',start);body=text[start:end]
        anchors=['memmove(dp->ordering, rp.best_ordering,',
            'memmove(defect->status, rp.status,',
            'fitness = mrisDefectPatchFitness(',
            'retessellateDefect(mris, mris_corrected, dvs, dp);',
            'detectDefectFaces(mris_corrected, dp);',
            'orientDefectFaces(mris_corrected, dp);',
            'defectMatch(mri, mris_corrected, dp, parms->smooth, parms->match);',
            'if (parms->check_surface_intersection)',
            'TPfree(&dp->tp);',
            'mris_corrected->vertices[dp->etable->edges[i].vno1].marked = FINAL_VERTEX;',
            'dp->defect->status[i] = DISCARD_VERTEX;',
            'mris_corrected->vertices[vertex_trans[dp->defect->border[i]]].marked = 0;']
        positions=[]
        for anchor in anchors:
            # The optional flag appears twice (counting and verbose logging).
            if anchor not in body:raise ValueError('missing final source rule: '+anchor)
            positions.append(body.index(anchor))
        if positions!=sorted(positions):raise ValueError('fixed final source control order differs')
        audit=dict(source_commit='d932c45b7941662ea380a05efef580568b98d41a',
            source_sha256=expected,ordered_rules_checked=len(anchors),passed=True,
            full_final_reference_execution='not_assessed')
    suite=unittest.defaultTestLoader.discover(str(root/'tests/recon_all'),pattern='test_topology_ga_finalize.py')
    tick=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(suite);elapsed=time.perf_counter()-tick
    for p,record in files.items():
        if hashlib.sha256((root/p).read_bytes()).hexdigest()!=record['sha256']:raise RuntimeError('tested source changed')
    report=dict(status='complete_local_finalization_contracts_only',
        observed_utc=datetime.now(timezone.utc).isoformat(),git_head_before_tests=before,git_head_after_tests=head(),
        source_files=files,source_rule_audit=audit,python=platform.python_version(),
        tests_run=result.testsRun,tests_failed=len(result.failures)+len(result.errors),tests_skipped=len(result.skipped),
        contract_runner_seconds=elapsed,whole_ga='not_assessed',whole_recon_all='not_assessed',
        real_final_source_comparison='not_assessed',overall_metric_equivalence='not_assessed',
        process_gpu_memory_bytes=None,production_default_changed=False,original_program_runtime_dependency=False)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-source',type=Path,default=None)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(**vars(parser.parse_args())))
