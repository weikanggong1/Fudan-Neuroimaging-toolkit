"""独立邻近缺陷合并的本地合同及固定源码版本审计。"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,platform,subprocess,sys,time,unittest
from pathlib import Path


def run(*,fixed_source,output,fixed_startup=None):
    """可选只读d932源码+不存在的JSON Path → 本地合同报告/exit0或1。

    fixed_startup为可选只读d932 mris_fix_topology.cpp Path，核对中心化
    前后法向的调用顺序；None时该相位源码核对未评估。不执行原版，
    不写参考、不读影像；报告HEAD前后、SHA、测试数和
    墙钟。新输出路径已存在报FileExistsError；源码/锚点不符ValueError，
    测试期间实现变化RuntimeError。真实连续影像及整例未评估。
    """
    root=Path(__file__).resolve().parents[4]
    if output.exists():raise FileExistsError('new immutable report path required')
    sys.path.insert(0,str(root/'src'))
    head=lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    before=head()
    paths=['src/fnit/recon_all/topology_defect_merge.py','src/fnit/recon_all/topology_preflight_python.py',
        'src/fnit/recon_all/smooth_surface_python.py','tests/recon_all/test_topology_defect_merge.py',
        'validation/recon_all/optimizations/20261010_topology_defect_merge/run_local_contracts.py']
    files={p:dict(size=(root/p).stat().st_size,sha256=hashlib.sha256((root/p).read_bytes()).hexdigest()) for p in paths}
    audit=None
    if fixed_source is not None:
        expected='55ace70ab09b1868a8cb152a9547879abdce790635b484bef1bf056986958187'
        data=fixed_source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('fixed source SHA differs')
        text=data.decode();start=text.index('static DEFECT_LIST *mrisMergeNeighboringDefects(MRIS *mris, DEFECT_LIST *dl)\n{')
        stop=text.index('static int defectIdentifyDefectiveVertices',start);body=text[start:stop]
        anchors=['if (nd[j] >= 2)','defect->nvertices = nadded;',
            'defect->area /= (float)defect->nvertices;',
            'nv->marked = -1; /* border vertex */','i--;','if (defect->nvertices) {']
        for anchor in anchors:
            if body.count(anchor)!=1:raise ValueError('merge source rule not unique')
        audit=dict(source_commit='d932c45b7941662ea380a05efef580568b98d41a',
            source_sha256=expected,unique_rules_checked=len(anchors),passed=True,
            val_write_before_merge=True,merge_body_does_not_assign_vertex_val='->val' not in body)
    phase_audit=None
    if fixed_startup is not None:
        expected='55cca5a4f8a82baee768a94bd545955bd805a4ad446fa19995511240a5e09be1'
        startup=fixed_startup.read_bytes()
        if hashlib.sha256(startup).hexdigest()!=expected:raise ValueError('fixed startup SHA differs')
        text=startup.decode()
        entry=text.index('MRIScorrectTopology(mris, mri,')
        restore=text.rfind('MRISrestoreVertexPositions(mris, CANONICAL_VERTICES)',0,entry)
        metric=text.index('MRIScomputeMetricProperties(mris)',restore)
        if not 0<restore<metric<entry:raise ValueError('startup initial metric phase differs')
        if fixed_source is None:raise ValueError('defect source required for two-phase audit')
        defect_text=fixed_source.read_text()
        correct=defect_text.index('MRI_SURFACE *MRIScorrectTopology(')
        center=defect_text.index('MRIScenterSphere(mris)',correct)
        segment=defect_text.index('dl = MRISsegmentDefects(',center)
        metric=defect_text.index('MRIScomputeMetricProperties(mris)',segment)
        merge=defect_text.index('mrisMergeNeighboringDefects(mris, dl)',metric)
        if not center<segment<metric<merge:raise ValueError('defect two-phase ordering differs')
        phase_audit=dict(startup_sha256=expected,source_commit='d932c45b7941662ea380a05efef580568b98d41a',
            initial_normals='canonical metric before centerSphere',merged_normals='canonical metric after centerSphere',
            source_order_verified=True,normal_consumer_audit='normal fields are not read by this source merge/retention/SFF/base chain')
    suite=unittest.defaultTestLoader.discover(str(root/'tests/recon_all'),pattern='test_topology_defect_merge.py')
    start=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(suite);elapsed=time.perf_counter()-start
    for p,record in files.items():
        if hashlib.sha256((root/p).read_bytes()).hexdigest()!=record['sha256']:raise RuntimeError('tested source changed')
    report=dict(status='complete_local_interface_contracts_only',observed_utc=datetime.now(timezone.utc).isoformat(),
        git_head_before_tests=before,git_head_after_tests=head(),source_files=files,source_rule_audit=audit,
        source_normal_phase_audit=phase_audit,
        python=platform.python_version(),tests_run=result.testsRun,tests_failed=len(result.failures)+len(result.errors),
        tests_skipped=len(result.skipped),contract_runner_seconds=elapsed,
        real_raw_to_base='not_assessed',whole_ga='not_assessed',whole_recon_all='not_assessed',
        overall_equivalence='not_assessed',process_gpu_memory_bytes=None,
        production_default_changed=False,original_program_runtime_dependency=False)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-source',type=Path,default=None)
    parser.add_argument('--fixed-startup',type=Path,default=None)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(**vars(parser.parse_args())))
