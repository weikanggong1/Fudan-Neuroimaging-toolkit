"""真实校准rawcounts的旧/新sigma2平滑回归，不读取影像或改写旧产物。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('input','old-source','new-source','source-manifest','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('fresh comparison report required')
    started=time.perf_counter()
    manifest=json.loads(args.source_manifest.read_text())
    old_sha,new_sha=sha256(args.old_source),sha256(args.new_source)
    if (manifest['old_source_sha256']!=old_sha or manifest['new_source_sha256']!=new_sha):
        raise ValueError('source manifest does not match executed old/new modules')
    old=load_source(args.old_source,'old_histogram')
    new=load_source(args.new_source,'new_histogram')
    with np.load(args.input,allow_pickle=False) as values:
        before={key:values['calibration_'+key+'_raw'].copy() for key in ('white','gray')}
        saved={key:values['calibration_'+key+'_smooth'].copy() for key in ('white','gray')}
        calibration_vertices=int(len(values['calibration_selected']))
    loaded=time.perf_counter()-started
    comparisons={}
    for name,counts in before.items():
        if counts.shape!=(256,) or counts.dtype!=np.float32 or not np.isfinite(counts).all():
            raise ValueError('actual finite float32 256-bin calibration raw counts required')
        original=counts.copy()
        start=time.perf_counter();baseline=old._smooth_histogram(counts);old_seconds=time.perf_counter()-start
        start=time.perf_counter();candidate=new._smooth_histogram(counts);new_seconds=time.perf_counter()-start
        repeat=new._smooth_histogram(counts)
        error=np.abs(candidate.astype(np.float64)-baseline.astype(np.float64))
        comparisons[name]=dict(
            input_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
            baseline_output_sha256=hashlib.sha256(baseline.tobytes()).hexdigest(),
            candidate_output_sha256=hashlib.sha256(candidate.tobytes()).hexdigest(),
            differing_values=int(np.count_nonzero(candidate!=baseline)),
            differing_bytes=int(np.count_nonzero(candidate.view(np.uint8)!=baseline.view(np.uint8))),
            max_abs=float(error.max()),p99_abs=float(np.percentile(error,99)),
            old_new_byte_equal=candidate.tobytes()==baseline.tobytes(),
            candidate_repeat_byte_equal=candidate.tobytes()==repeat.tobytes(),
            old_new_match_saved_calibration=(candidate.tobytes()==baseline.tobytes()==saved[name].tobytes()),
            input_unchanged=original.tobytes()==counts.tobytes(),
            old_seconds=old_seconds,new_seconds=new_seconds)
    cpu_model='unknown'
    cpuinfo=Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu_model=next((line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines()
                        if line.startswith('model name')),cpu_model)
    gate=all(row['old_new_byte_equal'] and row['candidate_repeat_byte_equal']
             and row['old_new_match_saved_calibration'] and row['input_unchanged']
             for row in comparisons.values())
    report=dict(schema_version=1,scope='frozen actual calibration rawcounts; sigma2 smoothing only',
        real_input=True,execution_complete=True,strict_smoothing_compatibility=gate,
        public_case='ds000114-sub07 LH first defect',calibration_vertices=calibration_vertices,
        input_npz_sha256=sha256(args.input),source_manifest_sha256=sha256(args.source_manifest),
        old_source_sha256=old_sha,new_source_sha256=new_sha,
        old_source_git_commit=manifest['old_source_git_commit'],comparisons=comparisons,
        precision=dict(float32_source_order=True,fp16_bf16=False,cuda_used=False,
                       tf32_policy_unchanged=True,gpu_memory='not_assessed'),
        environment=dict(role='candidate accelerator-b shared CPU',python=platform.python_version(),
                         numpy=np.__version__,cpu_model=cpu_model,
                         cpu_affinity=sorted(__import__('os').sched_getaffinity(0))),
        seconds=dict(import_load=loaded,script_before_report=time.perf_counter()-started),
        timing_scope='single calls on shared CPU; no speed ratio or whole-pipeline claim',
        dot_prior_real_validation='pending',global_sff_real_validation='pending',
        full_ga=False,mesh_quality='not_assessed',overall_equivalence='not_assessed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'strict_smoothing_compatibility':gate,
                      'calibration_vertices':calibration_vertices}))
    return 0 if gate else 1


if __name__=='__main__':
    raise SystemExit(main())
