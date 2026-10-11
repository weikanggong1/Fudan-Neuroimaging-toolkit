"""冻结全局 SFF 的同输入回归；实际 before 输入与 after 参考分开。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np

from fnit.recon_all.topology_global_sff import (global_second_fundamental_form,
    global_second_fundamental_form_state, global_curvature_priors)
from fnit.recon_all.topology_global_dot_prior import global_normal_dot_prior


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arrays(path):
    with np.load(path,allow_pickle=False) as values:
        return {key:values[key].copy() for key in values.files}


def _compare(candidate,reference):
    result={}
    for key,value in reference.items():
        if key not in candidate:
            result[key]=dict(missing=True,byte_equal=False);continue
        a,b=np.asarray(candidate[key]),np.asarray(value)
        if a.shape!=b.shape or a.dtype!=b.dtype:
            result[key]=dict(shape_equal=a.shape==b.shape,dtype_equal=a.dtype==b.dtype,byte_equal=False);continue
        finite=np.isfinite(a)&np.isfinite(b)
        errors=np.abs(a[finite].astype(np.float64)-b[finite].astype(np.float64))
        equal=(a==b)|(np.isnan(a)&np.isnan(b))
        result[key]=dict(shape_equal=True,dtype_equal=True,byte_equal=a.tobytes()==b.tobytes(),
            differing_values=int(np.count_nonzero(~equal)),max_abs=float(errors.max(initial=0)),
            p99_abs=float(np.percentile(errors,99)) if errors.size else 0,
            differing_bytes=int(np.count_nonzero(np.frombuffer(a.tobytes(),np.uint8)!=np.frombuffer(b.tobytes(),np.uint8))))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input','reference','source-manifest','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-repeat',type=Path)
    p.add_argument('--mode',choices=['sff','state','priors','dot'],default='sff')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('fresh global-SFF diagnostic output required')
    from fnit.recon_all.thread_budget import thread_budget
    started=time.perf_counter()
    with thread_budget(threads=a.threads) as budget:
        inputs=_arrays(a.input)
        names=['current_vertices','vertex_normals','neighbor_offsets','neighbors','vertex_rip','initial_k1','initial_k2']
        if a.mode=='state':
            names+=['initial_h','initial_gaussian_curvature','vertex_area']
        elif a.mode=='dot':
            names=['smoothed_original_vertices','metric_vertex_normals','neighbor_offsets','neighbors','vertex_rip']
        if set(inputs)!=set(names):raise ValueError('actual global-before fields only; no after-state/K input substitutions')
        loaded=time.perf_counter()-started
        compute={'sff':global_second_fundamental_form,'state':global_second_fundamental_form_state,
                 'priors':global_curvature_priors,'dot':global_normal_dot_prior}[a.mode]
        first=compute(**inputs);again=compute(**inputs)
        def outputs(value):
            if a.mode=='dot':
                return {key:np.asarray(entry) for key,entry in value.items()
                        if isinstance(entry,(np.ndarray,np.floating,np.integer))}
            sff=value if a.mode!='priors' else value['sff']
            fields={key:entry for key,entry in sff.items() if isinstance(entry,np.ndarray)}
            if a.mode=='priors':
                fields.update({'histogram_'+key:np.asarray(entry) for key,entry in value['histogram'].items()})
            elif a.mode=='state':
                fields.update({'surface_'+key:np.asarray(entry) for key,entry in value['surface_statistics'].items()})
            return fields
        candidate,repeat=outputs(first),outputs(again)
        # Source after arrays are loaded only after the FNIT producer has run.
        reference=_arrays(a.reference)
        required=['bins','counts'] if a.mode=='dot' else ['k1','k2','tangent_e1','tangent_e2']
        if not set(required).issubset(reference):
            raise ValueError('full scored reference fields required; empty/subset reference is not a pass')
        comparisons=_compare(candidate,reference)
        reference_repeat=_compare(reference,_arrays(a.reference_repeat)) if a.reference_repeat else None
        self_repeat=_compare(candidate,repeat)
        a.output.mkdir(parents=True)
        write_started=time.perf_counter();np.savez_compressed(a.output/'candidate.private.npz',**candidate)
        writing=time.perf_counter()-write_started
    strict=all(item['byte_equal'] for item in comparisons.values())
    report=dict(scope='frozen actual global SFF/PDF phase; not full GA or raw-MRI pipeline',
        execution_complete=True,strict_output_reproduction=strict,comparisons=comparisons,
        reference_repeat=reference_repeat,self_repeat=self_repeat,mode=a.mode,
        source_sha256={key:_sha(getattr(a,key.replace('-','_'))) for key in ['input','reference','source-manifest']},
        seconds=dict(loading=loaded,first_api=first['seconds'],repeat_api=again['seconds'],
            candidate_writing=writing,script_before_report=time.perf_counter()-started),
        environment=dict(python=platform.python_version(),numpy=np.__version__,thread_budget=budget),
        executed_module_sha256=_sha(__import__('sys').modules[compute.__module__].__file__),
        precision=dict(ordered_float32=True,eispack_float64=True,fastmath=False,
            cuda_executed=False,gpu_memory='not_assessed',tf32_policy_unchanged=True),
        mesh_quality='not_assessed',whole_ga=False,equivalence='not_assessed')
    (a.output/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if strict else 1


if __name__=='__main__':raise SystemExit(main())
