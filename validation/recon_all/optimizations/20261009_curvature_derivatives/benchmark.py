"""Pair a source-formula loop and Torch on frozen real principal maps."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numba
from numba import njit
import numpy as np
import torch
import fnit.recon_all.curvature_stats_torch as module
from fnit.recon_all.curvature_stats_torch import curvature_derivatives_tensor, write_curvature_derivatives

MAX_ALLOWED_ULPS = 2  # Predeclared for pointwise elementary functions, not whole-pipeline equivalence.
ORDER = ('BE', 'C', 'FI', 'S')


@njit(cache=True, fastmath=False)
def source_formula(k1, k2):
    result = np.empty((4, len(k1)), np.float32)
    for vertex in range(len(k1)):
        a, b = k1[vertex], k2[vertex]
        bending = np.float32(np.float32(a * a) + np.float32(b * b))
        difference = np.float32(a - b)
        first, second = abs(np.float64(a)), abs(np.float64(b))
        result[0, vertex] = bending
        result[1, vertex] = np.float32(np.sqrt(np.float64(bending) * 0.5))
        result[2, vertex] = np.float32(first * (first - second))
        result[3, vertex] = np.float32(difference * difference)
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(candidate, reference):
    a, b = np.asarray(candidate, np.float32), np.asarray(reference, np.float32)
    if a.shape != b.shape:
        raise ValueError('reference and candidate vertex counts differ')
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    bits_a, bits_b = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
    order_a = np.where(bits_a < 0, -2147483648-bits_a, bits_a)
    order_b = np.where(bits_b < 0, -2147483648-bits_b, bits_b)
    maximum_ulps = int(np.abs(order_a-order_b).max(initial=0))
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    bit_differences = a.view(np.uint32) != b.view(np.uint32)
    return {'different_vertices': int(np.count_nonzero(a != b)),
            'different_bit_patterns': int(np.count_nonzero(bit_differences)), 'max_abs': float(delta.max(initial=0)),
            'p99_abs': float(np.percentile(delta,99)) if len(delta) else 0.0,
            'rmse': float(np.sqrt(np.mean(delta*delta))) if len(delta) else 0.0,
            'max_ulps': maximum_ulps, 'finite': finite,
            'strict_float32_equal': bool(not bit_differences.any()),
            'within_fixed_tolerance': bool(finite and maximum_ulps <= MAX_ALLOWED_ULPS)}


def call(function, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = perf_counter()
    result = function()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return result, perf_counter()-started


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-program', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--code-version', required=True)
    args=parser.parse_args()
    torch.set_num_threads(args.threads); numba.set_num_threads(args.threads)
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    output=args.output
    output.mkdir(parents=True, exist_ok=True)
    report={'code_version':args.code_version, 'module_sha256':sha256(module.__file__),
            'script_sha256':sha256(__file__), 'upstream_commit':'d932c45b7941662ea380a05efef580568b98d41a',
            'reference_program_sha256':sha256(args.reference_program) if args.reference_program else None,
            'reference_kind':'frozen native post-principal maps plus isolated source-formula Numba loop',
            'reference_program_rerun':False, 'strict_reference_repeatability':'not_retested_this_run',
            'host':platform.node(), 'cpu':platform.processor(), 'platform':platform.platform(),
            'device':str(device), 'torch_version':torch.__version__, 'numba_version':numba.__version__,
            'threads':args.threads, 'torch_interop_threads':torch.get_num_interop_threads(),
            'affinity':sorted(os.sched_getaffinity(0)),
            'dtype':'float32 inputs/outputs; source-required double sqrt/fabs intermediate',
            'matmul_tf32':torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'autocast':False, 'max_allowed_ulps':MAX_ALLOWED_ULPS,
            'whole_pipeline':'not_run', 'process_memory_sampling':'not_measured', 'hemispheres':[],
            'nvidia_smi':subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout}
    for subject in args.subject:
        subject_name=subject.parent.parent.name if subject.name=='subject' else subject.name
        for hemi in ('lh','rh'):
            prefix=subject/'surf'/f'{hemi}.smoothwm'
            k1_path=prefix.with_name(prefix.name+'.K1.crv')
            k2_path=prefix.with_name(prefix.name+'.K2.crv')
            k1=np.ascontiguousarray(fsio.read_morph_data(str(k1_path)), dtype=np.float32)
            k2=np.ascontiguousarray(fsio.read_morph_data(str(k2_path)), dtype=np.float32)
            if not np.isfinite(k1).all() or not np.isfinite(k2).all():
                raise ValueError('nonfinite frozen inputs')
            t0=perf_counter(); scalar=source_formula(k1,k2); cold=perf_counter()-t0
            first,second=torch.as_tensor(k1,device=device),torch.as_tensor(k2,device=device)
            curvature_derivatives_tensor(first,second)
            source_samples=[]; torch_samples=[]; api_samples=[]; sequence=[]
            candidate=None
            for round_number in range(args.repeats):
                order=('source','torch','torch','source') if round_number%2==0 else ('torch','source','source','torch')
                for backend in order:
                    function=(lambda:source_formula(k1,k2)) if backend=='source' else (lambda:curvature_derivatives_tensor(first,second))
                    result,seconds=call(function,device)
                    (source_samples if backend=='source' else torch_samples).append(seconds)
                    sequence.append(backend)
                    if backend=='torch': candidate=result
            for _ in range(2*args.repeats):
                _,seconds=call(lambda:{name:value.cpu().numpy() for name,value in curvature_derivatives_tensor(torch.as_tensor(k1,device=device),torch.as_tensor(k2,device=device)).items()},device)
                api_samples.append(seconds)
            comparison={}; scalar_comparison={}; reference_hashes={}
            for index,name in enumerate(ORDER):
                reference_path=prefix.with_name(prefix.name+f'.{name}.crv')
                reference=fsio.read_morph_data(str(reference_path))
                actual=candidate[name].cpu().numpy()
                comparison[name]=metrics(actual,reference)
                scalar_comparison[name]=metrics(actual,scalar[index])
                reference_hashes[name]=sha256(reference_path)
            io_report=write_curvature_derivatives(k1_path=k1_path,k2_path=k2_path,
                output_prefix=output/subject_name/f'{hemi}.smoothwm',device=str(device))
            report['hemispheres'].append({'subject':subject_name,'hemisphere':hemi,'vertices':len(k1),
                'k1_sha256':sha256(k1_path),'k2_sha256':sha256(k2_path),'reference_hashes':reference_hashes,
                'source_first_call_including_jit_seconds':cold,'pair_sequence':sequence,
                'source_formula_seconds':source_samples,'source_formula_median_seconds':float(np.median(source_samples)),
                'torch_resident_seconds':torch_samples,'torch_resident_median_seconds':float(np.median(torch_samples)),
                'torch_api_including_transfers_seconds':api_samples,'torch_api_median_seconds':float(np.median(api_samples)),
                'io_api':io_report,'vs_frozen_native':comparison,'vs_source_formula':scalar_comparison})
            del first,second,candidate
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
    report['peak_reserved_bytes']=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None
    report['all_frozen_native_maps_within_tolerance']=all(m['within_fixed_tolerance'] for h in report['hemispheres'] for m in h['vs_frozen_native'].values())
    report['all_source_formula_maps_equal']=all(m['strict_float32_equal'] for h in report['hemispheres'] for m in h['vs_source_formula'].values())
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(output/'report.json'), 'all_frozen_native_maps_within_tolerance':report['all_frozen_native_maps_within_tolerance'],'all_source_formula_maps_equal':report['all_source_formula_maps_equal']}))


if __name__=='__main__': main()
