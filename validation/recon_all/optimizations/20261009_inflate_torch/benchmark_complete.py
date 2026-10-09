"""真实冻结smoothwm的完整native/NumPy/Torch配对，含inflated和sulc。

输入包manifest逐SHA检查；原生为明确Conda源码构建二进制。参考仅在
候选完成后读入，默认配对顺序ABC/CBA，保留strict诊断和原几何开发门。
无新整体等效阈值，不改生产默认。--output必须不存在。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all import inflate_python, inflate_torch, inflate_standard_run, inflate_topology
from fnit.recon_all.place_surface_normals import TorchFaceNormalTopology
from fnit.recon_all.mris_register_average_numba import RegistrationGradientAverager


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(candidate, reference):
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    return {"different_elements": int(np.count_nonzero(delta)),
            "max_abs": float(np.max(np.abs(delta), initial=0)),
            "p99_abs": float(np.percentile(np.abs(delta),99)),
            "rmse": float(np.sqrt(np.mean(delta*delta))), "mean_signed": float(np.mean(delta))}


def compare(candidate, reference, hemi):
    a, af, am = fsio.read_geometry(str(candidate/(hemi+".inflated")),read_metadata=True)
    b, bf, bm = fsio.read_geometry(str(reference/(hemi+".inflated")),read_metadata=True)
    if a.shape != b.shape or not np.array_equal(af,bf):
        return {"same_ordered_topology": False, "vertex_comparison": "not performed"}
    distance = np.linalg.norm(a.astype(np.float64)-b.astype(np.float64),axis=1)
    ac = fsio.read_morph_data(str(candidate/(hemi+".sulc")))
    bc = fsio.read_morph_data(str(reference/(hemi+".sulc")))
    geometry = {key: bool(np.array_equal(am.get(key),bm.get(key))) for key in set(am)|set(bm)}
    return {"same_ordered_topology": True, "vertices":len(a),"faces":len(af),
            "coordinate_float32":difference(a,b),
            "vertex_distance_mm":{"max":float(distance.max()),"p99":float(np.percentile(distance,99)),
                "rmse":float(np.sqrt(np.mean(distance*distance)))},
            "sulc_mm":difference(ac,bc),"volume_geometry_fields_equal":geometry,
            "existing_inflate_vertex_gate_0p001mm":bool(distance.max()<=.001),
            "strict_decoded_geometry_and_sulc":bool(np.array_equal(a,b) and np.array_equal(ac,bc)),
            "sulc_operator_equivalence_gate":"not established; strict diagnostic retained",
            "nonmanifold_connectivity":"unchanged ordered faces",
            "self_intersections":"not evaluated in this stage benchmark"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--case",action="append",help="可选case/lh或case/rh筛选，可重复")
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.threads<1 or args.repeats<1:raise ValueError("threads/repeats must be positive")
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    from numba import set_num_threads
    set_num_threads(args.threads)
    device=torch.device(args.device)
    if device.type!="cuda":raise ValueError("explicit CUDA required for GPU candidate")
    torch.empty(1,device=device);torch.cuda.synchronize(device)
    args.output.mkdir(parents=True)
    manifest=json.loads((args.data/"manifest.json").read_text())
    modules=(Path(__file__),Path(inflate_python.__file__),Path(inflate_torch.__file__),
             Path(inflate_standard_run.__file__),Path(__import__(TorchFaceNormalTopology.__module__,fromlist=['x']).__file__),
             Path(__import__(RegistrationGradientAverager.__module__,fromlist=['x']).__file__),Path(inflate_topology.__file__))
    report={"scope":"two_public_self_produced_frozen_smoothwm_complete_inflate_not_recon_all",
            "host":platform.node(),"device":str(device),"gpu":torch.cuda.get_device_name(device),
            "torch":torch.__version__,"cuda":torch.version.cuda,"cpu_affinity":sorted(os.sched_getaffinity(0)),
            "threads":args.threads,"matmul_tf32":torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32":torch.backends.cudnn.allow_tf32,"half_precision":False,
            "source_sha256":{str(p):sha(p) for p in modules},"native_sha256":sha(args.native),
            "data_manifest_sha256":sha(args.data/"manifest.json"),
            "native_ldd":subprocess.run(['ldd',str(args.native)],capture_output=True,text=True,check=True).stdout,
            "gpu_load_before":subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout,
            "production_default_changed":False,"whole_recon_all_acceleration":"not measured",
            "whole_metrics_equivalence":"not assessed","rows":[],"status":"running",
            "timing_scope":"same host ABC/CBA; API includes validation, setup, transfers, I/O; native process startup included; Python imports excluded",
            "process_gpu_memory":"PID namespace unresolved in this environment; tensor counters only, no zero-process claim"}
    started=time.perf_counter()
    def save():
        p=args.output/"summary.json";tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)
    try:
        save()
        for entry in manifest['cases']:
            key=entry['case']+'/'+entry['hemisphere']
            if args.case and key not in args.case:continue
            source=args.data/entry['surface'];hemi=entry['hemisphere']
            if sha(source)!=entry['sha256']:raise ValueError('input hash changed '+key)
            case_dir=args.output/entry['case']/hemi;case_dir.mkdir(parents=True)
            row={'case':entry['case'],'hemisphere':hemi,'input_sha256':entry['sha256'],'runs':[]}
            report['rows'].append(row);save()
            for repeat in range(args.repeats):
                order=('native','numpy','torch') if repeat%2==0 else ('torch','numpy','native')
                destinations={}
                for backend in order:
                    output=case_dir/(backend+'_'+str(repeat+1));output.mkdir()
                    destinations[backend]=output
                    torch.cuda.synchronize(device)
                    if backend=='torch':torch.cuda.reset_peak_memory_stats(device)
                    tick=time.perf_counter();result=None
                    if backend=='native':
                        command=[str(args.native),'-threads',str(args.threads),str(source),str(output/(hemi+'.inflated'))]
                        with (output/'native.log').open('w') as log:
                            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                    else:
                        result=inflate_standard_run.run_standard_inflate(input_surface=source,
                            inflated_output=output/(hemi+'.inflated'),sulc_output=output/(hemi+'.sulc'),
                            backend=backend,device=args.device if backend=='torch' else 'cpu',profile=repeat==0)
                    torch.cuda.synchronize(device)
                    wall=time.perf_counter()-tick
                    item={'repeat':repeat+1,'backend':backend,'wall_seconds':wall,'api':result,
                          'inflated_sha256':sha(output/(hemi+'.inflated')),'sulc_sha256':sha(output/(hemi+'.sulc'))}
                    if backend=='torch':
                        item['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
                        item['peak_reserved_bytes']=torch.cuda.max_memory_reserved(device)
                    row['runs'].append(item);save()
                    print('DONE',key,backend,repeat+1,wall,flush=True)
                for backend in ('numpy','torch'):
                    row.setdefault('comparisons',[]).append({'repeat':repeat+1,'backend':backend,
                        'to_same_repeat_native':compare(destinations[backend],destinations['native'],hemi)})
                save()
            native=[r for r in row['runs'] if r['backend']=='native']
            row['native_repeat_sulc_byte_exact']=len({r['sulc_sha256'] for r in native})==1
            row['native_repeat_decoded']=compare(case_dir/'native_2',case_dir/'native_1',hemi) if args.repeats>=2 else None
            save()
        if not report['rows']:raise ValueError('no selected cases')
        report['status']='complete_stage_pair'
    except Exception as error:
        report['status']='failed';report['error']=repr(error);save();raise
    report['control_wall_seconds']=time.perf_counter()-started
    report['gpu_load_after']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
    save()


if __name__=='__main__':main()
