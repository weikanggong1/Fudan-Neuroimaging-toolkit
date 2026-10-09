"""固定真实white/pial/inflated的GPU指标重复性；只读成熟模块，不修改精度。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import nibabel.freesurfer.io as fs
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_report(actual,reference):
    a,b=np.asarray(actual),np.asarray(reference)
    same=a.shape==b.shape and a.dtype==b.dtype
    if not same:return {'same_shape_dtype':False,'shape':list(a.shape),'dtype':str(a.dtype)}
    error=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return {'same_shape_dtype':True,'shape':list(a.shape),'dtype':str(a.dtype),
        'finite_elements':int(np.isfinite(a).sum()),'different_elements':int(np.count_nonzero(a!=b)),
        'max_absolute_difference':float(error.max(initial=0)),
        'p99_absolute_difference':float(np.percentile(error,99)),
        'mean_absolute_difference':float(error.mean()),
        'array_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
        'reference_array_sha256':hashlib.sha256(b.tobytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',action='append',nargs=3,required=True,metavar=('SUBJECT','HEMI','FOCUS_SURFACE'))
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--native-binary',type=Path,required=True)
    parser.add_argument('--assets-directory',type=Path,required=True)
    parser.add_argument('--code-base-commit',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--repeat',type=int,choices=(2,3),default=3)
    args=parser.parse_args()
    if args.threads<1:raise ValueError('positive thread count required')
    device=torch.device(args.device)
    if device.type!='cuda' or device.index is None:raise ValueError('indexed CUDA required')
    if args.output_directory.exists():raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    sys.path.insert(0,str(args.source_root/'src'))
    from fnit.recon_all.native_free import _run_surface_metrics
    from fnit.recon_all.surface_roi_gpu import vertex_volume_map
    from fnit.recon_all.surface_curvature_gpu import _neighbours
    from fnit.recon_all.surface_roi_curvature_gpu import principal_curvatures,_principal_curvatures_tensor
    from fnit.recon_all.surface_thickness_gpu import _normals
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    report={'scope':'same_input_same_device_existing_GPU_surface_metric_repeats_only_not_pipeline',
        'hostname':platform.node(),'code_base_commit':args.code_base_commit,'script_sha256':sha(__file__),
        'torch':torch.__version__,'numpy':np.__version__,'device':str(device),'threads':args.threads,
        'cpu_affinity_count':len(os.sched_getaffinity(0)),'repeat':args.repeat,'cases':[],
        'cuda_allocator_environment':{name:os.environ.get(name) for name in
            ('PYTORCH_NO_CUDA_MEMORY_CACHING','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF')},
        'cublas_workspace_environment':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),
        'half_precision':False,'RNG_use':'no stochastic algorithm or random seed changes in these metrics',
        'module_mutation':False,'TF32_disabled':False,'metrics_meaning':'TH3 vertex volume is not no-th3 regional GrayVol',
        'official_equivalence':'not_assessed; this probe compares current operator to itself',
        'process_GPU_memory':'external sampler; allocator peaks separately reported',
        'status':'started'}

    def precision():
        return {'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                'float32_matmul_precision':torch.get_float32_matmul_precision()}

    def save():
        report['source_sha256']={name:sha(module.__file__) for name,module in list(sys.modules.items())
            if name.startswith('fnit.recon_all') and getattr(module,'__file__',None) and Path(module.__file__).is_file()}
        p=args.output_directory/'report.json';temporary=p.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temporary.replace(p)

    save()
    try:
        setup=time.perf_counter();torch.cuda.synchronize(device)
        report['CUDA_setup_seconds']=time.perf_counter()-setup;report['gpu']=torch.cuda.get_device_name(device)
        for case_index,(subject_string,hemi,focus_name) in enumerate(args.case):
            if hemi not in ('lh','rh') or focus_name not in ('white','pial','inflated'):
                raise ValueError('invalid hemisphere/focus')
            subject=Path(subject_string);focus=subject/'surf'/f'{hemi}.{focus_name}'
            label=subject/'label'/f'{hemi}.cortex.label'
            input_paths=[subject/'surf'/f'{hemi}.{name}' for name in ('white','pial')]+[focus,label]
            hashes={str(p):sha(p) for p in input_paths}
            case={'input_sha256':hashes,'hemisphere':hemi,'focus_surface':focus_name,'API_runs':[],
                  'normal_runs':[],'frozen_normal_principal_runs':[]}
            report['cases'].append(case);reference_maps=None
            for iteration in range(args.repeat):
                isolated=args.output_directory/f'case{case_index}-repeat{iteration}'
                (isolated/'surf').mkdir(parents=True)
                for name in ('white','pial'):
                    shutil.copy2(subject/'surf'/f'{hemi}.{name}',isolated/'surf'/f'{hemi}.{name}')
                torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
                before=precision();started=time.perf_counter()
                seconds=_run_surface_metrics(args.native_binary,isolated,hemi,args.assets_directory,device=str(device))
                vertex_volume_map(white=isolated/'surf'/f'{hemi}.white',pial=isolated/'surf'/f'{hemi}.pial',
                    cortex_label=label,output=isolated/'surf'/f'{hemi}.volume',device=str(device))
                vertices,faces=fs.read_geometry(str(focus))
                k1,k2=principal_curvatures(vertices,faces,device=str(device))
                fs.write_morph_data(str(isolated/'surf'/f'{hemi}.focus.H'),np.float32((k1+k2)/2))
                fs.write_morph_data(str(isolated/'surf'/f'{hemi}.focus.K'),np.float32(k1*k2))
                torch.cuda.synchronize(device);elapsed=time.perf_counter()-started
                values={name:fs.read_morph_data(str(isolated/'surf'/f'{hemi}.{name}')) for name in
                    ('thickness','area','area.pial','curv','curv.pial','volume','focus.H','focus.K')}
                if reference_maps is None:reference_maps={name:x.copy() for name,x in values.items()}
                case['API_runs'].append({'iteration':iteration,'seconds_including_read_compute_write':elapsed,
                    'existing_run_surface_metric_timings':seconds,'precision_before':before,'precision_after':precision(),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
                    'maps':{name:array_report(x,reference_maps[name]) for name,x in values.items()}});save()
            # 保持同一真实几何，重复原法线函数；不替换成熟模块。
            vertices,faces=fs.read_geometry(str(focus));xyz=torch.as_tensor(np.float32(vertices),device=device)
            triangles=torch.as_tensor(np.int64(faces),device=device)
            normals=[]
            with torch.inference_mode():
                for iteration in range(args.repeat):
                    torch.cuda.synchronize(device);started=time.perf_counter();normal=_normals(xyz,triangles)
                    torch.cuda.synchronize(device);elapsed=time.perf_counter()-started
                    normals.append(normal)
                    case['normal_runs'].append({'iteration':iteration,'seconds':elapsed,
                        **array_report(normal.cpu().numpy(),normals[0].cpu().numpy())})
                _,_,neighbors,valid=_neighbours(faces,len(vertices))
                neighbors=torch.as_tensor(neighbors,device=device);valid=torch.as_tensor(valid,device=device)
                frozen=normals[0].clone();first=None
                for iteration in range(args.repeat):
                    torch.cuda.synchronize(device);started=time.perf_counter()
                    result=_principal_curvatures_tensor(xyz,frozen,neighbors,valid)
                    torch.cuda.synchronize(device);elapsed=time.perf_counter()-started
                    values=result.cpu().numpy();first=values.copy() if first is None else first
                    case['frozen_normal_principal_runs'].append({'iteration':iteration,'seconds':elapsed,
                        **array_report(values,first)})
            case['input_sha256_after']={str(p):sha(p) for p in input_paths}
            if hashes!=case['input_sha256_after']:raise RuntimeError('input geometry/label changed')
            save()
        report['status']='complete';save()
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
