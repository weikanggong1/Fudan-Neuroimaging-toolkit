"""完整自有数字拓扑真实输入回归；不调用原生程序，不称原始T1整例。"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import time

import nibabel.freesurfer.io as fsio
import numpy as np
from numba import set_num_threads
import torch

from fnit.recon_all.topology_volume_python import run_volume_topology
from fnit.recon_all.topology_preflight_python import topology_counts
from dataclasses import asdict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject',type=Path,action='append',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--code-commit',required=True)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.threads<1:
        raise ValueError('threads must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    modules=['topology_volume_python','tessellate_gpu','extract_main_component_python',
             'topology_preflight_python','mris_remove_intersection_python']
    report={'scope':'complete frozen pretess-volume topology candidate fileAPI; not rawT1 wholecase',
            'code_commit':args.code_commit,'script_sha256':sha(__file__),
            'source_sha256':{n:sha(importlib.import_module('fnit.recon_all.'+n).__file__) for n in modules},
            'host':platform.node(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'threads':args.threads,'device':args.device,'native_program_called':False,
            'algorithm':'different from original GA; ordered6/26 simple-point addition',
            'overall_equivalence':'not_assessed','cases':[],'execution_status':'running'}
    save=lambda: (args.output_dir/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for case,subject in enumerate(args.subject):
            for hemi,value in [('lh',255),('rh',127)]:
                target=args.output_dir/f'case{case}_{hemi}';target.mkdir()
                stage=run_volume_topology(input_volume=subject/'mri'/f'filled-pretess{value}.mgz',value=value,
                    output_volume=target/'filled.corrected.mgz',output_surface=target/f'{hemi}.orig.premesh',
                    device=args.device,source_surface=subject/'surf'/f'{hemi}.orig.nofix',
                    output_defect_labels=target/f'{hemi}.defect_labels',report_path=target/'topology.private.json')
                originals={}
                for name in ['orig.nofix','orig.premesh','orig']:
                    path=subject/'surf'/f'{hemi}.{name}'
                    xyz,faces=fsio.read_geometry(str(path))
                    originals[name]={'input_sha256':sha(path),'mesh':asdict(topology_counts(faces,len(xyz))),
                                     'coordinate_min_mm':xyz.min(axis=0).tolist(),'coordinate_max_mm':xyz.max(axis=0).tolist()}
                report['cases'].append({'case_number':case,'hemisphere':hemi,'subject':str(subject),
                    'candidate':stage,'preserved_source_and_frozen_GA_diagnostics':originals,
                    'vertex_correspondence':'different topology; no same-index comparison',
                    'triangle_surface_distance':'pending separate complete point-to-triangle comparison',
                    'downstream_sphere_white_pial_metrics':'not_run'})
                save()
        report['execution_status']='complete'
    except Exception as error:
        report['execution_status']='failed';report['failure_type']=type(error).__name__;report['failure']=str(error)
        raise
    finally:
        report['total_benchmark_seconds_including_io_and_quality']=time.perf_counter()-started
        save()


if __name__=='__main__':
    main()
