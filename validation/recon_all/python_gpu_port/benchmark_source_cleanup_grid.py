"""真实冻结surface的完整有向清理file API：GPU2/3桶ABBA及当前源码构建参考。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

import nibabel.freesurfer.io as fs
import numpy as np
import torch

from benchmark_placement_full_white import compare_geometry,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-surface",type=Path,required=True)
    parser.add_argument("--candidate-directory",type=Path,required=True)
    parser.add_argument("--output-directory",type=Path,required=True)
    parser.add_argument("--code-base-commit",required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--native-binary",type=Path)
    parser.add_argument("--native-repeat",type=int,default=2)
    args=parser.parse_args()
    if args.threads<1 or args.native_repeat<1:raise ValueError("positive thread and repeat counts required")
    device=torch.device(args.device)
    if device.type!="cuda" or device.index is None:raise ValueError("explicitly indexed CUDA device required")
    if args.output_directory.exists():raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(args.candidate_directory.resolve()))
    from fnit.recon_all import place_surface_final_cleanup as cleanup
    from fnit.recon_all.place_pial_python import _write_vertices_like
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    report={"scope":"frozen_real_initial_source_cleanup_file_API_grid_ABBA_not_white_or_recon_all",
        "hostname":platform.node(),"threads":args.threads,"cpu_affinity_count":len(os.sched_getaffinity(0)),
        "code_base_commit":args.code_base_commit,"script_sha256":sha256(__file__),
        "input_surface_sha256":sha256(args.input_surface),"device":str(device),
        "torch":torch.__version__,"numpy":np.__version__,"tf32_matmul":True,"tf32_cudnn":True,
        "half_precision":False,"cuda_allocator_environment":{name:os.environ.get(name) for name in
            ("PYTORCH_NO_CUDA_MEMORY_CACHING","PYTORCH_CUDA_ALLOC_CONF","PYTORCH_ALLOC_CONF")},
        "stage_timing_scope":"read actual frozen mesh, complete source cleanup and diagnostic surface write, synchronized GPU; CUDA context/import separate",
        "required_tolerance_before_run":{"coordinates_mm":0,"ordered_faces":0,"source_cleanup_trace":0},
        "intermediate_cleanup_residual_rule":"source best-state stopping allowed; complete white final zero gate unchanged",
        "reference_kind":"current independently Conda source-built program, fresh same-host same-input; not installed official",
        "overall_metric_equivalence":"not_assessed","whole_recon_all":"not_run","runs":[],"native_runs":[]}

    def save(status):
        report["status"]=status
        report["source_sha256"]={name:sha256(module.__file__) for name,module in list(sys.modules.items())
            if name.startswith('fnit.recon_all') and getattr(module,'__file__',None) and Path(module.__file__).is_file()}
        path=args.output_directory/'report.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)

    save('started');setup=time.perf_counter()
    try:
        torch.cuda.synchronize(device)
        report['cuda_setup_seconds']=time.perf_counter()-setup;report['gpu']=torch.cuda.get_device_name(device)
        outputs=[]
        for index,(kind,grid) in enumerate((('cold',2),('cold',3),('paired',2),('paired',3),('paired',3),('paired',2))):
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            started=time.perf_counter();vertices,faces=fs.read_geometry(str(args.input_surface));loaded=time.perf_counter()
            result,stats=cleanup.repair_intersections(vertices,faces,np.zeros(len(vertices),bool),
                marking_backend='source_torch',device=str(device),candidate_grid_cells_per_axis=grid)
            torch.cuda.synchronize(device);computed=time.perf_counter()
            output=args.output_directory/f'cleanup-{index}-grid{grid}.diagnostic-surface'
            _write_vertices_like(args.input_surface,output,result);torch.cuda.synchronize(device)
            finished=time.perf_counter();outputs.append(output)
            report['runs'].append({'kind':kind,'grid_cells_per_axis':grid,'file_API_seconds':finished-started,
                'read_seconds':loaded-started,'compute_seconds':computed-loaded,'write_seconds':finished-computed,
                'cleanup':stats,'coordinate_sha256':hashlib.sha256(result.tobytes()).hexdigest(),
                'surface_sha256':sha256(output),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
                'comparison_to_first_control':compare_geometry(outputs[0],output)})
            save('running')
        first=report['runs'][0]
        report['strict_backend_reproduction']='passed' if all(
            row['comparison_to_first_control']['different_coordinate_elements']==0
            and row['comparison_to_first_control']['same_vertex_count_and_ordered_faces']
            and row['cleanup']==first['cleanup'] for row in report['runs']) else 'failed'
        medians={str(grid):statistics.median(row['file_API_seconds'] for row in report['runs']
            if row['kind']=='paired' and row['grid_cells_per_axis']==grid) for grid in (2,3)}
        report['paired_median_file_API_seconds']=medians
        report['wall_reduction_percent_observation']=100*(1-medians['3']/medians['2'])
        if args.native_binary is not None:
            report['native_program_sha256']=sha256(args.native_binary)
            native_outputs=[]
            for index in range(args.native_repeat):
                output=args.output_directory/f'native-{index}.diagnostic-surface'
                command=[str(args.native_binary.resolve()),str(args.input_surface.resolve()),str(output.resolve())]
                started=time.perf_counter()
                with (args.output_directory/f'native-{index}.private.log').open('w') as stream:
                    run=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
                row={'command':command,'seconds_cold_CLI_including_read_compute_write':time.perf_counter()-started,
                    'returncode':run.returncode}
                report['native_runs'].append(row)
                if run.returncode!=0:save('failed_native');raise RuntimeError('native cleanup failed; private log retained')
                native_outputs.append(output);row['comparison_to_GPU']=compare_geometry(outputs[0],output)
                row['comparison_to_first_native']=compare_geometry(native_outputs[0],output)
                row['output_sha256']=sha256(output);save('running')
        if sha256(args.input_surface)!=report['input_surface_sha256']:raise RuntimeError('frozen input changed')
        save('complete')
        if report['strict_backend_reproduction']!='passed':raise SystemExit(1)
    except Exception as exc:
        report.update(error=str(exc),traceback=traceback.format_exc());save('failed');raise


if __name__=='__main__':main()
