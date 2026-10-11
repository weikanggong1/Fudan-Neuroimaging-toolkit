"""真实网格完整F64相交检查：输入/源码哈希、完整进程与分步时间。"""
import time
START = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vertices-npy', type=Path, help='真实冻结坐标 (N,3)；面与顺序仍取 --input，不修改原文件')
    parser.add_argument('--backend', choices=('python', 'numba', 'torch'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-uuid', default='')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--oracle-source', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import nibabel.freesurfer.io as fs
    import numba
    import numpy as np
    import fnit
    import fnit.recon_all.mris_remove_intersection_python as marker
    numba.set_num_threads(args.threads)
    samples, finished = [], threading.Event()
    def sample():
        while not finished.is_set():
            tick = time.perf_counter()
            row = {'elapsed': tick-START}
            try:
                r = subprocess.run(['nvidia-smi','-i',args.gpu_uuid,'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4)
                if r.returncode:
                    raise RuntimeError('GPU context query failed')
                uuid, mib = r.stdout.strip().split(',')
                row.update(device_uuid=uuid.strip(), device_total_context_bytes=int(mib)*1024*1024)
            except Exception as exc:
                row['query_error']=type(exc).__name__
            samples.append(row)
            finished.wait(.5)
    monitor = threading.Thread(target=sample, daemon=True) if args.gpu_uuid else None
    if monitor:
        monitor.start()
    report = {'status':'running','backend':args.backend,'source_commit':args.source_commit,
              'benchmark_script_sha256':sha(__file__),'input_sha256':sha(args.input),
              'input_bytes':args.input.stat().st_size,'threads':args.threads,
              'cpu_affinity':sorted(os.sched_getaffinity(0)),'load_before':os.getloadavg(),
              'overall_equivalence':'not_assessed','scope':'complete real same-input F64 intersection marking; not surface repair, GA or whole T1',
              'gpu_uuid':args.gpu_uuid,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'numpy_version':np.__version__,'numba_version':numba.__version__,
              'source_sha256':{'marker':sha(marker.__file__)},'fnit_module_location_private':fnit.__file__,
              'env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')}}
    if args.backend != 'python':
        import fnit.recon_all.topology_triangle_plane_filter as plane
        report['source_sha256']['plane_filter']=sha(plane.__file__)
    torch = None
    if args.backend == 'torch':
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        report['torch_device']={'name':torch.cuda.get_device_properties(args.device).name,'uuid':str(getattr(torch.cuda.get_device_properties(args.device),'uuid','unavailable')),'logical_device':str(args.device)}
        report['forward_precision']={'dtype':'float64','autocast':False,'tf32_matmul_actual':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn_actual':torch.backends.cudnn.allow_tf32}
    try:
        tick=time.perf_counter();vertices,faces=fs.read_geometry(str(args.input));report['load_seconds']=time.perf_counter()-tick
        if args.vertices_npy is not None:
            coordinate_start=time.perf_counter()
            frozen=np.load(args.vertices_npy,allow_pickle=False)
            if frozen.shape != vertices.shape or not np.isfinite(frozen).all():
                raise ValueError('frozen coordinates must match ordered (N,3) mesh')
            report.update(coordinate_sha256=sha(args.vertices_npy),coordinate_bytes=args.vertices_npy.stat().st_size,coordinate_dtype=str(frozen.dtype))
            vertices=frozen
            report['coordinate_load_seconds']=time.perf_counter()-coordinate_start
        report.update(vertices=len(vertices),faces=len(faces))
        calls, eliminated = {'original_predicate':0,'original_predicate_seconds':0.0}, {'certified':0,'eligible_pairs':0,'plane_filter_seconds':0.0}
        predicate=marker._triangles_intersect
        def observed(a,b):
            calls['original_predicate']+=1
            predicate_start=time.perf_counter()
            try:
                return predicate(a,b)
            finally:
                calls['original_predicate_seconds']+=time.perf_counter()-predicate_start
        marker._triangles_intersect=observed
        if args.backend != 'python':
            original_filter=plane.separated_triangle_pairs
            def observed_filter(**kwargs):
                filter_start=time.perf_counter()
                output=original_filter(**kwargs)
                eliminated['plane_filter_seconds']+=time.perf_counter()-filter_start
                eliminated['eligible_pairs']+=len(output);eliminated['certified']+=int(output.sum())
                return output
            plane.separated_triangle_pairs=observed_filter
        tick=time.perf_counter()
        if args.profile:
            import cProfile
            profiler=cProfile.Profile();profiler.enable()
        if args.oracle_source:
            marks,count=marker.mark_intersections(vertices,faces)
        else:
            marks,count=marker.mark_intersections(vertices,faces,separation_backend=args.backend,device=args.device)
        if torch is not None:
            torch.cuda.synchronize(args.device)
        if args.profile:
            profiler.disable();profiler.dump_stats(str(args.output/'profile.private.pstats'))
        report['complete_marking_seconds']=time.perf_counter()-tick
        report.update(marked_faces=int(count),marked_vertices=int(marks.sum()),predicate_calls=calls['original_predicate'],original_predicate_seconds=calls['original_predicate_seconds'],**eliminated)
        report['marking_remainder_seconds']=report['complete_marking_seconds']-report['original_predicate_seconds']-report['plane_filter_seconds']
        np.save(args.output/'marks.private.npy',marks,allow_pickle=False)
        report['marked_vertices_sha256']=sha(args.output/'marks.private.npy')
        if torch is not None:
            report['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
            report['torch_peak_reserved_bytes']=torch.cuda.max_memory_reserved(args.device)
        report['status']='complete'
    except Exception as exc:
        report.update(status='failed',failure=repr(exc))
        raise
    finally:
        finished.set()
        if monitor:
            monitor.join(5)
        report.update(script_wall_before_final_report_seconds=time.perf_counter()-START,maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,load_after=os.getloadavg())
        report['gpu_observation']={'sampling_interval_requested_seconds':.5,'maximum_sampling_gap_seconds':max((b['elapsed']-a['elapsed'] for a,b in zip(samples,samples[1:])),default=None),'query_failures':sum('query_error'in s for s in samples),'process_tree_bytes':None,'process_attribution':'not_verified; namespace mismatch cannot be interpreted as zero; device totals are context only','continuous_process_peak_verified':False,'samples_private':samples}
        (args.output/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report.get(k)for k in ('status','backend','complete_marking_seconds','script_wall_before_final_report_seconds','marked_faces','marked_vertices','predicate_calls','certified')},indent=2),flush=True)


if __name__=='__main__':
    main()
