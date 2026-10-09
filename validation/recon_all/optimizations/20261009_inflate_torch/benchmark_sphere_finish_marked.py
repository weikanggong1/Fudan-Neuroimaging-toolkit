"""同一实际finish checkpoint比较完整dense/marked CPU/GPU，各两次。

输入是benchmark_sphere_finish从真实完整球面链捕获的NPZ与原inflated头。
候选全部完成后才读CPU参考文件比较。顺序CPU dense/CPU marked/GPU
dense/GPU marked，再反向；保留原负面/扩张/停止/全体投影并逐轮核验。
不代表原始T1整例，也不扩展验收门。只读trace成本包含在阶段时间内。
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

from benchmark_sphere_finish import array_sha, sha


def observed_finish(vertices,faces,*,iteration,device,backend):
    """记录实际marked行SOAP位移与完整投影；不写计算状态。"""
    from fnit.recon_all.sphere_standard_finish import finish_standard_sphere
    if backend=="dense":
        from fnit.recon_all.mris_register_overlap import remove_overlap_sphere as function
    else:
        from fnit.recon_all.mris_register_overlap_marked import remove_overlap_sphere_marked as function
    target=inspect.unwrap(function)
    lines,first=inspect.getsourcelines(target)
    marker=[first+n for n,line in enumerate(lines) if line.lstrip().startswith("old_count = count")]
    if len(marker)!=1 or sys.gettrace() is not None:
        raise RuntimeError("cannot identify complete overlap update boundary")
    trace=[]
    def observer(frame,event,arg):
        if frame.f_code is not target.__code__:return None
        if event=="line" and frame.f_lineno==marker[0]:
            values=frame.f_locals
            soap=(values["displacement"][values["marked"]] if backend=="dense"
                  else values["displacement"])
            trace.append({"iteration_before_increment":values["iteration"],
                "negative_count_before_step":values["count"],"dt":values["dt"],
                "max_neighbors":values["max_neighbors"],"minimum_negative":values["min_negative"],
                "minimum_iteration":values["min_iteration"],"last_expand":values["last_expand"],
                "same":values["same"],"negative_faces_sha256":array_sha(values["negative"]),
                "marked_sha256":array_sha(values["marked"]),"marked_soap_float32_sha256":array_sha(soap),
                "projected_coordinates_float32_sha256":array_sha(values["xyz"])})
        return observer
    try:
        sys.settrace(observer)
        xyz,counts=finish_standard_sphere(vertices,faces,start_iteration=iteration,
            device=device,overlap_backend=backend)
    finally:sys.settrace(None)
    if len(trace)!=len(counts):raise RuntimeError("iteration capture differs from actual loop count")
    return xyz,counts,trace


def main():
    entry=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--checkpoint-sha256",required=True)
    parser.add_argument("--case",required=True)
    parser.add_argument("--hemisphere",choices=("lh","rh"),required=True)
    parser.add_argument("--inflated-template",type=Path,required=True)
    parser.add_argument("--dense-reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    if args.output.exists() or sha(args.checkpoint)!=args.checkpoint_sha256:
        raise ValueError("new output and exact immutable checkpoint required")
    args.output.mkdir(parents=True)
    report={"status":"running","scope":"fixed-input full standard-sphere finish CPU/GPU dense/marked; not end-to-end recon-all",
        "case":args.case,"hemisphere":args.hemisphere,
        "checkpoint_sha256":sha(args.checkpoint),"inflated_template_sha256":sha(args.inflated_template),
        "script_sha256":sha(__file__),"runs":[],"production_defaults_changed":False,
        "overall_metric_equivalence":"not assessed","threads":args.threads,
        "cpu_affinity":sorted(os.sched_getaffinity(0))}
    def save():(args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    save()
    try:
        import numpy as np
        import torch
        from numba import set_num_threads
        from fnit.recon_all.profiling import ProcessTreeDeviceSampler,configure_cuda_allocator
        from fnit.recon_all.sphere_standard_python import write_standard_sphere_surface
        from benchmark_sphere_prepare_group import compare_sphere
        torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        selected=torch.device(args.device)
        if selected.type!="cuda" or selected.index is None:raise ValueError("explicit CUDA index required")
        allocator=configure_cuda_allocator(args.device,"enabled")
        with np.load(args.checkpoint,allow_pickle=False) as c:
            vertices=c["vertices"].copy();faces=c["faces"].copy();iteration=int(c["start_iteration"])
        report.update(allocator=allocator,gpu=torch.cuda.get_device_name(selected),
            checkpoint_vertices_sha256=array_sha(vertices),checkpoint_faces_sha256=array_sha(faces),
            checkpoint_vertices=len(vertices),checkpoint_faces=len(faces),start_iteration=iteration,
            torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
            autocast=torch.is_autocast_enabled("cuda"))
        strategies=[("cpu","dense"),("cpu","marked"),(args.device,"dense"),(args.device,"marked")]
        for index,(device,backend) in enumerate(strategies+list(reversed(strategies))):
            sampler=ProcessTreeDeviceSampler(device=args.device,parent_pid=os.getpid(),interval=.5)
            sampler.sample_if_due(force=True);stop=threading.Event()
            def sample_loop():
                while not stop.wait(.05):sampler.sample_if_due()
            thread=threading.Thread(target=sample_loop,daemon=True);thread.start()
            torch.cuda.synchronize(selected);torch.cuda.reset_peak_memory_stats(selected)
            tick=time.perf_counter()
            try:
                xyz,counts,trace=observed_finish(vertices,faces,iteration=iteration,device=device,backend=backend)
                torch.cuda.synchronize(selected);elapsed=time.perf_counter()-tick
            finally:stop.set();thread.join()
            path=args.output/(str(index+1)+"_"+backend+"_"+device.replace(":","_")+".sphere")
            tick=time.perf_counter();write_standard_sphere_surface(path,xyz,faces,args.inflated_template)
            report["runs"].append({"backend":backend,"device":device,"full_api_seconds_with_transfer_and_readonly_trace":elapsed,
                "write_seconds":time.perf_counter()-tick,"output":str(path),"output_sha256":sha(path),
                "negative_counts":counts,"readonly_iterations":trace,"device_process_tree":sampler.report(),
                "torch_allocated_peak_bytes":torch.cuda.max_memory_allocated(selected),
                "torch_reserved_peak_bytes":torch.cuda.max_memory_reserved(selected)})
            save();print("DONE",device,backend,elapsed,"iterations",len(counts),flush=True)
        baseline=report["runs"][0];comparisons=[]
        for run in report["runs"]:
            c=compare_sphere(Path(run["output"]),Path(baseline["output"]))
            c.update(backend=run["backend"],device=run["device"],
                negative_counts_equal=run["negative_counts"]==baseline["negative_counts"],
                every_iteration_trace_equal=run["readonly_iterations"]==baseline["readonly_iterations"],
                existing_geometry_diagnostic_gate_0p001mm=bool(c["max_mm"]<=.001))
            comparisons.append(c)
        report["comparisons"]=comparisons
        report["prior_dense_checkpoint_comparison"]=compare_sphere(Path(baseline["output"]),args.dense_reference)
        report.update(status="complete_full_finish_eight_runs",
            strict_reproduction="passed" if all(c["coordinates_equal"] and c["negative_counts_equal"] and c["every_iteration_trace_equal"] for c in comparisons) else "failed",
            module_sha256={n:sha(m.__file__) for n,m in sorted(sys.modules.items()) if n.startswith("fnit.recon_all.") and getattr(m,"__file__",None) and Path(m.__file__).suffix==".py"},
            timing_scope="complete finish API and full readonly trajectory; I/O separately measured; first marked call includes CSR JIT; shared node")
    except Exception as error:
        report.update(status="failed",error=repr(error),traceback=traceback.format_exc());traceback.print_exc()
    report["main_seconds"]=time.perf_counter()-entry;save()
    return 0 if report["status"]=="complete_full_finish_eight_runs" else 1


if __name__=="__main__":raise SystemExit(main())
