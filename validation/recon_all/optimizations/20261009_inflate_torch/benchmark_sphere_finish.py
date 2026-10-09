"""冻结实际standard sphere清理入口并比较现有完整CPU/CUDA清理。

输入公开smoothwm包、声明资产/Conda程序和实际native_free SHA。
从实际完整球面链透明保存finish输入，仅用于诊断；候选不读参考修补。
每侧CPU/GPU/GPU/CPU保留完整投影、负面扩张、步长和停止规则。
只读trace记录每一步的负面/marked/位移/投影SHA，时间包含该诊断成本。
输出JSON、冻结NPZ和同序sphere；不代表原始T1整例或网格全通过。
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def trace_finish(function, vertices, faces, *, start_iteration, device):
    """只读逐轮诊断；不更改局部变量、顺序、设备或接受规则。"""
    from fnit.recon_all.mris_register_overlap import remove_overlap_sphere
    target = inspect.unwrap(remove_overlap_sphere)
    lines, first = inspect.getsourcelines(target)
    markers = [first + n for n, text in enumerate(lines)
               if text.lstrip().startswith("old_count = count")]
    if len(markers) != 1 or sys.gettrace() is not None:
        raise RuntimeError("cannot observe a unique frozen overlap boundary")
    history = []
    def observer(frame, event, arg):
        if frame.f_code is not target.__code__:
            return None
        if event == "line" and frame.f_lineno == markers[0]:
            values = frame.f_locals
            history.append({"iteration_before_increment": values["iteration"],
                "negative_count_before_step": values["count"],
                "dt": values["dt"], "max_neighbors": values["max_neighbors"],
                "minimum_negative": values["min_negative"],
                "minimum_iteration": values["min_iteration"],
                "last_expand": values["last_expand"], "same": values["same"],
                "negative_faces_sha256": array_sha(values["negative"]),
                "marked_sha256": array_sha(values["marked"]),
                "displacement_float32_sha256": array_sha(values["displacement"]),
                "projected_coordinates_float32_sha256": array_sha(values["xyz"])})
        return observer
    try:
        sys.settrace(observer)
        result, counts = function(vertices, faces, start_iteration=start_iteration,
                                  device=device)
    finally:
        sys.settrace(None)
    if len(history) != len(counts):
        raise RuntimeError("overlap trace count differs from actual complete iterations")
    return result, counts, history


def main():
    entered = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--native-free-sha256", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be a new directory")
    args.output.mkdir(parents=True)
    report = {"status": "running", "case": args.case, "hemisphere": args.hemisphere,
        "scope": "same-input complete standard-sphere finish after a new actual smoothwm chain; not raw-T1 recon-all",
        "device": args.device, "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "runs": [],
        "production_defaults_changed": False, "overall_metric_equivalence": "not assessed",
        "script_sha256": sha(__file__), "strict_input_manifest_sha256": sha(args.data / "manifest.json")}
    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        import numpy as np
        import torch
        from numba import set_num_threads
        from fnit.recon_all import native_free, sphere_standard_run
        from fnit.recon_all.sphere_standard_python import write_standard_sphere_surface
        from fnit.recon_all.profiling import ProcessTreeDeviceSampler, configure_cuda_allocator
        from benchmark_sphere_prepare_group import compare_sphere
        if sha(native_free.__file__) != args.native_free_sha256:
            raise ValueError("actual native_free differs from expected immutable SHA")
        torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
        set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        allocator = configure_cuda_allocator(args.device, "enabled")
        selected = torch.device(args.device)
        if selected.type != "cuda" or selected.index is None or args.threads < 1:
            raise ValueError("explicit CUDA device and positive thread budget required")
        report.update(allocator=allocator, native_free_sha256=sha(native_free.__file__),
            native_sha256=sha(args.native), torch_version=torch.__version__,
            cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(selected),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            autocast=torch.is_autocast_enabled("cuda"))
        rows=json.loads((args.data / "manifest.json").read_text())["cases"]
        row=next(row for row in rows if row["case"] == args.case
                 and row["hemisphere"] == args.hemisphere)
        original=args.data / row["surface"]
        if sha(original) != row["sha256"]:
            raise ValueError("public smoothwm SHA differs from manifest")
        subject=args.output / "captured_subject"
        (subject / "surf").mkdir(parents=True)
        (subject / "scripts").mkdir()
        smoothwm=subject / "surf" / (args.hemisphere + ".smoothwm")
        shutil.copy2(original, smoothwm)
        checkpoint=args.output / "finish_input.npz"
        original_finish=sphere_standard_run.finish_standard_sphere
        def capture(vertices, faces, *, start_iteration, device):
            if checkpoint.exists():
                raise RuntimeError("actual chain entered finish more than once")
            started=time.perf_counter()
            np.savez_compressed(checkpoint, vertices=np.asarray(vertices, np.float32),
                faces=np.asarray(faces, np.int32), start_iteration=np.int64(start_iteration))
            report["checkpoint_write_seconds"]=time.perf_counter() - started
            report["checkpoint_sha256"]=sha(checkpoint)
            save()
            return original_finish(vertices, faces, start_iteration=start_iteration, device=device)
        tick=time.perf_counter()
        try:
            sphere_standard_run.finish_standard_sphere=capture
            timings, captured=native_free._run_accurate_sphere_pair(
                inflate_binary=args.native, subject=subject, hemi=args.hemisphere,
                assets=args.assets, device=args.device, normals_backend="numba", inflate_backend="native")
        finally:
            sphere_standard_run.finish_standard_sphere=original_finish
        report.update(capture_chain_wall_seconds=time.perf_counter()-tick,
                      capture_timings=timings, capture_sphere=captured,
                      smoothwm_sha256=sha(smoothwm))
        save()
        print("CAPTURED",args.hemisphere, report["capture_chain_wall_seconds"], flush=True)
        inflated=subject / "surf" / (args.hemisphere+".inflated")
        for index, backend in enumerate(("cpu", args.device, args.device, "cpu")):
            full=time.perf_counter()
            read=time.perf_counter()
            with np.load(checkpoint,allow_pickle=False) as values:
                vertices=values["vertices"].copy(); faces=values["faces"].copy()
                iteration=int(values["start_iteration"])
            read_seconds=time.perf_counter()-read
            if sha(checkpoint) != report["checkpoint_sha256"]:
                raise ValueError("finish checkpoint changed during ABBA")
            sampler=ProcessTreeDeviceSampler(device=args.device,parent_pid=os.getpid(),interval=.5)
            sampler.sample_if_due(force=True)
            stop=threading.Event()
            def sample_loop():
                while not stop.wait(.05):sampler.sample_if_due()
            thread=threading.Thread(target=sample_loop,daemon=True);thread.start()
            torch.cuda.synchronize(selected);torch.cuda.reset_peak_memory_stats(selected)
            started=time.perf_counter()
            try:
                finished, counts, trace=trace_finish(original_finish,vertices,faces,
                    start_iteration=iteration,device=backend)
                torch.cuda.synchronize(selected)
                compute=time.perf_counter()-started
            finally:
                stop.set();thread.join()
            output=args.output / (str(index+1)+"_"+backend.replace(":","_")+".sphere")
            write=time.perf_counter()
            write_standard_sphere_surface(output,finished,faces,inflated)
            write_seconds=time.perf_counter()-write
            run={"backend":backend,"read_seconds":read_seconds,
                "complete_api_seconds_including_transfer_and_readonly_trace":compute,
                "write_seconds":write_seconds,"full_read_compute_write_sampler_seconds":time.perf_counter()-full,
                "output":str(output),"output_sha256":sha(output),"negative_counts":counts,
                "readonly_iterations":trace,"device_process_tree":sampler.report(),
                "torch_allocated_peak_bytes":torch.cuda.max_memory_allocated(selected),
                "torch_reserved_peak_bytes":torch.cuda.max_memory_reserved(selected),
                "torch_stats_scope":"target GPU allocator; CPU rows include only resident capture-chain CUDA context/tensors, not CPU memory"}
            report["runs"].append(run);save()
            print("DONE",args.hemisphere,backend,compute,"iterations",len(counts),flush=True)
        control=Path(report["runs"][0]["output"])
        comparisons=[]
        for run in report["runs"]:
            comparison=compare_sphere(Path(run["output"]),control)
            comparison.update(backend=run["backend"],
                negative_counts_equal=run["negative_counts"]==report["runs"][0]["negative_counts"],
                every_iteration_trace_equal=run["readonly_iterations"]==report["runs"][0]["readonly_iterations"],
                existing_geometry_diagnostic_gate_0p001mm=bool(comparison["max_mm"]<=.001))
            comparisons.append(comparison)
        report["comparisons"]=comparisons
        report["captured_chain_vs_same_cpu_checkpoint"]=compare_sphere(
            control,subject / "surf" / (args.hemisphere+".sphere"))
        report["status"]="complete_same_input_finish_abba"
        report["strict_reproduction"]="passed" if all(c["coordinates_equal"] and c["negative_counts_equal"] and c["every_iteration_trace_equal"] for c in comparisons) else "failed"
        report["module_sha256"]={name:sha(module.__file__) for name,module in sorted(sys.modules.items())
            if name.startswith("fnit.recon_all.") and getattr(module,"__file__",None)
            and Path(module.__file__).suffix==".py"}
        report["timing_scope"]="full finish API includes checkpoint-to-device/result-to-host and readonly trace; checkpoint read/write, cold actual-chain JIT/startup and sampler overhead separately retained"
    except Exception as error:
        report.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        traceback.print_exc()
    report["main_seconds"]=time.perf_counter()-entered;save()
    return 0 if report["status"]=="complete_same_input_finish_abba" else 1


if __name__=="__main__":
    raise SystemExit(main())
