"""真实WM冻结thicken输入：旧平面孔填充与Torch静态缓存/有序Numba ABBA。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import threading
import inspect

import nibabel as nib
import numpy as np
import torch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def compare(actual, expected):
    delta = np.abs(actual.astype(np.int16)-expected.astype(np.int16))
    labels = np.union1d(np.unique(actual),np.unique(expected))
    dice = {str(int(label)): float(2*np.count_nonzero((actual==label)&(expected==label)) /
        (np.count_nonzero(actual==label)+np.count_nonzero(expected==label))) for label in labels}
    return {"different_voxels":int(np.count_nonzero(delta)),"max_abs":int(delta.max()),
            "p99_abs":float(np.percentile(delta,99)),"label_dice":dice}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--native-output",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--module-dir",type=Path,required=True)
    p.add_argument("--device",default="cuda:1")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--code-commit",required=True)
    p.add_argument("--checkpoint-dir", type=Path,
                   help="已完成FNIT自产冻结阶段目录；复核输入/旧模块/检查点SHA后复用，不称原始T1整例")
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and device.index is None):
        raise ValueError("require CPU or an explicit target CUDA device")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.zeros(1,device=device);synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    memory_stop=threading.Event()
    memory_rows,memory_errors=[],[]
    memory_started=time.perf_counter()

    def sample_card():
        while not memory_stop.is_set():
            try:
                if device.type != "cuda":
                    return
                free,total=torch.cuda.mem_get_info(device)
                memory_rows.append({"t_seconds":time.perf_counter()-memory_started,
                    "free_bytes":int(free),"total_bytes":int(total),"used_bytes":int(total-free)})
            except RuntimeError as error:
                memory_errors.append(str(error))
            memory_stop.wait(.25)

    memory_thread=threading.Thread(target=sample_card,daemon=True)
    memory_thread.start()
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(args.module_dir.resolve()))
    from fnit.recon_all import mri_segment as wm
    from fnit.recon_all import mri_segment_planar_torch as planar
    old_planar=wm._fill_planar_holes
    old_thicken=wm._thicken_strands_core
    captured={}
    reference=None
    report={"scope":"frozen_real_complete_thicken_stage_ABBA; not whole recon timing",
        "code_commit":args.code_commit,"host":platform.node(),"device":str(device),
        "code_commit_role":"frozen package baseline plus executed module SHA-256 overlay",
        "cpu_affinity":sorted(os.sched_getaffinity(0)),"threads":args.threads,
        "modules_sha256":{Path(m.__file__).name:sha(m.__file__) for m in (wm,planar)},
        "script_sha256":sha(__file__),"source_sha256":sha(args.source),
        "native_output_sha256":sha(args.native_output),"records":[],
        "torch":torch.__version__,"python":platform.python_version(),"numpy":np.__version__,
        "matmul_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_tf32":torch.backends.cudnn.allow_tf32,
        "no_half":True,"gpu_uuid":str(torch.cuda.get_device_properties(device).uuid) if device.type == "cuda" else None}

    def capture(image,source,closed,segments,**parameters):
        captured.update(image=image.copy(),source=source.copy(),closed=closed.copy(),
                        segments=[[tuple(point) for point in segment] for segment in segments],parameters=parameters)
        return old_thicken(image,source,closed,segments,**parameters)

    prior_file=args.output_dir/"existing-torch-histogram.mgz"
    checkpoint_dir = args.output_dir
    if args.checkpoint_dir is None:
        wm._thicken_strands_core=capture
        tick=time.perf_counter()
        try:
            prior_api=wm.segment_white_matter_mgz(source_path=args.source,output_path=prior_file,
                device=device,histogram_backend="torch",histogram_batch_size=2048)
            synchronize()
        finally:
            wm._thicken_strands_core=old_thicken
        report["capture_full_API"]={"seconds":time.perf_counter()-tick,"api":prior_api,
            "timing_includes_diagnostic_input_copies":True}
        np.savez_compressed(args.output_dir/"wm_thicken_frozen.npz",image=captured["image"],
            source=captured["source"],closed=captured["closed"])
        (args.output_dir/"segments.json").write_text(json.dumps(captured["segments"]))
    else:
        checkpoint_dir = args.checkpoint_dir
        receipt = json.loads((checkpoint_dir/"report.json").read_text())
        if (receipt["source_sha256"] != sha(args.source)
                or receipt["modules_sha256"]["mri_segment.py"] != sha(wm.__file__)
                or receipt["checkpoint"]["arrays_sha256"] != sha(checkpoint_dir/"wm_thicken_frozen.npz")
                or receipt["checkpoint"]["segments_sha256"] != sha(checkpoint_dir/"segments.json")):
            raise ValueError("declared same-input checkpoint/source/module hash mismatch")
        with np.load(checkpoint_dir/"wm_thicken_frozen.npz", allow_pickle=False) as arrays:
            captured.update({name: arrays[name].copy() for name in ("image", "source", "closed")})
        captured["segments"] = [[tuple(point) for point in segment]
                                for segment in json.loads((checkpoint_dir/"segments.json").read_text())]
        captured["parameters"] = receipt["checkpoint"]["parameters"]
        prior_file=checkpoint_dir/"existing-torch-histogram.mgz"
        report["capture_full_API"]={"status":"not_run; hash-verified FNIT same-input checkpoint reused",
            "origin_report_sha256":sha(checkpoint_dir/"report.json")}
    report["checkpoint"]={"arrays_sha256":sha(checkpoint_dir/"wm_thicken_frozen.npz"),
        "segments_sha256":sha(checkpoint_dir/"segments.json"),"parameters":captured["parameters"],
        "shape":[int(axis) for axis in captured["source"].shape],"dtype":str(captured["source"].dtype),
        "candidate_input_policy":"FNIT current same-input outputs; no native intermediate read by candidate"}
    current_planar_rows=[]

    def fast_planar(result,strand):
        start=time.perf_counter()
        row=planar.fill_planar_holes_cached(result=result,strand=strand,
            device=str(device),batch_size=args.batch_size)
        synchronize()
        row["seconds"]=time.perf_counter()-start
        current_planar_rows.append(row)

    def call(backend):
        wm._fill_planar_holes=fast_planar if backend=="cached" else old_planar
        try:
            return old_thicken(captured["image"],captured["source"],captured["closed"],
                captured["segments"],**captured["parameters"])
        finally:
            wm._fill_planar_holes=old_planar

    tick=time.perf_counter()
    call("cached");synchronize()
    report["cold_cached_stage_seconds"]=time.perf_counter()-tick
    current_planar_rows.clear()
    for index,backend in enumerate(("python","cached","cached","python")):
        synchronize();tick=time.perf_counter()
        result,strands=call(backend);synchronize()
        seconds=time.perf_counter()-tick
        if reference is None:
            reference=(result.copy(),[x.copy() for x in strands])
        row={"backend":backend,"seconds":seconds,"output":compare(result,reference[0]),
             "ordered_component_count":len(strands),"component_maps_different_elements":
             sum(int(np.count_nonzero(lhs!=rhs)) for lhs,rhs in zip(strands,reference[1])),
             "planar_rows_nested_do_not_sum":list(current_planar_rows)}
        current_planar_rows.clear();report["records"].append(row)
        (args.output_dir/"partial_report.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(row),flush=True)
    report["median_seconds"]={backend:float(np.median([x["seconds"] for x in report["records"]
        if x["backend"]==backend])) for backend in ("python","cached")}
    report["stage_speedup"]=report["median_seconds"]["python"]/report["median_seconds"]["cached"]
    wm._fill_planar_holes=fast_planar
    final_file=args.output_dir/"cached-full-WM.mgz"
    synchronize();tick=time.perf_counter()
    public_options = {}
    if "planar_backend" in inspect.signature(wm.segment_white_matter_mgz).parameters:
        public_options = {"planar_backend": "cached", "planar_batch_size": args.batch_size}
        wm._fill_planar_holes=old_planar
    report["complete_API_wiring"] = "public explicit backend" if public_options else "benchmark-only wrapper on frozen old API"
    final_api=wm.segment_white_matter_mgz(source_path=args.source,output_path=final_file,
        device=device,histogram_backend="torch",histogram_batch_size=2048, **public_options)
    synchronize()
    report["cached_complete_API_seconds"]=time.perf_counter()-tick
    report["cached_complete_API"]=final_api
    final_image,prior_image,native_image=[nib.load(path) for path in (final_file,prior_file,args.native_output)]
    final=np.asarray(final_image.dataobj)
    report["complete_vs_existing_Torch_histogram"]=compare(final,np.asarray(prior_image.dataobj))
    report["existing_Torch_histogram_output_sha256"]=sha(prior_file)
    report["complete_vs_native"]=compare(final,np.asarray(native_image.dataobj))
    report["complete_affine_equal"]=bool(np.array_equal(final_image.affine,prior_image.affine))
    report["complete_dtype_equal"]=final_image.get_data_dtype()==prior_image.get_data_dtype()
    report["complete_MGH_header_equal"]=final_image.header.binaryblock==prior_image.header.binaryblock
    report["torch_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    report["torch_peak_reserved_bytes"]=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    memory_stop.set();memory_thread.join(timeout=5)
    intervals=np.diff([row["t_seconds"] for row in memory_rows])
    report["target_card_memory_upper_bound"]={"scope":"explicit target CUDA total-free; includes all processes/driver",
        "requested_interval_seconds":.25,"max_interval_seconds":float(intervals.max()) if len(intervals) else None,
        "peak_card_used_bytes":max((row["used_bytes"] for row in memory_rows),default=None),
        "errors":memory_errors,"samples":memory_rows,
        "process_memory":"not measurable from earlier nvidia-smi PID queries; no zero-memory claim"}
    report["whole_recon_speedup"]="not_measured"
    report["whole_metric_equivalence"]="not_assessed"
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
