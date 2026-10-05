"""Resume one saved FNIT decoder tail or replay its posterior; no full CNN."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):
            digest.update(block)
    return digest.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def tensor_sha(value):
    array=value.detach().numpy()
    assert array.flags.c_contiguous and array.shape[0]==1
    digest=hashlib.sha256()
    for channel in range(array.shape[1]):
        digest.update(memoryview(array[0,channel]).cast("B"))
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","source","plan","checkpoint","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--mode",choices=("resume","baseline","candidate"),required=True)
    p.add_argument("--preblur",type=Path)
    p.add_argument("--reference",type=Path)
    args=p.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    plan=json.loads(args.plan.read_text())
    root=args.root
    before={name:sha(args.source/"fnit/synthseg_parc"/name) for name in plan["source_files"]}
    assert before==plan["source_files"]
    sys.path.insert(0,str(args.source))
    import numpy as np
    import torch
    from torch.nn import functional as F
    import fnit
    from fnit.synthseg_parc import segment
    from fnit.synthseg_parc.cpu_join import join_nearest_cpu
    import channel_batch as trial
    assert Path(fnit.__file__).resolve()==(args.source/"fnit/__init__.py").resolve()
    assert sha(Path(trial.__file__))==plan["prototype_files"]["channel_batch.py"]
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    assert sorted(os.sched_getaffinity(0))==plan["cpu_affinity"]
    assert not torch.cuda.is_initialized()
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
    def flags():
        return {"matmul_tf32":bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_tf32":bool(torch.backends.cudnn.allow_tf32),
                "mkldnn":bool(torch.backends.mkldnn.enabled),
                "cpu_autocast":bool(torch.is_autocast_enabled("cpu")),
                "grad_enabled":bool(torch.is_grad_enabled()),
                "cuda_initialized":bool(torch.cuda.is_initialized())}
    initial=flags()
    report={"schema":"fnit_seg_cpu_blur_real_trial/v1","mode":args.mode,
        "scope":"Saved real FNIT terminal/prefilter diagnostic, not a whole T1 or native benchmark",
        "worker_sha256":sha(__file__),"plan_sha256":sha(args.plan),
        "helper_sha256":sha(trial.__file__),"source_before":before,
        "hostname":os.uname().nodename,"cpu_affinity":sorted(os.sched_getaffinity(0)),
        "torch_version":torch.__version__,"torch_parallel_info":torch.__config__.parallel_info(),
        "torch_configuration":torch.__config__.show(),"threads":torch.get_num_threads(),
        "interop_threads":torch.get_num_interop_threads(),"initial_flags":initial,
        "input_sha256":plan["input_sha256"],"weights":plan["weights"]}
    t0=time.perf_counter()
    if args.mode=="resume":
        manifest=json.loads((args.checkpoint/"manifest.private.json").read_text())
        assert manifest["status"]=="partial_capture_complete"
        assert manifest["source_files"]==plan["producer_files"]
        assert manifest["checkpoint_files"]==plan["checkpoint_files"]
        assert manifest["input_sha256"]==plan["input_sha256"]
        assert manifest["prepared_values_sha256"]==plan["prepared_values_sha256"]
        weights=root/"workspaces/smri_cpu_20261004/assets/weights"
        weight=weights/"synthseg_2.0.h5"
        assert sha(weight)==plan["weights"][weight.name]["sha256"]
        assert weight.stat().st_size==plan["weights"][weight.name]["bytes"]
        for name,info in plan["checkpoint_files"].items():
            assert sha(args.checkpoint/name)==info["sha256"]
            assert (args.checkpoint/name).stat().st_size==info["bytes"]
        model=segment.SegmentUNet().load_h5(weight).eval()
        skip=torch.from_numpy(np.load(args.checkpoint/"skip.npy"))
        value=torch.from_numpy(np.load(args.checkpoint/"value.npy"))
        assert tuple(skip.shape)==(1,24,192,224,256) and tuple(value.shape)==(1,48,96,112,128)
        assert skip.dtype==value.dtype==torch.float32
        report["checkpoint_files"]=manifest["checkpoint_files"]
        with torch.inference_mode(),torch.backends.mkldnn.flags(enabled=False):
            joined=join_nearest_cpu(skip,value)
            assert tensor_sha(joined)==plan["prior_join_bit_identity_sha256"]
            del skip,value
            usage_before=resource.getrusage(resource.RUSAGE_SELF)
            conv0_wall=time.perf_counter()
            # This CPU-only dispatcher profiler adds no Module hooks and no
            # shape, stack or memory collection. Native BLAS inner operations
            # might remain contained within the slow_conv3d event.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=False,with_stack=False,profile_memory=False) as profiler:
                first=model.up[3].conv0(joined)
            conv0_wall=time.perf_counter()-conv0_wall
            usage_after=resource.getrusage(resource.RUSAGE_SELF)
            del joined
            conv0_user=usage_after.ru_utime-usage_before.ru_utime
            conv0_system=usage_after.ru_stime-usage_before.ru_stime
            report["up3_conv0_observed"]={"wall_seconds":conv0_wall,
                "process_user_seconds":conv0_user,"process_system_seconds":conv0_system,
                "observed_process_CPU_seconds_per_wall_second":(conv0_user+conv0_system)/conv0_wall,
                "record_shapes":False,"with_stack":False,"profile_memory":False,"module_hooks":False,
                "operators":[{"name":event.key,"calls":event.count,
                    "cpu_inclusive_us":event.cpu_time_total,"cpu_exclusive_us":event.self_cpu_time_total}
                    for event in profiler.key_averages()],
                "limitations":"Instrumented sole decoder-tail resume; CPU ratio is this process total, not per-kernel occupancy; native GEMM/unfold may lack distinct ATen events."}
            profiler.export_chrome_trace(str(args.output/"conv0_trace.private.json"))
            x=F.elu(first)
            del first
            x=F.elu(model.up[3].conv1(x))
            x=model.up[3].bn(x)
            logits=model.likelihood(x)
            del x,model
            posterior=torch.softmax(logits,dim=1)
            del logits
        assert tuple(posterior.shape)==(1,33,192,224,256)
        assert posterior.dtype==torch.float32 and posterior.is_contiguous()
        assert bool(torch.isfinite(posterior).all())
        file=args.output/"preblur.private.npy"
        np.save(file,posterior.numpy())
        report.update(status="terminal_preblur_saved",terminal_observed_seconds=time.perf_counter()-t0,
            preblur_file={"sha256":sha(file),"bytes":file.stat().st_size,
                         "value_sha256":tensor_sha(posterior),"shape":list(posterior.shape),
                         "stride":list(posterior.stride()),"dtype":str(posterior.dtype)},
            full_CNN_executed=False,whole_map_CSV_assessed=False)
    else:
        assert args.preblur is not None
        preblur_manifest=json.loads((args.preblur.parent/"report.private.json").read_text())
        assert preblur_manifest["status"]=="terminal_preblur_saved"
        assert preblur_manifest["source_before"]==before
        assert sha(args.preblur)==preblur_manifest["preblur_file"]["sha256"]
        posterior=torch.from_numpy(np.load(args.preblur))
        assert tensor_sha(posterior)==preblur_manifest["preblur_file"]["value_sha256"]
        report["preblur_file"]=preblur_manifest["preblur_file"]
        usage_before=resource.getrusage(resource.RUSAGE_SELF)
        with torch.inference_mode(),torch.backends.mkldnn.flags(enabled=False):
            t1=time.perf_counter()
            actual=(trial.blur(posterior,segment._blur) if args.mode=="candidate"
                    else segment._blur(posterior))
            operation_seconds=time.perf_counter()-t1
        usage_after=resource.getrusage(resource.RUSAGE_SELF)
        assert tensor_sha(posterior)==preblur_manifest["preblur_file"]["value_sha256"]
        assert actual.shape==posterior.shape and actual.stride()==posterior.stride()
        assert actual.dtype==torch.float32 and actual.device.type=="cpu" and actual.is_contiguous()
        assert actual.untyped_storage().data_ptr()!=posterior.untyped_storage().data_ptr()
        array=actual.numpy()
        report["operation_seconds_with_trial_guard"]=operation_seconds
        report["operation_process_user_seconds"]=usage_after.ru_utime-usage_before.ru_utime
        report["operation_process_system_seconds"]=usage_after.ru_stime-usage_before.ru_stime
        report["output"]={"value_sha256":tensor_sha(actual),"shape":list(actual.shape),
                           "stride":list(actual.stride()),"dtype":str(actual.dtype)}
        bit_different=0
        maximum=0.
        finite=True
        reference=(np.load(args.reference,mmap_mode="r") if args.reference else None)
        if reference is not None:
            assert reference.shape==array.shape and reference.dtype==array.dtype
        for channel in range(33):
            for depth in range(0,192,8):
                part=array[:,channel,depth:depth+8]
                finite=finite and bool(np.isfinite(part).all())
                if reference is not None:
                    expected=reference[:,channel,depth:depth+8]
                    bit_different+=int(np.count_nonzero(part.view(np.uint32)!=expected.view(np.uint32)))
                    maximum=max(maximum,float(np.max(np.abs(part-expected))))
        report["bit_gate"]={"all_finite":finite,"bit_different_values":bit_different,
                            "max_absolute":maximum,"comparison_executed":reference is not None,
                            "input_unchanged":True,"shape_stride_dtype_same":True,"output_independent":True}
        if args.reference is None:
            assert args.mode=="baseline"
            np.save(args.output/"baseline_blur.private.npy",array)
        report["status"]="blur_bit_gate_passed" if finite and bit_different==0 else "blur_bit_gate_failed"
    report["maximum_RSS_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report["source_after"]={name:sha(args.source/"fnit/synthseg_parc"/name) for name in before}
    report["final_flags"]=flags()
    report["source_unchanged"]=report["source_after"]==before
    report["flags_unchanged"]=report["final_flags"]==initial
    report["RSS_gate_passed"]=report["maximum_RSS_bytes"]<=plan["gates"]["maximum_RSS_bytes"]
    report["worker_observed_seconds_with_load_save_compare"]=time.perf_counter()-t0
    write(args.output/"report.private.json",report)
    assert report["source_unchanged"] and report["flags_unchanged"] and report["RSS_gate_passed"]
    assert report["status"]!="blur_bit_gate_failed"
    print(json.dumps({"status":report["status"],"mode":args.mode,
        "maximum_RSS_bytes":report["maximum_RSS_bytes"]}))


if __name__=="__main__":
    main()
