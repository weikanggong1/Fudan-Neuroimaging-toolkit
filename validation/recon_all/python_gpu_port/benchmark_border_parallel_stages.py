"""同输入完整白质／pial的显式目标搜索并行回归。

候选仅复制自产输入；参考输出在阶段结束后读取比较。原有序目标平均、
试步接受和源清理全部保留；两后端在同一线程预算下完整运行，不替代整例。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from benchmark_python_surface_adapters import geometry, volume, publish, sha


def array_receipt(value):
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-subject", type=Path, required=True)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-base-commit", required=True)
    parser.add_argument("--stage", choices=("preaparc", "pial"), required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--order", choices=("serial_parallel", "parallel_serial"), default="serial_parallel")
    args = parser.parse_args()
    out = args.output_directory
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(mode=0o700, parents=True)
    old_path = args.baseline_directory / "report.json"
    old = json.loads(old_path.read_text())
    if old["status"] != "complete":
        raise RuntimeError("old complete stage receipt missing")
    hemi = args.hemisphere
    baseline = old["stages"][f"{hemi}_{args.stage}"]
    inputs = baseline["input_sha256"]
    if inputs != {name: sha(args.source_subject / name) for name in inputs}:
        raise RuntimeError("old stage input SHA mismatch")
    suffix = "white.preaparc" if args.stage == "preaparc" else "pial.T1"
    reference = args.baseline_directory / f"{hemi}-{args.stage}-reference" / "surf" / f"{hemi}.{suffix}"
    if sha(reference) != baseline["surface"]["reference_sha256"]:
        raise RuntimeError("reference output SHA mismatch")
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory))
    from fnit.recon_all import place_white_preaparc_python as white
    from fnit.recon_all import place_pial_python as pial
    from fnit.recon_all.autodet_gwstats_python import write_autodet_stats
    from fnit.recon_all.thread_budget import thread_budget
    report = {"scope": "frozen_same_input_complete_surface_parallel_target_search_not_raw_T1",
        "code_base_commit": args.code_base_commit, "script_sha256": sha(Path(__file__)),
        "baseline_receipt_sha256": sha(old_path), "input_sha256": inputs,
        "hostname": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "stage": args.stage, "hemisphere": hemi, "threads": args.threads, "device": args.device,
        "reference_program_sha256": old["reference_program_sha256"],
        "reference_reproducibility": "existing same-input reference reused by input/program/output SHA; no new native repeat",
        "allocator_environment": {name: os.environ.get(name) for name in
            ("PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")},
        "order": args.order.split("_"), "runs": {}, "status": "started",
        "performance_scope": "complete API includes loads/writes/target first JIT; imports, CUDA initialization and input copy/hash precede clock; shared load single pair",
        "whole_metric_equivalence": "not_assessed"}
    receipt = out / "report.json"
    publish(receipt, report)
    try:
        with thread_budget(threads=args.threads) as budget:
            report["thread_budget"] = budget
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.synchronize(args.device)
            for backend in report["order"]:
                dest = out / backend
                for relative in inputs:
                    target = dest / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(args.source_subject / relative, target)
                row = dict(status="running", trace=[], border_calls=[], prohibited_process_attempts=[], allowed_system_probes=[])
                report["runs"][backend] = row
                publish(receipt, report)
                def callback(step, pass_index, xyz, diagnostics):
                    row["trace"].append(dict(step=step, pass_index=pass_index,
                        coordinates=array_receipt(xyz), diagnostics=diagnostics))
                    publish(receipt, report)
                stage = white if args.stage == "preaparc" else pial
                original_border = stage.compute_border_values_first_pass
                def observed_border(*items, **keywords):
                    tick = perf_counter()
                    values = original_border(*items, **keywords)
                    elapsed = perf_counter()-tick
                    row["border_calls"].append(dict(surface=keywords["surface"], sigma=keywords.get("sigma",2.),
                        backend=keywords["target_search_backend"], seconds=elapsed,
                        fields=[array_receipt(value) for value in values]))
                    return values
                original_popen = subprocess.Popen
                def denied(*items, **keywords):
                    command = items[0] if items else keywords.get("args")
                    if command == "lscpu" or command == ["lscpu"] or command == ("lscpu",):
                        row["allowed_system_probes"].append("lscpu: NumPy CPU capability check")
                        return original_popen(*items, **keywords)
                    row["prohibited_process_attempts"].append(repr(command))
                    raise RuntimeError("producer attempted a prohibited external process")
                kwargs = dict(regularization_backend="cpu", sampling_backend="cpu", candidate_backend="torch_snapshot",
                    candidate_grid_cells_per_axis=3, retained_mht_backend="compiled",
                    cleanup_marking_backend="source_torch", cleanup_candidate_grid_cells_per_axis=3,
                    device=args.device, trace_callback=callback, target_search_backend=backend)
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                tick = perf_counter()
                stage.compute_border_values_first_pass = observed_border
                subprocess.Popen = denied
                try:
                    if args.stage == "preaparc":
                        write_autodet_stats(brain_path=dest/"mri/brain.finalsurfs.mgz", wm_path=dest/"mri/wm.mgz",
                            surface_path=dest/f"surf/{hemi}.orig.premesh",
                            output_path=dest/f"surf/autodet.gw.stats.{hemi}.dat", hemisphere=hemi)
                        result = white.place_white_preaparc(subject_dir=dest, hemi=hemi,
                            output=dest/f"surf/{hemi}.{suffix}", output_volume=dest/"mri/mrisps.wpa.mgz",
                            max_steps=400, **kwargs)
                    else:
                        result = pial.place_pial_t1(subject=dest, hemisphere=hemi,
                            output=dest/f"surf/{hemi}.{suffix}", profile=True, max_steps=200, **kwargs)
                finally:
                    stage.compute_border_values_first_pass = original_border
                    subprocess.Popen = original_popen
                torch.cuda.synchronize(args.device)
                row.update(status="complete", result=result, wall_seconds=perf_counter()-tick,
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                    process_tree_GPU_memory="not measured; Torch allocator is not process tree occupancy",
                    vs_reference=geometry(reference,dest/f"surf/{hemi}.{suffix}"),
                    input_unchanged=inputs=={name:sha(args.source_subject/name) for name in inputs})
                if args.stage == "preaparc":
                    row["MRI_vs_reference"] = volume(reference.parents[1]/"mri/mrisps.wpa.mgz",dest/"mri/mrisps.wpa.mgz")
                    row["threshold_bytes_equal"] = (dest/f"surf/autodet.gw.stats.{hemi}.dat").read_bytes() == (reference.parents[1]/f"surf/autodet.gw.stats.{hemi}.dat").read_bytes()
                publish(receipt, report)
            left,right=report["runs"]["serial"],report["runs"]["parallel"]
            report["comparison"] = dict(all_accepted_coordinate_SHA_and_decisions_exact=left["trace"]==right["trace"],
                all_border_fields_exact=[{k:v for k,v in item.items() if k not in ("backend","seconds")} for item in left["border_calls"]]
                                      ==[{k:v for k,v in item.items() if k not in ("backend","seconds")} for item in right["border_calls"]],
                pass_ends_exact=left["result"]["pass_ends"]==right["result"]["pass_ends"],
                final_cleanup_exact=left["result"]["cleanup"]==right["result"]["cleanup"],
                final_geometry=geometry(out/"serial"/f"surf/{hemi}.{suffix}",out/"parallel"/f"surf/{hemi}.{suffix}"),
                serial_seconds=left["wall_seconds"],parallel_seconds=right["wall_seconds"],
                source_serial_stage_seconds=left["result"].get("stage_seconds"),
                source_parallel_stage_seconds=right["result"].get("stage_seconds"))
            report["comparison"]["native_final_geometry_exact"] = all(
                row["vs_reference"]["ordered_faces"] and row["vs_reference"]["different_coordinate_elements"]==0
                for row in (left,right))
            report["comparison"]["final_intersections_zero"] = all(
                row["result"]["cleanup"]["intersecting_faces_after"]==0 for row in (left,right))
            if args.stage == "preaparc":
                report["comparison"]["prepared_MRI_and_threshold_exact"] = all(
                    row["MRI_vs_reference"]["different_voxels"]==0 and row["MRI_vs_reference"]["affine_equal"]
                    and row["MRI_vs_reference"]["dtype_equal"] and row["threshold_bytes_equal"] for row in (left,right))
            report["status"]="complete"
            report["regression_exact"] = all(report["comparison"][k] for k in
                ("all_accepted_coordinate_SHA_and_decisions_exact","all_border_fields_exact","pass_ends_exact","final_cleanup_exact"))
            report["regression_exact"] &= report["comparison"]["native_final_geometry_exact"] and report["comparison"]["final_intersections_zero"]
            report["regression_exact"] &= report["comparison"]["final_geometry"]["ordered_faces"] and report["comparison"]["final_geometry"]["different_coordinate_elements"]==0
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report["precision"]=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,half=False,autocast=False)
        report["loaded_source_sha256"]={name:sha(Path(module.__file__)) for name,module in list(sys.modules.items())
            if name.startswith("fnit.recon_all.place_") and getattr(module,"__file__",None)}
        publish(receipt,report)


if __name__=="__main__":
    main()
