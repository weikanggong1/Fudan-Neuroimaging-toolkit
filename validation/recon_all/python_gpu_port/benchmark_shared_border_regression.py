"""共用边界表达式修复的最终 white/pial 真输入完整回归。

逐项核验已完成旧控制的输入/参考 SHA，候选仅读取 FNIT 自产阶段输入。
参考及旧候选只在写出后比较；本工具不将冻结同输入回归称作连续整例。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import torch

from benchmark_python_surface_adapters import geometry, volume, publish, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-subject", type=Path, required=True,
                        help="保持不变的 FNIT 自产阶段输入目录")
    parser.add_argument("--baseline-directory", type=Path, required=True,
                        help="旧完整同输入 benchmark 的 case 目录及 report.json")
    parser.add_argument("--candidate-directory", type=Path, required=True,
                        help="只有新 border 模块的独立 overlay，不修改冻结源码")
    parser.add_argument("--output-directory", type=Path, required=True,
                        help="全新诊断目录；存在时抛 FileExistsError")
    parser.add_argument("--code-base-commit", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.output_directory
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(mode=0o700, parents=True)
    baseline_path = args.baseline_directory / "report.json"
    baseline = json.loads(baseline_path.read_text())
    if baseline["status"] != "complete":
        raise RuntimeError("baseline full-stage run did not complete")
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory))
    from fnit.recon_all import surface_placement_python as api
    from fnit.recon_all.thread_budget import thread_budget
    report = dict(scope="frozen_same_input_shared_border_white_and_pial_not_raw_T1",
        code_base_commit=args.code_base_commit, script_sha256=sha(Path(__file__)),
        baseline_receipt_sha256=sha(baseline_path), hostname=platform.node(),
        cpu_affinity=sorted(os.sched_getaffinity(0)), threads=args.threads,
        device=args.device, reference_program_sha256=baseline["reference_program_sha256"],
        reference_reproducibility="existing_exact_stage_reference_reused_by_SHA; no new native repeat",
        allocator_environment={key: os.environ.get(key) for key in (
            "PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")},
        stages={}, status="started", whole_metric_equivalence="not_assessed")
    receipt = out / "report.json"
    publish(receipt, report)
    started = perf_counter()
    try:
        with thread_budget(threads=args.threads):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            for hemi in ("lh", "rh"):
                for stage, runner, name in (
                    ("white", api.run_final_white_python, "white"),
                    ("pial", api.run_pial_python, "pial.T1"),
                ):
                    key = f"{hemi}_{stage}"
                    old = baseline["stages"][key]
                    if not old["surface"]["ordered_faces"] or old["surface"]["different_coordinate_elements"]:
                        raise RuntimeError(f"old same-input stage not exact: {key}")
                    hashes = {name: sha(args.source_subject / name) for name in old["input_sha256"]}
                    if hashes != old["input_sha256"]:
                        raise RuntimeError(f"frozen stage input changed: {key}")
                    reference = args.baseline_directory / f"{hemi}-{stage}-reference" / "surf" / f"{hemi}.{name}"
                    old_surface = args.baseline_directory / f"{hemi}-{stage}" / "surf" / f"{hemi}.{name}"
                    if sha(reference) != old["surface"]["reference_sha256"] or sha(old_surface) != old["surface"]["candidate_sha256"]:
                        raise RuntimeError(f"frozen reference or baseline output changed: {key}")
                    dest = out / f"{hemi}-{stage}"
                    for relative in hashes:
                        target = dest / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(args.source_subject / relative, target)
                    row = dict(input_sha256=hashes, status="running", external_process_attempts=[],
                               allowed_system_probes=[])
                    report["stages"][key] = row
                    publish(receipt, report)
                    original_popen = subprocess.Popen
                    def denied(*items, **kwargs):
                        # NumPy 的首次 SVE 特性检查只读调用 lscpu，不是脑影像命令。
                        # 仅放行这个精确命令并记录，其余候选启动进程仍全部拒绝。
                        command = items[0] if items else kwargs.get("args")
                        if command == "lscpu" or command == ["lscpu"] or command == ("lscpu",):
                            row["allowed_system_probes"].append("lscpu: NumPy SVE CPU capability check")
                            return original_popen(*items, **kwargs)
                        row["external_process_attempts"].append(repr(items[0]) if items else "unknown")
                        raise RuntimeError("candidate attempted an external process")
                    torch.cuda.synchronize(args.device)
                    torch.cuda.reset_peak_memory_stats(args.device)
                    tick = perf_counter()
                    subprocess.Popen = denied
                    try:
                        result = runner(subject_dir=dest, hemi=hemi, device=args.device)
                    finally:
                        subprocess.Popen = original_popen
                    torch.cuda.synchronize(args.device)
                    row.update(status="complete", result=result,
                        wall_seconds_including_IO_and_sync=perf_counter()-tick,
                        vs_reference=geometry(reference, dest / "surf" / f"{hemi}.{name}"),
                        vs_old_candidate=geometry(old_surface, dest / "surf" / f"{hemi}.{name}"),
                        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                        process_tree_GPU_memory="not measured; allocator is not process-tree occupancy",
                        input_unchanged=hashes == {name: sha(args.source_subject / name) for name in hashes})
                    if stage == "white":
                        ref_volume = reference.parents[1] / "mri/mrisps.white.mgz"
                        if sha(ref_volume) != old["volume"]["reference_sha256"]:
                            raise RuntimeError("frozen reference MRI changed")
                        row["volume"] = volume(ref_volume, dest / "mri/mrisps.white.mgz")
                    row["strict_stage_exact"] = (
                        row["vs_reference"]["ordered_faces"] and row["vs_reference"]["different_coordinate_elements"] == 0
                        and row["input_unchanged"] and not row["external_process_attempts"]
                        and ("volume" not in row or row["volume"]["different_voxels"] == 0
                             and row["volume"]["affine_equal"] and row["volume"]["dtype_equal"]))
                    publish(receipt, report)
            report["status"] = "complete"
            report["strict_stage_exact"] = all(row["strict_stage_exact"] for row in report["stages"].values())
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["total_diagnostic_seconds"] = perf_counter()-started
        report["precision"] = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32, half=False, autocast=False)
        report["loaded_source_sha256"] = {name: sha(Path(module.__file__)) for name, module in list(sys.modules.items())
            if name.startswith("fnit.recon_all.place_") and getattr(module, "__file__", None)}
        publish(receipt, report)


if __name__ == "__main__":
    main()
