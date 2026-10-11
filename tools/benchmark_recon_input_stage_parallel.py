"""N4重建线程契约及N4/SynthSeg完整组真实对照；诊断不读参考进生产。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n")


def reconstruction_gate(args) -> dict:
    """完整N4文件API拟合1/重建1，与已有重建4的同输入完整输出比较。"""
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.module_directory.resolve()))
    from fnit.recon_all import n4_simpleitk, input_n4_chain, compare_native_free
    import nibabel as nib
    import numpy as np
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = {"status": "running", "scope": "same-input full SimpleITK N4 fit1 reconstruction1 vs existing reconstruction4; not pipeline",
        "script_sha256": digest(Path(__file__)), "candidate_module_sha256": digest(Path(n4_simpleitk.__file__)),
        "input_sha256": digest(args.input), "hostname": platform.node(),
        "CPU_affinity": sorted(os.sched_getaffinity(0)), "load_before": list(os.getloadavg()),
        "production_default_changed": False}
    reference = json.loads((args.reference_directory / "run.json").read_bytes())
    if (reference["status"] != "complete" or reference["fitting_threads"] != 1
            or reference["details"]["reconstruction_threads"] != 4
            or reference["input_sha256_before"] != report["input_sha256"]
            or reference["source_module_sha256"] != report["candidate_module_sha256"]):
        raise ValueError("declared complete reconstruction4 stage is not this same input/source")
    report["reference_producer_sha256"] = digest(args.reference_directory / "run.json")
    report["reference_float_sha256"] = digest(args.reference_directory / "corrected_float.npy")
    report["reference_uint8_sha256"] = digest(args.reference_directory / "nu0.mgz")
    captured = {}
    original = n4_simpleitk.correct_array
    def capture(*positional, **keywords):
        array, details = original(*positional, **keywords)
        captured["float"] = array  # 仅保留引用；诊断写出不计入API墙钟
        return array, details
    n4_simpleitk.correct_array = capture
    tick = time.perf_counter()
    try:
        result = input_n4_chain.run_n4_stage(input_path=args.input, output_path=args.output / "nu0.mgz",
            n4_backend="simpleitk", n4_execution="in-process", native_binary=None,
            device="cpu", threads=1, profile=False, report_path=args.output / "N4-profile.json")
    finally:
        n4_simpleitk.correct_array = original
    report["complete_API_seconds"] = time.perf_counter()-tick
    left = np.load(args.reference_directory / "corrected_float.npy")
    right = captured["float"]
    error = np.abs(left.astype(np.float64)-right.astype(np.float64))
    report["float_comparison"] = {"different_elements": int(np.count_nonzero(left != right)),
        "max_abs": float(error.max()), "p99_abs": float(np.quantile(error,.99)),
        "reference_dtype": str(left.dtype), "candidate_dtype": str(right.dtype),
        "reference_shape": list(left.shape), "candidate_shape": list(right.shape)}
    report["uint8_comparison"] = compare_native_free._volume(args.reference_directory / "nu0.mgz", args.output / "nu0.mgz")
    headers = [nib.load(str(path)).header for path in (args.reference_directory / "nu0.mgz", args.output / "nu0.mgz")]
    report["uint8_comparison"]["vox2ras_tkr_max_abs"] = float(np.abs(headers[0].get_vox2ras_tkr()-headers[1].get_vox2ras_tkr()).max())
    report.update(status="complete", actual_N4=result, final_input_sha256=digest(args.input),
        load_after=list(os.getloadavg()), comparison_seconds=time.perf_counter()-tick-report["complete_API_seconds"],
        safe_for_group=(report["float_comparison"]["different_elements"] == 0
            and report["uint8_comparison"]["different_voxels"] == 0
            and report["uint8_comparison"]["affine_max_abs"] == 0.0
            and report["uint8_comparison"]["vox2ras_tkr_max_abs"] == 0.0))
    save(args.output / "gate.json", report)
    return report


def complete_group(args) -> dict:
    """完整组 API/CLI；仅显式 already-initialized mode 初始化父 CUDA。"""
    tick = time.perf_counter()
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.module_directory.resolve()))
    from fnit.recon_all.input_stage_parallel import run_input_stage_group
    import torch
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parent_buffer = None
    if args.initialized_parent:
        parent_buffer = torch.ones(1024, dtype=torch.float32, device=args.device)
        torch.cuda.synchronize(torch.device(args.device))
    import_and_parent_seconds = time.perf_counter()-tick
    report = run_input_stage_group(orig_path=args.input, n4_output=args.output / "nu0.mgz",
        segmentation_output=args.output / "synthseg.mgz", volumes_output=args.output / "synthseg.vol.csv",
        weights_dir=args.weights, color_lut=args.color_lut, report_path=args.output / "group.json",
        device=args.device, threads=4, execution=args.execution, profile_stages=True,
        code_version=args.code_version)
    result = {"status": "complete", "scope": "same-input N4/SynthSeg complete group; not raw T1 whole",
        "execution": args.execution, "parent_initialized": args.initialized_parent,
        "import_and_parent_setup_seconds": import_and_parent_seconds,
        "script_wall_seconds": time.perf_counter()-tick,
        "group_wall_seconds": report["group_wall_seconds"],
        "script_sha256": digest(Path(__file__)), "group_helper_sha256": report["runtime_helper_sha256"],
        "source_manifest_sha256": digest(args.source_manifest), "code_version": args.code_version,
        "hostname": platform.node(), "CPU_affinity": sorted(os.sched_getaffinity(0)),
        "input_sha256": digest(args.input), "group_report_sha256": digest(args.output / "group.json"),
        "parent_buffer_preserved": bool(parent_buffer is None or torch.equal(parent_buffer, torch.ones_like(parent_buffer))),
        "precision_preserved": report["parent_state_preserved"], "load_after": list(os.getloadavg())}
    save(args.output / "run.json", result)
    return result


def compare_groups(reference: Path, candidate: Path) -> dict:
    """完整组 uint8/分割/CSV及几何同输入比较，不读入生产函数。"""
    import nibabel as nib
    import numpy as np
    from fnit.recon_all.compare_native_free import _volume
    result = {}
    for name in ("nu0.mgz", "synthseg.mgz"):
        left, right = reference / name, candidate / name
        row = _volume(left, right)
        headers = [nib.load(str(path)).header for path in (left, right)]
        row["vox2ras_tkr_max_abs"] = float(np.abs(headers[0].get_vox2ras_tkr()-headers[1].get_vox2ras_tkr()).max())
        if name == "synthseg.mgz":
            a, b = [np.asanyarray(nib.load(str(path)).dataobj) for path in (left, right)]
            row["label_dice"] = {}
            for label in np.union1d(np.unique(a), np.unique(b)):
                if label == 0:
                    continue
                mask_a, mask_b = a == label, b == label
                denominator = int(mask_a.sum()) + int(mask_b.sum())
                row["label_dice"][str(int(label))] = 2*int((mask_a & mask_b).sum()) / denominator
        result[name] = row
    result["csv_byte_identical"] = (reference / "synthseg.vol.csv").read_bytes() == (candidate / "synthseg.vol.csv").read_bytes()
    result["strict_same_input_passed"] = (result["csv_byte_identical"] and all(
        result[name]["different_voxels"] == 0 and result[name]["affine_max_abs"] == 0
        and result[name]["vox2ras_tkr_max_abs"] == 0 for name in ("nu0.mgz", "synthseg.mgz")))
    return result


def abba(args) -> dict:
    """四次完整冷 exec ABBA；外层墙钟包含脚本导入、API与写出。"""
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    results = []
    for index, execution in enumerate(("serial", "parallel", "parallel", "serial"), 1):
        path = args.output / f"{index}-{execution}"
        command = [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()),
            "--mode", "group", "--module-directory", str(args.module_directory), "--input", str(args.input),
            "--output", str(path), "--weights", str(args.weights), "--color-lut", str(args.color_lut),
            "--device", args.device, "--execution", execution, "--source-manifest", str(args.source_manifest),
            "--code-version", args.code_version]
        with (args.output / f"{index}-{execution}.log").open("w") as stream:
            tick = time.perf_counter()
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        elapsed = time.perf_counter()-tick
        row = {"index": index, "execution": execution, "complete_CLI_seconds": elapsed,
               "returncode": process.returncode}
        if (path / "run.json").is_file():
            row["run"] = json.loads((path / "run.json").read_bytes())
        results.append(row)
        save(args.output / "ABBA.json", {"status": "running", "runs": results})
        if process.returncode:
            raise RuntimeError(f"complete group {index}-{execution} failed; preserved log/report")
    comparisons = {f"1-serial_vs_{index}-{execution}": compare_groups(args.output / "1-serial", args.output / f"{index}-{execution}")
                   for index, execution in ((2, "parallel"), (3, "parallel"), (4, "serial"))}
    result = {"status": "complete", "runs": results, "comparisons": comparisons,
              "strict_same_input_passed": all(row["strict_same_input_passed"] for row in comparisons.values()),
              "overall_metric_equivalence": "not_assessed", "full_workflow_measured": False}
    save(args.output / "ABBA.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reconstruction-gate", "group", "abba"), required=True)
    parser.add_argument("--module-directory", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--color-lut", type=Path)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--execution", choices=("serial", "parallel"), default="serial")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--code-version", default="FNIT-source-hashes")
    parser.add_argument("--initialized-parent", action="store_true")
    args = parser.parse_args()
    if args.mode == "reconstruction-gate":
        if args.reference_directory is None:
            parser.error("reconstruction-gate needs --reference-directory")
        report = reconstruction_gate(args)
    else:
        if any(value is None for value in (args.weights, args.color_lut, args.source_manifest)):
            parser.error("group/abba need --weights --color-lut --source-manifest")
        report = complete_group(args) if args.mode == "group" else abba(args)
    print(json.dumps({key: report[key] for key in ("status", "safe_for_group", "complete_API_seconds",
        "float_comparison", "group_wall_seconds", "strict_same_input_passed") if key in report}))
    if not report.get("safe_for_group", report.get("strict_same_input_passed", True)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
