"""在同主机固定预算串行执行两例 N4 native 重复、完整 Torch 和除法变体。

所有路径为具名参数，不保存登录地址。新 --output 必须不存在。
--native/--diagnostic 必须是声明 Conda 源码构建产物；不调用系统原软件。
完整 Torch 结果只在算完之后与 reference 比较；没有参考补写或阈值修改。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True, help="声明 Conda Python")
    parser.add_argument("--environment", type=Path, required=True, help="Conda prefix；实际加载库记录到收据")
    parser.add_argument("--native", type=Path, required=True, help="FNIT 当前 N4 native 参考")
    parser.add_argument("--diagnostic", type=Path, required=True, help="同源码 final-capture 诊断")
    parser.add_argument("--workspace", type=Path, required=True, help="包含冻结 v1/ 和独立 control/ 的工作区")
    parser.add_argument("--data", type=Path, required=True, help="含带哈希 manifest.json 的公开 stage bundle")
    parser.add_argument("--output", type=Path, required=True, help="新运行目录；拒绝覆盖")
    parser.add_argument("--device", default="cuda:0", help="显式目标GPU")
    parser.add_argument("--threads", type=int, default=1, help="CPU/Torch预算；native fitting固定1线程")
    parser.add_argument("--cpu-list", help="可选逗号分隔允许CPU；不改变其他进程")
    parser.add_argument("--skip-cpu-torch", action="store_true", help="仅在CPU完整反馈已单独验收时省略；报告明确记录")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.cpu_list:
        requested = {int(value) for value in args.cpu_list.split(",")}
        if not requested <= os.sched_getaffinity(0):
            raise ValueError("requested CPUs outside permitted affinity")
        os.sched_setaffinity(0, requested)
    args.output.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads), "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
        "LD_LIBRARY_PATH": str(args.environment / "lib") + ":" + environment.get("LD_LIBRARY_PATH", "")})
    dataset = json.loads((args.data / "manifest.json").read_text())
    report = {"kind": "two_real_frozen_complete_n4_same_host_not_recon_all", "host": platform.node(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": args.threads,
        "native_fitting_threads": 1, "native_reconstruction_threads": 1, "device": args.device,
        "program_sha256": {str(p): sha(p) for p in (args.native, args.diagnostic, args.python, Path(__file__))},
        "data_manifest_sha256": sha(args.data / "manifest.json"), "status": "running", "jobs": [],
        "cases": [], "cpu_torch_skipped": args.skip_cpu_torch,
        "production_default_changed": False, "whole_n4_equivalence": "not established",
        "whole_recon_all_acceleration": "not measured"}
    for program in (args.native, args.diagnostic):
        report.setdefault("dynamic_libraries", {})[str(program)] = subprocess.run(
            ["ldd", str(program)], env=environment, capture_output=True, text=True, check=True).stdout
    report_path = args.output / "summary.json"
    def save():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)
    def run(label, command, *, source=None):
        local_env = environment.copy()
        if source is not None:
            local_env["PYTHONPATH"] = str(source / "src")
        start = time.perf_counter()
        print("START " + label, flush=True)
        with (args.output / (label + ".log")).open("w") as stream:
            completed = subprocess.run(list(map(str, command)), env=local_env, stdout=stream, stderr=subprocess.STDOUT)
        entry = {"label": label, "command": list(map(str, command)), "wall_seconds": time.perf_counter()-start,
                 "returncode": completed.returncode, "source_workspace": str(source) if source else None}
        report["jobs"].append(entry); save()
        print("END " + label + " " + str(entry["wall_seconds"]), flush=True)
        if completed.returncode:
            raise RuntimeError(label + " failed; see preserved log")
    try:
        save()
        v1 = args.workspace / "v1"
        control = args.workspace / "control"
        variant = args.workspace / "division_variant"
        generator = v1 / "validation/recon_all/optimizations/20261009_n4_torch_substages/prepare_division_variant.py"
        run("prepare_division_variant", [args.python, generator, "--source-workspace", v1,
                                           "--output-workspace", variant])
        for case in dataset["cases"]:
            case_name = case["id"]
            directory = args.output / case_name
            directory.mkdir()
            input_raw = args.data / case["input_raw"]
            if sha(input_raw) != case["input_raw_sha256"]:
                raise ValueError("frozen input hash changed")
            common = list(map(str, case["shape"] + case["spacing_mm"]))
            references = []
            for repeat in (1, 2):
                destination = directory / f"native_{repeat}.final.raw"
                run(case_name + f"_native_{repeat}", [args.native, input_raw, destination, *common,
                    "1", directory / f"native_{repeat}.profile.json"])
                references.append(destination)
            native_stable = sha(references[0]) == sha(references[1])
            diagnostic_prefix = directory / "diagnostic.profile.json"
            diagnostic_output = directory / "diagnostic.final.raw"
            run(case_name + "_diagnostic", [args.diagnostic, input_raw, diagnostic_output, *common,
                                           "1", diagnostic_prefix])
            entry = {"case": case_name, "input_sha256": sha(input_raw), "native_repeat_byte_exact": native_stable,
                     "diagnostic_byte_exact_to_native": sha(diagnostic_output) == sha(references[0]),
                     "native_output_sha256": sha(references[0]), "historical_reference_kind": case["reference_kind"]}
            if case.get("historical_reference_raw"):
                entry["same_input_historical_native_byte_exact"] = sha(args.data / case["historical_reference_raw"]) == sha(references[0])
            report["cases"].append(entry); save()
            if not native_stable or not entry["diagnostic_byte_exact_to_native"]:
                raise RuntimeError("native repeat or diagnostic changed; investigate before candidate interpretation")
            runs = [("v1_cpu", v1, "cpu", True)] if not args.skip_cpu_torch else []
            runs += [(f"{name}_gpu_{repeat}", source, args.device, repeat == 1)
                     for name, source in (("v1", v1), ("division", variant)) for repeat in (1, 2)]
            for label, source, device, profile in runs:
                destination = directory / label / "report.json"
                command = [args.python, control / "benchmark_full.py", "--input-raw", input_raw,
                    "--shape", *map(str, case["shape"]), "--spacing", *map(str, case["spacing_mm"]),
                    "--reference-raw", references[0], "--output", destination, "--device", device,
                    "--threads", str(args.threads), "--export-final-fields"]
                if profile:
                    command.append("--profile")
                run(case_name + "_" + label, command, source=source)
            for source_name, source in (("v1", v1), ("division", variant)):
                for device_name, device in (("cpu", "cpu"), ("gpu", args.device)):
                    run(case_name + "_frozen_" + source_name + "_" + device_name,
                        [args.python, control / "benchmark_frozen_fields.py", "--prefix", diagnostic_prefix,
                         "--input-raw", input_raw, "--corrected-reference-raw", references[0],
                         "--device", device, "--threads", str(args.threads), "--output",
                         directory / ("frozen_" + source_name + "_" + device_name + ".json")], source=source)
        report["status"] = "complete_stage_pair"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save()
        raise
    save()


if __name__ == "__main__":
    main()
