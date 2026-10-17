"""冻结自产 smoothwm，核验共享平均器对完整 Torch inflation 的影响。

复用既有 run_standard_inflate、完整源码 SHA 门和 inflate 比较器，不改算法。
prepare 从两份成功生产 benchmark 提取 smoothwm、历史 inflated/sulc、实际
每侧线程与 API 参数。run 在空目录中为首侧运行 A/B，其余三侧运行 B；
首侧以只读 trace 记录真实每步坐标/梯度/动量/sulc/RMS，故该次墙钟仅供
诊断，不能作为提速证据。历史报告仅有末次积分状态，不能声称历史逐步
轨迹一致。每次独立解释器、独立 JIT 缓存，CUDA 目标在 API 前后同步，
计时包含读取、写出和传输。缓存政策匹配实际生产 surface worker。

所有输入坐标为 surface RAS/mm；sulc 为同序 (N,) float32/mm。严格门为
保存几何、sulc、体积头和已有积分字段 exact；结构门复用生产闭合流形
检查，不执行通用三角面碰撞扫描。显存没有驱动归属采样时保留未知。
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[4]
COMMON_PATH = ROOT / "validation/recon_all/optimizations/20261017_sphere_normals/benchmark_standard_sphere.py"
COMPARE_PATH = ROOT / "validation/recon_all/optimizations/20261009_inflate_torch/benchmark_complete.py"
STATE_FIELDS = ("steps", "final_rms", "stopped_by_rms")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load(COMMON_PATH, "_fnit_inflate_source_tools")


def verify_inputs(path, expected):
    data = C.verify_inputs(path, expected)
    for row in data["cases"]:
        runtime = C.read_bound(row["runtime"], row["runtime_sha256"])
        api = runtime["surfaces"][row["hemisphere"]]["standard_sphere_report"]["inflate_runtime"]
        if api != row["historical_api"]:
            raise ValueError("historical inflation API changed")
        if row["threads"] != 2 or not row["allocator_cache_enabled"]:
            raise ValueError("this regression requires the recorded two-thread cached surface worker")
        for name in ("smoothwm", "inflated", "sulc"):
            item = row["inputs"][name]
            required = Path(row["benchmark"]).parent / "subject/surf" / (row["hemisphere"] + "." + name)
            if Path(item["path"]) != required or C.sha(C.regular(required)) != item["sha256"]:
                raise ValueError("frozen inflation input/output SHA changed")
    return data


def prepare(args):
    with contextlib.redirect_stdout(io.StringIO()):
        C.prepare(args)
    data = json.loads(args.output.read_text())
    for row in data["cases"]:
        runtime = json.loads(Path(row["runtime"]).read_text())
        group = runtime["hemisphere_scheduling"]["groups"][0]
        api = runtime["surfaces"][row["hemisphere"]]["standard_sphere_report"]["inflate_runtime"]
        if (group["worker_threads"] != 2 or group["worker_allocator_selection"]["selected_policy"] != "enabled"
                or api["backend"] != "torch" or api["device"] != "cuda:0" or not api["save_sulc"]
                or api["profile_synchronizes_substages"]):
            raise ValueError("producer actual inflation recipe differs from this regression")
        path = C.regular(Path(row["benchmark"]).parent / "subject/surf" / (row["hemisphere"] + ".sulc"))
        row["inputs"]["sulc"] = {"path": str(path), "sha256": C.sha(path)}
        row.update(historical_api=api, threads=2, allocator_cache_enabled=True,
                   historical_integrator_state={key: api["integration"][key] for key in STATE_FIELDS},
                   historical_trajectory_scope="terminal steps/RMS/stop only; no saved per-step trajectory")
    data["scope"] = "two successful FNIT subjects: own smoothwm input and own saved inflated/sulc diagnostic references"
    C.save(args.output, data)
    verify_inputs(args.output, C.sha(args.output))
    print(json.dumps({"manifest": str(args.output), "sha256": C.sha(args.output), "cases": 4}))


def observer_for(operation):
    from fnit.recon_all.inflate_torch import TorchInflationContext
    function = inspect.unwrap(TorchInflationContext.integrate)
    lines, start = inspect.getsourcelines(function)
    matches = [start + i for i, line in enumerate(lines) if line.lstrip().startswith("if rms < rms_target:")]
    if len(matches) != 1:
        raise ValueError("cannot identify unchanged complete inflation trace point")
    rows = []

    def observer(frame, event, arg):
        if frame.f_code is not function.__code__:
            return None
        if event == "line" and frame.f_lineno == matches[0]:
            state = frame.f_locals
            row = {key: C.plain(state[key]) for key in ("step", "averages", "weight", "rms")}
            row.update({key + "_sha256": C.array_sha(state[key]) for key in
                        ("xyz", "gradient", "previous", "sulc", "normals", "area", "current_dist")})
            rows.append(row)
        return observer

    return observer, rows


def strict_comparison(value):
    return bool(value.get("same_ordered_topology") and value.get("strict_decoded_geometry_and_sulc")
                and all(value.get("volume_geometry_fields_equal", {}).values()))


def worker(args):
    started = time.perf_counter()
    payload = C.read_bound(args.payload, args.payload_sha256)
    output = Path(payload["output"])
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    report_path = output / "report.private.json"
    report = {"status": "running", "arm": payload["arm"], "case": payload["row"]["case"],
              "hemisphere": payload["row"]["hemisphere"], "payload_sha256": args.payload_sha256,
              "script_sha256": C.sha(__file__), "phase": "source_and_input_validation"}
    observer, trace = None, []
    try:
        source = C.verify_source(payload["source"], payload["source_manifest"], payload["source_manifest_sha256"])
        data = verify_inputs(payload["inputs_manifest"], payload["inputs_manifest_sha256"])
        if payload["row"] not in data["cases"]:
            raise ValueError("payload row absent from bound input manifest")
        sys.path.insert(0, str(Path(payload["source"]) / "src"))
        import numba
        import torch
        import nibabel.freesurfer.io as fsio
        import numpy as np
        from fnit.recon_all import inflate_standard_run as operation
        from fnit.recon_all.native_free import _native_final_mesh_structure
        comparator = load(COMPARE_PATH, "_fnit_inflate_existing_comparison")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        numba.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if "PYTORCH_NO_CUDA_MEMORY_CACHING" in os.environ:
            raise ValueError("cache policy must match recorded enabled surface worker")
        device = torch.device(payload["device"])
        if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
            raise ValueError("explicit available target CUDA required")
        imported = {}
        for name, module in tuple(sys.modules.items()):
            if name.startswith("fnit.") and getattr(module, "__file__", None):
                path = Path(module.__file__).resolve()
                if not path.is_relative_to(Path(payload["source"]).resolve()):
                    raise ValueError("mixed frozen source import: " + name)
                relative = str(path.relative_to(Path(payload["source"]).resolve()))
                if relative.endswith(".py") and C.sha(path) != source["files"].get(relative):
                    raise ValueError("loaded module differs from source manifest")
                imported[name] = {"path": str(path), "sha256": C.sha(path)}
        row = payload["row"]
        hemi = row["hemisphere"]
        kwargs = {"input_surface": row["inputs"]["smoothwm"]["path"],
                  "inflated_output": output / (hemi + ".inflated"),
                  "sulc_output": output / (hemi + ".sulc"), "backend": "torch",
                  "device": str(device), "profile": False, "save_sulc": True}
        if payload["trace"]:
            if sys.gettrace() is not None:
                raise RuntimeError("existing Python trace would invalidate read-only observer")
            observer, trace = observer_for(operation)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        report.update(code_version=source["commit"], source_manifest_sha256=payload["source_manifest_sha256"],
                      python_executable_sha256=C.sha(Path(sys.executable).resolve()), imported_modules=imported,
                      source_utility_sha256=C.sha(COMMON_PATH), existing_compare_sha256=C.sha(COMPARE_PATH),
                      threads={"torch": torch.get_num_threads(), "numba": numba.get_num_threads(), "interop": 1},
                      cpu_affinity=sorted(os.sched_getaffinity(0)), allocator_cache_enabled=True,
                      precision={"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                                 "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                                 "autocast_cuda": torch.is_autocast_enabled("cuda"), "half_precision": False},
                      target_gpu_name=torch.cuda.get_device_name(device),
                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), phase="actual_inflate_api_running")
        C.save(report_path, report)
        tick = time.perf_counter()
        try:
            if observer is not None:
                sys.settrace(observer)
            api = operation.run_standard_inflate(**kwargs)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - tick
        finally:
            if observer is not None:
                sys.settrace(None)
        report.update(api=api, stage_seconds_including_io_and_synchronization=elapsed,
                      timing_qualification="read_only_trace_diagnostic_not_performance_sample" if observer else "single_cold_API_not_whole_pipeline_speedup",
                      memory={"pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                              "pytorch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                              "owned_simultaneous_driver_peak_bytes": None, "driver_status": "not_sampled"},
                      phase="api_returned_before_posthoc_validation")
        C.save(report_path, report)
        if observer is not None:
            if len(trace) != api["integration"]["steps"]:
                raise RuntimeError("missed real inflation integration steps")
            C.save(output / "trace.private.json", trace)
            report["trace_sha256"] = C.sha(output / "trace.private.json")
        historical = comparator.compare(output, Path(row["inputs"]["inflated"]["path"]).parent, hemi)
        historical.update(inflated_bytes_equal=C.sha(kwargs["inflated_output"]) == row["inputs"]["inflated"]["sha256"],
                          sulc_bytes_equal=C.sha(kwargs["sulc_output"]) == row["inputs"]["sulc"]["sha256"])
        xyz, faces, metadata = fsio.read_geometry(str(kwargs["inflated_output"]), read_metadata=True)
        current_state = {key: api["integration"][key] for key in STATE_FIELDS}
        report.update(historical_saved_output=historical,
                      historical_recorded_integrator_state={"recorded_fields": list(STATE_FIELDS),
                          "old": row["historical_integrator_state"], "current": current_state,
                          "exact_equal": current_state == row["historical_integrator_state"],
                          "per_step_trajectory": "not_available_in_historical_report"},
                      quality={"structure": _native_final_mesh_structure(xyz, faces),
                               "all_finite": bool(np.isfinite(xyz).all()),
                               "scope": "production closed manifold/genus0/orientation/zero-area gate; no general intersection scan"})
        report["strict_historical_output"] = strict_comparison(historical) and historical["inflated_bytes_equal"] and historical["sulc_bytes_equal"]
        report["quality"]["passed"] = report["quality"]["structure"]["status"] == "passed" and report["quality"]["all_finite"]
        C.verify_source(payload["source"], payload["source_manifest"], payload["source_manifest_sha256"])
        verify_inputs(payload["inputs_manifest"], payload["inputs_manifest_sha256"])
        report.update(status="complete" if report["quality"]["passed"] else "failed_quality", phase="all_checks_returned")
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    report["worker_seconds_including_imports_validation_and_quality"] = time.perf_counter() - started
    C.save(report_path, report)
    return 0 if report["status"] == "complete" else 1


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source_bindings = {}
    for arm, prefix in (("A", "baseline"), ("B", "candidate")):
        source_bindings[arm] = C.verify_source(getattr(args, prefix + "_source"), getattr(args, prefix + "_manifest"), getattr(args, prefix + "_manifest_sha256"))
    data = verify_inputs(args.inputs_manifest, args.inputs_manifest_sha256)
    args.output.mkdir(parents=True, mode=0o700)
    report = {"status": "running", "scope": "complete same-input Torch inflation shared-averager regression; not raw-T1 recon-all",
              "script_sha256": C.sha(__file__), "utility_sha256": C.sha(COMMON_PATH), "compare_sha256": C.sha(COMPARE_PATH),
              "code_versions": {arm: value["commit"] for arm, value in source_bindings.items()},
              "inputs_manifest_sha256": args.inputs_manifest_sha256, "runs": [],
              "same_input_first_side_AB": None, "overall_metric_equivalence": "not_assessed"}
    summary = args.output / "summary.private.json"
    C.save(summary, report)
    try:
        first_outputs = {}
        for index, row in enumerate(data["cases"]):
            for arm in ("AB" if index == 0 else "B"):
                prefix = "baseline" if arm == "A" else "candidate"
                destination = args.output / (row["case"] + "_" + row["hemisphere"] + "_" + arm)
                payload = {"row": row, "arm": arm, "trace": index == 0, "output": str(destination),
                           "source": str(getattr(args, prefix + "_source").absolute()),
                           "source_manifest": str(getattr(args, prefix + "_manifest").absolute()),
                           "source_manifest_sha256": getattr(args, prefix + "_manifest_sha256"),
                           "inputs_manifest": str(args.inputs_manifest.absolute()),
                           "inputs_manifest_sha256": args.inputs_manifest_sha256, "device": args.device}
                payload_path = destination.with_suffix(".payload.private.json")
                C.save(payload_path, payload)
                environment = os.environ.copy()
                environment.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
                environment.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2",
                                   NUMBA_CACHE_DIR=str(destination) + ".numba-cache", TRITON_CACHE_DIR=str(destination) + ".triton-cache")
                tick = time.perf_counter()
                with destination.with_suffix(".log").open("w") as stream:
                    completed = subprocess.run([args.python_interpreter, "-X", "faulthandler", "-u", str(Path(__file__).absolute()),
                        "worker", "--payload", str(payload_path), "--payload-sha256", C.sha(payload_path)],
                        env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
                path = destination / "report.private.json"
                value = json.loads(path.read_text()) if path.exists() else {"status": "failed_before_report"}
                binding = {"case": row["case"], "hemisphere": row["hemisphere"], "arm": arm,
                           "exit_code": completed.returncode, "parent_full_wall_seconds": time.perf_counter() - tick,
                           "worker_report": str(path), "worker_report_sha256": C.sha(path) if path.exists() else None,
                           "status": value["status"]}
                report["runs"].append(binding)
                C.save(summary, report)
                if completed.returncode or value["status"] != "complete":
                    raise RuntimeError("worker failed; preserve outputs: " + str(destination))
                if index == 0:
                    first_outputs[arm] = (destination, value)
        a, av = first_outputs["A"]
        b, bv = first_outputs["B"]
        hemi = data["cases"][0]["hemisphere"]
        report["same_input_first_side_AB"] = {
            "inflated_bytes_equal": C.sha(a / (hemi + ".inflated")) == C.sha(b / (hemi + ".inflated")),
            "sulc_bytes_equal": C.sha(a / (hemi + ".sulc")) == C.sha(b / (hemi + ".sulc")),
            "actual_full_step_trace_equal": json.loads((a / "trace.private.json").read_text()) == json.loads((b / "trace.private.json").read_text()),
            "terminal_state_equal": {key: av["api"]["integration"][key] for key in STATE_FIELDS} == {key: bv["api"]["integration"][key] for key in STATE_FIELDS}}
        report["strict_current_AB"] = all(report["same_input_first_side_AB"].values())
        worker_reports = [json.loads(Path(row["worker_report"]).read_text()) for row in report["runs"]]
        report["strict_historical_saved_output"] = all(row["strict_historical_output"] for row in worker_reports)
        report["historical_recorded_terminal_state_equal"] = all(row["historical_recorded_integrator_state"]["exact_equal"] for row in worker_reports)
        report["historical_per_step_trajectory"] = "not_available; historical report only terminal steps/RMS/stop"
        report["status"] = "complete" if report["strict_current_AB"] and report["strict_historical_saved_output"] and report["historical_recorded_terminal_state_equal"] else "completed_with_differences"
    except BaseException as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
    C.save(summary, report)
    return 0 if report["status"] == "complete" else 3 if report["status"] == "completed_with_differences" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--benchmark", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=prepare)
    p = commands.add_parser("run")
    for arm in ("baseline", "candidate"):
        p.add_argument("--" + arm + "-source", type=Path, required=True)
        p.add_argument("--" + arm + "-manifest", type=Path, required=True)
        p.add_argument("--" + arm + "-manifest-sha256", required=True)
    p.add_argument("--inputs-manifest", type=Path, required=True)
    p.add_argument("--inputs-manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--python-interpreter", default=sys.executable)
    p.set_defaults(function=run)
    p = commands.add_parser("worker")
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--payload-sha256", required=True)
    p.set_defaults(function=worker)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
