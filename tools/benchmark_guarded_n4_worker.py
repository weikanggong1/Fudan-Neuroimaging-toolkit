"""固定已安装wheel的两例自产orig：冷CLI与已初始化父API，完整200轮回归。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import nibabel as nib
import numpy as np
import torch

from fnit.recon_all import n4_torch_worker, n4_itk_torch_experimental, n4_bspline_torch
from fnit.recon_all import profiling, nvml_process_memory, worker_program_guard, execution_recipe


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison(candidate, reference):
    left, right = nib.load(str(candidate)), nib.load(str(reference))
    a, b = np.asarray(left.dataobj), np.asarray(right.dataobj)
    if a.shape != b.shape:
        raise ValueError("N4 output shape changed")
    difference = a.astype(np.float64) - b.astype(np.float64)
    return {"same_shape": True, "same_affine": bool(np.array_equal(left.affine, right.affine)),
            "same_dtype": str(a.dtype) == str(b.dtype), "shape": list(a.shape), "dtype": str(a.dtype),
            "different_voxels": int(np.count_nonzero(difference)),
            "max_abs": float(np.max(np.abs(difference))), "rmse": float(np.sqrt(np.mean(difference**2))),
            "negative_differences": int(np.count_nonzero(difference < 0)),
            "positive_differences": int(np.count_nonzero(difference > 0))}


def precision():
    return {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32, "autocast": torch.is_autocast_enabled()}


class Monitor:
    """单采样线程独占sampler；direct映射查询与sampler快照分开保存。"""
    def __init__(self, uuid, diagnostic, phase):
        self.query = nvml_process_memory.NvmlProcessQuery()
        self.query.devices()
        self.sampler = profiling.ProcessTreeDeviceSampler(device="cuda:0", parent_pid=os.getpid(),
                                                          interval=.25, query_backend="nvml")
        self.sampler.uuid = uuid
        self.uuid, self.diagnostic, self.phase = uuid, diagnostic, phase
        self.frames, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                raw, used, error = self.query.snapshot(self.uuid)
                raw_ended = time.monotonic()
                tree = self.sampler._tree(os.getpid())
                mapping, evidence = self.diagnostic.build_owned_sched_pid_map(self.query, self.uuid, tree, raw)
                self.sampler.trusted_host_pid_map = self.sampler._validate_pid_map(mapping)
                row, call = self.diagnostic.sample_current_call(self.sampler)
                self.frames.append({"phase": self.phase, "raw_query_start_monotonic": started,
                    "raw_query_end_monotonic": raw_ended, "raw_query_span_seconds": raw_ended-started,
                    "direct_processes": raw, "direct_device_bytes": used,
                    "direct_error_type": error.split("(", 1)[0] if error else None,
                    "mapping_evidence": evidence, "sampler_call": call,
                    "current_tree_total_bytes": row["tree_total_bytes"] if row is not None else None,
                    "current_ownership": row["ownership"] if row is not None else "unavailable"})
            except Exception as error:
                # 映射或直接查询失败不采样、不复用历史映射/历史合计。
                self.sampler.trusted_host_pid_map = {}
                self.frames.append({"phase": self.phase, "monotonic": started,
                    "current_tree_total_bytes": None, "current_ownership": "unavailable",
                    "error_type": type(error).__name__})
            self.stop.wait(.25)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError("NVML monitor did not finish")
        self.sampler.close()
        self.query.close()
        available = [f["current_tree_total_bytes"] for f in self.frames if f["current_tree_total_bytes"] is not None]
        times = [f.get("raw_query_start_monotonic", f.get("monotonic")) for f in self.frames]
        missing = any(f["current_tree_total_bytes"] is None for f in self.frames)
        return {"sampling_interval_seconds": .25, "frames": self.frames,
            "sampler_report": self.sampler.report(), "all_frames_owned": bool(self.frames) and not missing,
            "full_sequence_peak_bytes": None if missing else max(available, default=None),
            "verified_window_observed_max_bytes": max(available, default=None),
            "max_observed_interval_seconds": max((b-a for a,b in zip(times,times[1:])), default=None),
            "scope": "sampler snapshot sums; verified window is not whole run or continuous peak"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "output", "diagnostic", "wheel", "install-report", "installed-target", "audit-helper"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--resource-protocol", type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--idle-baseline-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == args.gpu_uuid
    assert os.environ["FNIT_RECON_FORBID_ORIGINAL_STAGES"] == "1"
    assert not torch.cuda.is_initialized()
    inputs = json.loads(args.inputs.read_text())
    assert inputs["status"] == "input_receipts_verified_waiting_wheel_and_start"
    install_report = json.loads(args.install_report.read_text())
    assert install_report["status"] == "passed"
    assert sha(args.wheel) == install_report["wheel"]["sha256"]
    modules = [n4_torch_worker, n4_itk_torch_experimental, n4_bspline_torch,
               profiling, nvml_process_memory, worker_program_guard, execution_recipe]
    module_hashes = {}
    for module in modules:
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(args.installed_target.resolve())
        module_hashes[module.__name__] = sha(path)
    for row in install_report["package_checks"]:
        assert sha(args.installed_target / row["path"]) == row["wheel_sha256"]
    for case in inputs["cases"]:
        assert sha(case["input"]) == case["input_sha256"]
        for old in case["old_workers"]:
            assert old["source_sha256"]["full_n4"] == module_hashes[n4_itk_torch_experimental.__name__]
            assert old["source_sha256"]["bspline_fit"] == module_hashes[n4_bspline_torch.__name__]
    query = nvml_process_memory.NvmlProcessQuery()
    devices = query.devices()
    handle = query.handles[args.gpu_uuid]
    rows, used, error = query.snapshot(args.gpu_uuid)
    graphics = query.module.nvmlDeviceGetGraphicsRunningProcesses(handle)
    utilization = query.module.nvmlDeviceGetUtilizationRates(handle)
    query.close()
    memory = None
    protocol = None
    if args.resource_protocol is not None:
        protocol = json.loads(args.resource_protocol.read_text())
        assert protocol["policy"] == "shared_low_utilization_v2"
        assert protocol["source_commit"] == inputs["source_commit"]
        assert protocol["input_receipt_sha256"] == sha(args.inputs)
        physical = int(next(k for k,v in devices.items() if v == args.gpu_uuid))
        # query句柄已关闭，此处用独立NVML引用取得启动前内存门。
        gate_query = nvml_process_memory.NvmlProcessQuery()
        gate_query.devices()
        memory = gate_query.module.nvmlDeviceGetMemoryInfo(gate_query.handles[args.gpu_uuid])
        gate_query.close()
        if physical == 4 or error or utilization.gpu > 10 or memory.free < 32000000000:
            raise RuntimeError("shared resource gate failed; no CUDA allocation or N4 execution")
    elif rows or graphics or utilization.gpu != 0 or error or used != args.idle_baseline_bytes:
        raise RuntimeError("fresh idle gate failed; no CUDA allocation or N4 execution")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    before = precision()
    assert sorted(os.sched_getaffinity(0)) == [88, 89, 90, 91]
    args.output.mkdir(parents=True)
    report = {"status": "running", "source_commit": inputs["source_commit"],
        "wheel_sha256": sha(args.wheel), "wheel_bytes": args.wheel.stat().st_size,
        "root_install_report_sha256": sha(args.install_report),
        "input_receipt_sha256": sha(args.inputs), "harness_sha256": sha(__file__),
        "monitor_helper_sha256": sha(args.diagnostic), "module_sha256": module_hashes,
        "scope": "two self-produced orig, four full N4 exec runs; not whole reconstruction",
        "gpu_uuid": args.gpu_uuid, "physical_gpu": next(k for k,v in devices.items() if v == args.gpu_uuid),
        "threads": 4, "cpu_affinity": sorted(os.sched_getaffinity(0)), "half_requested": False,
        "precision_before": before, "parent_allocator_environment": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
        "original_stage_guard_requested": True, "production_default_changed": False, "rows": [],
        "resource_protocol": protocol, "resource_protocol_sha256": sha(args.resource_protocol) if protocol else None,
        "startup_resource_observation": {"resident_compute_processes": rows, "resident_graphics_count": len(graphics),
            "utilization_percent": int(utilization.gpu), "device_used_bytes": used,
            "device_total_bytes": int(memory.total) if memory else None,
            "device_free_bytes": int(memory.free) if memory else None, "nonexclusive": protocol is not None}}
    spec = importlib.util.spec_from_file_location("verified_nvml_diagnostic", args.diagnostic)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    def save():
        (args.output / "report.private.json").write_text(json.dumps(report, indent=2)+"\n")
    try:
        for mode in ("cold_cli", "initialized_api"):
            for case in inputs["cases"]:
                directory = args.output / case["case"] / mode
                directory.mkdir(parents=True)
                output, receipt = directory / "nu0.mgz", directory / "worker.private.json"
                os.environ["FNIT_RECON_AUDIT_DIR"] = str(directory / "execution-audit")
                monitor = Monitor(args.gpu_uuid, diagnostic, case["case"]+":"+mode)
                retained = None
                initialized_before = torch.cuda.is_initialized()
                monitor.start()
                parent_initialization_seconds = None
                if mode == "initialized_api":
                    initialization_started = time.perf_counter()
                    torch.cuda.set_device("cuda:0")
                    torch.cuda.synchronize("cuda:0")
                    parent_initialization_seconds = time.perf_counter()-initialization_started
                    initialized_before = torch.cuda.is_initialized()
                begin = time.perf_counter()
                try:
                    if mode == "cold_cli":
                        assert not initialized_before
                        command = [sys.executable, "-m", "fnit.recon_all.n4_torch_worker", "--input", case["input"],
                            "--output", str(output), "--report", str(receipt), "--device", "cuda:0", "--threads", "4",
                            "--matmul-tf32", "1", "--cudnn-tf32", "1", "--code-version", inputs["source_commit"]]
                        with (directory / "cli.private.log").open("w") as log:
                            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                        assert not torch.cuda.is_initialized()
                    else:
                        # API已初始化父CUDA；外部计时另包含加载真实orig及64MiB传输。
                        retained = torch.from_numpy(np.asarray(nib.load(case["input"]).dataobj, dtype=np.float32).copy()).to("cuda:0")
                        torch.cuda.synchronize("cuda:0")
                        assert retained.numel()*retained.element_size() == 64*2**20
                        n4_torch_worker.run_isolated_n4(input_path=case["input"], output_path=output,
                            report_path=receipt, device="cuda:0", threads=4, profile=False, code_version=inputs["source_commit"])
                        torch.cuda.synchronize("cuda:0")
                    wall = time.perf_counter()-begin
                    result = json.loads(receipt.read_text())
                    assert result["input_sha256"] == case["input_sha256"] and result["output_sha256"] == sha(output)
                    assert result["api"]["iterations"] == 200
                    assert result["precision"]["matmul_tf32"] and result["precision"]["cudnn_tf32"]
                    assert result["precision"]["autocast"]["enabled"] is False
                    assert result["threads"]["requested"] == 4 and result["threads"]["cpu_affinity"] == [88,89,90,91]
                    assert result["cuda_allocator"]["torch_stats_known_valid"]
                    parent_stats = {"cuda_initialized": torch.cuda.is_initialized(),
                        "torch_stats_valid_for_peak": "PYTORCH_NO_CUDA_MEMORY_CACHING" not in os.environ,
                        "allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_initialized() else None,
                        "reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_initialized() else None,
                        "retained_real_input_bytes": retained.numel()*retained.element_size() if retained is not None else 0}
                finally:
                    monitoring = monitor.finish()
                comparisons = {"old_direct": comparison(output, case["old_direct_output"])}
                for old in case["old_workers"]:
                    assert sha(old["output"]) == old["output_sha256"]
                    comparisons["old_worker_"+old["mode"]] = comparison(output, old["output"])
                for compared in comparisons.values():
                    assert compared["same_affine"] and compared["same_dtype"] and compared["different_voxels"] == 0
                report["rows"].append({"case": case["case"], "mode": mode, "input_sha256": case["input_sha256"],
                    "parent_cuda_initialized_before_exec": initialized_before,
                    "parent_initialization_seconds_outside_api_wall": parent_initialization_seconds, "worker": result,
                    "wall_seconds_including_import_exec_transfer_io_exit": wall, "parent_stats": parent_stats,
                    "monitoring": monitoring, "comparisons": comparisons,
                    "historical_to_native": case["old_workers"][0]["historical_to_native"],
                    "historical_reference_scope": "preserved prior metrics; no upstream image read or new native run"})
                del retained
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize("cuda:0")
                assert precision() == before
                save()
        report["precision_after"] = precision()
        report["status"] = "completed_four_full200_guarded_n4_regressions"
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
