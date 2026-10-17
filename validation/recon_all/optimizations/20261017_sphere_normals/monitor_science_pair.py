"""外层只读监测显式科学子命令；复用既有PID归属和同期显存采样。

不改冻结重建，不初始化CUDA，不设精度/线程/allocator，不自动重试或终止
子任务。每帧增量JSONL；无法归属或采样失败时完整自有峰值保持null。
输入、输出、时钟范围与具名示例见MONITOR_SCIENCE_PAIR.md。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback


ENVIRONMENT_KEYS = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS",
    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF")
PAIR_SCOPE = "two-raw-T1-four-empty-directory-paired-whole-runs"


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def bound_file(path, expected):
    """普通显式文件与SHA必须匹配；不读取文件内容以外的凭据或环境。"""
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None or file_sha(path) != expected:
        raise ValueError("bound metadata/source SHA mismatch")
    return path



def source_manifest_identity(*, source_root, source_manifest, source_manifest_sha256,
                             source_commit, profiling_sha256, nvml_sha256):
    """核对既有源码清单提交和两份监测模块，不写marker或修改冻结树。"""
    path = bound_file(source_manifest, source_manifest_sha256)
    value = json.loads(path.read_text())
    if (value.get("schema") != "fnit-sphere-source-v1" or
            re.fullmatch(r"[0-9a-f]{40}", source_commit) is None or
            value.get("commit") != source_commit or not isinstance(value.get("files"), dict)):
        raise ValueError("source manifest schema/commit differs")
    expected = {"src/fnit/recon_all/profiling.py": profiling_sha256,
                "src/fnit/recon_all/nvml_process_memory.py": nvml_sha256}
    for relative, expected_sha in expected.items():
        if value["files"].get(relative) != expected_sha:
            raise ValueError("monitor module SHA differs from existing source manifest")
        bound_file(Path(source_root) / relative, expected_sha)
    return {"source_commit": source_commit, "source_manifest_sha256": source_manifest_sha256,
            "source_manifest_declared_files": len(value["files"]),
            "source_manifest_check_scope": "existing manifest SHA/commit plus two loaded monitoring modules; full science source integrity remains the original job gate"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def finite_number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def empty_totals():
    return {"sampled_frames": 0, "resolved_frames": 0, "unknown_or_failed_frames": 0,
        "first_sample_monotonic": None, "last_sample_monotonic": None,
        "maximum_observed_interval_seconds": None, "owned_simultaneous_sampled_peak_bytes": None,
        "partial_attributed_same_snapshot_peak_bytes": None, "target_device_sampled_peak_bytes": None,
        "target_compute_process_sum_sampled_peak_bytes": None}


def update_totals(totals, frame):
    """只汇总既有sampler本帧数值；任一未知帧禁止宣称完整自有峰值。"""
    totals["sampled_frames"] += 1
    tick = frame["capture_started_monotonic"]
    if totals["first_sample_monotonic"] is None:
        totals["first_sample_monotonic"] = tick
    previous = totals["last_sample_monotonic"]
    if previous is not None:
        totals["maximum_observed_interval_seconds"] = max(totals["maximum_observed_interval_seconds"] or 0., tick - previous)
    totals["last_sample_monotonic"] = tick
    row = frame.get("sampler_row")
    resolved = isinstance(row, dict) and row.get("ownership") == "resolved" and finite_number(row.get("tree_total_bytes")) is not None
    totals["resolved_frames" if resolved else "unknown_or_failed_frames"] += 1
    if isinstance(row, dict):
        for output, source in (("partial_attributed_same_snapshot_peak_bytes", "known_tree_bytes"),
                ("target_device_sampled_peak_bytes", "target_device_used_bytes"),
                ("target_compute_process_sum_sampled_peak_bytes", "target_compute_process_sum_bytes")):
            value = finite_number(row.get(source))
            if value is not None:
                totals[output] = max(totals[output], value) if totals[output] is not None else value
    if totals["unknown_or_failed_frames"]:
        totals["owned_simultaneous_sampled_peak_bytes"] = None
    elif resolved:
        value = row["tree_total_bytes"]
        totals["owned_simultaneous_sampled_peak_bytes"] = max(totals["owned_simultaneous_sampled_peak_bytes"] or 0, value)


class RunObserver:
    """只读实际pair/benchmark元数据，保存可证明的CLI存活观测窗口。

    冻结工具没有持久化精确CLI起止monotonic，故不编造该边界。不绑定的
    wrapper空档帧不进入任何病例；首次/末次观测不是精确CLI起止。
    """
    def __init__(self, pair_report):
        self.path = Path(pair_report).absolute() if pair_report else None
        self.cache = {}
        self.run_metadata = {}

    def read(self, path):
        path = Path(path)
        stat = path.stat()
        key = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        previous = self.cache.get(str(path))
        if previous is None or previous[0] != key:
            payload = path.read_bytes()
            self.cache[str(path)] = (key, json.loads(payload), hashlib.sha256(payload).hexdigest())
        return self.cache[str(path)][1:]

    def poll(self, tree, identity_function):
        if self.path is None or not self.path.is_file():
            return {"status": "not_observed"}
        try:
            pair, pair_sha = self.read(self.path)
            if pair.get("schema_version") != 1 or pair.get("scope") != PAIR_SCOPE:
                return {"status": "unrecognized_pair_schema"}
            live = []
            for row in pair.get("runs", []):
                benchmark_path = Path(row.get("benchmark_path", "")).absolute()
                if not benchmark_path.is_relative_to(self.path.parent) or not benchmark_path.is_file():
                    continue
                benchmark, benchmark_sha = self.read(benchmark_path)
                same = (isinstance(row.get("input_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"]) is not None and
                        isinstance(row.get("code_version"), str) and re.fullmatch(r"[0-9a-f]{40}", row["code_version"]) is not None and
                        benchmark.get("input_sha256") == row["input_sha256"] and benchmark.get("code_version") == row["code_version"])
                run_id = row.get("run_id")
                if not same or not isinstance(run_id, str):
                    continue
                metadata = {key: row.get(key) for key in ("run_id", "case", "arm", "code_version", "input_sha256", "status", "exit_code")}
                metadata.update(benchmark_report_sha256_observed=benchmark_sha, pair_report_sha256_observed=pair_sha,
                    benchmark_execution=benchmark.get("execution"), benchmark_cli_wall_seconds=finite_number(benchmark.get("cli_wall_seconds")),
                    benchmark_full_harness_seconds=finite_number(benchmark.get("full_harness_seconds")))
                self.run_metadata[run_id] = metadata
                pid = benchmark.get("cli_pid")
                if type(pid) is int and pid in tree and row.get("status") == "running" and benchmark.get("execution") == "running":
                    identity = identity_function(pid)
                    command = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
                    if (identity.get("identity_error") is None and command == benchmark.get("cli_command") and
                            "fnit.recon_all.native_free" in command):
                        live.append({**metadata, "status": "verified_live_cli_metadata", "cli_identity": identity})
            return live[0] if len(live) == 1 else {"status": "ambiguous_live_runs" if live else "no_verified_live_cli"}
        except (OSError, ValueError, TypeError, AttributeError):
            return {"status": "metadata_unavailable"}


def capture_frame(*, query, sampler, helper, gpu_uuid, observer):
    """分开记录映射用raw快照与sampler新快照，不将两个时刻混成同一帧。"""
    started = time.monotonic()
    frame = {"capture_started_monotonic": started, "sampler_row": None}
    # 只保留当前帧；完整原行已逐帧写JSONL，不在内存累计数小时样本。
    sampler.samples.clear()
    sampler.errors.clear()
    try:
        tree = sampler._tree(os.getpid())
        before = observer.poll(tree, sampler._process_identity)
        raw_started = time.monotonic()
        rows, device_bytes, device_error = query.snapshot(gpu_uuid)
        raw_ended = time.monotonic()
        mapping, evidence = helper.build_owned_sched_pid_map(query, gpu_uuid, tree, rows)
        sampler.trusted_host_pid_map = sampler._validate_pid_map(mapping)
        mapping_ended = time.monotonic()
        row, call = helper.sample_current_call(sampler)
        if sampler.uuid != gpu_uuid:
            raise ValueError("sampler actual target UUID differs")
        after = observer.poll(sampler._tree(os.getpid()), sampler._process_identity)
        same_run = (before.get("status") == after.get("status") == "verified_live_cli_metadata" and
                    before.get("run_id") == after.get("run_id") and before.get("cli_identity") == after.get("cli_identity"))
        frame.update(mapping_raw_snapshot={"started_monotonic": raw_started, "ended_monotonic": raw_ended,
                "rows": rows, "device_used_bytes": device_bytes, "device_query_error": device_error,
                "scope": "mapping discovery snapshot only; not the sampler snapshot"},
            mapping_finished_monotonic=mapping_ended, mapping_evidence=evidence, sampler_call=call,
            sampler_row=row, run_binding=after if same_run else {"status": "not_bound_across_sample", "before_status": before.get("status"), "after_status": after.get("status")})
    except Exception as error:
        sampler.trusted_host_pid_map = {}
        frame.update(status="unknown_sample", error_type=type(error).__name__,
                     run_binding={"status": "not_assessed"})
    frame["capture_ended_monotonic"] = time.monotonic()
    return frame


def monitor(options, command):
    """原命令仅Popen一次；采样只读。监测异常不终止已启动科学子树。"""
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("explicit command after -- required")
    if socket.gethostname() != options.expected_host or os.getuid() != options.expected_uid:
        raise ValueError("monitor host/account differs")
    if not os.uname().release.startswith("3.10."):
        raise ValueError("owned sched mapping requires validated Linux 3.10 source")
    if not 0.05 <= options.interval <= 60:
        raise ValueError("sampling interval must be 0.05..60 seconds")
    os.umask(0o077)
    source = options.source_root.resolve(strict=True)
    files = {"monitor": bound_file(__file__, options.wrapper_sha256),
        "diagnose_tool": bound_file(options.diagnose_tool, options.diagnose_sha256),
        "profiling": bound_file(source / "src/fnit/recon_all/profiling.py", options.profiling_sha256),
        "nvml": bound_file(source / "src/fnit/recon_all/nvml_process_memory.py", options.nvml_sha256),
        "command_script": bound_file(options.command_script, options.command_script_sha256)}
    source_identity = source_manifest_identity(source_root=source,
        source_manifest=options.source_manifest, source_manifest_sha256=options.source_manifest_sha256,
        source_commit=options.source_commit, profiling_sha256=options.profiling_sha256, nvml_sha256=options.nvml_sha256)
    files["source_manifest"] = options.source_manifest.resolve(strict=True)
    if not any(Path(item).absolute() == files["command_script"] for item in command if item and not item.startswith("-")):
        raise ValueError("declared command script is not in explicit argv")
    if os.path.lexists(options.output_root):
        raise FileExistsError("monitor output must not exist")
    sys.path.insert(0, str(source / "src"))
    from fnit.recon_all.profiling import ProcessTreeDeviceSampler
    from fnit.recon_all.nvml_process_memory import NvmlProcessQuery
    import torch
    for key, module_name in (("profiling", "fnit.recon_all.profiling"), ("nvml", "fnit.recon_all.nvml_process_memory")):
        if Path(sys.modules[module_name].__file__).resolve() != files[key]:
            raise ValueError("loaded monitoring module is outside bound source")
    if torch.cuda.is_initialized():
        raise ValueError("monitor parent must not have initialized CUDA")
    helper = load_module("_bound_owned_sched_monitor", files["diagnose_tool"])
    options.output_root.mkdir(parents=True, mode=0o700)
    report_path = options.output_root / "report.private.json"
    state = {"scope": "independent outer read-only process-tree monitor; original science reports remain unchanged",
        "status": "preflight", "host": socket.gethostname(), "uid": os.getuid(), "pid": os.getpid(),
        **source_identity,
        "files_sha256": {name: file_sha(path) for name, path in files.items()},
        "python_executable": str(Path(sys.executable).resolve()), "python_executable_sha256": file_sha(Path(sys.executable).resolve()),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "target_gpu_uuid": options.gpu_uuid, "logical_device": options.device, "sampling_interval_seconds": options.interval,
        "command": command, "child_started": False, "command_exit_code": None,
        "cuda_initialized_at_entry": False, "wrapper_totals": empty_totals(), "per_run_observed_samples": {},
        "timing_scope": "wrapper monotonic observations include launch, gaps and shutdown; never substitute for scientific CLI/harness clocks",
        "run_boundary_scope": "only stable live CLI metadata/identity across an entire sample; observed first/last frames are not exact CLI start/end monotonic",
        "memory_scope": "existing sampler attribution unchanged; partial known-tree sums are lower-bound observations, never a complete own peak; no continuous-peak guarantee",
        "20GB_owned_memory_acceptance": "not_assessed"}
    save(report_path, state)
    query, sampler, child = None, None, None
    observer = RunObserver(options.speed_pair_report)
    try:
        query = NvmlProcessQuery()
        if options.gpu_uuid not in query.devices().values():
            raise ValueError("declared target UUID absent")
        sampler = ProcessTreeDeviceSampler(device=options.device, parent_pid=os.getpid(), interval=options.interval, query_backend="nvml")
        initial = capture_frame(query=query, sampler=sampler, helper=helper, gpu_uuid=options.gpu_uuid, observer=observer)
        if sampler.uuid != options.gpu_uuid or initial.get("sampler_row") is None:
            raise ValueError("prelaunch target query did not succeed")
        with (options.output_root / "samples.private.jsonl").open("x", buffering=1) as samples, (options.output_root / "child.private.log").open("x") as log:
            def record(frame):
                samples.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
                update_totals(state["wrapper_totals"], frame)
                binding = frame.get("run_binding", {})
                if binding.get("status") == "verified_live_cli_metadata":
                    update_totals(state["per_run_observed_samples"].setdefault(binding["run_id"], empty_totals()), frame)
            record(initial)
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            state.update(status="monitoring", child_started=True, child_pid=child.pid, child_launch_observed_monotonic=time.monotonic())
            save(report_path, state)
            checkpoint = time.monotonic()
            while child.poll() is None:
                tick = time.monotonic()
                record(capture_frame(query=query, sampler=sampler, helper=helper, gpu_uuid=options.gpu_uuid, observer=observer))
                if time.monotonic() - checkpoint >= 30:
                    save(report_path, state)
                    checkpoint = time.monotonic()
                time.sleep(max(0., options.interval - (time.monotonic() - tick)))
            state["command_exit_code"] = child.wait()
            state["child_exit_observed_monotonic"] = time.monotonic()
            record(capture_frame(query=query, sampler=sampler, helper=helper, gpu_uuid=options.gpu_uuid, observer=observer))
        state["status"] = "complete_monitoring" if child.returncode == 0 else "science_command_failed"
    except BaseException as error:
        state.update(status="monitor_failed", error_type=type(error).__name__, traceback=traceback.format_exc())
        if child is not None and child.poll() is None:
            # 只等自有科学命令；不发送任何终止信号或重试。
            state["waiting_for_original_child_after_monitor_failure"] = True
            save(report_path, state)
            state["command_exit_code"] = child.wait()
    finally:
        state["ended_monotonic"] = time.monotonic()
        state["cuda_initialized_after"] = torch.cuda.is_initialized()
        if sampler is not None:
            state["query_backend"] = sampler.query_backend
            state["query_backend_errors"] = sampler.backend_errors
            sampler.close()
        if query is not None:
            query.close()
        state["run_metadata_observed"] = observer.run_metadata
        for run_id in observer.run_metadata:
            state["per_run_observed_samples"].setdefault(run_id, empty_totals())
        state["files_unchanged_after_monitoring"] = {
            name: file_sha(path) == state["files_sha256"][name] for name, path in files.items()}
        if not all(state["files_unchanged_after_monitoring"].values()):
            state["status"] = "monitor_source_binding_changed"
        sample_file = options.output_root / "samples.private.jsonl"
        state["samples_jsonl_sha256"] = file_sha(sample_file) if sample_file.exists() else None
        save(report_path, state)
    return 0 if state["status"] == "complete_monitoring" and not state["cuda_initialized_after"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "source-manifest", "diagnose-tool", "command-script", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("wrapper-sha256", "source-commit", "source-manifest-sha256", "diagnose-sha256", "profiling-sha256", "nvml-sha256", "command-script-sha256", "expected-host", "gpu-uuid"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-uid", type=int, required=True)
    parser.add_argument("--speed-pair-report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--interval", type=float, default=.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    return monitor(options, options.command)


if __name__ == "__main__":
    raise SystemExit(main())
