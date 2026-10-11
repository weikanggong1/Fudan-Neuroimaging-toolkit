"""独立任务后台空闲门：不终止进程，取得同卡锁后再运行已授权四组N4。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from fnit.recon_all.nvml_process_memory import NvmlProcessQuery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    root, task = args.fnit_root, args.task
    source, run = root / "workspaces" / task, root / "runs" / task
    status_path = run / "launch.private.json"
    query = NvmlProcessQuery()
    devices = query.devices()
    history = []
    def save(status, **extra):
        status_path.write_text(json.dumps({"status": status, "time_unix": time.time(),
            "MRI_started": False if status in {"waiting_idle_gpu", "resource_query_failed"} else True if status == "completed" else None,
            "harness_dispatched": status in {"dispatching_after_idle_gate", "completed", "failed"},
            "resource_observations": history, **extra}, indent=2)+"\n")
    try:
        while True:
            observations = []
            for index in sorted(devices, key=lambda x: (x != "6", int(x))):
                uuid, lock = devices[index], None
                try:
                    rows, used, error = query.snapshot(uuid)
                    handle = query.handles[uuid]
                    graphics = query.module.nvmlDeviceGetGraphicsRunningProcesses(handle)
                    utilization = query.module.nvmlDeviceGetUtilizationRates(handle)
                    state = {"physical_index": int(index), "uuid": uuid,
                        "compute_process_count": len(rows), "graphics_process_count": len(graphics),
                        "gpu_utilization_percent": int(utilization.gpu), "device_used_bytes": used,
                        "idle": not rows and not graphics and utilization.gpu == 0 and not error}
                    observations.append(state)
                    if not state["idle"]:
                        continue
                    lock = Path("/tmp", "fnit-recon-gpu"+index+".lock").open("a")
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        state["lock_available"] = False
                        continue
                    state["lock_available"] = True
                    # harness在CUDA初始化前再次检查进程/利用率/精确基线。
                    environment = dict(os.environ)
                    environment["CUDA_VISIBLE_DEVICES"] = uuid
                    environment["FNIT_RECON_FORBID_ORIGINAL_STAGES"] = "1"
                    environment["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
                    command = [sys.executable, str(source / "tools/benchmark_guarded_n4_worker.py"),
                        "--inputs", str(run / "inputs.private.json"), "--output", str(run / "execution_v1"),
                        "--diagnostic", str(source / "tools/diagnose_recon_nvml_namespace.py"),
                        "--wheel", str(root / "archive/transfers/fudan_neuroimaging_toolkit-0.16.0-py3-none-any.whl"),
                        "--install-report", str(root / "archive/transfers/recon-fnit5-root-install-v3.public.json"),
                        "--installed-target", str(run / "install_v1"),
                        "--audit-helper", str(source / "tools/recon_python_execution_audit"),
                        "--gpu-uuid", uuid, "--idle-baseline-bytes", str(used)]
                    history.append({"time_unix": time.time(), "states": observations})
                    save("dispatching_after_idle_gate", device=state)
                    with (run / "execution_v1.private.log").open("w") as log:
                        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    (run / "exit_execution_v1").write_text(str(completed.returncode)+"\n")
                    save("completed" if completed.returncode == 0 else "failed", device=state,
                         exit_code=completed.returncode)
                    if completed.returncode == 0:
                        (run / "COMPLETE_execution_v1").write_text("four real N4 calls completed; collect and verify before delivery\n")
                    return completed.returncode
                except Exception as error:
                    observations.append({"physical_index": int(index), "error_type": type(error).__name__, "idle": False})
                finally:
                    if lock is not None:
                        lock.close()
            history.append({"time_unix": time.time(), "states": observations})
            # 完整等待历史另追加；状态只保留最近60轮，避免文件无界增长。
            with (run / "idle_history.private.jsonl").open("a") as stream:
                stream.write(json.dumps(history[-1])+"\n")
            history = history[-60:]
            save("waiting_idle_gpu")
            time.sleep(30)
    finally:
        query.close()


if __name__ == "__main__":
    raise SystemExit(main())
