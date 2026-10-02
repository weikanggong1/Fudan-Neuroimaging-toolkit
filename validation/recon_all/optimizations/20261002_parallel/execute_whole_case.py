"""从原始 T1 和空目录运行固定版本整例；可用于 CLI 或预初始化 CUDA API。

输入 --config JSON：python/code_root/code_commit/source_archive_sha256/input/
output/weights/assets/native_bin_dir/device/threads/invocation/gpu_uuid/
fs_license/diagnostic_root；可选 pipeline_kwargs 和 pipeline_cli_args。
输出 launch.json、command.log、completion.json；与 run_monitored.py 配合记录
指定 GPU 上同一时刻父子进程显存。pipeline_kwargs 仅用于 API，CLI 使用明确
pipeline_cli_args。计时含本入口参数校验、导入、加载、搬运、计算和读写。
参考结果不进入本入口。输入/路径/后端错误非零退出，失败报告保留。
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["diagnostic_root"])
    root.mkdir(parents=True, exist_ok=False)
    completion = {"execution_status": "failed", "code_commit": config["code_commit"],
                  "source_archive_sha256": config["source_archive_sha256"],
                  "strict_reproduction": "not_assessed",
                  "new_degradation": "not_assessed",
                  "overall_metric_equivalence": "not_assessed; no confirmed whole-case gates"}
    try:
        output = Path(config["output"])
        if output.exists():
            raise FileExistsError("whole-case output must not exist")
        for name in ("input", "fs_license"):
            if not Path(config[name]).is_file():
                raise FileNotFoundError("declared " + name + " unavailable")
        source = Path(config["code_root"]).resolve()
        if not (source / "src/fnit/recon_all/native_free.py").is_file():
            raise FileNotFoundError("declared candidate source unavailable")
        threads = int(config["threads"])
        if threads != 4:
            raise ValueError("paired protocol freezes total CPU budget at 4")
        overrides = config.get("pipeline_kwargs", {})
        protected = {"t1", "subject_dir", "weights_dir", "assets_dir", "native_bin_dir",
                     "device", "threads", "profile_stages", "cuda_allocator_cache"}
        if not isinstance(overrides, dict) or protected.intersection(overrides):
            raise ValueError("pipeline_kwargs cannot override declared inputs/resources/timing")
        extras = config.get("pipeline_cli_args", [])
        protected_cli = {"--weights-dir", "--assets-dir", "--native-bin-dir", "--device",
                         "--threads", "--profile-stages", "--cuda-allocator-cache"}
        if not isinstance(extras, list) or not all(isinstance(item, str) for item in extras):
            raise ValueError("pipeline_cli_args must contain strings")
        if any(item.split("=", 1)[0] in protected_cli for item in extras):
            raise ValueError("pipeline_cli_args cannot override declared resources/timing")
        env = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = str(threads)
        env.update(PYTHONPATH=str(source / "src"), CUDA_VISIBLE_DEVICES=config["gpu_uuid"],
                   FS_LICENSE=config["fs_license"], FREESURFER_HOME=config["assets"],
                   PYTORCH_NO_CUDA_MEMORY_CACHING="1",
                   NUMBA_CACHE_DIR=str(root / "numba_cache"))
        argv = [config["python"], "-m", "fnit.recon_all.native_free", config["input"],
                config["output"], "--weights-dir", config["weights"],
                "--assets-dir", config["assets"], "--native-bin-dir", config["native_bin_dir"],
                "--device", config["device"], "--threads", str(threads), "--profile-stages",
                "--cuda-allocator-cache", "disabled", *extras]
        if config["invocation"] == "initialized_cuda_api":
            argv = [config["python"], str(Path(__file__).resolve()), "--api-child",
                    str(args.config.resolve())]
        elif config["invocation"] != "cli":
            raise ValueError("invocation must be cli or initialized_cuda_api")
        launch = {**config, "command": argv, "host": os.uname().nodename,
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "loadavg_at_launch": os.getloadavg(),
                  "script_sha256": digest(__file__),
                  "config_sha256": digest(args.config),
                  "candidate_native_free_sha256": digest(source / "src/fnit/recon_all/native_free.py"),
                  "timing_scope": "entry validation/import/loading/transfers/compute/output IO",
                  "environment": {k: env[k] for k in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES",
                                  "OMP_NUM_THREADS", "PYTORCH_NO_CUDA_MEMORY_CACHING")}}
        (root / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
        with (root / "command.log").open("w") as stream:
            code = subprocess.call(argv, env=env, stdout=stream, stderr=subprocess.STDOUT)
        completion["child_exit_code"] = code
        if code:
            raise RuntimeError("whole-case child returned " + str(code))
        report_path = output / "fnit-native-free-run.json"
        report = json.loads(report_path.read_text())
        completion["pipeline_status"] = report.get("status")
        if report.get("status") != "complete":
            raise RuntimeError("pipeline did not record execution complete")
        completion["output_validation"] = report.get("output_validation")
        completion["pipeline_total_seconds"] = report.get("total_seconds")
        completion.update(exit_code=0, execution_status="complete")
    except BaseException as error:
        completion.update(execution_status="failed",
                          exit_code=completion.get("child_exit_code") or 1)
        completion["error"] = repr(error)
        raise
    finally:
        completion["command_seconds"] = time.perf_counter() - started
        completion["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (root / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")

def api_child(path):
    config = json.loads(Path(path).read_text())
    import torch
    from fnit.recon_all.profiling import configure_cuda_allocator
    from fnit.recon_all.native_free import run_recon_all_python
    torch.set_num_threads(config["threads"])
    allocator = configure_cuda_allocator(device=config["device"], policy="disabled")
    retained = torch.ones(1, dtype=torch.float32, device=config["device"])
    torch.cuda.synchronize(config["device"])
    kwargs = dict(t1=config["input"], subject_dir=config["output"], weights_dir=config["weights"],
                  assets_dir=config["assets"], native_bin_dir=config["native_bin_dir"],
                  device=config["device"], threads=config["threads"], profile_stages=True,
                  cuda_allocator_cache="auto")
    kwargs.update(config.get("pipeline_kwargs", {}))
    metadata = {"cuda_initialized_before_api": torch.cuda.is_initialized(),
                "device_uuid": str(torch.cuda.get_device_properties(config["device"]).uuid),
                "retained_tensor_bytes": retained.numel() * retained.element_size(),
                "allocator_before_initialization": allocator}
    run_recon_all_python(**kwargs)
    torch.cuda.synchronize(config["device"])
    (Path(config["output"]) / "run-api-invocation.json").write_text(json.dumps(metadata, indent=2) + "\n")

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--api-child":
        api_child(sys.argv[2])
    else:
        main()
