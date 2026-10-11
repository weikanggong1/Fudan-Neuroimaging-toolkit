"""独立安装采样模块并只读本进程身份；不创建 CUDA/MRI 工作负载。"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, source_directory: Path, source_manifest_sha256: str,
        installation_directory: Path, output_directory: Path) -> dict:
    """四个具名参数：冻结源码、清单SHA、全新临时导入目标、全新收据目录。"""
    started = time.perf_counter()
    source = source_directory.resolve()
    target = installation_directory.resolve()
    output = output_directory.resolve()
    manifest_path = source / "SOURCE_MANIFEST.json"
    if sha256(manifest_path) != source_manifest_sha256:
        raise ValueError("source manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    expected_names = {"modules/profiling.py", "modules/nvml_process_memory.py",
                      "verify_installed_sampler.py"}
    if set(manifest["payloads"]) != expected_names:
        raise ValueError("exact sampler module and verifier payload required")
    for name, item in manifest["payloads"].items():
        path = source / name
        if sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError("source payload SHA or size mismatch")
    if (target.exists() or output.exists() or target == output or
            target.is_relative_to(source) and target.name != "python_target" or
            output.is_relative_to(source)):
        raise ValueError("fresh independent installation and receipt targets required")
    target.mkdir(mode=0o700, parents=True)
    output.mkdir(mode=0o700, parents=True)
    # 只安装两个真实模块；轻量命名空间不导入整个 FNIT 的调度入口。
    package_name = "fnit_sampler_install_probe"
    package = target / package_name
    package.mkdir(mode=0o700)
    (package / "__init__.py").write_text('"""Standalone sampler installation probe."""\n')
    installed = {}
    for name in ("profiling.py", "nvml_process_memory.py"):
        destination = package / name
        shutil.copyfile(source / "modules" / name, destination)
        expected = manifest["payloads"]["modules/" + name]
        if sha256(destination) != expected["sha256"]:
            raise ValueError("installed module SHA mismatch")
        installed[name] = {"sha256": sha256(destination),
                           "bytes": destination.stat().st_size}
    installed_seconds = time.perf_counter() - started
    environment_before = dict(os.environ)
    import_started = time.perf_counter()
    import torch

    def controls():
        return {"cuda_initialized": bool(torch.cuda.is_initialized()),
                "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
                "torch_intraop_threads": int(torch.get_num_threads()),
                "torch_interop_threads": int(torch.get_num_interop_threads())}

    before = controls()
    pynvml_loaded_before = "pynvml" in sys.modules
    sys.path.insert(0, str(target))  # 仅当前核验进程的临时导入路径。
    module = importlib.import_module(package_name + ".profiling")
    if Path(module.__file__).resolve() != (package / "profiling.py").resolve():
        raise ValueError("actual imported sampler is outside installed target")
    if sha256(Path(module.__file__)) != installed["profiling.py"]["sha256"]:
        raise ValueError("actual imported sampler SHA mismatch")
    import_seconds = time.perf_counter() - import_started
    sampler_type = module.ProcessTreeDeviceSampler
    own_pid = os.getpid()
    identity = sampler_type._process_identity(own_pid)
    sampler = sampler_type(device="cpu", parent_pid=own_pid, query_backend="auto")
    mapping = sampler._pid_mapping({own_pid})
    sampler.sample_if_due(force=True)
    cpu_report = sampler.report()
    nvml_instance_created = sampler._nvml is not None
    sampler.close()
    after = controls()
    changed_environment_keys = sorted(key for key in set(environment_before) | set(os.environ)
                                      if environment_before.get(key) != os.environ.get(key))
    try:
        nvml_package_version = importlib.metadata.version("nvidia-ml-py")
    except importlib.metadata.PackageNotFoundError:
        nvml_package_version = None
    checks = {"module_bytes_equal": True,
              "cuda_not_initialized": not before["cuda_initialized"] and not after["cuda_initialized"],
              "controls_unchanged": before == after,
              "environment_unchanged": not changed_environment_keys,
              "no_automatic_driver_pid_aliases": mapping["aliases"] == {},
              "cpu_device_not_applicable": cpu_report["status"] == "not_applicable",
              "cpu_samples_empty": cpu_report["samples"] == [],
              "query_backend_not_selected": cpu_report["query_backend"] is None,
              "nvml_instance_not_created": not nvml_instance_created,
              "no_pynvml_import_by_sampler": pynvml_loaded_before or "pynvml" not in sys.modules}
    report = {"schema": "fnit.gpu_pid_namespace.installed_metadata.v1",
              "status": "passed" if all(checks.values()) else "failed",
              "source_manifest_sha256": source_manifest_sha256,
              "module_source_commit": manifest["module_source_commit"],
              "git_head_at_bundle": manifest["git_head_at_bundle"],
              "installation": {"mode": "temporary_PYTHONPATH_two_module_target",
                               "full_fnit_package_installation": False,
                               "installed_modules": installed,
                               "verifier_sha256": sha256(Path(__file__))},
              "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                           "kernel": platform.release(), "nvidia_ml_py": nvml_package_version},
              "process_identity": identity,
              "mapping": {key: mapping[key] for key in ("status", "source", "unavailable_reasons")},
              "mapping_aliases_count": len(mapping["aliases"]),
              "controls_before": before, "controls_after": after,
              "changed_environment_keys": changed_environment_keys,
              "cpu_sampler": {key: cpu_report[key] for key in ("status", "query_backend",
                              "peak_tree_total_bytes", "peak_target_compute_process_sum_bytes",
                              "peak_target_device_used_bytes")},
              "checks": checks,
              "seconds": {"validation_and_installation": installed_seconds,
                          "torch_and_actual_module_import": import_seconds,
                          "complete_probe": time.perf_counter() - started},
              "gpu_query_performed": False, "gpu_uuid_exported": False,
              "new_cuda_workload": False, "new_mri_computation": False,
              "old_environment_prefix_modified": False,
              "frozen_raw_sources_modified": False,
              "read_cmdline": False, "read_private_logs": False,
              "scope": "actual installed sampler with allowed own proc identity; no live GPU ownership mapping or MRI validation"}
    path = output / "report.public.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    complete = {"schema": "fnit.gpu_pid_namespace.installed_completion.v1",
                "status": report["status"], "report_file": path.name,
                "report_sha256": sha256(path), "source_manifest_sha256": source_manifest_sha256}
    (output / "COMPLETE.public.json").write_text(json.dumps(complete, indent=2) + "\n")
    if report["status"] != "passed":
        raise RuntimeError("installed sampler metadata check failed; safe report retained")
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--installation-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
