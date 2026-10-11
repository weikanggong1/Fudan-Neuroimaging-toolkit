"""校验实际安装收据，只导出预先声明的版本、数值和哈希字段。"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re


def export_installation(*, run_directory: Path, expected_report: Path,
                        output_file: Path) -> dict:
    """从已完成安装导出安全 JSON；不读取影像或运行模型。

    三个必填 Path 分别为含 report/COMPLETE 的实际运行目录、本地冻结
    安装报告、新输出文件。所有时间单位秒；显存未测保持 null。
    返回并保存 report 字典与原始报告 SHA。字段、完成状态、十二模块
    或来源不符及已有输出时抛异常；原报告、日志和绝对路径不导出。
    完整结构和具名参数中文示例见同目录 README 的第二、三节。
    """
    if output_file.exists():
        raise FileExistsError(output_file)
    data = (run_directory / "report.json").read_bytes()
    original_sha = hashlib.sha256(data).hexdigest()
    complete = json.loads((run_directory / "COMPLETE.json").read_text())
    if complete != {"status": "complete", "report_sha256": original_sha}:
        raise ValueError("completed report binding required")
    actual = json.loads(data)
    expected = json.loads(expected_report.read_text())
    safe = {}
    for key in ("git_base", "source_overlay", "source_manifest_sha256", "wheel"):
        if actual[key] != expected[key]:
            raise ValueError("frozen source/wheel differs: " + key)
        safe[key] = expected[key]
    if not re.fullmatch(r"[0-9a-f]{40}", safe["git_base"]):
        raise ValueError("Git commit required")
    safe["status"] = "passed"
    if actual["status"] != "passed":
        raise ValueError("installation did not pass")
    if len(actual["package_checks"]) != 12:
        raise ValueError("twelve installed module checks required")
    safe["package_checks"] = []
    for checked, frozen in zip(actual["package_checks"], expected["package_checks"]):
        if checked != frozen:
            raise ValueError("source/wheel/installed module differs")
        safe["package_checks"].append(frozen)

    def duration(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative seconds required")
        return value

    for key in ("installation", "entry"):
        if actual[key]["exit_code"] != 0:
            raise ValueError("successful subprocess required")
        safe[key] = {"exit_code": 0, "seconds_including_io": duration(actual[key]["seconds_including_io"])}
    observed = actual["entry"]["checks"]
    local = expected["entry_checks"]
    entry = {}
    for key in ("status", "ga_library_blocker_scan_backend_default", "selected_profile",
                "n4_runtime_dependencies", "precision_and_cuda_at_entry",
                "precision_and_cuda_after_import_and_profile_resolution"):
        if observed[key] != local[key]:
            raise ValueError("runtime entry differs: " + key)
        entry[key] = local[key]
    for key in ("python_version", "torch_version"):
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}(?:\+[A-Za-z0-9.]+)?", observed[key]):
            raise ValueError("numeric package version required")
        entry[key] = observed[key]
    entry["cli"] = []
    if len(observed["cli"]) != len(local["cli"]):
        raise ValueError("both fresh CLI checks required")
    for checked, frozen in zip(observed["cli"], local["cli"]):
        item = {}
        for key in ("module", "exit_code", "verified_options", "output_sha256"):
            if checked[key] != frozen[key]:
                raise ValueError("actual CLI differs: " + key)
            item[key] = frozen[key]
        item["fresh_cli_seconds_including_imports"] = duration(checked["fresh_cli_seconds_including_imports"])
        entry["cli"].append(item)
    safe["entry"]["checks"] = entry
    safe["total_seconds_including_install_and_entry"] = duration(actual["total_seconds_including_install_and_entry"])
    for key in ("verification_script_sha256", "entry_script_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", actual[key]):
            raise ValueError("full program SHA required")
        safe[key] = actual[key]
    if actual["python_version"] != entry["python_version"]:
        raise ValueError("interpreter version differs")
    safe["python_version"] = entry["python_version"]
    if not all(type(x) is int and x >= 0 for x in actual["cpu_affinity"]):
        raise ValueError("numeric CPU affinity required")
    safe["cpu_affinity"] = actual["cpu_affinity"]
    for key in ("runtime_gpu_precision", "whole_recon_all", "clean_conda_full_reconstruction"):
        if actual[key] != "not_assessed":
            raise ValueError("unmeasured scope cannot be passed")
        safe[key] = "not_assessed"
    if actual["gpu_memory_bytes"] is not None:
        raise ValueError("GPU memory was not measured")
    safe["gpu_memory_bytes"] = None
    safe["scope"] = "independent wheel/import/fresh CLI; existing dependencies; no MRI/GPU"
    result = {"report": safe, "private_report_sha256": original_sha}
    output_file.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run_directory", "expected_report", "output_file"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    export_installation(**vars(parser.parse_args()))
