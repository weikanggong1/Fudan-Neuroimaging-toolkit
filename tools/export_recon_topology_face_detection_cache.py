"""导出已完成真实 ABBA 的数值、版本和哈希；私有路径/命令/数据留在原目录。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hashes(mapping):
    result = {}
    for key, value in mapping.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key) or not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError("identifier and SHA-256 required")
        result[key] = value
    return result


def _numbers(value):
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        return [_numbers(item) for item in value]
    if isinstance(value, dict):
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", key) for key in value):
            raise ValueError("numeric dictionary identifier required")
        return {key: _numbers(item) for key, item in value.items()}
    raise ValueError("public numeric data must not contain path or command strings")


def run(*, run_directory, source_manifest, plot_directory, output):
    """已完成run目录、冻结manifest、已完成plot目录、新output目录→public报告路径。

    全部必填Path；输入只读。先要求两个进程EXIT0、全部四次严格比较成功，
    导出白名单数字/哈希，记录私有原件SHA但不复制内容。公开图须匹配brain及
    report SHA。元数据版本为导出时探测，不冒充ABBA期间的负载/显存采样。
    非法结构/未完成/含路径字段/已有output抛ValueError；不替换失败状态。
    """
    run_directory = Path(run_directory)
    source_manifest = Path(source_manifest)
    plot_directory = Path(plot_directory)
    output = Path(output)
    raw = run_directory / "candidate/report.private.json"
    report = json.loads(raw.read_text())
    started = json.loads((run_directory / "STARTED.json").read_text())
    process = json.loads((run_directory / "process.private.json").read_text())
    exits = [json.loads((run_directory / name).read_text())["exit_code"]
             for name in ("EXIT.json", "candidate/EXIT.json")]
    plot = json.loads((plot_directory / "plot_receipt.public.json").read_text())
    if (output.exists() or exits != [0, 0] or report["status"] != "passed" or
            len(report["records"]) != 4 or len(report["comparisons"]) != 4 or
            plot["benchmark_report_sha256"] != _sha(raw) or
            plot["brain_sha256"] != started["input_sha256"]["brain"] or
            json.loads((plot_directory / "EXIT.json").read_text())["exit_code"] != 0 or
            _sha(source_manifest) != started["manifest_sha256"]):
        raise ValueError("fresh export requires complete bound ABBA and actual plot")
    comparisons = []
    for row in report["comparisons"]:
        groups = {}
        for name, group in row["groups"].items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise ValueError("public group name must not contain private paths")
            groups[name] = _numbers(group)
            if not all(value is True for value in group.get("fields", {}).values()
                       for value in value.values()) or group.get("keys_equal", True) is not True or group.get("equal", True) is not True:
                raise ValueError("all full comparisons must pass")
        comparisons.append(dict(index=row["index"], groups=groups))
    records = []
    for row in report["records"]:
        if row["backend"] not in ("control", "candidate"):
            raise ValueError("ABBA backend required")
        times = {key: _numbers(row[key]) for key in (
            "index", "fresh_cli_seconds", "complete_api_seconds", "child_seconds_including_imports",
            "calibration_seconds", "face_detection_calls", "context_preparations", "cpu_affinity")}
        times.update(backend=row["backend"],
            first_candidate_source_gate=row["first_candidate_source_gate"],
            final_repeat_gate=row["final_repeat_gate"],
            input_sha256=_hashes(row["input_sha256"]),
            actual_fnit_loaded_source_sha256=_hashes(row["actual_fnit_loaded_source_sha256"]),
            driver_sha256=row["driver_sha256"],
            thread_environment={key: int(value) for key, value in row["thread_environment"].items()})
        if times["first_candidate_source_gate"] != "strict_bounded_candidate_pipeline_passed" or times["final_repeat_gate"] != "bounded_real_finalization_repeat_passed":
            raise ValueError("complete bounded source and self-repeat gates required")
        records.append(times)
    summaries = {}
    for key in ("fresh_cli_seconds", "complete_api_seconds"):
        medians = {backend: statistics.median(row[key] for row in records if row["backend"] == backend)
                   for backend in ("control", "candidate")}
        summaries[key] = dict(**medians, reduction_percent=100 * (1-medians["candidate"]/medians["control"]))
    cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), None)
    versions = {}
    for name in ("numpy", "numba", "torch", "nibabel", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    public = dict(status="passed", scope="full same-input candidate/finalization ABBA; not full GA or raw-T1",
        records=records, comparisons=comparisons, median_seconds=summaries,
        complete_driver_api_seconds=report["complete_driver_seconds"],
        complete_driver_cli_seconds=process["complete_driver_cli_seconds"],
        child_peak_rss_kib=process["child_peak_rss_kib"], exit_codes=exits,
        input_sha256=_hashes(started["input_sha256"]), source_manifest_sha256=_sha(source_manifest),
        private_report_sha256=_sha(raw), private_process_sha256=_sha(run_directory / "process.private.json"),
        private_started_sha256=_sha(run_directory / "STARTED.json"),
        baseline_git=started["baseline_git"], export_script_sha256=_sha(__file__),
        version_probe=dict(time_utc=datetime.now(timezone.utc).isoformat(),
                           cpu_model=cpu_model, python=platform.python_version(), libraries=versions,
                           timing="after completed ABBA; not a benchmark load measurement"),
        load_during_benchmark="not_recorded", cuda_visible=False, gpu_memory_bytes=None,
        pytorch_allocated_bytes=None, pytorch_reserved_bytes=None,
        complete_ga="not_assessed", whole_recon_all="not_assessed", final_mesh_quality="not_assessed",
        overall_metric_equivalence="not_assessed", production_default_changed=False,
        reference_after_used_to_generate=False)
    text = json.dumps(public, indent=2, ensure_ascii=False, allow_nan=False)+"\n"
    if re.search(r"/cpfs01|/cwStorage|gongwk|gwk_44019|10\.193\.|10\.190\.|password=|token=|BEGIN PRIVATE", text):
        raise ValueError("public report contains a protected string")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "report.json").write_text(text)
    (output / "SOURCE_MANIFEST.json").write_bytes(source_manifest.read_bytes())
    for name in ("plot_receipt.public.json", "candidate_t1_and_difference.png", "EXIT.json"):
        source = plot_directory / name
        (output / ("plot_EXIT.json" if name == "EXIT.json" else name)).write_bytes(source.read_bytes())
    return dict(status="exported", files={p.name: dict(bytes=p.stat().st_size, sha256=_sha(p))
                                          for p in sorted(output.iterdir())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-directory", "source-manifest", "plot-directory", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2, allow_nan=False))
