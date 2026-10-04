"""汇总已完成的同例官方评估；只读JSON，不执行影像算法或建立新阈值。

输入：--current-evaluation 当前评估目录、--previous-evaluation 旧版本同例评估目录、
--whole-metadata 当前整例元数据目录、--previous-binding-correction 已审计旧绑定纠正目录、--report 新JSON路径。
输出：版本/输入绑定、原始报告SHA、最终指标、旧新差额和独立验收状态。
不读取MRI、网格、权重或许可证；无对应独立官方CLI，参考完整recon-all -all。
任一绑定不符、18阶段未完成或report已存在时抛异常。
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

def summarize(*, current_evaluation, previous_evaluation, whole_metadata, previous_binding_correction, report):
    """目录参数为Path；距离单位mm，统计沿原报告单位；返回写出的字典。"""
    started = time.perf_counter()
    sources = {}
    def read(root, name):
        path = Path(root) / name
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    now = read(current_evaluation, "execution_binding.json")
    old = read(previous_evaluation, "execution_binding.json")
    correction = read(previous_binding_correction, "correction.json")
    fixed_old = read(previous_binding_correction, "verification/execution_binding.json")
    if (correction["status"] != "metadata_correction_verified" or
            correction["changed_fields"] != ["input_sha256"]):
        raise ValueError("historical binding correction not verified")
    if {k for k in old if old.get(k) != fixed_old.get(k)} != {"input_sha256"} or set(old) != set(fixed_old):
        raise ValueError("historical binding changed beyond input SHA")
    original_sha = sources[str(Path(previous_evaluation) / "execution_binding.json")]
    fixed_sha = sources[str(Path(previous_binding_correction) / "verification/execution_binding.json")]
    if (original_sha != correction["frozen_receipts"]["original_binding"]["sha256"] or
            fixed_sha != correction["frozen_receipts"]["verification_binding"]["sha256"]):
        raise ValueError("historical binding correction SHA mismatch")
    old = fixed_old
    if now["input_sha256"] != old["input_sha256"]:
        raise ValueError("different raw T1")
    if now["official"]["subject"] != old["official"]["subject"]:
        raise ValueError("different official reference")
    checkpoint = read(current_evaluation, "checkpoint.json")
    if checkpoint["status"] != "complete" or len(checkpoint["phases"]) != 18:
        raise ValueError("current evaluation incomplete")
    if any(p["status"] != "complete" for p in checkpoint["phases"].values()):
        raise ValueError("a comparison phase incomplete")
    strict = read(current_evaluation, "strict_precision_candidate_vs_official.json")
    previous_strict = read(previous_evaluation, "strict_startup_only_candidate_vs_official.json")
    if strict["checked"] != 138 or previous_strict["checked"] != 138:
        raise ValueError("wrong strict diagnostic scope")
    current = read(current_evaluation, "region_precision_candidate_vs_official.json")
    previous = read(previous_evaluation, "region_startup_only_candidate_vs_official.json")
    metric_fields = ("matched_regions", "mae", "median_absolute_relative_error_percent",
                     "p90_absolute_relative_error_percent", "maximum_absolute_error",
                     "worst_regions_by_relative_error")
    metrics = {}
    for name, result in current["aparc_68"].items():
        if result["matched_regions"] != 68 or previous["aparc_68"][name]["matched_regions"] != 68:
            raise ValueError("incomplete aparc regions")
        metrics[name] = {
            "current": {k: result[k] for k in metric_fields},
            "previous": {k: previous["aparc_68"][name][k] for k in metric_fields},
            "mae_delta_current_minus_previous": result["mae"] - previous["aparc_68"][name]["mae"],
        }
    dice = read(current_evaluation, "dice_precision_candidate_vs_official.json")
    previous_dice = read(previous_evaluation, "dice_startup_only_candidate_vs_official.json")
    dice_fields = ("different_voxels", "minimum_dice", "p05_dice", "median_dice", "worst_labels", "stored_dtypes")
    label_metrics = {
        name: {"current": {k: values[k] for k in dice_fields},
               "previous": {k: previous_dice["files"][name][k] for k in dice_fields}}
        for name, values in dice["files"].items()
    }
    surfaces = {
        name: read(current_evaluation, f"surface_{name}_precision_candidate_vs_official.json")
        for name in ("white", "pial")
    }
    whole = read(whole_metadata, "summary.json")
    candidate = now["precision_candidate"]["completion"]
    previous_candidate = old["startup_only_candidate"]["completion"]
    if whole["completion"]["code_commit"] != candidate["code_commit"]:
        raise ValueError("whole/evaluation source mismatch")
    wall = candidate["command_seconds"]
    previous_wall = previous_candidate["command_seconds"]
    value = {
        "schema": "fnit-same-case-precision-summary-v1", "case": "ds000114_sub-06",
        "current_code_commit": candidate["code_commit"],
        "previous_code_commit": previous_candidate["code_commit"],
        "source_archive_sha256": candidate["source_archive_sha256"],
        "raw_t1_sha256": now["input_sha256"], "official": now["official"],
        "strict_reproduction": {"status": "failed", "checked": 138, "passed": strict["passed"],
                                "previous_passed": previous_strict["passed"]},
        "optimization_degradation": "not_assessed; versions also differ in main changes",
        "overall_metric_equivalence": "not_assessed; no confirmed whole-case gates",
        "entry_seconds": wall, "previous_entry_seconds": previous_wall,
        "observed_entry_delta_seconds": wall - previous_wall,
        "observed_entry_delta_percent": 100 * (wall / previous_wall - 1),
        "runtime_comparison_scope": "single same-host same-T1 same-thread-budget observation; shared load not controlled",
        "timing": whole["timing"], "monitor": whole["monitor"],
        "whole_own_simultaneous_sampled_gpu_peak_bytes": whole["own_sampled_parent_children_peak_bytes"],
        "aparc_68": metrics, "label_dice": label_metrics, "surfaces": surfaces,
        "comparison_phase_seconds": sum(p["seconds"] for p in checkpoint["phases"].values()),
        "comparison_lock_wait_seconds": sum(p.get("lock_wait_seconds", 0) for p in checkpoint["phases"].values()),
        "source_reports_sha256": sources,
    }
    value["report_summary_seconds"] = time.perf_counter() - started
    with Path(report).open("x") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-evaluation", type=Path, required=True)
    parser.add_argument("--previous-evaluation", type=Path, required=True)
    parser.add_argument("--whole-metadata", type=Path, required=True)
    parser.add_argument("--previous-binding-correction", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    result = summarize(current_evaluation=arguments.current_evaluation,
                       previous_evaluation=arguments.previous_evaluation,
                       whole_metadata=arguments.whole_metadata, previous_binding_correction=arguments.previous_binding_correction, report=arguments.report)
    print(json.dumps({"case": result["case"], "strict": result["strict_reproduction"],
                      "report": str(arguments.report)}, ensure_ascii=False))
