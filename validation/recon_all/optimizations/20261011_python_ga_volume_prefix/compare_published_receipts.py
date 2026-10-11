"""比较已发布旧前段收据和当前安全收据，文件SHA与聚合指标分别记录。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def compare_published_receipts(*, historical_report: Path, current_report: Path,
        historical_whole_receipt: Path, historical_exit_receipt: Path,
        expected_input_sha256: str, output: Path) -> dict:
    """两份已发布JSON及旧终态收据→固定字段的历史/当前对照。

    四份Path输入和原始T1 SHA必填，output须不存在。只复用项目显式
    _volume白名单读取14MRI数值、shape/dtype/几何、标签Dice、文件SHA；
    验证旧整例failed/1。保留每文件候选SHA及共同control SHA，聚合
    指标相同不证明实际数组相同；实际数组结论另由直接诊断给出。
    不加载影像、不重跑生产、不使用旧失败整例作为完整control。
    输出原JSON/旧退出证据/当前源码SHA及逐MRI匹配项；失败抛异常。
    """
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / "tools"))
    from export_recon_genetic_prefix import VOLUMES, _digest, _volume
    _digest(expected_input_sha256)
    paths = dict(historical_report=historical_report, current_report=current_report,
                 historical_whole_receipt=historical_whole_receipt,
                 historical_exit_receipt=historical_exit_receipt)
    records = {key: json.loads(path.read_bytes()) for key, path in paths.items()}
    old, current = records["historical_report"], records["current_report"]
    if (old["input_sha256"] != expected_input_sha256 or current["input_sha256"] != expected_input_sha256
            or old["status"] != "complete_prefix_diagnostic" or current["status"] != "complete_prefix_diagnostic"
            or set(old["volumes"]) != set(VOLUMES) or set(current["volumes"]) != set(VOLUMES)):
        raise ValueError("same declared T1 and fourteen published prefix outputs required")
    terminal, exit_receipt = records["historical_whole_receipt"], records["historical_exit_receipt"]
    if terminal["execution"] != "failed" or terminal["exit_code"] != 1 or exit_receipt["exit_code"] != 1:
        raise ValueError("historical failed/1 whole boundary differs")
    rows = {}
    for name in VOLUMES:
        left, right = _volume(old["volumes"][name]), _volume(current["volumes"][name])
        values = sorted((set(left) | set(right)) - {"file_sha256"})
        rows[name] = dict(historical_candidate_sha256=left["file_sha256"]["candidate"],
            current_candidate_sha256=right["file_sha256"]["candidate"],
            candidate_file_bytes_equal=left["file_sha256"]["candidate"] == right["file_sha256"]["candidate"],
            historical_control_sha256=left["file_sha256"]["reference"],
            current_control_sha256=right["file_sha256"]["reference"],
            same_control_file_bytes=left["file_sha256"]["reference"] == right["file_sha256"]["reference"],
            retained_field_equal={key: left.get(key) == right.get(key) for key in values},
            retained_fields_equal=all(left.get(key) == right.get(key) for key in values),
            historical_candidate_shape=left["candidate_shape"], current_candidate_shape=right["candidate_shape"],
            historical_candidate_dtype=left["candidate_dtype"], current_candidate_dtype=right["candidate_dtype"])
    lta_keys = ("reference", "candidate", "different_elements", "max_abs", "type", "file_sha256")
    result = dict(schema_version=1, status="complete_published_receipt_comparison",
        validation_layer="published receipt hashes and retained metrics; not direct array evaluation",
        input_sha256=expected_input_sha256,
        historical_whole_execution="failed", historical_whole_exit_code=1,
        historical_prefix_capture_execution=old["candidate_whole_execution"],
        current_prefix_capture_execution=current["candidate_whole_execution_at_capture"],
        actual_array_equivalence_claim=False, overall_metric_equivalence="not_assessed",
        receipt_sha256={key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
        source_sha256=dict(runner=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            explicit_schema=hashlib.sha256((root/"tools/export_recon_genetic_prefix.py").read_bytes()).hexdigest()),
        volumes=rows, candidate_file_bytes_equal_count=sum(row["candidate_file_bytes_equal"] for row in rows.values()),
        retained_fields_equal_count=sum(row["retained_fields_equal"] for row in rows.values()),
        same_control_file_bytes_count=sum(row["same_control_file_bytes"] for row in rows.values()),
        talairach_lta_retained_fields_equal={key: old["talairach_lta"][key] == current["talairach_lta"][key] for key in lta_keys})
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("historical-report", "current-report", "historical-whole-receipt", "historical-exit-receipt", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    result = compare_published_receipts(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("status", "candidate_file_bytes_equal_count", "retained_fields_equal_count")}))


if __name__ == "__main__":
    main()
