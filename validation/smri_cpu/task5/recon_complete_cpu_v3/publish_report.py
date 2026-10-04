"""匿名导出现有完整 CPU recon-all 比较结果；不执行重建或重算指标。

输入为既有私密 JSON 目录和独立计时导出 JSON，输出为逐项匿名 JSON、
脑区 CSV、逐标签 CSV、源文件 SHA。绝对服务器路径、命令和许可证字段
不发布；源影像及模型也不复制。整体数值等价保持 not_assessed。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


PRIVATE_KEYS = {"reference", "candidate", "subject", "subject_dir", "output", "output_dir",
                "command", "commands", "argv", "environment", "env", "fnit_import",
                "pythonpath", "fs_license", "license", "executable", "python_executable",
                "proper_pair_details", "host"}
REMOVE_ALL_KEYS = {"source_paths", "proper_pair_details"}
REPORTS = ("status.json", "strict.private.json", "geometry.private.json", "vertices.private.json",
           "regions.private.json", "dice.private.json", "surfaces.private.json",
           "quality_fnit/report.json", "quality_official/report.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_member(value: str) -> str:
    """绝对字典键仅保留标准输出相对路径；未知绝对键不发布。"""
    arm = ("official/" if "task5_official_recon" in value else
           "fnit/" if "task5_fnit_recon" in value else "")
    for marker in ("/mri/", "/surf/", "/stats/", "/scripts/"):
        if marker in value:
            return arm + marker.strip("/") + "/" + value.split(marker, 1)[1]
    return arm + Path(value).name


def anonymize(value, audit: dict, key: str = ""):
    if isinstance(value, dict):
        result = {}
        for original_key, item in value.items():
            lower = original_key.lower()
            if lower in REMOVE_ALL_KEYS or (lower in PRIVATE_KEYS and isinstance(item, (str, list))):
                audit["removed_path_or_command_fields"] += 1
                continue
            new_key = portable_member(original_key) if original_key.startswith("/") else original_key
            if new_key in result:
                raise ValueError("anonymized member collision: " + new_key)
            result[new_key] = anonymize(item, audit, new_key)
        return result
    if isinstance(value, list):
        return [anonymize(item, audit, key) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        audit["nonfinite_values_to_null"] += 1
        return None
    if isinstance(value, str):
        if value.startswith(("/cwStorage/", "/home/", "/mnt/", "/public/", "/tmp/")):
            audit["removed_absolute_string_values"] += 1
            return portable_member(value)
        # 比较器错误信息可能在文本中嵌入路径，避免只过滤顶层字段。
        return re.sub(r"/(?:cwStorage|home|mnt|public|tmp)/[^\s\"']+", "[private_path]", value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def table(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty requested table: " + path.name)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--timing-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--runtime-report", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--figure-record", type=Path)
    args = parser.parse_args()
    status = json.loads((args.comparison_dir / "status.json").read_text())
    if status.get("status") != "scored":
        raise ValueError("full comparison has not finished")
    if status.get("strict_checked") != 138:
        raise ValueError("full comparison must include all 138 outputs")
    timing = json.loads(args.timing_report.read_text())
    for arm in ("official", "candidate"):
        if timing[arm].get("status") != "complete" or timing[arm].get("returncode") != 0:
            raise ValueError("incomplete full-process receipt: " + arm)
    if timing.get("candidate_report_status") != "complete":
        raise ValueError("candidate pipeline not complete")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {"removed_path_or_command_fields": 0, "removed_absolute_string_values": 0,
             "nonfinite_values_to_null": 0}
    source_hashes = {}
    public = {}
    for member in REPORTS:
        path = args.comparison_dir / member
        name = member.replace(".private", "").replace("/report", "")
        raw = json.loads(path.read_text())
        value = anonymize(raw, audit)
        write_json(args.output_dir / name, value)
        source_hashes[member] = sha256(path)
        public[name] = value
    write_json(args.output_dir / "timing.public.json", anonymize(timing, audit))
    source_hashes["timing.private.json"] = sha256(args.timing_report)
    table(args.output_dir / "candidate_stages.public.csv", timing["candidate_stages"])
    table(args.output_dir / "official_command_timers.public.csv", timing["official_command_timers"])
    region_rows = []
    for metric, entry in public["regions.json"]["aparc_68"].items():
        if (entry.get("matched_regions") != 68 or entry.get("missing_in_candidate") or
                entry.get("extra_in_candidate")):
            raise ValueError("cortical metric must include all 68 matched regions")
        for region, row in entry["per_region"].items():
            region_rows.append({"metric": metric, "region": region, **row})
    table(args.output_dir / "cortical_regions.public.csv", region_rows)
    label_rows = []
    for filename, entry in public["dice.json"]["files"].items():
        for label, row in entry["per_label"].items():
            label_rows.append({"file": filename, "label": label, **row})
    table(args.output_dir / "segmentation_labels.public.csv", label_rows)
    if args.runtime_report:
        runtime = json.loads(args.runtime_report.read_text())
        write_json(args.output_dir / "runtime.public.json", anonymize(runtime, audit))
        source_hashes["runtime.private.json"] = sha256(args.runtime_report)
    if bool(args.figure_dir) != bool(args.figure_record):
        raise ValueError("figure directory and receipt must be provided together")
    if args.figure_dir:
        figure_receipt = json.loads(args.figure_record.read_text())
        if figure_receipt.get("status") != "complete" or figure_receipt.get("returncode") != 0:
            raise ValueError("figure rendering did not complete")
        provenance_path = args.figure_dir / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        figures = args.output_dir / "figures"
        figures.mkdir(exist_ok=True)
        for name, expected in provenance["outputs_sha256"].items():
            if Path(name).name != name or Path(name).suffix != ".png":
                raise ValueError("unexpected figure member")
            path = args.figure_dir / name
            if sha256(path) != expected:
                raise ValueError("figure transfer hash differs: " + name)
            shutil.copy2(path, figures / name)
        figure_manifest = {"receipt": anonymize(figure_receipt, audit),
                           "provenance": anonymize(provenance, audit),
                           "private_receipt_sha256": sha256(args.figure_record),
                           "private_provenance_sha256": sha256(provenance_path)}
        write_json(figures / "manifest.public.json", figure_manifest)
        source_hashes["figure_record.private.json"] = sha256(args.figure_record)
        source_hashes["figure_provenance.private.json"] = sha256(provenance_path)
    manifest = {"schema_version": 1, "scope": "posthoc reporting of one finished same-input CPU pair",
                "input_sha256": timing["input_sha256"], "canonical_index_sha256": args.index_sha256,
                "source_label": timing["source_label"],
                "source_archive_sha256": timing["source_archive_sha256"],
                "private_result_sha256": source_hashes, "anonymization": audit,
                "overall_numerical_equivalence": "not_assessed",
                "equivalent_reconstruction_speedup": None,
                "publisher_sha256": sha256(Path(__file__)),
                "public_files_sha256": {p.name: sha256(p) for p in sorted(args.output_dir.iterdir())
                                        if p.is_file() and p.suffix in (".json", ".csv") and
                                        p.name != "manifest.public.json"}}
    write_json(args.output_dir / "manifest.public.json", manifest)
    print(json.dumps({"status": "published", "strict_checked": status["strict_checked"],
                      "strict_passed": status["strict_passed"], "cortical_rows": len(region_rows),
                      "label_rows": len(label_rows), "anonymization": audit}))


if __name__ == "__main__":
    main()
