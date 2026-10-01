"""更正既有 Dice JSON 中的 filled 名称，保留所有数值和原始报告。

输入 --input 为冻结比较报告，--template 为已核验默认 255/127 语义的
派生报告，--output 为不存在的新 JSON。仅修改 filled.mgz 的名称字符串；
坐标、dtype、计数、Dice、输入/计算程序哈希均不改变。不读取影像、不执行
配准或分割，结果不是重新运行 benchmark。缺字段/模板语义错误/输出已存在
时抛异常。没有独立原软件 CLI，命名依据标准 mri_fill 默认半球编码。

具名参数示例（路径分别为原报告、已核验语义模板、新派生报告）：
  python correct_dice_label_metadata.py \
    --input whole/sub02/paired/dice_vs_baseline.json \
    --template official_candidate_sub02/dice_vs_official_semantics_corrected.json \
    --output whole/sub02/paired/dice_vs_baseline_semantics_corrected.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def file_sha256(*, path: Path) -> str:
    """读取指定文件原始字节并返回 SHA-256 字符串；读取失败传播异常。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correct_report(*, input_report: Path, template_report: Path,
                   output_report: Path) -> dict:
    """派生 filled 名称更正报告，返回含哈希及数值不变证明的 JSON 字典。

    三个 Path 均必须显式提供；前两者是已完成比较 JSON，后者必须不存在。
    没有影像坐标变换。只允许改 filled 的 127/255 名称，其他未知编码置
    None，防止误解释为其他标签空间；数值/形状/存储dtype与单位原样保留。
    验证除名称与新增来源说明之外的 JSON 完全相同，然后写入新文件。
    """
    if output_report.exists():
        raise FileExistsError(output_report)
    original = json.loads(input_report.read_text())
    template = json.loads(template_report.read_text())
    template_names = {
        label: row["name"]
        for label, row in template["files"]["filled.mgz"]["per_label"].items()
    }
    if template_names.get("255") != "Left-Hemisphere-Fill" \
            or template_names.get("127") != "Right-Hemisphere-Fill":
        raise ValueError("Template is not the verified default hemisphere-fill namespace")
    corrected = copy.deepcopy(original)
    changes = {}
    for label, row in corrected["files"]["filled.mgz"]["per_label"].items():
        name = template_names.get(label) if label in {"127", "255"} else None
        if row["name"] != name:
            changes[label] = {"old_name": row["name"], "corrected_name": name}
            row["name"] = name
    restored = copy.deepcopy(corrected)
    for label, row in restored["files"]["filled.mgz"]["per_label"].items():
        row["name"] = original["files"]["filled.mgz"]["per_label"][label]["name"]
    if restored != original:
        raise AssertionError("A non-name comparison value changed")
    verified = template["label_semantics_correction"]
    corrected["label_semantics_correction"] = {
        "scope": "metadata-only filled namespace correction; no image/Dice rerun",
        "original_report": os.path.relpath(input_report.resolve(), output_report.parent.resolve()),
        "original_report_sha256": file_sha256(path=input_report),
        "template_report": os.path.relpath(template_report.resolve(), output_report.parent.resolve()),
        "template_report_sha256": file_sha256(path=template_report),
        "candidate_calculation_commit": original["code_commit"],
        "corrected_comparator_source_sha256": verified["corrected_comparator_source_sha256"],
        "upstream_fixed_source": verified["upstream_fixed_source"],
        "fnit_source": verified["fnit_source"],
        "changed_names": changes,
        "dice_values_counts_hashes_geometry_unchanged": True,
        "script_sha256": file_sha256(path=Path(__file__)),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(corrected, ensure_ascii=False, indent=2) + "\n")
    return corrected


def main() -> None:
    """读取具名参数并保存新报告；不覆盖原始报告或已有派生结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="原始 Dice JSON")
    parser.add_argument("--template", required=True, type=Path, help="已核验半球编码的派生 JSON")
    parser.add_argument("--output", required=True, type=Path, help="新的元数据派生 JSON")
    args = parser.parse_args()
    report = correct_report(input_report=args.input, template_report=args.template,
                            output_report=args.output)
    print(json.dumps(report["label_semantics_correction"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
