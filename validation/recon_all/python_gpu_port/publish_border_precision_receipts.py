"""导出白质目标表达式诊断收据；保留实测失败、完整轨迹和原始报告 SHA。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from publish_pial_norm_receipts import sanitize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    receipts = {
        "before_full_stage.json": "runs/preaparc_first_difference_20261010_v1/sub06-lh/report.json",
        "native_target_passes.json": "runs/preaparc_target_diagnostic_20261010_v1/sub06-lh/report.json",
        "expression_variants.json": "runs/preaparc_target_expression_variants_20261010_v1/report.json",
        "native_probe_build.json": "workspaces/preaparc_target_diagnostic_20261010_v1/build-receipt.json",
        "native_probe_relocation.json": "workspaces/preaparc_target_diagnostic_20261010_v1/relocation-receipt.json",
        "source_manifest.json": "workspaces/preaparc_border_precision_fix_20261010_v1/SOURCE_MANIFEST.json",
    }
    for case in ("sub06-lh", "sub06-rh", "sub07-lh", "sub07-rh"):
        for kind in ("report", "postcheck"):
            receipts[f"{case}_{kind}.json"] = (
                f"runs/preaparc_border_precision_fix_20261010_v1/{case}/{kind}.json"
            )
    for version in ("v1", "v2"):
        receipts[f"shared_{version}_source_manifest.json"] = (
            f"workspaces/shared_border_white_pial_precision_20261010_{version}/SOURCE_MANIFEST.json"
        )
        for case in ("sub06", "sub07"):
            receipts[f"shared_{version}_{case}.json"] = (
                f"runs/shared_border_white_pial_precision_20261010_{version}/{case}/report.json"
            )
    receipts["parallel_targets_report.json"] = "runs/border_parallel_targets_20261010_v1/report.json"
    receipts["parallel_targets_source_manifest.json"] = (
        "workspaces/border_parallel_targets_20261010_v1/SOURCE_MANIFEST.json"
    )
    for name, relative in receipts.items():
        path = args.private_root / relative
        if not path.is_file():
            continue
        raw = json.loads(path.read_text())
        published = {
            "private_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "private_relative_artifact": relative,
            "report": sanitize(raw),
        }
        if name == "native_target_passes.json":
            # 原始 caller 标签保留；实际源码版本由冻结清单另行核对，不能改写旧报告。
            published["verified_base_commit"] = "54f681dcab210a22c2e6862b2854a7033b5a208d"
            published["verified_base_manifest_sha256"] = (
                "a14094ad60b915e64d323095e135c272e1416ef7b11675212a6079a92b086347"
            )
            published["original_code_base_commit_is_caller_label"] = True
        destination = args.output_directory / name
        destination.write_text(json.dumps(published, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
