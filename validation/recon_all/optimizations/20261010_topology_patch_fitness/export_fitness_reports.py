"""公开评分收据去敏：只替换主机标识，保留所有数值/失败/版本字段。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(*, downloaded_private_directory, output_directory):
    """下载的五份收据→去敏比较报告与原字节 SHA 清单。

    输出目录必须不存在。比较报告只替换 environment.host 为稳定角色；
    两份 reference、dtype 和 driver 报告保持原字节。原始路径、主机、
    许可证和原影像不公开；本函数不调整任何精度、时间或失败结果。
    """
    if output_directory.exists():
        raise FileExistsError("new public report directory required")
    output_directory.mkdir(parents=True)
    manifest = {"schema": "fnit-public-fitness-report-export-v1",
        "scope": "privacy export only; numeric results and source bindings unchanged",
        "files": {}}
    for name in ["comparison.json", "input_type_comparison.json", "driver_receipt.json",
                 "repeat0.json", "repeat1.json"]:
        source = downloaded_private_directory / name
        target = output_directory / ("reference/" + name if name.startswith("repeat") else name)
        target.parent.mkdir(exist_ok=True)
        if name == "comparison.json":
            report = json.loads(source.read_text())
            before = report["environment"].pop("host")
            report["environment"]["host_role"] = "same authorized CPU host as reference"
            target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            # 检查去敏没有意外改变任何数值字段。
            restored = json.loads(target.read_text())
            restored["environment"].pop("host_role")
            restored["environment"]["host"] = before
            if restored != json.loads(source.read_text()):
                raise RuntimeError("unexpected report change during privacy export")
            changed = ["environment.host -> environment.host_role"]
        else:
            shutil.copyfile(source, target)
            changed = []
        manifest["files"][str(target.relative_to(output_directory))] = {
            "original_private_sha256": sha(source), "public_sha256": sha(target),
            "privacy_changes": changed, "numeric_fields_unchanged": True}
    manifest["export_script_sha256"] = sha(__file__)
    (output_directory / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloaded-private-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    export(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
