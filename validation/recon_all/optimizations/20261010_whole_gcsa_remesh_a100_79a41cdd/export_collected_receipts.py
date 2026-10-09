"""复用已有数值合同导出器，公开本轮JSON并按字节复制六张诊断PNG。

显式输入私有收集目录、新公开目录、私有字符串替换表和已有导出器
路径；不读取MRI、不更改指标、时间、阈值或坐标。原始报告保留。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


def export_collected_receipts(*, input_directory: Path, output_directory: Path,
                             replacements_file: Path, exporter_path: Path) -> dict:
    """返回公开JSON数值合同和PNG SHA；任何替换残留或SHA不符报错。

    参数均为必填Path：已有收据目录/尚不存在的输出目录/JSON字符串
    替换表/已验证export_receipts.py。数字、bool、null的类型和值由原
    导出器逐项核验；PNG原样复制，无重采样、几何变换或再绘制。
    失败可留下未完成目录，但不会修改输入；无原软件对应命令。
    """
    spec = importlib.util.spec_from_file_location("fnit_receipt_exporter", exporter_path)
    if spec is None or spec.loader is None:
        raise ImportError(str(exporter_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.export_receipts(input_directory=input_directory,
        output_directory=output_directory, replacements_file=replacements_file)
    pictures = {}
    for source in sorted(input_directory.rglob("*.png")):
        if source.is_symlink():
            raise ValueError("symlink image receipt")
        relative = source.relative_to(input_directory)
        target = output_directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        original = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != original:
            raise ValueError("image byte content changed")
        pictures[str(relative)] = original
    private_strings = json.loads(replacements_file.read_text())
    for path in output_directory.rglob("*.json"):
        content = path.read_text()
        if any(secret in content for secret in private_strings):
            raise ValueError("private string remains in " + str(path.relative_to(output_directory)))
    report = {"scope": "public JSON receipts and unchanged PNG only",
        "json_file_count": len(manifest["files"]),
        "numeric_boolean_null_values_unchanged": manifest["numeric_boolean_null_values_unchanged"],
        "image_sha256": pictures,
        "base_exporter_sha256": hashlib.sha256(exporter_path.read_bytes()).hexdigest(),
        "collector_exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "private_string_scan": "passed"}
    (output_directory / "PUBLIC_IMAGE_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-directory", "output-directory", "replacements-file", "exporter-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = export_collected_receipts(input_directory=args.input_directory,
        output_directory=args.output_directory, replacements_file=args.replacements_file,
        exporter_path=args.exporter_path)
    print(json.dumps({k: result[k] for k in
        ("json_file_count", "numeric_boolean_null_values_unchanged", "private_string_scan")}))
