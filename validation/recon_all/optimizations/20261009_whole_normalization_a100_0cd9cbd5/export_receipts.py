"""脱敏显式指定目录中的JSON数值收据，保留原始及公开SHA和数值合同。

仅处理JSON，不复制影像、权重或许可证。替换表是本地JSON对象，键为
私有字符串、值为公开占位符；输出目录须不存在。失败不修改源收据。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def export_receipts(*, input_directory: Path, output_directory: Path,
                    replacements_file: Path) -> dict:
    """读取完整原始JSON树及字符串替换表，返回公开文件哈希清单。

    输入和输出均为目录路径；替换表为JSON str→str对象；没有隐含服务器
    路径。无坐标变换、量纲或计算门槛变更。逐文件比较数字/布尔/null
    的类型及遍历序列；失败抛异常，可能留下未完成的公开目录。
    """
    if not input_directory.is_dir():
        raise NotADirectoryError(input_directory)
    if output_directory.exists():
        raise FileExistsError(output_directory)
    replacements = json.loads(replacements_file.read_text())
    if not isinstance(replacements, dict) or any(not isinstance(k, str) or not k or
            not isinstance(v, str) for k, v in replacements.items()):
        raise ValueError("replacements must be nonempty-string to string mapping")
    def numeric_values(value):
        if isinstance(value, dict):
            return [n for v in value.values() for n in numeric_values(v)]
        if isinstance(value, list):
            return [n for v in value for n in numeric_values(v)]
        if value is None or isinstance(value, (bool, int, float)):
            return [(type(value).__name__, value)]
        return []
    files, count = {}, 0
    paths = sorted(input_directory.rglob("*.json"))
    if not paths:
        raise ValueError("no JSON receipts")
    for source in paths:
        if source.is_symlink():
            raise ValueError("symlink receipts are not allowed")
        raw = source.read_bytes()
        original = json.loads(raw)
        content = raw.decode("utf-8")
        for private, public in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
            # 对JSON编码后的片段替换，避免反斜杠、引号破坏收据。
            content = content.replace(json.dumps(private, ensure_ascii=False)[1:-1],
                                      json.dumps(public, ensure_ascii=False)[1:-1])
        decoded = json.loads(content)
        before, after = numeric_values(original), numeric_values(decoded)
        if json.dumps(before, allow_nan=False) != json.dumps(after, allow_nan=False):
            raise ValueError("numeric receipt changed: " + str(source))
        public_bytes = content.encode("utf-8")
        relative = source.relative_to(input_directory)
        output = output_directory / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(public_bytes)
        files[str(relative)] = {"raw_sha256": hashlib.sha256(raw).hexdigest(),
            "published_sha256": hashlib.sha256(public_bytes).hexdigest(),
            "numeric_boolean_null_values": len(before)}
        count += len(before)
    manifest = {"scope": "JSON receipts only; no MRI, weights, licenses or credentials",
        "policy": "explicit private strings replaced; numeric values and types unchanged",
        "files": files, "numeric_boolean_null_values_unchanged": count,
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output_directory / "PUBLIC_EXPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--replacements-file", type=Path, required=True)
    args = parser.parse_args()
    result = export_receipts(input_directory=args.input_directory,
        output_directory=args.output_directory, replacements_file=args.replacements_file)
    print(json.dumps({"files": len(result["files"]),
        "numeric_boolean_null_values_unchanged": result["numeric_boolean_null_values_unchanged"]}))
