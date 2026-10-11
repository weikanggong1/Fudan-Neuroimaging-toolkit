"""发布完整球面诊断JSON/脑图，保留原SHA并替换授权私有根路径。

只处理JSON与PNG，不复制影像、参考程序、原C++或许可证。结果仍绑定
实际trace内的加载模块SHA，不能用当前Git版本覆盖冻结代码身份。
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sanitize(value, replacements):
    if isinstance(value, dict):
        return {key: sanitize(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, replacements) for item in value]
    if isinstance(value, str):
        for source, target in replacements:
            value = value.replace(source, target)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--private-root", action="append", default=[],
                        help="private root=public placeholder; may be repeated")
    args = parser.parse_args()
    replacements = [tuple(item.split("=", 1)) for item in args.private_root]
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for original in sorted(args.raw_directory.rglob("*")):
        if not original.is_file() or original.suffix not in {".json", ".png"}:
            continue
        relative = original.relative_to(args.raw_directory)
        destination = args.output_directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if original.suffix == ".json":
            payload = sanitize(json.loads(original.read_text()), replacements)
            payload["public_receipt_raw_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
            payload["public_receipt_scope"] = "root paths substituted; measured values and frozen code hashes retained"
            destination.write_text(json.dumps(payload, indent=2) + "\n")
        else:
            shutil.copyfile(original, destination)
        entries.append({"path": relative.as_posix(),
                        "raw_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                        "public_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    (args.output_directory / "publication_manifest.json").write_text(json.dumps({
        "scope": "own diagnostic JSON and permitted public-derived brain plot only",
        "publisher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "entries": entries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
