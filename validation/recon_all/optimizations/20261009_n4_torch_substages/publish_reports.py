"""生成 N4 公开收据副本，仅替换私有目录前缀和主机名。

输入是已经核验的私有报告目录，不含影像、许可证或凭据。--output 必须
不存在；只允许 JSON、CSV、日志、JUnit XML 和 PNG。--private-prefix 与
--hostname 可重复传入，原字符串不会写入公开映射。数值、源码和程序哈希
保持原样；每个文件记录原始/公开 SHA-256，原始收据必须另行私有保留。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def publish(*, source: Path, output: Path, private_prefixes: list[str],
            hostnames: list[str]) -> dict:
    """保留报告相对结构；遇到意外格式、链接或残留私有字符串直接失败。"""
    source = source.resolve(strict=True)
    if not source.is_dir() or output.exists():
        raise ValueError("source must be a directory and output must not exist")
    if output.resolve().is_relative_to(source):
        raise ValueError("public output must be outside private input directory")
    prefixes = sorted(set(private_prefixes), key=len, reverse=True)
    names = sorted(set(hostnames), key=len, reverse=True)
    if any(not value or not value.startswith("/") for value in prefixes):
        raise ValueError("private prefixes must be nonempty absolute paths")
    if any(not value for value in names):
        raise ValueError("hostnames must be nonempty")
    manifest = {"scope": "public path-and-host redaction only; numerical results unchanged",
                "originals_retained_privately": True, "files": {}}
    paths = sorted(source.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("report input must not contain symbolic links")
    output.mkdir(parents=True)
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".json", ".csv", ".log", ".xml", ".png"):
            raise ValueError("unexpected report format: " + path.suffix)
        original = path.read_bytes()
        published = original
        for value in prefixes:
            published = published.replace(value.encode(), b"FNIT_ROOT")
        for value in names:
            published = published.replace(value.encode(), b"BENCHMARK_HOST")
        # Binary plots must stay byte-identical; confidential metadata needs a
        # separately reviewed image export, not blind PNG byte replacement.
        if path.suffix.lower() == ".png" and published != original:
            raise ValueError("PNG contains private text; export it without that metadata")
        if path.suffix.lower() == ".json":
            json.loads(published)
        relative = str(path.relative_to(source))
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(published)
        manifest["files"][relative] = {
            "original_sha256": digest(original), "original_bytes": len(original),
            "published_sha256": digest(published), "published_bytes": len(published),
            "private_text_replaced": original != published}
    (output / "public_export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="已核验、私有保留的原始收据目录")
    parser.add_argument("--output", type=Path, required=True, help="新公开目录；不覆盖既有文件")
    parser.add_argument("--private-prefix", action="append", default=[], help="需要移除的目录前缀，可重复")
    parser.add_argument("--hostname", action="append", default=[], help="需要移除的主机名，可重复")
    arguments = parser.parse_args()
    result = publish(source=arguments.source, output=arguments.output,
                     private_prefixes=arguments.private_prefix, hostnames=arguments.hostname)
    print(json.dumps({"files": len(result["files"]), "output": str(arguments.output)}))


if __name__ == "__main__":
    main()
