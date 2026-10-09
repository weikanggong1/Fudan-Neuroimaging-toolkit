"""无损压缩已去敏感的完整 detail.json，保持公开收据的逻辑 SHA。

--reports 必须包含 public_export_manifest.json。只压缩大于 --minimum-bytes
（默认 1 MB）的 detail.json；保留全部 JSON 事件和数值，原始私有报告不动。
压缩前核对公开 SHA，压缩后逐字节解压回验，成功后才移除本公开副本的
无压缩重复文件。记录实际 .json.gz 路径/SHA/字节数；原 published_sha256
仍对应解压内容。零时间戳 gzip 不包含路径、账号或时间元数据。无新增依赖。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--minimum-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    manifest_path = args.reports / "public_export_manifest.json"
    receipt = json.loads(manifest_path.read_text())
    if args.minimum_bytes < 1:
        raise ValueError("positive minimum size required")
    digest = lambda data: hashlib.sha256(data).hexdigest()
    compressed = {}
    for name, row in receipt["files"].items():
        path = args.reports / name
        if path.name != "detail.json" or path.stat().st_size < args.minimum_bytes:
            continue
        original = path.read_bytes()
        if digest(original) != row["published_sha256"]:
            raise ValueError("public logical content changed: " + name)
        packed = gzip.compress(original, compresslevel=9, mtime=0)
        if gzip.decompress(packed) != original:
            raise ValueError("gzip roundtrip failed: " + name)
        target = path.with_suffix(".json.gz")
        if target.exists():
            raise ValueError("new compressed path required: " + str(target))
        target.write_bytes(packed)
        if gzip.decompress(target.read_bytes()) != original:
            raise ValueError("stored gzip roundtrip failed: " + name)
        stored_path = str(target.relative_to(args.reports))
        row.update(storage="gzip", stored_path=stored_path, stored_sha256=digest(packed),
                   stored_bytes=len(packed))
        compressed[name] = {"stored_path": stored_path, "stored_sha256": digest(packed),
                            "stored_bytes": len(packed), "logical_sha256": digest(original),
                            "logical_bytes": len(original)}
        path.unlink()
    receipt["large_detail_storage"] = "lossless gzip; published_sha256 is uncompressed content; all original private reports retained"
    manifest_path.write_text(json.dumps(receipt, indent=2) + "\n")
    (args.reports / "compressed_report_manifest.json").write_text(json.dumps({
        "compression_script_sha256": digest(Path(__file__).read_bytes()),
        "scope": "lossless storage only; no omitted callback, numerical or decision events",
        "files": compressed}, indent=2) + "\n")
    print(json.dumps({"compressed_files": len(compressed),
                      "logical_bytes": sum(row["logical_bytes"] for row in compressed.values()),
                      "stored_bytes": sum(row["stored_bytes"] for row in compressed.values())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
