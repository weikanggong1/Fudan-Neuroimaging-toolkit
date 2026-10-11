"""仅核对本公开报告的文件 SHA 与禁止发布的路径/资源；不读取服务器。"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=Path(__file__).resolve().parent,
                        help="本任务已脱敏报告目录")
    parser.add_argument("--create", action="store_true", help="在最终内容固定后生成清单；默认只读核对")
    args = parser.parse_args()
    root = args.report_root
    manifest_path = root / "PUBLICATION_MANIFEST.json"
    forbidden_extensions = {".mgz", ".mgh", ".nii", ".gz", ".npz", ".cpp", ".h", ".so", ".a"}
    # 仅检查公开字符串；服务器前缀、账号/认证标记及 IPv4 地址不得发布。
    private_pattern = re.compile(rb"/(?:cpfs01|cwStorage)/|gwk_[0-9]{4,}|BEGIN (?:RSA|OPENSSH) PRIVATE|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path or "__pycache__" in path.parts:
            continue
        assert path.suffix not in forbidden_extensions, path
        data = path.read_bytes()
        if path.suffix != ".png":
            assert private_pattern.search(data) is None, path
        entries.append({"path": str(path.relative_to(root)), "bytes": len(data),
                        "published_sha256": hashlib.sha256(data).hexdigest()})
    expected = {"scope": "published frozen registration regression, failure retained; not whole-pipeline validation",
                "raw_report_hashes": "reports/v1 and reports/v2 PUBLIC_EXPORT_MANIFEST original_sha256",
                "public_payload_hashes": "published_sha256 below hashes actual public bytes, not raw server files",
                "MRI_NPZ_original_CPP_licenses_credentials_published": False,
                "overall_equivalence": "not_assessed", "files": entries}
    if args.create:
        manifest_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n")
    else:
        assert json.loads(manifest_path.read_text()) == expected, "公开文件与冻结清单不一致"
    print(json.dumps({"verified_public_payloads": len(entries), "private_string_check": "pass"}))


if __name__ == "__main__":
    main()
