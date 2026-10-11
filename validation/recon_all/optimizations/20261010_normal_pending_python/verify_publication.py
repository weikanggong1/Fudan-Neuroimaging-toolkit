"""校验本版发布字节及实际测试的候选/测试 SHA，不重新运行 MRI。"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.report_root.resolve()
    manifest = json.loads((root / "PUBLICATION_MANIFEST.json").read_text())
    for item in manifest["payloads"]:
        if digest(root / item["file"]) != item["sha256"]:
            raise ValueError(f"published payload changed: {item['file']}")
    exported = json.loads((root / "reports/PUBLIC_EXPORT_MANIFEST.json").read_text())
    for item in exported["entries"]:
        if digest(root / "reports" / item["file"]) != item["published_sha256"]:
            raise ValueError("public SHA differs from exported bytes")
    repository = root.parents[3]
    benchmark = json.loads((root / "reports/benchmark.json").read_text())
    if digest(repository / "src/fnit/recon_all/normal_pending_python.py") != benchmark["source_module_sha256"]["candidate"]:
        raise ValueError("candidate differs from the actual real-state benchmark")
    contracts = json.loads((root / "reports/unit_v2.json").read_text())
    if digest(repository / "tests/recon_all/test_normal_pending_python.py") != contracts["test_file_sha256"]:
        raise ValueError("contracts differ from the actual final test run")
    print(json.dumps({"payloads_checked": len(manifest["payloads"]), "published_SHA_valid": True,
                      "actual_candidate_and_final_test_SHA_valid": True}))


if __name__ == "__main__":
    main()
