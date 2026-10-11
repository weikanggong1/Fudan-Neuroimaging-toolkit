"""导出本任务真实收据：仅脱敏字符串，保留全部数值、布尔、null 与原始 SHA。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def numeric_leaves(value, trail=()):
    if isinstance(value, dict):
        return [item for key in sorted(value) for item in numeric_leaves(value[key], trail + (key,))]
    if isinstance(value, list):
        return [item for index, child in enumerate(value) for item in numeric_leaves(child, trail + (index,))]
    if value is None or isinstance(value, (bool, int, float)):
        return [(trail, type(value).__name__, value)]
    return []


def sanitize(value, private_root: str):
    if isinstance(value, dict):
        return {key: sanitize(child, private_root) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize(child, private_root) for child in value]
    if isinstance(value, str):
        return value.replace(private_root, "$FNIT_ROOT")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="私有 FNIT 根目录")
    parser.add_argument("--run", required=True, help="runs 下本任务已完成目录名")
    parser.add_argument("--workspace", required=True, help="workspaces 下旧新冻结源码目录名")
    parser.add_argument("--output", type=Path, required=True, help="全新的公开收据目录")
    args = parser.parse_args()
    versions = {"recon_registration_zero_corner_20261010_v1": ("old", "new"),
                "recon_registration_zero_corner_20261010_v2": ("new2",)}
    if args.run not in versions or args.workspace != args.run:
        raise ValueError("本版 exporter 只绑定独立已完成 v1/v2，不能改标其他运行")
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.root / "runs" / args.run
    driver = json.loads((run / "driver.json").read_bytes())
    if driver["status"] not in {"complete", "finished_with_stage_failures"}:
        raise ValueError("驱动尚未完成")
    sources = [("driver.json", run / "driver.json")]
    for case in driver["cases"]:
        if case["name"] == "79-comparison":
            sources.append(("79-comparison.json", run / "79-comparison.json"))
        else:
            sources.append((case["name"] + ".json", run / case["name"] / "report.json"))
    if args.run.endswith("_v1") and (run / "79-comparison.json").exists():
        sources.append(("79-comparison.json", run / "79-comparison.json"))
    workspace = args.root / "workspaces" / args.workspace
    sources += [("source-" + kind + ".json", workspace / kind / "STAGE_SOURCE_BINDING.json") for kind in versions[args.run]]
    sources += [("first-nonfinite-diagnostic.json", args.root / "runs/diagnose_registration_nonfinite_20261010_v1/sub06-lh/diagnostic.json"),
                ("second-nonfinite-diagnostic.json", args.root / "runs/diagnose_registration_nonfinite_20261010_v2/sub06-lh/diagnostic.json")]
    manifest = {"schema_version": 1, "run": args.run, "workspace": args.workspace,
                "scope": "frozen stage regression, including failures; not recon-all end to end",
                "redaction": "private FNIT prefix replaced only in strings; no numeric leaves changed",
                "overall_equivalence": "not_assessed", "files": []}
    for name, path in sources:
        raw = path.read_bytes()
        original = json.loads(raw)
        public = sanitize(original, str(args.root))
        unchanged = numeric_leaves(original) == numeric_leaves(public)
        if not unchanged:
            raise AssertionError("数值脱敏保真失败: " + name)
        encoded = (json.dumps(public, ensure_ascii=False, indent=2) + "\n").encode()
        (args.output / name).write_bytes(encoded)
        manifest["files"].append({"published_path": name, "original_sha256": sha256(raw),
                                  "published_sha256": sha256(encoded), "original_bytes": len(raw),
                                  "published_bytes": len(encoded), "numeric_leaves_preserved": unchanged,
                                  "numeric_leaf_count": len(numeric_leaves(original))})
    (args.output / "PUBLIC_EXPORT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(sources), "numeric_leaves_preserved": True}))


if __name__ == "__main__":
    main()
