"""公开本轮JSON报告和两张公开T1派生PNG；不复制体积、权重或许可证。

输入为只含本轮验证的runs_root及冻结源码manifest。显式列举JSON/PNG，
校验两例COMPLETE，再生成新目录；数值、bool、null及哈希必须保真。
私有根目录、主机和账号由具名参数指定替换，原始报告保持只读。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def leaves(value):
    if isinstance(value, dict):
        return [x for child in value.values() for x in leaves(child)]
    if isinstance(value, list):
        return [x for child in value for x in leaves(child)]
    return [value] if isinstance(value, (int, float, bool)) or value is None else []


def export(*, runs_root: Path, source_manifest: Path, output: Path,
           private_root: str, private_host: str, private_account: str) -> dict:
    """返回report/PNG来源SHA清单；异常不修改原报告，已有output报错。"""
    for case in ("sub06", "sub07"):
        value = json.loads((runs_root / case / "COMPLETE.json").read_bytes())
        if value.get("status") != "complete_prefixes":
            raise ValueError("pair completion gate failed: " + case)
    paths = sorted(runs_root.rglob("*.json"))
    # 所有输入固定在本次运行目录，且只读取报告格式。
    if not paths or any(p.is_symlink() for p in paths):
        raise ValueError("missing reports or symlink report")
    images = [runs_root / "diagnostics/compare_plot_v1/plots" /
              (case + "_raw_conform_wm_filled.png") for case in ("sub06", "sub07")]
    if not all(p.is_file() and not p.is_symlink() for p in images):
        raise FileNotFoundError("two public-derived brain PNGs required")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"scope": "raw-T1 to filled receipts and public-derived PNGs; not whole recon-all",
                "numeric_boolean_null_leaves_preserved": True,
                "raw_originals_unchanged": True, "reports": {}, "images": {}}

    def sanitize(value):
        if isinstance(value, str):
            return value.replace(private_root, "[PRIVATE_FNIT]").replace(
                private_host, "A100-8").replace(private_account, "[PRIVATE_ACCOUNT]")
        if isinstance(value, list):
            return [sanitize(child) for child in value]
        if isinstance(value, dict):
            return {sanitize(key): sanitize(child) for key, child in value.items()}
        return value

    inputs = [(p, str(p.relative_to(runs_root))) for p in paths]
    inputs.append((source_manifest, "SOURCE_MANIFEST.json"))
    for path, name in inputs:
        raw = path.read_bytes()
        original = json.loads(raw)
        result = sanitize(original)
        if leaves(original) != leaves(result):
            raise RuntimeError("numeric leaf changed: " + name)
        encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        if any(token.encode() in encoded for token in
               (private_root, private_host, private_account)):
            raise RuntimeError("private identifier survived: " + name)
        destination = output / name.replace(".private.json", ".public.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_bytes(encoded)
        manifest["reports"][str(destination.relative_to(output))] = {
            "raw_sha256": sha(raw), "raw_bytes": len(raw),
            "public_sha256": sha(encoded), "public_bytes": len(encoded),
            "numeric_boolean_null_leaf_count": len(leaves(original))}
    for path in images:
        destination = output / "plots" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        data = destination.read_bytes()
        manifest["images"][str(destination.relative_to(output))] = {
            "raw_sha256": sha(path.read_bytes()), "public_sha256": sha(data),
            "bytes": len(data), "kind": "public ds000114 derived diagnostic PNG"}
    manifest["report_count"] = len(manifest["reports"])
    manifest["image_count"] = len(manifest["images"])
    (output / "PUBLIC_EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs-root", "source-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("private-root", "private-host", "private-account"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = export(runs_root=args.runs_root, source_manifest=args.source_manifest,
                    output=args.output, private_root=args.private_root,
                    private_host=args.private_host, private_account=args.private_account)
    print(json.dumps({key: result[key] for key in ("report_count", "image_count")}))


if __name__ == "__main__":
    main()
