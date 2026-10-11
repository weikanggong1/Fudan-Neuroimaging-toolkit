"""导出本轮完整比较JSON和公开T1派生图；不复制MRI、权重或许可证。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def numeric_leaves(value):
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in numeric_leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in numeric_leaves(child)]
    return [value] if isinstance(value, (float, int, bool)) or value is None else []


def export(*, runs_root: Path, output: Path, private_root: str,
           private_host: str, private_account: str) -> dict:
    """只导出明确8个JSON和2个PNG；数值/bool/null保真，原始字节不改。

    runs_root含本轮comparisons_v1和plots_v1；output必须新目录。三个
    private参数是现场核验的字符串标识，仅在字符串/字典键中替换；错误
    标识残留、数字变化或失败完成门抛异常。返回完整原/公开SHA清单。
    """
    if output.exists():
        raise FileExistsError(output)
    if (runs_root / "COMPARE_V1_EXIT_CODE").read_text().strip() != "0" or (runs_root / "PLOT_V1_EXIT_CODE").read_text().strip() != "0":
        raise ValueError("actual comparison and plot exit 0 required")
    summary = json.loads((runs_root / "comparisons_v1/summary.json").read_bytes())
    if summary["status"] != "complete_n4_prefix_comparison" or summary["full_workflow_complete"]:
        raise ValueError("complete two-case prefix comparison required")
    if not all((private_root, private_host, private_account)):
        raise ValueError("explicit private identity strings required")
    sources = [(runs_root / "comparisons_v1" / name, "comparisons/" + name)
               for name in ("summary.json", "sub06.json", "sub07.json")]
    for role in ("reference", "candidate"):
        sources.append((runs_root / "comparisons_v1" / f"sub06.{role}.SOURCE_MANIFEST.json.private",
                        role.upper() + "_SOURCE_MANIFEST.json"))
    sources.append((runs_root / "plots_v1/plot_provenance.json", "plots/plot_provenance.json"))
    for case in ("sub06", "sub07"):
        sources.append((runs_root / "comparisons_v1" / f"{case}.candidate.benchmark.json.private",
                        "candidate_producer/" + case + ".json"))
    output.mkdir(parents=True)
    report = {"scope": "two real raw-T1 volume prefixes and public-derived diagnostic PNGs; not whole recon-all",
        "originals_unchanged": True, "numeric_boolean_null_leaves_preserved": True,
        "comparison_and_plot_exit_codes": [0, 0], "reports": {}, "images": {}}

    def sanitize(value):
        if isinstance(value, str):
            return value.replace(private_root, "[PRIVATE_FNIT]").replace(private_host, "A100-8").replace(private_account, "[PRIVATE_ACCOUNT]")
        if isinstance(value, dict):
            return {sanitize(key): sanitize(child) for key, child in value.items()}
        if isinstance(value, list):
            return [sanitize(child) for child in value]
        return value

    for source, name in sources:
        before = source.read_bytes()
        original = json.loads(before)
        public = sanitize(original)
        if numeric_leaves(original) != numeric_leaves(public):
            raise RuntimeError("numeric leaf changed")
        data = (json.dumps(public, indent=2, allow_nan=False) + "\n").encode()
        if any(token.encode() in data for token in (private_root, private_host, private_account)):
            raise RuntimeError("private identity survived")
        target = output / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        if source.read_bytes() != before:
            raise RuntimeError("raw receipt changed")
        report["reports"][name] = {"raw_sha256": sha(before), "public_sha256": sha(data),
            "raw_bytes": len(before), "public_bytes": len(data), "numeric_boolean_null_leaf_count": len(numeric_leaves(original))}
    for case in ("sub06", "sub07"):
        source = runs_root / "plots_v1" / (case + "_n4_wm_filled.png")
        target = output / "plots" / source.name
        shutil.copyfile(source, target)
        original, public = source.read_bytes(), target.read_bytes()
        if original != public:
            raise RuntimeError("PNG byte changed")
        report["images"]["plots/" + source.name] = {"raw_sha256": sha(original),
            "public_sha256": sha(public), "bytes": len(public), "kind": "public ds000114 derived diagnostic PNG"}
    (output / "PUBLIC_EXPORT_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("private-root", "private-host", "private-account"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(export(**vars(parser.parse_args()))))
