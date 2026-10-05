"""Check FNIT user-manual structure and local links without running MRI."""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
import unicodedata
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]
EXPECTED = ["1. 功能简介", "2. Python 调用", "3. 命令行调用", "4. 原软件调用", "5. 最新精度和运行时间", "6. 最近版本和 benchmark", "7. 参考文献、原软件和资源"]
PIPELINES = {"fmri", "connectome", "dmri_pipeline", "recon_all", "fast_vbm", "subregions", "ukb_vbm"}


def slug(title: str) -> str:
    title = re.sub(r"<[^>]+>", "", title).strip().lower().replace("`", "")
    title = "".join(char for char in title if not unicodedata.category(char).startswith(("P", "S")) or char in "-_")
    return re.sub(r"\s", "-", title)


@lru_cache(maxsize=None)
def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def anchors(path: Path) -> set[str]:
    text = read(path)
    result = set(re.findall(r'<a\s+(?:[^>]*?\s+)?id=["\']([^"\']+)', text))
    counts: dict[str, int] = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", text, re.M):
        key = slug(heading)
        index = counts.get(key, 0)
        counts[key] = index + 1
        result.add(key if index == 0 else f"{key}-{index}")
    return result


def check() -> dict:
    manuals = sorted(ROOT.glob("docs/*/README.md")) + [ROOT / "docs/fmri/surface.md"]
    auxiliary = [ROOT / "README.md", ROOT / "docs/WEIGHTS.md", ROOT / "docs/RESOURCE_MANIFEST.md", ROOT / "docs/README_TEMPLATE.md", Path(__file__).with_name("README.md")]
    archives = sorted(ROOT.glob("validation/*/*archive_20261005.md"))
    errors: list[dict] = []
    pages: list[dict] = []
    link_count = 0
    for path in manuals:
        text = read(path)
        lines = text.splitlines()
        headings = re.findall(r"^## (.*)$", text, re.M)
        fifth = next((index for index, line in enumerate(lines) if line == "## " + EXPECTED[4]), None)
        sixth = next((index for index, line in enumerate(lines) if line == "## " + EXPECTED[5]), None)
        ratio = (sixth - fifth) / len(lines) if fifth is not None and sixth is not None else None
        images = len(re.findall(r"!\[[^\]]*\]\(", text))
        gallery = path.parent.name == "figures"
        pipeline = path.parent.name in PIPELINES or path.name == "surface.md"
        lower, upper = (300, 700) if pipeline else (200, 500)
        if headings != EXPECTED:
            errors.append({"path": str(path.relative_to(ROOT)), "reason": "seven_section_structure"})
        if not gallery and not lower <= len(lines) <= upper:
            errors.append({"path": str(path.relative_to(ROOT)), "reason": "line_budget", "lines": len(lines)})
        if ratio is None or ratio > 0.25:
            errors.append({"path": str(path.relative_to(ROOT)), "reason": "benchmark_fraction", "fraction": ratio})
        if images > 3:
            errors.append({"path": str(path.relative_to(ROOT)), "reason": "too_many_images", "images": images})
        if pipeline and "```mermaid" not in text:
            errors.append({"path": str(path.relative_to(ROOT)), "reason": "pipeline_flowchart_missing"})
        pages.append({"path": str(path.relative_to(ROOT)), "lines": len(lines), "utf8_bytes": len(text.encode()), "benchmark_fraction": ratio, "images": images, "navigation_only": gallery})
    for path in manuals + auxiliary + archives:
        text = read(path)
        if len(re.findall(r"^```", text, re.M)) % 2:
            errors.append({"path": str(path.relative_to(ROOT)), "reason": "unclosed_fence"})
        for match in re.finditer(r"!?\[[^\]]*\]\(([^\s)]+)\)", text):
            link = match.group(1).strip("<>")
            if re.match(r"\w+://|mailto:|data:", link):
                continue
            target, _, fragment = link.partition("#")
            destination = (path.parent / unquote(target)).resolve() if target else path
            link_count += 1
            reason = None
            if not destination.exists():
                reason = "missing_path"
            elif fragment and destination.suffix == ".md" and unquote(fragment) not in anchors(destination):
                reason = "missing_anchor"
            if reason:
                errors.append({"path": str(path.relative_to(ROOT)), "line": text[:match.start()].count("\n") + 1, "target": link, "reason": reason})
    return {"schema": "fnit.readme_manual_audit.v1", "baseline_commit": "140c3739ac6c7a6826bf9421202ef59bec7ffe67", "readme_count": len(manuals) - 1, "additional_surface_manual": True, "homepage_lines": len(read(ROOT / "README.md").splitlines()), "local_links_checked": link_count, "manuals": pages, "errors": errors, "passed": not errors, "scope": "Static structure/link checks only; per-module audit records contain real public imports, signatures and actual CLI help. No new scientific benchmark."}


if __name__ == "__main__":
    result = check()
    Path(__file__).with_name("summary.public.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manuals": len(result["manuals"]), "local_links_checked": result["local_links_checked"], "errors": result["errors"], "passed": result["passed"]}, ensure_ascii=False, indent=2))
    sys.exit(0 if result["passed"] else 1)
