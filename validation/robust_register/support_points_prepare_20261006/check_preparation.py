"""Standard-library preparation checks; no MRI or numerical code executes."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

from common import digest, validate_plan


def check(private_plan, previous_manifest):
    source = Path(__file__).parent
    plan = json.loads(Path(private_plan).read_text())
    validate_plan(plan)
    compiled = []
    for path in sorted(source.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=path.name)
        compile(tree, path.name, "exec")
        compiled.append(path.name)
    for name, expected in plan["harness_bindings"].items():
        path = source / name
        if {"bytes": int(path.stat().st_size), "sha256": digest(path)} != {
                key: expected[key] for key in ("bytes", "sha256")}:
            raise ValueError("prepared private PLAN harness differs: " + name)
    old = Path(previous_manifest)
    old_data = json.loads(old.read_text())
    if digest(old) != "70029f922222ca77cafe2f3e5f3ec78d2a6bd7c244afb930237b33dc20befb00":
        raise ValueError("the original report manifest changed")
    for name, expected in old_data["files"].items():
        path = old.parent / name
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("an original report file changed: " + name)
    missing = []
    links = 0
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", (source / "README.md").read_text()):
        if target.startswith(("https://", "http://", "#")):
            continue
        links += 1
        if not (source / target.split("#")[0]).exists():
            missing.append(target)
    if missing:
        raise ValueError("new README links missing: " + repr(missing))
    public_summary = json.loads((source / "PLAN.public.json").read_text())
    if public_summary["private_PLAN"] != {
            "bytes": int(Path(private_plan).stat().st_size), "sha256": digest(private_plan)}:
        raise ValueError("public PLAN does not bind the actual private PLAN")
    texts = [(path.name, path.read_text()) for path in source.iterdir()
             if path.is_file() and path.suffix in (".py", ".md", ".json")]
    for name, text in texts:
        for expression in (
            r"(?<![\d.])(?:10\.190\.248\.(?:228|214)|10\.193\.2\.99)(?![\d.])",
            r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
            r"(?:/home1/gongwk|/cwStorage/home/gongwk|/mnt/c/Users/admin|/home/wgong)/",
        ):
            if re.search(expression, text):
                raise ValueError("sensitive text in prepared public file: " + name)
    worker = ast.parse((source / "inspect_saved_points.py").read_text())
    scientific_calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
    prohibited = []
    for node in scientific_calls:
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else "")
        if name in ("robust_register", "fit", "optimize", "Popen", "subprocess",
                    "system", "exec", "eval"):
            prohibited.append(name)
    if prohibited:
        raise ValueError("registration/optimizer/native-process calls found in worker")
    shape_lists = [node for node in ast.walk(worker)
                   if isinstance(node, ast.ListComp)
                   and isinstance(node.elt, ast.Call)
                   and isinstance(node.elt.func, ast.Name)
                   and node.elt.func.id == "int"]
    return {
        "schema": 1, "status": "metadata_preparation_checks_passed",
        "Python_AST_compiled_files": compiled, "Python_AST_compile_count": len(compiled),
        "runtime_harness_SHA_exact": len(plan["harness_bindings"]),
        "private_PLAN": {"bytes": int(Path(private_plan).stat().st_size),
                         "sha256": digest(private_plan)},
        "original_final_report_files_SHA_unchanged": len(old_data["files"]) + 1,
        "local_README_links_checked": links,
        "local_README_broken_links": [], "public_sensitive_text_hits": 0,
        "worker_registration_optimizer_native_process_calls": 0,
        "explicit_builtin_int_metadata_comprehensions": len(shape_lists),
        "bindings_declared": len(plan["bindings"]), "harness_declared": len(plan["harness_bindings"]),
        "actual_sampling_or_registration_jobs": 0, "actual_uploads": 0,
        "scope": "source AST, metadata/hash/privacy/link checks only; not numerical or MRI validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-plan", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.private_plan, args.previous_manifest), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
