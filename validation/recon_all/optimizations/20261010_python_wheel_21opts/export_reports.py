"""公开本次wheel私有收据：只脱敏字符串，保留每个数值、bool和null及原SHA。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def numeric_leaves(value, path=()):
    if isinstance(value, dict):
        return [item for key in sorted(value) for item in numeric_leaves(value[key], path+(key,))]
    if isinstance(value, list):
        return [item for index, child in enumerate(value) for item in numeric_leaves(child, path+(index,))]
    if value is None or isinstance(value, (bool, int, float)):
        return [(path, type(value).__name__, value)]
    return []


def sanitize(value, replacements):
    if isinstance(value, dict):
        return {key: sanitize(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize(child, replacements) for child in value]
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-output", type=Path, required=True, help="已经完成的独立wheel运行目录")
    parser.add_argument("--public-output", type=Path, required=True, help="新的公开报告目录")
    args = parser.parse_args()
    root, output = args.private_output, args.public_output
    output.mkdir(parents=True, exist_ok=False)
    binding = json.loads((root/"SOURCE_BINDING.private.json").read_bytes())
    check = json.loads((root/"CHECK.private.json").read_bytes())
    assert check["status"] == "complete" and check["profile_option_count"] == 21
    assert check["cuda_initialized"] is False and check["no_CUDA_init_after_imports_profile_help"] is True
    assert hashlib.sha256((root/"SOURCE_BINDING.private.json").read_bytes()).hexdigest() == check["source_binding_sha256"]
    wheel = root/"wheel"/check["wheel"]["name"]
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == check["wheel"]["sha256"]
    assert wheel.stat().st_size == check["wheel"]["size_bytes"]
    build = json.loads((root/"BUILD.private.json").read_bytes())
    conda_prefix = str(Path(build["command"][0]).parent.parent)
    replacements = [(binding["repo"], "$REPO_ROOT"), (str(root), "$TASK_OUTPUT"),
                    (conda_prefix, "$EXISTING_CONDA_PREFIX")]
    manifest = {"scope": "actual21opts wheel/independent target only; noMRI or cleanConda",
                "redaction": "private absolute paths only; numeric/bool/null leaves unchanged",
                "files": []}
    for name in ("SOURCE_BINDING", "BUILD", "INSTALL", "CHECK", "CLI_AUDIT"):
        raw = (root/(name+".private.json")).read_bytes()
        original = json.loads(raw);public = sanitize(original,replacements)
        assert numeric_leaves(original) == numeric_leaves(public)
        encoded = (json.dumps(public, indent=2, ensure_ascii=False)+"\n").encode()
        (output/(name+".json")).write_bytes(encoded)
        manifest["files"].append({"path":name+".json", "original_sha256":hashlib.sha256(raw).hexdigest(),
            "published_sha256":hashlib.sha256(encoded).hexdigest(), "original_bytes":len(raw),
            "published_bytes":len(encoded), "numeric_leaves_preserved":True,
            "numeric_leaf_count":len(numeric_leaves(original))})
    (output/"PUBLIC_EXPORT_MANIFEST.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"public_reports":5,"wheel_sha256":check["wheel"]["sha256"],
                      "numeric_leaves_preserved":True,"git_only_snapshot":binding.get("git_only_snapshot",False)}))


if __name__ == "__main__":
    main()
