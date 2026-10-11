"""服务器侧导出拓扑数值收据；复用现有发布器，不包含MRI/程序/NPZ。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--exporter", required=True, type=Path)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--private-host", required=True)
    parser.add_argument("--private-value", action="append", default=[])
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("existing_fnit_receipt_exporter", args.exporter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def clean(value):
        if isinstance(value, str):
            # 根目录可能包含用户名；先替换完整根，避免短用户名破坏后续匹配。
            value = value.replace(args.private_root, "[PRIVATE_FNIT]").replace(args.private_host, "A100-8")
            for private in sorted(args.private_value, key=len, reverse=True):
                if not private:
                    raise ValueError("empty private-value is not permitted")
                value = value.replace(private, "[PRIVATE_BUILD]")
            return value
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {clean(key): clean(item) for key, item in value.items()}
        return value

    with TemporaryDirectory(prefix="fnit-topology-public-json-") as directory:
        temporary = Path(directory)
        originals = {}
        for path in sorted(args.input_dir.rglob("*.json")):
            raw = path.read_bytes()
            original = json.loads(raw)
            transformed = clean(original)
            if module.numeric_leaves(original) != module.numeric_leaves(transformed):
                raise RuntimeError("numeric/boolean/null report changed")
            relative = path.relative_to(args.input_dir)
            output = temporary / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(transformed, indent=2) + "\n")
            originals[str(relative)] = hashlib.sha256(raw).hexdigest()
        subprocess.run([sys.executable, str(args.exporter), "--input-dir", str(temporary),
            "--output-dir", str(args.output_dir), "--private-root", args.private_root,
            "--private-host", args.private_host], check=True)
        manifest_path = args.output_dir / "PUBLIC_EXPORT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        for relative, entry in manifest["reports"].items():
            entry["intermediate_sanitized_sha256"] = entry["raw_sha256"]
            entry["raw_sha256"] = originals[relative]
        manifest["multi_prefix_preclean"] = True
        manifest["scope"] = "derived JSON only; original receipt hashes retained; all numeric/boolean/null leaves preserved"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
