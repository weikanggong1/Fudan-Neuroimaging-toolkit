"""Lossless storage of GEMS validation JSON with shared local metadata.

Only validation artifacts use this format. Runtime code and scoring are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def compact(value, depth=0):
    """One scalar metric row per line; retain file-hash maps one key per line."""
    padding = "  " * depth
    if isinstance(value, dict):
        if len(value) <= 30 and all(not isinstance(v, (dict, list)) for v in value.values()):
            return json.dumps(value, allow_nan=False)
        rows = ["  " * (depth + 1) + json.dumps(k) + ": " + compact(v, depth + 1)
                for k, v in value.items()]
        return "{\n" + ",\n".join(rows) + "\n" + padding + "}"
    if isinstance(value, list):
        if all(not isinstance(v, (dict, list)) for v in value):
            return json.dumps(value, allow_nan=False)
        rows = ["  " * (depth + 1) + compact(v, depth + 1) for v in value]
        return "[\n" + ",\n".join(rows) + "\n" + padding + "]"
    return json.dumps(value, allow_nan=False)


def load_report(path):
    """Expand local JSON pointers and source-map base/override records."""
    path = Path(path).resolve()
    cache = {}

    def target(reference, owner, active):
        filename, pointer = reference.split("#", 1)
        if Path(filename).name != filename or not pointer.startswith("/"):
            raise ValueError("References must be local filenames and JSON pointers")
        referred = owner.parent / filename
        token = (str(referred), pointer)
        if token in active:
            raise ValueError("Cyclic JSON reference")
        if referred not in cache:
            cache[referred] = json.loads(referred.read_text())
        node = cache[referred]
        for part in pointer[1:].split("/"):
            node = node[part.replace("~1", "/").replace("~0", "~")]
        return expand(node, referred, active | {token})

    def expand(node, owner, active):
        if isinstance(node, dict):
            if set(node) == {"$ref"}:
                return target(node["$ref"], owner, active)
            if set(node) == {"$base", "$overrides", "$remove"}:
                base = target(node["$base"], owner, active)
                for key in node["$remove"]:
                    del base[key]
                base.update(expand(node["$overrides"], owner, active))
                return base
            return {k: expand(v, owner, active) for k, v in node.items()}
        if isinstance(node, list):
            return [expand(v, owner, active) for v in node]
        return node

    return expand(json.loads(path.read_text()), path, set())


def pack_directory(root):
    root = Path(root)
    score_names = ["gpu_crop_v2_official_score.public.json",
                   "gpu_crop_mask_v3_official_score.public.json",
                   "gpu_ha_v4_official_score.public.json"]
    source_names = ["cache_v1_source.public.json", "crop_mask_v3_source.public.json",
                    "ha_v4_source.public.json"]
    names = score_names + source_names + ["regional_delta_summary.public.json"]
    originals = {name: json.loads((root / name).read_text()) for name in names}
    baseline = originals[score_names[0]]["baseline"]
    if any(originals[name]["baseline"] != baseline for name in score_names):
        raise ValueError("Baseline records differ; cannot consolidate")
    base_files = originals[source_names[0]]["files"]
    empty = {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    shared = {"encoding": "fnit-validation-local-json-refs-v1", "empty_stats": empty,
              "source_files": base_files, "grids": {}}
    grid_keys = {}

    def pack(node, key=None):
        if isinstance(node, dict):
            if node == empty:
                return {"$ref": "gems_shared.public.json#/empty_stats"}
            if key == "grid":
                fingerprint = canonical(node)
                if fingerprint not in grid_keys:
                    name = "g" + str(len(grid_keys))
                    grid_keys[fingerprint] = name
                    shared["grids"][name] = node
                return {"$ref": "gems_shared.public.json#/grids/" + grid_keys[fingerprint]}
            return {k: pack(v, k) for k, v in node.items()}
        if isinstance(node, list):
            return [pack(v, key) for v in node]
        return node

    encoded = {name: pack(value) for name, value in originals.items()}
    shared["baseline"] = pack(baseline)
    for name in score_names:
        encoded[name]["baseline"] = {"$ref": "gems_shared.public.json#/baseline"}
    for name in source_names:
        key = "files" if name == source_names[0] else "source_sha256"
        files = originals[name][key]
        encoded[name][key] = {
            "$base": "gems_shared.public.json#/source_files",
            "$overrides": {k: v for k, v in files.items() if base_files.get(k) != v},
            "$remove": [k for k in base_files if k not in files]}
    original_stats = {name: {"bytes": (root / name).stat().st_size,
                             "lines": len((root / name).read_text().splitlines()),
                             "decoded_canonical_sha256": digest(originals[name])} for name in names}
    duplicate = root / "gpu_crop_v2_official_score.compact.public.json"

    def subset(a, b):
        if isinstance(a, dict):
            return isinstance(b, dict) and all(k in b and subset(v, b[k]) for k, v in a.items())
        if isinstance(a, list):
            return isinstance(b, list) and len(a) == len(b) and all(subset(x, y) for x, y in zip(a, b))
        return a == b

    old_compact = json.loads(duplicate.read_text())
    if not subset(old_compact, originals[score_names[0]]):
        raise ValueError("The duplicate compact report contains unique values")
    removed = {"name": duplicate.name, "bytes": duplicate.stat().st_size,
               "lines": len(duplicate.read_text().splitlines()),
               "decoded_canonical_sha256": digest(old_compact),
               "reason": "Verified exact projection of the retained complete crop-v2 report"}
    (root / "gems_shared.public.json").write_text(compact(shared) + "\n")
    for name, value in encoded.items():
        (root / name).write_text(compact(value) + "\n")
        if load_report(root / name) != originals[name]:
            raise AssertionError("Lossless decode failed: " + name)
    duplicate.unlink()
    rows = [{"name": name, **original_stats[name],
             "encoded_bytes": (root / name).stat().st_size,
             "encoded_lines": len((root / name).read_text().splitlines()),
             "encoded_sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
             "lossless_decode_equal": True} for name in names]
    manifest = {"encoding": "fnit-validation-local-json-refs-v1", "reports": rows,
                "shared": {"name": "gems_shared.public.json",
                           "bytes": (root / "gems_shared.public.json").stat().st_size,
                           "lines": len((root / "gems_shared.public.json").read_text().splitlines()),
                           "sha256": hashlib.sha256((root / "gems_shared.public.json").read_bytes()).hexdigest()},
                "removed_duplicate_projection": removed,
                "all_scientific_values_and_source_hashes_preserved": True}
    (root / "report_encoding_manifest.public.json").write_text(compact(manifest) + "\n")
    print(json.dumps({"old_bytes": sum(r["bytes"] for r in rows) + removed["bytes"],
                      "new_bytes": sum(r["encoded_bytes"] for r in rows) + manifest["shared"]["bytes"],
                      "old_lines": sum(r["lines"] for r in rows) + removed["lines"],
                      "new_lines": sum(r["encoded_lines"] for r in rows) + manifest["shared"]["lines"],
                      "all_lossless_equal": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack-directory")
    pack.add_argument("directory", type=Path)
    decode = commands.add_parser("decode")
    decode.add_argument("input", type=Path)
    decode.add_argument("output", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "pack-directory":
        pack_directory(args.directory)
    elif args.command == "decode":
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(load_report(args.input), indent=2, allow_nan=False) + "\n")
    else:
        manifest = json.loads((args.directory / "report_encoding_manifest.public.json").read_text())
        for row in manifest["reports"]:
            assert digest(load_report(args.directory / row["name"])) == row["decoded_canonical_sha256"], row["name"]
        shared = args.directory / manifest["shared"]["name"]
        assert hashlib.sha256(shared.read_bytes()).hexdigest() == manifest["shared"]["sha256"]
        print("All decoded scientific records and source hashes match their original canonical SHA-256.")


if __name__ == "__main__":
    main()
