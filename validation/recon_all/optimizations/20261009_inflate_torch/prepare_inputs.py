"""复制明确声明的公开T1自产smoothwm到新私有包；不修改源表面。

--config 为 JSON 数组：case、hemisphere、surface、public_source_url、
source_recipe，各项明确指定。--output/--archive不得存在，包只含四个
声明表面和SHA清单，不递归复制被试、权重或许可证。
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.archive.exists():
        raise FileExistsError("input bundle refuses overwrite")
    entries = json.loads(args.config.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("config must be a nonempty list")
    if args.archive.resolve().is_relative_to(args.output.resolve()):
        raise ValueError("archive must be outside bundle")
    keys = [(entry["case"], entry["hemisphere"]) for entry in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("case/hemisphere must be unique")
    for case, hemi in keys:
        if Path(case).name != case or case in ("", ".", "..") or hemi not in ("lh", "rh"):
            raise ValueError("invalid case/hemisphere")
    args.output.mkdir(parents=True)
    manifest = {"scope": "public_T1_FNIT_self_produced_frozen_smoothwm_not_raw_T1_pipeline",
                "config_sha256": sha(args.config), "cases": []}
    for entry in entries:
        source = Path(entry["surface"])
        destination = args.output / entry["case"] / (entry["hemisphere"] + ".smoothwm")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest["cases"].append({"case": entry["case"], "hemisphere": entry["hemisphere"],
            "surface": str(destination.relative_to(args.output)), "sha256": sha(destination),
            "bytes": destination.stat().st_size, "public_source_url": entry["public_source_url"],
            "source_recipe": entry["source_recipe"]})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as output:
        output.add(args.output, arcname=args.output.name)
    print(json.dumps({"archive_sha256": sha(args.archive), "archive_bytes": args.archive.stat().st_size,
                      "manifest_sha256": sha(args.output / "manifest.json")}))


if __name__ == "__main__":
    main()
