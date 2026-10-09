"""两例公开冻结WM/fill最小输入包；只复制明确列出的文件，不含许可证。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--frozen-run-root", type=Path, required=True)
    parser.add_argument("--histogram-diagnostic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cohort = json.loads(args.cohort_manifest.read_text())
    manifest = {"scope": "public frozen same-input WM/fill only; raw T1 whole reconstruction not included",
        "cohort_manifest_sha256": sha(args.cohort_manifest), "script_sha256": sha(__file__),
        "reference_policy": "references are benchmark-only; production candidate consumes inputs only",
        "cases": {}, "files": {}}
    for case in ("sub-07", "sub-06"):
        case_id = f"ds000114_{case}"
        entry = next(row for row in cohort["cases"] if row["id"] == case_id)
        if entry["license"] != "CC0-1.0" or sha(entry["server_input"]) != entry["sha256"]:
            raise ValueError("public source license/hash is not the declared frozen cohort")
        mri = args.frozen_run_root / case_id / "attempt_01/subject/mri"
        provenance = json.loads((mri.parent / "fnit-native-free-run.json").read_text())
        if Path(provenance["input"]).resolve() != Path(entry["server_input"]).resolve():
            raise ValueError("frozen source does not trace to declared public raw input")
        manifest["cases"][case_id] = {key: entry[key] for key in
            ("dataset", "subject", "session", "snapshot", "license", "source_git_commit", "source_path", "sha256")}
        manifest["cases"][case_id]["source_URL"] = entry["source_url"]
        sources = {f"inputs/{case}/antsdn.brain.mgz": mri / "antsdn.brain.mgz",
                   f"inputs/{case}/wm.mgz": mri / "wm.mgz",
                   f"inputs/{case}/aseg.presurf.mgz": mri / "aseg.presurf.mgz",
                   f"inputs/{case}/transforms/talairach.lta": mri / "transforms/talairach.lta",
                   f"references/{case}/filled_fnit_frozen.mgz": mri / "filled.mgz"}
        for index in (1, 2):
            for stem in ("int", "histo"):
                name = f"wmseg.{stem}.{index}.mgz"
                sources[f"references/{case}/{name}"] = args.histogram_diagnostic_root / case / "native_diag_v1" / name
        for relative, source in sources.items():
            destination = args.output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha(source) != sha(destination):
                raise ValueError("copied file hash differs")
            row = {"sha256": sha(destination), "bytes": destination.stat().st_size,
                   "role": "candidate_input" if relative.startswith("inputs/") else "diagnostic_reference_only"}
            if destination.suffix == ".mgz":
                image = nib.load(destination)
                row.update({"shape": [int(axis) for axis in image.shape],
                            "dtype": str(image.get_data_dtype()), "affine_mm": image.affine.tolist()})
            manifest["files"][relative] = row
    manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"].values())
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
