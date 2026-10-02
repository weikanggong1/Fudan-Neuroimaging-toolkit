"""记录本轮真实 T1、权重、资产及候选原生程序的大小和 SHA-256。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


BINARIES = ("fnit_n4_itk", "mri_em_register", "mri_segment",
            "mri_edit_wm_with_aseg", "mris_fix_topology_fnit",
            "mris_remove_intersection", "mris_inflate", "mris_place_surface")
REFERENCE_BINARIES = ("mri_convert", "AntsN4BiasFieldCorrectionFs",
                      "mri_em_register", "mri_ca_normalize", "mri_segment",
                      "mri_edit_wm_with_aseg")


def _record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub01-t1", type=Path, required=True)
    parser.add_argument("--sub02-t1", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--native-bin", type=Path, required=True)
    parser.add_argument("--reference-bin", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu = next((line.split(":", 1)[1].strip() for line in
                Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "unknown")
    report = {"code_commit": args.code_commit, "host": platform.node(), "cpu": cpu,
              "inputs": {name: _record(getattr(args, name)) for name in
                         ("sub01_t1", "sub02_t1")},
              "weights": {str(path.relative_to(args.weights)): _record(path)
                          for path in sorted(args.weights.rglob("*")) if path.is_file()
                          and path.name.lower() not in ("license.txt", ".license")},
              "assets": {str(path.relative_to(args.assets)): _record(path)
                         for path in sorted(args.assets.rglob("*")) if path.is_file()
                         and path.name.lower() not in ("license.txt", ".license")},
              "binaries": {name: _record(args.native_bin / name) for name in BINARIES},
              "reference_binaries": {name: _record(args.reference_bin / name)
                                     for name in REFERENCE_BINARIES}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({key: len(report[key]) for key in
           ("weights", "assets", "binaries", "reference_binaries")})


if __name__ == "__main__":
    main()
