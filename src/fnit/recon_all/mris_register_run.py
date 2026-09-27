"""Register a conventional sphere through sulc and smoothwm without FreeSurfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from .mris_register_sulc_run import run_register_sulc
from .mris_register_smoothwm_run import run_register_smoothwm


def run_register_sphere(sphere: str | Path, smoothwm: str | Path,
                        sulc: str | Path, atlas: str | Path,
                        output: str | Path, *, overlap_device: str = "cpu") -> dict:
    """Read ordered sphere/smoothwm meshes, per-vertex sulc and atlas TIFF.

    Write the registered sphere to output. Return input/output paths and
    hashes, temporary sulc-seed hash, device, nested sulc/smoothwm pass
    trajectories and timings, and total seconds; remove the temporary seed.
    """
    started = time.perf_counter()
    output = Path(output)
    input_hashes = {name: hashlib.sha256(Path(value).read_bytes()).hexdigest()
                    for name, value in (("sphere", sphere), ("smoothwm", smoothwm),
                                        ("sulc", sulc), ("atlas", atlas))}
    with TemporaryDirectory(prefix="fnit-sphere-reg-", dir=output.parent) as temp:
        seed = Path(temp) / "sulc_seed"
        sulc_report = run_register_sulc(sphere, smoothwm, sulc, atlas, seed)
        seed_hash = hashlib.sha256(seed.read_bytes()).hexdigest()
        smoothwm_report = run_register_smoothwm(
            sphere, smoothwm, seed, atlas, output,
            seed_iteration=sulc_report["last_iteration"],
            overlap_device=overlap_device)
    sulc_report["output"] = "(temporary sulc seed)"
    smoothwm_report["sulc_seed"] = "(temporary sulc seed)"
    return {"sphere": str(sphere), "smoothwm": str(smoothwm),
            "sulc": str(sulc), "atlas": str(atlas), "output": str(output),
            "overlap_device": overlap_device, "input_sha256": input_hashes,
            "sulc_seed_sha256": seed_hash,
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "sulc_pass": sulc_report,
            "smoothwm_pass": smoothwm_report,
            "total_seconds_including_io": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sphere", "smoothwm", "sulc", "atlas", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--overlap-device", default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = run_register_sphere(
        args.sphere, args.smoothwm, args.sulc, args.atlas, args.output,
        overlap_device=args.overlap_device)
    content = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
