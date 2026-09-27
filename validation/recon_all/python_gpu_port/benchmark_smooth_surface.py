"""Paired mris_smooth -nw -seed 1234 replay on identical triangle inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

from fnit.recon_all.smooth_surface_python import smooth_surface


def compare(native: Path, candidate: Path) -> dict:
    original = native.read_bytes()
    actual = candidate.read_bytes()
    if original[:3] != actual[:3] or original[:3] != b"\xff\xff\xfe":
        raise ValueError("unexpected triangle surface magic")
    native_body = original[original.index(b"\n\n", 3) + 2:]
    python_body = actual[actual.index(b"\n\n", 3) + 2:]
    nvertices = int.from_bytes(native_body[:4], "big")
    nfaces = int.from_bytes(native_body[4:8], "big")
    core_end = 8 + 12 * (nvertices + nfaces)
    return {"vertices": nvertices, "triangles": nfaces,
            "geometry_bytes_identical": native_body[:core_end] == python_body[:core_end]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-surface", type=Path, required=True)
    parser.add_argument("--right-surface", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {"lh": args.left_surface.resolve(), "rh": args.right_surface.resolve()}
    rows = []
    for hemi, surface in inputs.items():
        timings = {"native": [], "python": []}
        comparison = None
        for repeat in range(args.repeats):
            outputs = {side: args.output_dir / f"{hemi}.{side}.{repeat}.surf"
                       for side in timings}
            for side in (("native", "python") if repeat % 2 == 0
                         else ("python", "native")):
                start = time.perf_counter()
                if side == "native":
                    subprocess.run([str(args.native_binary), "-n", str(args.iterations),
                                    "-nw", "-seed", "1234",
                                    str(surface), str(outputs[side])],
                                   capture_output=True, check=True)
                else:
                    smooth_surface(surface, outputs[side], iterations=args.iterations,
                                   device=args.device)
                timings[side].append(time.perf_counter() - start)
            comparison = compare(outputs["native"], outputs["python"])
            if not comparison["geometry_bytes_identical"]:
                raise ValueError(f"{hemi} repeat {repeat} differs: {comparison}")
        rows.append({"hemi": hemi, "comparison": comparison,
                     "wall_seconds": timings,
                     "median_seconds": {side: statistics.median(values)
                                        for side, values in timings.items()}})
    report = {"host": platform.node(), "device": args.device,
              "iterations": args.iterations,
              "native_binary": str(args.native_binary),
              "input_sha256": {hemi: hashlib.sha256(path.read_bytes()).hexdigest()
                               for hemi, path in inputs.items()},
              "repeats": args.repeats,
              "timing_scope": "same-host file I/O; native process startup included; Python imports excluded",
              "rows": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
