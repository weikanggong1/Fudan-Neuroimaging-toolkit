"""Check raw FP32 bit patterns using the saved same-input CPU/CUDA outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import nibabel.freesurfer.io as fsio
import numpy as np
from benchmark import ORDER, metrics, source_formula


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--subjects-root", type=Path, required=True)
    args = parser.parse_args()
    path = args.run / "report.json"
    report = json.loads(path.read_text())
    for row in report["hemispheres"]:
        prefix = args.subjects_root / row["subject"] / "attempt_01" / "subject" / "surf" / (row["hemisphere"] + ".smoothwm")
        k1 = np.ascontiguousarray(fsio.read_morph_data(str(prefix) + ".K1.crv"), dtype=np.float32)
        k2 = np.ascontiguousarray(fsio.read_morph_data(str(prefix) + ".K2.crv"), dtype=np.float32)
        reference_formula = source_formula(k1, k2)
        for index, name in enumerate(ORDER):
            candidate = fsio.read_morph_data(row["io_api"]["outputs"][name])
            native = fsio.read_morph_data(str(prefix) + "." + name + ".crv")
            row["vs_frozen_native"][name] = metrics(candidate, native)
            row["vs_source_formula"][name] = metrics(candidate, reference_formula[index])
    report["all_source_formula_maps_equal"] = all(m["strict_float32_equal"] for h in report["hemispheres"] for m in h["vs_source_formula"].values())
    report["precision_refresh"] = {"scope": "saved output raw FP32 bit patterns; no timing rerun",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics_script_sha256": hashlib.sha256(Path(__file__).with_name("benchmark.py").read_bytes()).hexdigest()}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"run": str(args.run), "all_source_formula_bitwise_equal": report["all_source_formula_maps_equal"]}))


if __name__ == "__main__":
    main()
