"""Prepare full real-data input representations without shortening the main run."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(cfg["candidate_source"]) / "src"))
    import numpy as np
    from fnit.mshbm.core import load_assets
    from fnit.mshbm.cli import read_cortex
    assets = load_assets(cfg["binding"]["assets"])
    series = read_cortex(cfg["binding"]["timeseries"][0], assets["cortex_mask"])
    if series.shape != (490, 59412):
        raise ValueError("Only the complete real 490-frame run is admitted")
    full = np.zeros((490, 64984), dtype=np.float32)
    full[:, assets["cortex_mask"]] = series
    arrays = {"cortex": series, "cortex_transposed": series.T,
              "full": full, "full_transposed": full.T,
              "same_run_half1": series[:245], "same_run_half2": series[245:]}
    checks, private = [], {}
    for name, array in arrays.items():
        path = args.output_dir / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        private[name] = {"path": str(path), "shape": list(array.shape),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if not name.startswith("same_run"):
            read = read_cortex(path, assets["cortex_mask"])
            checks.append({"format": name, "shape": list(read.shape),
                           "different_values": int(np.count_nonzero(read != series)),
                           "byte_values_exact": bool(np.array_equal(read, series))})
    (args.output_dir / "inputs.private.json").write_text(json.dumps(private, indent=2) + "\n")
    report = {"complete_frames": 490, "cortex_vertices": 59412,
              "representation_controls": checks,
              "same_run_split": {"frames": [245, 245], "all_490_frames_preserved": True,
                                 "independent_acquisitions": False}}
    (args.output_dir / "representation_controls.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
