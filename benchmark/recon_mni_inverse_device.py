"""真实同前向场：旧逆场current1与修复逆场current0，均在目标GPU1计算。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "overlay", "forward", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_dir))
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.overlay))
    import nibabel as nib
    import numpy as np
    import torch
    from fnit.recon_all.mni_warp_inverse import invert_mni_warp
    if torch.cuda.device_count() < 2: raise ValueError("requires two visible CUDA devices")
    old_path = args.source_dir / "fnit/recon_all/mni_warp_inverse.py"
    spec = importlib.util.spec_from_file_location("fnit.recon_all._old_inverse_device_probe", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"input_sha256": sha(args.forward), "old_source_sha256": sha(old_path),
              "new_source_sha256": sha(Path(sys.modules[invert_mni_warp.__module__].__file__)),
              "benchmark_sha256": sha(__file__), "target_device": "cuda:1",
              "scope": "same-input complete inverse API, old current1/new current0, same physical GPU1; no model rerun"}
    for name, function, current in (("old", old.invert_mni_warp, 1), ("new", invert_mni_warp, 0)):
        with torch.cuda.device(current):
            before = torch.cuda.current_device()
            torch.cuda.synchronize("cuda:1")
            tick = time.monotonic()
            report = function(args.forward, args.output / f"{name}.nii.gz", device="cuda:1")
            torch.cuda.synchronize("cuda:1")
            result[name] = {"complete_api_wall_seconds": time.monotonic()-tick, "api": report,
                           "current_device_before": before, "current_device_after": torch.cuda.current_device()}
            assert before == torch.cuda.current_device()
    first, second = (nib.load(str(args.output / f"{name}.nii.gz")) for name in ("old", "new"))
    a, b = np.asarray(first.dataobj), np.asarray(second.dataobj)
    error = np.abs(a.astype(np.float64)-b.astype(np.float64))
    result["comparison"] = {"different_voxels": int(np.count_nonzero(a != b)),
        "maximum": float(error.max()), "p99": float(np.quantile(error,.99)),
        "affine_equal": bool(np.array_equal(first.affine,second.affine)),
        "dtype_equal": first.get_data_dtype()==second.get_data_dtype(),
        "header_equal": first.header.binaryblock==second.header.binaryblock,
        "file_sha_equal": sha(args.output/"old.nii.gz")==sha(args.output/"new.nii.gz")}
    result["iteration_stats_equal"] = all(result["old"]["api"][k]==result["new"]["api"][k]
                                          for k in ("voronoi_iterations","soap_iterations_xyz"))
    (args.output/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    assert result["comparison"]["different_voxels"]==0 and result["iteration_stats_equal"]
    print(json.dumps(result["comparison"]))


if __name__ == "__main__": main()
