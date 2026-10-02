"""真实体积 Python API 计时、峰值显存及与 CLI 输出的一致性核查。"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.mshbm import parcellate_volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("volume", "left-surface", "right-surface", "cortical-mask",
                 "output-dir", "cli-output-dir", "source-root", "source-commit", "report"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    labels, history = parcellate_volume(
        volume=args.volume, left_surface=args.left_surface, right_surface=args.right_surface,
        cortical_mask=args.cortical_mask, output_dir=args.output_dir, device=args.device,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started
    cli = Path(args.cli_output_dir)
    np.testing.assert_array_equal(labels, np.load(cli / "labels_fslr32k_64984.npy"))
    first = nib.load(Path(args.output_dir) / "labels_mni.nii.gz")
    second = nib.load(cli / "labels_mni.nii.gz")
    np.testing.assert_array_equal(np.asarray(first.dataobj), np.asarray(second.dataobj))
    files = ["src/fnit/mshbm/" + name for name in
             ("__init__.py", "core.py", "cli.py", "output.py", "volume.py", "assets_setup.py",
              "assets/hcp40_fslr32k_17.npz")]
    files += ["src/fnit/connectome/atlas_surface.py"]
    hashes = {name: hashlib.sha256((Path(args.source_root) / name).read_bytes()).hexdigest()
              for name in files}
    report = {"source_commit": args.source_commit, "source_sha256": hashes,
              "python": platform.python_version(), "numpy": np.__version__,
              "pytorch": torch.__version__, "cuda": torch.version.cuda,
              "cpu": next(line.split(":", 1)[1].strip() for line in
                          Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")),
              "gpu": (torch.cuda.get_device_name(args.device)
                      if args.device.startswith("cuda") else None),
              "blas_threads": 8, "api_seconds": elapsed,
              "cuda_peak_allocated_gib": (torch.cuda.max_memory_allocated(args.device) / 2**30
                                          if args.device.startswith("cuda") else None),
              "cuda_peak_reserved_gib": (torch.cuda.max_memory_reserved(args.device) / 2**30
                                         if args.device.startswith("cuda") else None),
              "cli_vs_python_surface_label_differences": 0,
              "cli_vs_python_volume_label_differences": 0,
              "outer_iterations": len(history), "tf32": torch.backends.cuda.matmul.allow_tf32,
              "precision": "float32 signal, float64 geometric distance; no fp16"}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
