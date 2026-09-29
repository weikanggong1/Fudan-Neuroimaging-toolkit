"""在真实 MNI152 模板上比较 FNIT 体积/表面映射的 CPU 与 CUDA 路径。"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit import convert_space


def _values(paths):
    return [np.stack([np.asarray(frame.data) for frame in nib.load(str(path)).darrays], axis=1)
            for path in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mni", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_per_process_memory_fraction(0.15, device=0)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    results = {"gpu": torch.cuda.get_device_name(0),
               "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
               "gpu_free_before_bytes": torch.cuda.mem_get_info(0)[0]}
    surfaces = {}
    for device in ("cpu", "cuda:0"):
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        surfaces[device] = convert_space(
            source=args.mni, source_space="MNI152", target_space="fsaverage",
            target_density="164k", output_dir=args.work_dir / f"forward_{device.split(':')[0]}",
            assets_dir=args.assets_dir, device=device,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(0)
        results[f"forward_{device.split(':')[0]}_seconds"] = time.perf_counter() - start
    cpu, gpu = _values(surfaces["cpu"]), _values(surfaces["cuda:0"])
    results["forward_mae"] = [float(np.mean(np.abs(a - b))) for a, b in zip(cpu, gpu)]
    results["forward_max_abs"] = [float(np.max(np.abs(a - b))) for a, b in zip(cpu, gpu)]
    volumes = {}
    for device in ("cpu", "cuda:0"):
        start = time.perf_counter()
        volumes[device] = convert_space(
            source=surfaces["cpu"], source_space="fsaverage", source_density="164k",
            target_space="MNI152", output_dir=args.work_dir / f"reverse_{device.split(':')[0]}",
            assets_dir=args.assets_dir, device=device,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(0)
        results[f"reverse_{device.split(':')[0]}_seconds"] = time.perf_counter() - start
    a = np.asarray(nib.load(str(volumes["cpu"])).dataobj, dtype=np.float32)
    b = np.asarray(nib.load(str(volumes["cuda:0"])).dataobj, dtype=np.float32)
    results["reverse_mae"] = float(np.mean(np.abs(a - b)))
    results["reverse_max_abs"] = float(np.max(np.abs(a - b)))
    results["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
    results["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
    results["gpu_free_after_bytes"] = torch.cuda.mem_get_info(0)[0]
    (args.work_dir / "gpu_benchmark.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
