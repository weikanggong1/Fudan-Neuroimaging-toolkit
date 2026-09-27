"""Measure one exact-input FP32 SynthSeg stage without modifying production code."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

import fnit.synthseg_parc.segment as segment_module
import fnit.synthseg_parc.synthseg as synthseg_module
from fnit.synthseg_parc import SynthSeg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--benchmark", choices=("true", "false"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--empty-after-first-blur", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = args.benchmark == "true"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    marks = []
    started = time.perf_counter()

    def mark(name: str) -> None:
        torch.cuda.synchronize(device)
        own = None
        try:
            lines = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"], text=True).splitlines()
            own = next((int(line.split(",")[1].strip()) for line in lines
                        if int(line.split(",")[0].strip()) == os.getpid()), None)
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
            pass
        marks.append({"point": name, "seconds": round(time.perf_counter() - started, 3),
                      "allocated_mib": round(torch.cuda.memory_allocated(device) / 2**20, 1),
                      "reserved_mib": round(torch.cuda.memory_reserved(device) / 2**20, 1),
                      "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
                      "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
                      "nvidia_smi_mib": own})

    def instrument(owner, name: str) -> None:
        original = getattr(owner, name)
        def measured(*pos, **kw):
            mark(f"{name}_begin")
            torch.cuda.reset_peak_memory_stats(device)
            result = original(*pos, **kw)
            mark(f"{name}_end")
            return result
        setattr(owner, name, measured)

    instrument(synthseg_module, "preprocess_t1")
    instrument(segment_module.SegmentUNet, "forward")
    instrument(segment_module.SynthSegSegmenter, "posterior")
    instrument(synthseg_module, "postprocess_segmentation")
    instrument(synthseg_module, "_official_soft_volumes")
    if args.empty_after_first_blur:
        original_blur = segment_module._blur
        blur_count = [0]
        def measured_blur(posterior):
            result = original_blur(posterior)
            blur_count[0] += 1
            if blur_count[0] == 1:
                mark("first_blur_before_empty_cache")
                torch.cuda.empty_cache()
                mark("first_blur_after_empty_cache")
            return result
        segment_module._blur = measured_blur

    mark("start")
    synthseg = SynthSeg(weights=args.weights_dir, device=args.device, threads=4)
    mark("model_loaded")
    torch.cuda.reset_peak_memory_stats(device)
    result = synthseg(args.image, keep_geometry=True, color_lut=args.lut)
    mark("result_returned")
    hard = np.ascontiguousarray(np.asarray(result.segmentation.data).astype(np.int16))
    metrics = {"segmentation_sha256": hashlib.sha256(hard.tobytes()).hexdigest(),
               "segmentation_shape": list(hard.shape),
               "volumes_mm3": result.volumes_mm3,
               "total_intracranial_mm3": result.total_intracranial_mm3,
               "near_tie_voxels": result.near_tie_voxels}
    del result, hard, synthseg
    gc.collect()
    mark("after_gc")
    torch.cuda.empty_cache()
    mark("after_empty_cache")
    output = {"image": str(args.image), "device": args.device,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "empty_after_first_blur": args.empty_after_first_blur,
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "marks": marks, "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({"output": str(args.output), "max_allocated_mib": max(
        m["peak_allocated_mib"] for m in marks), "max_reserved_mib": max(
        m["peak_reserved_mib"] for m in marks)}))


if __name__ == "__main__":
    main()
