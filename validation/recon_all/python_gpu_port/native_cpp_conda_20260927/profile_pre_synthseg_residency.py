"""Diagnose CUDA memory retained by recon-all stages before SynthSeg.

Run only after the active end-to-end benchmark finishes. This reproduces its
GPU prefix using an existing ``nu.mgz``; N4 itself runs in a CPU subprocess.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import nibabel as nib
import numpy as np
import torch

import fnit.recon_all.input_talairach_chain as input_module
from fnit.recon_all.input_talairach_chain import run_input_talairach_chain
from fnit.recon_all.mri_mask_gpu import mask_volume
from fnit.recon_all.normalization import normalize_t1
from fnit.synthseg_parc import SynthSeg
from fnit.synthseg_parc.segment import SegmentUNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--nu", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-only", action="store_true")
    parser.add_argument("--isolate-synthmorph", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    marks = []
    started = time.perf_counter()

    def mark(stage: str) -> None:
        torch.cuda.synchronize(device)
        try:
            lines = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"], text=True).splitlines()
            resident = next(int(line.split(",")[1].strip()) for line in lines
                            if int(line.split(",")[0].strip()) == os.getpid())
        except (OSError, subprocess.CalledProcessError, StopIteration):
            resident = None
        marks.append({
            "stage": stage,
            "seconds": round(time.perf_counter() - started, 3),
            "allocated_mib": round(torch.cuda.memory_allocated(device) / 2**20, 1),
            "reserved_mib": round(torch.cuda.memory_reserved(device) / 2**20, 1),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
            "nvidia_smi_mib": resident,
            "outside_torch_allocator_mib": (
                round(resident - torch.cuda.memory_reserved(device) / 2**20, 1)
                if resident is not None else None),
        })

    def stage(name, function, *pos, keep_result=False, **kw):
        torch.cuda.reset_peak_memory_stats(device)
        result = function(*pos, **kw)
        mark(name)
        if not keep_result:
            del result
        gc.collect()
        torch.cuda.empty_cache()
        mark(name + "_after_empty_cache")
        return result if keep_result else None

    if args.isolate_synthmorph:
        child_code = (
            "import sys, torch; "
            "torch.backends.cudnn.benchmark=True; "
            "torch.backends.cudnn.deterministic=True; "
            "torch.backends.cudnn.allow_tf32=True; "
            "torch.backends.cuda.matmul.allow_tf32=True; "
            "from fnit.recon_all.talairach_synthmorph import register_talairach; "
            "register_talairach(*sys.argv[1:6], device=sys.argv[6], threads=int(sys.argv[7]))"
        )

        def isolated_register(moving, template, weights, xfm, lta, *, device, threads):
            subprocess.run([sys.executable, "-c", child_code, str(moving),
                            str(template), str(weights), str(xfm), str(lta),
                            str(device), str(threads)], check=True)
            mark("SynthMorph_child_exit")

        input_module.register_talairach = isolated_register

    if args.input_only:
        for owner, name, label in (
            (input_module, "run_input_chain", "conform_end"),
            (input_module.SynthStrip, "__call__", "SynthStrip_end"),
            (input_module, "register_talairach", "SynthMorph_end"),
        ):
            original = getattr(owner, name)

            def measured(*pos, _original=original, _label=label, **kw):
                result = _original(*pos, **kw)
                mark(_label)
                return result

            setattr(owner, name, measured)

    mark("fresh_context")
    with tempfile.TemporaryDirectory(prefix="fnit_gpu_prefix_", dir=args.output.parent) as work:
        subject = Path(work) / "subject"
        initial = stage("input_talairach", run_input_talairach_chain,
                        args.t1, subject, args.weights_dir, args.assets_dir,
                        device=str(device), threads=4, keep_result=True)
        transforms = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                      for name, path in (("xfm", initial["talairach_xfm"]),
                                         ("lta", initial["talairach_affine_lta"]),
                                         ("synthstrip", initial["synthstrip"]),
                                         ("orig", initial["conformed"]))}
        xfm_values = np.fromstring(Path(initial["talairach_xfm"]).read_text()
                                   .split("Linear_Transform =")[1].split(";")[0],
                                   sep=" ").tolist()
        voxel_hashes = {name: hashlib.sha256(np.ascontiguousarray(
            np.asarray(nib.load(str(path)).dataobj)).tobytes()).hexdigest()
            for name, path in (("synthstrip", initial["synthstrip"]),
                               ("orig", initial["conformed"]))}
        if args.input_only:
            args.output.write_text(json.dumps({
                "device": str(device), "torch_version": torch.__version__,
                "cudnn_plan_cache_limit": os.environ.get("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"),
                "marks": marks, "isolated_synthmorph": args.isolate_synthmorph,
                "transforms": transforms, "voxel_hashes": voxel_hashes,
                "xfm_values": xfm_values}, indent=2))
            print(args.output)
            return
        mri = subject / "mri"
        stage("T1_normalize", normalize_t1, args.nu,
              initial["talairach_xfm"], mri / "T1.mgz", device=str(device))
        stage("brainmask", mask_volume, mri / "T1.mgz",
              initial["synthstrip"], mri / "brainmask.mgz", device=str(device))
        torch.backends.cudnn.allow_tf32 = False
        original_forward = SegmentUNet.forward
        forward_count = [0]

        def measured_forward(model, image):
            output = original_forward(model, image)
            forward_count[0] += 1
            mark(f"SynthSeg_forward_{forward_count[0]}")
            return output

        SegmentUNet.forward = measured_forward
        result = stage("SynthSeg", lambda: SynthSeg(
            weights=args.weights_dir, device=str(device), threads=4)(
                mri / "orig.mgz", keep_geometry=True,
                color_lut=args.assets_dir / "FreeSurferColorLUT.txt"),
                keep_result=True)
        hard = np.ascontiguousarray(result.segmentation.data.astype(np.int16))
        metrics = {"segmentation_sha256": hashlib.sha256(hard.tobytes()).hexdigest(),
                   "volumes_mm3": result.volumes_mm3,
                   "total_intracranial_mm3": result.total_intracranial_mm3}
        del hard, result
        gc.collect()
        torch.cuda.empty_cache()
        mark("after_result_deleted")
    args.output.write_text(json.dumps({
        "device": str(device), "torch_version": torch.__version__,
        "cudnn_plan_cache_limit": os.environ.get("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"),
        "marks": marks, "metrics": metrics, "transforms": transforms,
        "voxel_hashes": voxel_hashes, "xfm_values": xfm_values,
        "isolated_synthmorph": args.isolate_synthmorph}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
