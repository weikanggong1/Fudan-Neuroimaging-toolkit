"""Paired real-nine-map benchmark against a frozen pre-plan sampler module.

Inputs stay on the validation host. Reports contain hashes, scalar gates and
wall/memory measurements; no image, affine matrix or subject path is exported.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import time

import nibabel as nib
import numpy as np
import torch

from fnit._sampling_plan import SamplingGeometry, _header_bytes
from fnit.applywarp import TorchApplyWarp
from fnit.mmorf import prepare_mmorf_warp
from fnit.dmri_pipeline.tbss import preprocess_fa

MAP_NAMES = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _materialize(path):
    image = nib.load(str(path))
    # Keep the decoded original header/affine exactly; only replace ArrayProxy
    # with its materialized array, so gzip reads are outside all timers.
    image._dataobj = np.asanyarray(image.dataobj)
    return image


def _load_legacy(path, backend):
    name = f"fnit.{'applywarp' if backend == 'tbss' else 'mmorf'}._frozen_map_benchmark"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_state():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()


def _memory(device):
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "free_whole_device_bytes": free, "total_device_bytes": total}


def _image_gate(before, after):
    left, right = np.asarray(before.dataobj), np.asarray(after.dataobj)
    gate = {"voxels_exact": bool(np.array_equal(left, right)),
            "header_exact": _header_bytes(before) == _header_bytes(after),
            "affine_exact": bool(np.array_equal(before.affine, after.affine)),
            "shape_exact": before.shape == after.shape,
            "dtype_exact": before.get_data_dtype() == after.get_data_dtype(),
            "maximum_absolute_difference": float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))}
    return gate


def _mask_images(outputs, reference, skeleton):
    arrays = {name: np.asarray(image.dataobj, dtype=np.float32).squeeze(axis=3)
              if image.ndim == 4 else np.asarray(image.dataobj, dtype=np.float32)
              for name, image in outputs.items()}
    valid = (arrays["FA"] != 0) & (np.asarray(reference.dataobj, dtype=np.float32) != 0)
    arrays["FA"] = arrays["FA"] * valid
    skeleton_mask = (np.asarray(skeleton.dataobj, dtype=np.float32) >= 2000) & valid
    return arrays, {name: values * skeleton_mask for name, values in arrays.items()}, valid, skeleton_mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("tbss", "mmorf"), required=True)
    parser.add_argument("--native-dir", type=Path, required=True,
                        help="Real completed native/ directory with six dti_* and three NODDI_* maps")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--warp", type=Path, required=True)
    parser.add_argument("--affine", type=Path, help="MMORF tensor affine .mat; TBSS coefficient includes affine")
    parser.add_argument("--fa-skeleton", type=Path, help="Required for TBSS FA/skeleton mask gates")
    parser.add_argument("--legacy-module", type=Path, required=True,
                        help="Frozen 954ad source: applywarp/core.py or mmorf/core.py; never the edited module")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit-bytes", type=int, default=20_000_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.backend == "tbss" and args.fa_skeleton is None:
        parser.error("--fa-skeleton is required for TBSS")
    if args.backend == "mmorf" and args.affine is None:
        parser.error("--affine is required for MMORF")
    if args.output.exists():
        raise FileExistsError("output report already exists")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.memory_limit_bytes / torch.cuda.get_device_properties(device).total_memory,
            device,
        )
    native_paths = {
        name: args.native_dir / (f"dti_{name}.nii.gz" if name in MAP_NAMES[:6] else f"NODDI_{name}.nii.gz")
        for name in MAP_NAMES
    }
    maps = {name: _materialize(path) for name, path in native_paths.items()}
    reference, warp = _materialize(args.reference), _materialize(args.warp)
    affine = np.loadtxt(args.affine) if args.affine is not None else None
    skeleton = _materialize(args.fa_skeleton) if args.fa_skeleton else None
    if args.backend == "tbss":
        maps["FA"] = preprocess_fa(maps["FA"])[0]
    legacy = _load_legacy(args.legacy_module, args.backend)
    current_module = sys.modules["fnit.applywarp.core" if args.backend == "tbss" else "fnit.mmorf.core"]
    if _sha256(args.legacy_module) == _sha256(current_module.__file__):
        raise ValueError("legacy module is identical to the current sampler; supply the frozen pre-change source")
    if args.backend == "tbss":
        old_apply = legacy.TorchApplyWarp(device)
        current_apply = TorchApplyWarp(device)
    _sync(device)
    report = {
        "schema_version": 1,
        "scope": "real saved native nine maps; fixed existing warp and affine; no registration re-estimation",
        "backend": args.backend,
        "timing_scope": "materialized input; synchronized prepare + nine independent image calls including D2H; no input reads or output NIfTI writes",
        "sampling": (
            "same single-map channel shape/order; original float64 transform geometry and float32 image sampling; no FP16/BF16"
            if args.backend == "tbss" else
            "same single-map channel shape/order; original mixed float64 transforms and float32 sampling coordinates/images; no FP16/BF16"
        ),
        "device": str(device), "python": sys.version.split()[0], "torch": torch.__version__,
        "threads": torch.get_num_threads(), "memory_limit_bytes": args.memory_limit_bytes,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "legacy_module_sha256": _sha256(args.legacy_module),
        "current_module_sha256": _sha256(current_module.__file__),
        "geometry_guard_sha256": _sha256(sys.modules["fnit._sampling_plan"].__file__),
        "input_sha256": {name: _sha256(path) for name, path in native_paths.items()},
        "reference_sha256": _sha256(args.reference), "warp_sha256": _sha256(args.warp),
        "affine_sha256": _sha256(args.affine) if args.affine else None,
        "fa_skeleton_sha256": _sha256(args.fa_skeleton) if args.fa_skeleton else None,
        "runs": [], "all_gates_passed": True,
    }
    for repeat in range(args.repeats):
        run = {"repeat": repeat, "condition": "first_call" if repeat == 0 else "warm"}
        run["gpu_before"] = _gpu_state() if device.type == "cuda" else None
        outputs = {}
        masks = {}
        for route in ("legacy", "prepared") if repeat % 2 == 0 else ("prepared", "legacy"):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            run[f"{route}_memory_before"] = _memory(device)
            _sync(device)
            started = time.perf_counter()
            plans = {}
            prepare_seconds = 0.0
            images = {}
            masks[route] = {}
            for name, source in maps.items():
                if route == "prepared":
                    key = SamplingGeometry.capture(source)
                    if key not in plans:
                        prepare_started = time.perf_counter()
                        plans[key] = (current_apply.prepare(
                            source, reference, warp=warp, interpolation="trilinear", warp_convention="relative"
                        ) if args.backend == "tbss" else prepare_mmorf_warp(
                            source, reference, warp, affine=affine, device=device, interpolation="linear"
                        ))
                        _sync(device)
                        prepare_seconds += time.perf_counter() - prepare_started
                    result = plans[key].apply(source, reference=reference)
                else:
                    result = (old_apply(source, reference, warp=warp, interpolation="trilinear", warp_convention="relative")
                              if args.backend == "tbss" else legacy.apply_mmorf_warp(
                                  source, reference, warp, affine=affine, device=device, interpolation="linear"))
                images[name] = result.image if args.backend == "tbss" else result
                if args.backend == "tbss":
                    masks[route][name] = result.valid_mask
            _sync(device)
            run[f"{route}_seconds"] = time.perf_counter() - started
            run[f"{route}_prepare_seconds"] = prepare_seconds if route == "prepared" else None
            run[f"{route}_plan_count"] = len(plans) if route == "prepared" else None
            run[f"{route}_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            run[f"{route}_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            run[f"{route}_memory_after"] = _memory(device)
            outputs[route] = images
            del plans
        run["image_gates"] = {name: _image_gate(outputs["legacy"][name], outputs["prepared"][name]) for name in MAP_NAMES}
        passed = all(all(value for key, value in gate.items() if key != "maximum_absolute_difference")
                     for gate in run["image_gates"].values())
        if args.backend == "tbss":
            run["valid_mask_exact"] = {name: bool(np.array_equal(masks["legacy"][name], masks["prepared"][name])) for name in MAP_NAMES}
            old_standard, old_skeleton, old_valid, old_skeleton_mask = _mask_images(outputs["legacy"], reference, skeleton)
            new_standard, new_skeleton, new_valid, new_skeleton_mask = _mask_images(outputs["prepared"], reference, skeleton)
            run["tbss_postprocessing_exact"] = {
                "fa_valid_mask": bool(np.array_equal(old_valid, new_valid)),
                "skeleton_mask": bool(np.array_equal(old_skeleton_mask, new_skeleton_mask)),
                "standard_nine": all(np.array_equal(old_standard[name], new_standard[name]) for name in MAP_NAMES),
                "skeleton_nine": all(np.array_equal(old_skeleton[name], new_skeleton[name]) for name in MAP_NAMES),
            }
            passed = passed and all(run["valid_mask_exact"].values()) and all(run["tbss_postprocessing_exact"].values())
        run["gates_passed"] = passed
        run["gpu_after"] = _gpu_state() if device.type == "cuda" else None
        report["all_gates_passed"] &= passed
        report["runs"].append(run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_gates_passed": report["all_gates_passed"], "runs": len(report["runs"])}, indent=2))
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
