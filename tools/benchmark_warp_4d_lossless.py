#!/usr/bin/env python3
"""Full-image real-data warp benchmark against frozen FNIT source.

The private JSON input is {"cases": [{"backend": "applywarp" | "synthmorph",
"input": "/real/image.nii.gz", "reference": "/real/reference.nii.gz",
"warp": "/real/nonzero_warp.nii.gz", "devices": ["cpu", "cuda:0"]}]}.
Optional per-case keys: interpolation, warp_convention, premat, postmat,
output_dtype, fill, official_output. No image is cropped or frame subsampled.
Cases may be 3D for a separately labelled fixed official comparison.

Example:
  PYTHONPATH=src python tools/benchmark_warp_4d_lossless.py \
    --legacy-root /path/to/frozen_7473452 --case-json /private/cases.json \
    --output-dir /private/warp_benchmark --repeats 3 --frame-chunks 32 64 128

Only report.public.json and template-space PNGs are publication candidates.
The .private directory contains subject outputs/configuration and stays local.
Input decoding is timed once, outside materialized-image API timings. Saving
is timed separately. Profiler diagnostics never contribute to runtime claims.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
import torch

import fnit.applywarp.core as current_applywarp
import fnit.synthmorph.pipeline as current_synthmorph
from fnit._sampling_plan import _header_bytes
from fnit._transforms import DenseWarp


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_modules(root):
    root = Path(root)
    if not (root / "src/fnit").is_dir():
        raise ValueError("legacy-root must contain src/fnit")
    apply = load_module("fnit.applywarp._frozen_4d_core", root / "src/fnit/applywarp/core.py")
    spatial = load_module("fnit.synthmorph._frozen_4d_spatial", root / "src/fnit/synthmorph/spatial.py")
    synth = load_module("fnit.synthmorph._frozen_4d_pipeline", root / "src/fnit/synthmorph/pipeline.py")
    # Relative imports initially resolve to the installed package. Bind the
    # old spatial functions explicitly so the oracle cannot use new kernels.
    for name in ("compose", "surfa_nearest", "transform"):
        setattr(synth, name, getattr(spatial, name))
    return apply, synth, spatial


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def materialize(path):
    started = time.perf_counter()
    image = nib.load(str(path))
    image._dataobj = np.asanyarray(image.dataobj)
    return image, time.perf_counter() - started


def blocks(image, frames=16):
    if image.ndim == 3:
        yield np.asanyarray(image.dataobj)
    else:
        for start in range(0, image.shape[3], frames):
            yield np.asanyarray(image.dataobj[..., start:start + frames])


def voxel_hash(image):
    digest = hashlib.sha256()
    digest.update(str(tuple(image.shape)).encode())
    digest.update(np.dtype(image.get_data_dtype()).str.encode())
    for values in blocks(image):
        digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def metadata_gate(reference, candidate):
    return {
        "shape_exact": tuple(reference.shape) == tuple(candidate.shape),
        "dtype_exact": reference.get_data_dtype() == candidate.get_data_dtype(),
        "header_exact": _header_bytes(reference) == _header_bytes(candidate),
        "affine_exact": bool(np.array_equal(reference.affine, candidate.affine)),
        "tr_exact": reference.header.get_zooms()[3:] == candidate.header.get_zooms()[3:],
        "time_units_exact": reference.header.get_xyzt_units() == candidate.header.get_xyzt_units(),
    }


def image_gate(reference, candidate):
    metadata = metadata_gate(reference, candidate)
    same_shape = metadata["shape_exact"]
    same_dtype = metadata["dtype_exact"]
    exact = same_shape and same_dtype
    negative_zero_exact = same_shape
    changed = 0
    maximum = 0.0
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    if same_shape:
        for left, right in zip(blocks(reference), blocks(candidate)):
            left, right = np.asarray(left), np.asarray(right)
            count += left.size
            if same_dtype:
                # A same-size integer view works on strided frame arrays.
                # It avoids two full XYZ/T layout copies while checking every
                # value's bits, including signed zero. Equal bits imply all
                # numerical errors are zero, so float64 work is unnecessary.
                bits = np.dtype(f"u{left.dtype.itemsize}")
                block_exact = bool(np.array_equal(left.view(bits), right.view(bits)))
                exact &= block_exact
                if block_exact:
                    continue
            negative_zero_exact &= bool(np.array_equal(
                (left == 0) & np.signbit(left), (right == 0) & np.signbit(right),
            ))
            difference = left.astype(np.float64) - right.astype(np.float64)
            changed += int(np.count_nonzero(difference))
            maximum = max(maximum, float(np.max(np.abs(difference))))
            squared_error += float(np.sum(difference * difference))
            absolute_error += float(np.sum(np.abs(difference)))
    return {
        "voxel_bits_exact": bool(exact), "negative_zero_exact": bool(negative_zero_exact),
        **metadata,
        "changed_values": changed, "values": count,
        "maximum_absolute_difference": maximum,
        "mae": absolute_error / count if count else None,
        "rmse": (squared_error / count) ** 0.5 if count else None,
    }


@contextmanager
def phase_timers(module, backend, device):
    times = {"prepare_seconds": None, "sampling_seconds": None}
    changed = []

    def hook(owner, name, key):
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            sync(device)
            started = time.perf_counter()
            output = original(*args, **kwargs)
            sync(device)
            times[key] = (times[key] or 0.0) + time.perf_counter() - started
            return output

        setattr(owner, name, measured)
        changed.append((owner, name, original))

    if backend == "applywarp":
        hook(module.TorchApplyWarp, "prepare", "prepare_seconds")
        hook(module.ApplyWarpPlan, "_apply_loaded", "sampling_seconds")
    elif hasattr(module, "_resampled_frames"):
        hook(module, "_prepare_transform", "prepare_seconds")
        hook(module, "_sample_prepared", "sampling_seconds")
    else:
        # The frozen implementation has no separated coordinate preparation;
        # do not manufacture a stage time by altering its sampling algorithm.
        hook(module, "_resampled_image", "sampling_seconds")
    try:
        yield times
    finally:
        for owner, name, original in reversed(changed):
            setattr(owner, name, original)


def execute(module, backend, image, reference, warp, case, device, chunk, legacy):
    if backend == "applywarp":
        kwargs = {} if legacy else {"frame_chunk_size": chunk}
        warper = module.TorchApplyWarp(str(device), **kwargs)
        return warper(
            image, reference, warp=warp, premat=case.get("premat"), postmat=case.get("postmat"),
            interpolation=case.get("interpolation", "trilinear"),
            warp_convention=case.get("warp_convention", "auto"),
            output_dtype=case.get("output_dtype", "float"),
        )
    source_geometry = nib.load(str(image)) if isinstance(image, (str, Path)) else image
    transformation = DenseWarp(np.asarray(warp.dataobj), source=source_geometry, target=reference)
    kwargs = {} if legacy else {"device": str(device), "frame_chunk_size": chunk}
    return module.apply_transform(
        image, transformation, method=case.get("interpolation", "linear"),
        fill=case.get("fill", 0), dtype=case.get("output_dtype", "float32"), **kwargs,
    )


def output_image(result):
    return result.image if hasattr(result, "image") else result


def figure(reference, baseline, candidate, path):
    # Only central slices on the reference grid are rendered; raw/native
    # anatomy and subject labels are never included in the publication PNG.
    template = np.asanyarray(reference.dataobj)
    if template.ndim == 4:
        template = template[..., 0]
    first = np.asanyarray(baseline.dataobj[..., 0] if baseline.ndim == 4 else baseline.dataobj)
    second = np.asanyarray(candidate.dataobj[..., 0] if candidate.ndim == 4 else candidate.dataobj)
    canvas = Image.new("RGB", (960, 750), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("Reference template", "Frozen FNIT: first frame", "Current FNIT: first frame", "Absolute difference")):
        draw.text((column * 240 + 5, 8), title, fill="black")
    for row, axis in enumerate((2, 1, 0)):
        slices = [np.rot90(np.take(volume, volume.shape[axis] // 2, axis=axis))
                  for volume in (template, first, second, np.abs(first - second))]
        shared = max(float(np.percentile(np.abs(first), 99.5)), 1e-12)
        for column, panel in enumerate(slices):
            scale = max(float(np.percentile(np.abs(panel), 99.5)), 1e-12) if column in (0, 3) else shared
            gray = (np.clip(np.abs(panel) / scale, 0, 1) * 255).astype(np.uint8)
            tile = Image.fromarray(gray).convert("RGB").resize((240, 240))
            canvas.paste(tile, (column * 240, 25 + row * 240))
    canvas.save(path)


def measured_run(
    module, backend, image, reference, warp, case, device, chunk, legacy, private_path,
    *, diagnostic_phases=False, save_output=True,
):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    # Stage hooks synchronize CUDA and are restricted to diagnostic warm-up.
    # Measured API calls have no stage hooks or intermediate synchronizations.
    context = phase_timers(module, backend, device) if diagnostic_phases else nullcontext(
        {"prepare_seconds": None, "sampling_seconds": None},
    )
    with context as stages:
        started = time.perf_counter()
        result = execute(module, backend, image, reference, warp, case, device, chunk, legacy)
        sync(device)
        elapsed = time.perf_counter() - started
    output = output_image(result)
    memory = None if device.type != "cuda" else {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    save_elapsed = None
    if save_output:
        save_started = time.perf_counter()
        if hasattr(result, "valid_mask"):
            result.save(private_path)
        else:
            nib.save(output, private_path)
        save_elapsed = time.perf_counter() - save_started
    metrics = {"materialized_api_seconds": elapsed, "save_seconds": save_elapsed,
               "api_plus_save_seconds": elapsed + save_elapsed if save_elapsed is not None else None,
               "phase_instrumented_diagnostic_only": diagnostic_phases,
               **stages, "cuda_memory": memory}
    return result, metrics


def profiler_diagnostic(module, backend, image, reference, warp, case, device, chunk, path):
    if device.type != "cuda":
        return None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        result = execute(module, backend, image, reference, warp, case, device, chunk, False)
        sync(device)
    del result
    profile.export_chrome_trace(str(path))
    events = sorted(profile.key_averages(), key=lambda item: item.device_time_total, reverse=True)[:15]
    return [{"operation": item.key, "device_time_total_us": item.device_time_total,
             "cpu_time_total_us": item.cpu_time_total, "calls": item.count} for item in events]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frame-chunks", type=int, nargs="*", default=[32, 64, 128])
    parser.add_argument("--auto-only", action="store_true",
                        help="Validate/time only legacy_all_channels and current_auto; retain full images and SynthMorph CUDA all-channel oracle")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--profile-cuda", action="store_true")
    parser.add_argument("--path-api-repeats", type=int, default=0,
                        help="0 disables path API; 1 adds one full warmup validation and one timing-only repeat for legacy/current_auto")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.path_api_repeats not in (0, 1) or any(chunk < 1 for chunk in args.frame_chunks):
        parser.error("repeats and frame chunks must be positive")
    torch.set_num_threads(args.threads)
    configuration = json.loads(args.case_json.read_text())
    legacy_apply, legacy_synth, legacy_spatial = frozen_modules(args.legacy_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    private = args.output_dir / ".private"
    private.mkdir(mode=0o700)
    (private / "cases.json").write_text(args.case_json.read_text())
    source_paths = {
        "candidate_applywarp": current_applywarp.__file__, "candidate_synthmorph": current_synthmorph.__file__,
        "candidate_spatial": Path(current_synthmorph.__file__).with_name("spatial.py"),
        "legacy_applywarp": legacy_apply.__file__, "legacy_synthmorph": legacy_synth.__file__,
        "legacy_spatial": legacy_spatial.__file__, "benchmark": __file__,
    }
    report = {
        "schema_version": 1, "source_sha256": {key: sha256(value) for key, value in source_paths.items()},
        "scope": "full real images; fixed nonzero transforms; no crop, frame reduction or re-estimation",
        "timing": "one load separately; every configuration warmup validates all bits/metadata; only materialized current_auto times save/reload; formal repeats time API/peak only",
        "validation_policy": "all frames and voxels validated once per configuration/device/input mode; serializer roundtrip is measured on current_auto per device and explicitly referenced by other bit-identical outputs; formal timing outputs are not individually validated",
        "variant_selection": "auto_only" if args.auto_only else "auto_and_explicit_chunks",
        "phase_note": "Only warmup has synchronized diagnostic phase hooks; measured repeat API has no hooks. ApplyWarp sample phase includes layout/H2D/D2H; SynthMorph current sample covers kernels only; legacy SynthMorph prepare is unavailable and sample includes geometry/H2D/D2H",
        "python": sys.version.split()[0], "torch": torch.__version__, "threads": args.threads,
        "cases": [], "all_required_lossless_gates_passed": True,
    }
    exit_code = 0
    for index, case in enumerate(configuration["cases"], 1):
        anonymous = f"case_{index:03d}"
        entry = {"case": anonymous, "backend": case["backend"], "runs": []}
        report["cases"].append(entry)
        try:
            backend = case["backend"]
            if backend not in ("applywarp", "synthmorph"):
                raise ValueError("unsupported backend")
            image, input_load = materialize(case["input"])
            reference, reference_load = materialize(case["reference"])
            warp, warp_load = materialize(case["warp"])
            if image.ndim not in (3, 4) or (image.ndim == 4 and image.shape[3] < 2):
                raise ValueError("benchmark requires 3D official case or multi-frame full 4D image")
            if not np.any(np.asarray(warp.dataobj) != 0):
                raise ValueError("benchmark warp must be nonzero")
            entry.update({
                "shape": list(image.shape), "reference_shape": list(reference.shape[:3]),
                "scope": "full_4d" if image.ndim == 4 else "separate_fixed_3d_case",
                "input_sha256": {key: sha256(case[key]) for key in ("input", "reference", "warp")},
                "load_seconds": {"input": input_load, "reference": reference_load, "warp": warp_load},
                "interpolation": case.get("interpolation", "trilinear" if backend == "applywarp" else "linear"),
            })
            for key in ("premat", "postmat"):
                if case.get(key) is not None:
                    entry["input_sha256"][key] = sha256(case[key])
            current = current_applywarp if backend == "applywarp" else current_synthmorph
            frozen = legacy_apply if backend == "applywarp" else legacy_synth
            official = None
            if case.get("official_output"):
                official, official_load = materialize(case["official_output"])
                entry["load_seconds"]["official_output"] = official_load
                entry["input_sha256"]["official_output"] = sha256(case["official_output"])
                entry["official_comparison_scope"] = (
                    "fixed_3d_official_output" if official.ndim == 3 else "fixed_4d_official_output"
                )
            for device_string in case.get("devices", ["cpu"]):
                device = torch.device(device_string)
                if device.type == "cuda":
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA requested but unavailable")
                    torch.cuda.set_per_process_memory_fraction(
                        min(1.0, 19_000_000_000 / torch.cuda.get_device_properties(device).total_memory), device,
                    )
                baseline_device = device if backend == "applywarp" else torch.device("cpu")
                canonical = private / f"{anonymous}_{device.type}_baseline.nii"
                canonical_mask = private / f"{anonymous}_{device.type}_valid.npy"
                warm_file = private / f"{anonymous}_{device.type}_warmup.nii.gz"
                result, warm_metrics = measured_run(
                    frozen, backend, image, reference, warp, case, baseline_device, None, True, warm_file,
                    save_output=False,
                )
                base_image = output_image(result)
                base_header = _header_bytes(base_image)
                base_affine = np.array(base_image.affine, copy=True)
                base_voxel_hash = voxel_hash(base_image)
                checkpoint_started = time.perf_counter()
                nib.save(base_image, canonical)
                warm_metrics["checkpoint_save_seconds"] = time.perf_counter() - checkpoint_started
                if hasattr(result, "valid_mask"):
                    np.save(canonical_mask, result.valid_mask)
                del result, base_image
                warm_file.unlink(missing_ok=True)
                gc.collect()
                baseline = nib.load(str(canonical), mmap=True)
                baseline._dataobj = np.asanyarray(baseline.dataobj)
                entry["runs"].append({"device": str(baseline_device), "comparison_device": str(device),
                                      "variant": "legacy_all_channels_warmup", **warm_metrics})
                variants = [("legacy_all_channels", frozen, None, True)]
                variants += [("current_auto", current, None, False)]
                if not args.auto_only:
                    variants += [(f"current_chunk_{chunk}", current, chunk, False) for chunk in args.frame_chunks]
                gpu_baseline = None
                if backend == "synthmorph" and device.type == "cuda":
                    # CPU and CUDA can round differently. Establish a separate
                    # full-channel CUDA oracle for the lossless chunking gate.
                    frame_count = image.shape[3] if image.ndim == 4 else 1
                    gpu_result, gpu_metrics = measured_run(
                        current, backend, image, reference, warp, case, device,
                        frame_count, False, warm_file,
                        save_output=False,
                    )
                    gpu_image = output_image(gpu_result)
                    gpu_header = _header_bytes(gpu_image)
                    gpu_affine = np.array(gpu_image.affine, copy=True)
                    gpu_checkpoint = private / f"{anonymous}_cuda_all_channels.nii"
                    checkpoint_started = time.perf_counter()
                    nib.save(gpu_image, gpu_checkpoint)
                    gpu_metrics["checkpoint_save_seconds"] = time.perf_counter() - checkpoint_started
                    del gpu_result, gpu_image
                    warm_file.unlink(missing_ok=True)
                    gc.collect()
                    gpu_baseline = nib.load(str(gpu_checkpoint), mmap=True)
                    gpu_baseline._dataobj = np.asanyarray(gpu_baseline.dataobj)
                    entry["runs"].append({"device": str(device), "variant": "current_cuda_all_channels_warmup",
                                          "frame_chunk_size": frame_count, **gpu_metrics})
                    if not args.auto_only:
                        variants.insert(1, ("current_all_channels", current, frame_count, False))
                for repeat in range(args.repeats + 1):
                    ordered = variants if repeat % 2 == 0 else list(reversed(variants))
                    for name, module, chunk, legacy in ordered:
                        actual_device = baseline_device if legacy else device
                        # Only one checkpoint and one candidate output remain
                        # on disk per case/device; variants overwrite this file.
                        output_path = private / f"{anonymous}_{device.type}_candidate.nii.gz"
                        result, metrics = measured_run(
                            module, backend, image, reference, warp, case, actual_device, chunk, legacy, output_path,
                            diagnostic_phases=repeat == 0,
                            save_output=repeat == 0 and name == "current_auto",
                        )
                        if repeat > 0:
                            del result
                            gc.collect()
                            entry["runs"].append({
                                "device": str(actual_device), "comparison_device": str(device), "variant": name,
                                "frame_chunk_size": chunk, "repeat": repeat, "warmup": False,
                                "input_mode": "materialized", "validation_seconds": None,
                                "output_validation": "warmup_same_configuration",
                                "warmup_validation_reference": {"input_mode": "materialized", "variant": name,
                                                                "comparison_device": str(device), "repeat": 0},
                                "lossless_gate_required": False, "lossless_gate_passed": None, "gate": None,
                                **metrics,
                            })
                            (args.output_dir / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
                            continue
                        validation_started = time.perf_counter()
                        output = output_image(result)
                        comparison = gpu_baseline if gpu_baseline is not None and not legacy else baseline
                        comparison_header = gpu_header if gpu_baseline is not None and not legacy else base_header
                        comparison_affine = gpu_affine if gpu_baseline is not None and not legacy else base_affine
                        legacy_cpu_comparison = (
                            image_gate(baseline, output) if gpu_baseline is not None and not legacy else None
                        )
                        gate = image_gate(comparison, output)
                        # Serialized headers may normalize scaling; compare
                        # unsaved candidate header to unsaved frozen header.
                        gate["header_exact"] = _header_bytes(output) == comparison_header
                        gate["affine_exact"] = bool(np.array_equal(output.affine, comparison_affine))
                        gate["valid_mask_exact"] = (
                            bool(np.array_equal(result.valid_mask, np.load(canonical_mask, mmap_mode="r")))
                            if hasattr(result, "valid_mask") else None
                        )
                        own_roundtrip = metrics["save_seconds"] is not None
                        before_save_hash = voxel_hash(output) if own_roundtrip else None
                        if not legacy and chunk is None and repeat == 0:
                            figure(reference, baseline, output, args.output_dir / f"{anonymous}_{device.type}.png")
                        del result, output
                        gc.collect()
                        gate["save_reload_voxel_bits_exact"] = None
                        gate["saved_header_exact"] = None
                        gate["saved_affine_exact"] = None
                        official_comparison = None
                        if own_roundtrip:
                            reloaded, _ = materialize(output_path)
                            gate["save_reload_voxel_bits_exact"] = voxel_hash(reloaded) == before_save_hash
                            # Hash covers all saved values; metadata is checked
                            # without a second full-image numerical pass.
                            saved_gate = metadata_gate(comparison, reloaded)
                            gate["saved_header_exact"] = saved_gate["header_exact"]
                            gate["saved_affine_exact"] = saved_gate["affine_exact"]
                            official_comparison = image_gate(official, reloaded) if official is not None else None
                            del reloaded
                        required = True
                        passed = all(gate[key] for key in (
                            "voxel_bits_exact", "negative_zero_exact", "shape_exact", "dtype_exact",
                            "header_exact", "affine_exact", "tr_exact", "time_units_exact",
                        )) and gate["valid_mask_exact"] is not False
                        if own_roundtrip:
                            passed &= all(gate[key] for key in (
                                "save_reload_voxel_bits_exact", "saved_header_exact", "saved_affine_exact",
                            ))
                        if required and not passed:
                            report["all_required_lossless_gates_passed"] = False
                            exit_code = 1
                        entry["runs"].append({
                            "device": str(actual_device), "comparison_device": str(device), "variant": name,
                            "frame_chunk_size": chunk, "repeat": repeat, "warmup": repeat == 0,
                            "input_mode": "materialized",
                            "validation_seconds": time.perf_counter() - validation_started,
                            "output_validation": "complete_warmup_output_bits_metadata",
                            "roundtrip_validation": "this_output" if own_roundtrip else "current_auto_same_serializer_reference",
                            "roundtrip_validation_reference": None if own_roundtrip else {
                                "input_mode": "materialized", "variant": "current_auto",
                                "comparison_device": str(device), "repeat": 0,
                            },
                            "lossless_gate_required": required, "lossless_gate_passed": passed,
                            "gate": gate, **metrics,
                            "official_comparison": official_comparison,
                            "legacy_cpu_comparison_not_a_lossless_gate": legacy_cpu_comparison,
                            "lossless_reference": "current_cuda_all_channels" if gpu_baseline is not None and not legacy else "legacy_all_channels",
                        })
                        (args.output_dir / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
                if args.path_api_repeats:
                    path_variants = [variant for variant in variants if variant[0] in ("legacy_all_channels", "current_auto")]
                    for repeat in range(args.path_api_repeats + 1):
                        ordered = path_variants if repeat % 2 == 0 else list(reversed(path_variants))
                        for name, module, chunk, legacy in ordered:
                            actual_device = baseline_device if legacy else device
                            output_path = private / f"{anonymous}_{device.type}_candidate.nii.gz"
                            result, metrics = measured_run(
                                module, backend, case["input"], reference, warp, case,
                                actual_device, chunk, legacy, output_path,
                                save_output=False,
                            )
                            metrics["input_path_api_seconds"] = metrics.pop("materialized_api_seconds")
                            if repeat > 0:
                                del result
                                gc.collect()
                                entry["runs"].append({
                                    "device": str(actual_device), "comparison_device": str(device), "variant": name,
                                    "frame_chunk_size": chunk, "repeat": repeat, "warmup": False,
                                    "input_mode": "input_path_reference_and_warp_materialized", "validation_seconds": None,
                                    "output_validation": "warmup_same_configuration",
                                    "warmup_validation_reference": {
                                        "input_mode": "input_path_reference_and_warp_materialized", "variant": name,
                                        "comparison_device": str(device), "repeat": 0,
                                    },
                                    "lossless_gate_required": False, "lossless_gate_passed": None, "gate": None,
                                    **metrics,
                                })
                                (args.output_dir / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
                                continue
                            validation_started = time.perf_counter()
                            output = output_image(result)
                            comparison = gpu_baseline if gpu_baseline is not None and not legacy else baseline
                            comparison_header = gpu_header if gpu_baseline is not None and not legacy else base_header
                            comparison_affine = gpu_affine if gpu_baseline is not None and not legacy else base_affine
                            legacy_cpu_comparison = (
                                image_gate(baseline, output) if gpu_baseline is not None and not legacy else None
                            )
                            gate = image_gate(comparison, output)
                            gate["header_exact"] = _header_bytes(output) == comparison_header
                            gate["affine_exact"] = bool(np.array_equal(output.affine, comparison_affine))
                            gate["valid_mask_exact"] = (
                                bool(np.array_equal(result.valid_mask, np.load(canonical_mask, mmap_mode="r")))
                                if hasattr(result, "valid_mask") else None
                            )
                            del result, output
                            gc.collect()
                            gate["save_reload_voxel_bits_exact"] = None
                            gate["saved_header_exact"] = None
                            gate["saved_affine_exact"] = None
                            required = True
                            passed = all(gate[key] for key in (
                                "voxel_bits_exact", "negative_zero_exact", "shape_exact", "dtype_exact",
                                "header_exact", "affine_exact", "tr_exact", "time_units_exact",
                            )) and gate["valid_mask_exact"] is not False
                            if required and not passed:
                                report["all_required_lossless_gates_passed"] = False
                                exit_code = 1
                            entry["runs"].append({
                                "device": str(actual_device), "comparison_device": str(device), "variant": name,
                                "frame_chunk_size": chunk, "repeat": repeat, "warmup": repeat == 0,
                                "input_mode": "input_path_reference_and_warp_materialized",
                                "validation_seconds": time.perf_counter() - validation_started,
                                "output_validation": "complete_warmup_output_bits_metadata",
                                "roundtrip_validation": "current_auto_same_serializer_reference",
                                "roundtrip_validation_reference": {
                                    "input_mode": "materialized", "variant": "current_auto",
                                    "comparison_device": str(device), "repeat": 0,
                                },
                                "lossless_gate_required": required, "lossless_gate_passed": passed,
                                "gate": gate, **metrics,
                                "legacy_cpu_comparison_not_a_lossless_gate": legacy_cpu_comparison,
                                "lossless_reference": "current_cuda_all_channels" if gpu_baseline is not None and not legacy else "legacy_all_channels",
                            })
                            (args.output_dir / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
                if args.profile_cuda and device.type == "cuda":
                    entry.setdefault("profiler_diagnostics_not_speed_measurements", {})[str(device)] = profiler_diagnostic(
                        current, backend, image, reference, warp, case, device, None,
                        private / f"{anonymous}_{device.type}_trace.json",
                    )
                entry.setdefault("baseline_decoded_voxel_sha256", {})[str(device)] = base_voxel_hash
                if gpu_baseline is not None:
                    del gpu_baseline
                del baseline
            del image, reference, warp
        except Exception as error:
            entry["error_type"] = type(error).__name__
            with (private / "errors.log").open("a") as stream:
                traceback.print_exc(file=stream)
            report["all_required_lossless_gates_passed"] = False
            exit_code = 1
        (args.output_dir / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
