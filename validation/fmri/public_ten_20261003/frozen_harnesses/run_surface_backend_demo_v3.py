"""验证分阶段准备的真实 backend；不把复用的 volume/recon 计为 cold whole。"""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import nibabel as nib
import numpy as np
import torch

import fnit
from fnit.fmri import surface_reconstruction as reconstruction_adapter
from fnit.fmri import fMRISurface_pipeline, locate_bids_inputs
from fnit.fmri.surface_reconstruction import _subject_files, prepare_surface_reconstruction
from fnit.fmri.surface_prepare import load_fsnative_to_t1w
from fnit.fmri.surface_volume import inspect_surface_volume


def verify_owned_cache(config, source_t1w, options):
    """只读核对实际缓存；失败直接终止，绝不让示例触发冷重建。"""
    root = Path(config["recon_all_output_dir"]).expanduser().resolve()
    subject = root / "subject"
    manifest = json.loads((root / reconstruction_adapter._MANIFEST).read_text())
    request = dict(manifest["request"])
    expected = {"backend": config["recon_all_backend"],
                "source_t1w": str(Path(source_t1w).resolve()),
                "source_sha256": reconstruction_adapter._sha256(source_t1w),
                "device": str(config["device"]),
                "options": reconstruction_adapter._json_value(options),
                "producer": reconstruction_adapter._producer_fingerprint(config["recon_all_backend"])}
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("owned reconstruction request differs; refusing a cold rebuild")
    effective = request.get("effective", {})
    if config["recon_all_backend"] == "provided":
        archive = Path(config["recon_all"]).expanduser().resolve()
        if request.get("provided") != {"path": str(archive), "sha256": reconstruction_adapter._sha256(archive)}:
            raise ValueError("provided ZIP differs from the owned import cache")
    elif config["recon_all_backend"] == "fnit":
        weights, assets = reconstruction_adapter._fnit_resources(options)
        native = reconstruction_adapter._native_dir(options)
        if (effective.get("weights_dir") != str(weights) or effective.get("assets_dir") != str(assets)
                or effective.get("native_bin_dir") != str(native)
                or effective.get("resources") != reconstruction_adapter._fnit_resource_fingerprint(weights, assets)):
            raise ValueError("effective FNIT resource identity differs from the owned cache")
        binaries = {name: reconstruction_adapter._sha256(native / name)
                    for name in reconstruction_adapter._FNIT_BIN_NAMES if (native / name).is_file()}
        if effective.get("native_binaries") != binaries:
            raise ValueError("effective FNIT native binary identity differs from the owned cache")
    for label in ("command", "mris_expand"):
        if label in effective and reconstruction_adapter._sha256(effective[label]) != effective[label + "_sha256"]:
            raise ValueError("effective native command bytes changed")
    if (manifest.get("owner") != reconstruction_adapter._OWNER or manifest.get("schema") != 1
            or manifest.get("subject_dir") != str(subject)
            or not reconstruction_adapter._cached_report(manifest, request, subject)):
        raise ValueError("owned reconstruction closure is not a valid complete cache")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if Path(fnit.__file__).resolve() != (args.source_root / "src/fnit/__init__.py").resolve():
        raise ValueError("the actually imported FNIT package differs from the frozen source root")
    config_bytes = args.config.read_bytes()
    config_sha_before = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    if config.get("auto_volume") is not False:
        raise ValueError("this staged demo requires explicit auto_volume=False")
    if "PYTORCH_NO_CUDA_MEMORY_CACHING" in os.environ:
        raise ValueError("unset the allocator flag before this cache-enabled demo process starts")
    device = torch.device(config["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("the staged validator supports explicit CPU or CUDA devices")
    if device.type == "cpu" and (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available()):
        raise ValueError("CPU demo must hide CUDA before Python starts")
    torch.set_num_threads(config.get("cpu_threads", 8))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, 20e9 / torch.cuda.get_device_properties(device).total_memory), device)
        torch.cuda.synchronize(device)
    helper_path = args.source_root / "validation/fmri/public_ten_20261003/run_fnit_subject.py"
    spec = importlib.util.spec_from_file_location("frozen_real_validation", helper_path)
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    helper_sha_before = validation.sha256(helper_path)
    runner_sha_before = validation.sha256(Path(__file__))
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    state = {"status": "running", "source_revision": args.source_revision,
             "backend": config["recon_all_backend"], "subject": config["subject"],
             "validation_helper_sha256": helper_sha_before,
             "demo_runner_sha256": runner_sha_before,
             "configuration_sha256": config_sha_before,
             "actual_fnit_source_identity_verified": True,
             "surface_device": str(device), "cpu_threads": torch.get_num_threads(),
             "torch_cuda_available": torch.cuda.is_available(),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "allocator_cache": "enabled", "allocator_flag_present": False,
             "parent_torch_allocator_limit_bytes": 20_000_000_000 if device.type == "cuda" else None,
             "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
             "scope": "Complete surface API and saved-output checks after CPU preparation; original volume is verified and reused; reconstruction is already complete and selected through the verified provided-input or owned-manifest contract. Cold reconstruction and preparation are reported separately.",
             "queue_seconds_excluded": True}
    validation.write(args.output / "report.public.json", state)
    monitor = validation.Monitor(args.output / "gpu_samples.private.jsonl") if device.type == "cuda" else None
    if monitor is not None:
        monitor.thread.start()
    started = time.perf_counter()
    try:
        inputs = locate_bids_inputs(config["bids_root"], subject=config["subject"],
                                    session=config.get("session"), task=config.get("task", "rest"))
        volume = inspect_surface_volume(
            inputs, config["derivatives_root"], signal=config.get("signal", "preproc"),
            hcp_assets_dir=config["hcp_assets_dir"],
            mni_template=config.get("volume_options", {}).get("mni_template"))
        if volume.state != "ready":
            raise RuntimeError(f"staged volume is not ready: {volume}")
        provided_input = Path(config["recon_all"]) if config["recon_all_backend"] == "provided" else None
        provided_directory = provided_input is not None and provided_input.is_dir()
        readonly_subject = provided_input if provided_directory else Path(config["recon_all_output_dir"]) / "subject"
        guard_paths = {"raw/" + p.relative_to(inputs.bids_root).as_posix(): p
                       for p in (inputs.bold, volume.source_t1w)}
        guard_paths.update({"volume/" + p.relative_to(config["derivatives_root"]).as_posix(): p
                            for p in volume.expected_paths})
        guard_paths.update({"readonly_subject/" + name: readonly_subject / name
                            for name in _subject_files(readonly_subject, require_middle=True)})
        if provided_input is not None and not provided_directory:
            guard_paths["provided_archive/recon_all.zip"] = provided_input
        owned_manifest = Path(config["recon_all_output_dir"]) / reconstruction_adapter._MANIFEST
        if not provided_directory:
            guard_paths["owned_reconstruction/manifest"] = owned_manifest
        if isinstance(config.get("fsnative_to_t1w"), (str, Path)):
            guard_paths["affine/verified_forward_scanner_ras"] = Path(config["fsnative_to_t1w"])
        before = {name: validation.sha256(path) for name, path in guard_paths.items()}
        source_before = validation.source_hashes(args.source_root)
        adapter_options = dict(config.get("recon_all_options") or {})
        if config.get("fsnative_to_t1w") is not None:
            adapter_options["fsnative_to_t1w"] = load_fsnative_to_t1w(config["fsnative_to_t1w"])
        if not provided_directory:
            verify_owned_cache(config, volume.source_t1w, adapter_options)
        lookup_started = time.perf_counter()
        cached = prepare_surface_reconstruction(
            volume.source_t1w, args.output / "adapter_preflight",
            recon_all=config.get("recon_all"), backend=config["recon_all_backend"],
            output_dir=config["recon_all_output_dir"], device=config["device"],
            options=adapter_options)
        lookup_seconds = time.perf_counter() - lookup_started
        if not cached.reused:
            raise RuntimeError("staged reconstruction was not ready to reuse before the API")
        api_started = time.perf_counter()
        result = fMRISurface_pipeline(**config)
        api_seconds = time.perf_counter() - api_started
        if result.volume_executed:
            raise RuntimeError("staged demo unexpectedly recomputed volume")
        metadata = json.loads(result.metadata.read_text())
        prerequisite = metadata["FNIT"]["VolumePrerequisite"]
        reconstruction = metadata["FNIT"]["Reconstruction"]
        if prerequisite["Executed"] or not prerequisite["Reused"]:
            raise RuntimeError("surface did not record the ready volume reuse")
        if Path(result.recon_all).resolve() != cached.subject_dir.resolve():
            raise RuntimeError("full API selected a different reconstruction from cache preflight")
        direct_provided = (provided_directory
                           and cached.subject_dir.resolve() == readonly_subject.resolve()
                           and not reconstruction.get("commands"))
        if reconstruction.get("reused") is not True and not direct_provided:
            raise RuntimeError("staged adapter cache did not reuse its owned reconstruction")
        frames = nib.load(str(inputs.bold)).shape[3]
        outputs = {"dtseries": validation.image_check(result.dtseries, frames, inputs.tr)}
        for hemi, path in (("L", result.left), ("R", result.right)):
            image = nib.load(str(path))
            if len(image.darrays) != frames or any(
                    array.data.shape != (32492,) or not np.isfinite(array.data).all()
                    for array in image.darrays):
                raise ValueError(f"{hemi} saved full-run GIFTI has invalid frames/vertices/data")
            outputs[hemi] = {"shape": [frames, 32492], "sha256": validation.sha256(path),
                             "all_finite": True}
        for path in (result.metadata, result.qc_report, *result.registered_spheres):
            if path is None or not path.is_file():
                raise ValueError("persistent metadata/QC/registered sphere is missing")
        after = {name: validation.sha256(path) for name, path in guard_paths.items()}
        source_after = validation.source_hashes(args.source_root)
        helper_sha_after = validation.sha256(helper_path)
        runner_sha_after = validation.sha256(Path(__file__))
        config_sha_after = validation.sha256(args.config)
        if (before != after or source_before != source_after or helper_sha_before != helper_sha_after
                or runner_sha_before != runner_sha_after or config_sha_before != config_sha_after):
            raise RuntimeError("readonly input or frozen source changed during the demo")
        private = {"result": dataclasses.asdict(result), "configuration": config,
                   "source_hashes_before": source_before, "source_hashes_after": source_after}
        (args.output / "files.private.json").write_text(json.dumps(private, indent=2, default=str) + "\n")
        state.update(status="complete", full_api_seconds=api_seconds,
                     driver_through_saved_output_validation_seconds=time.perf_counter() - started,
                     timing_seconds=result.timing_seconds, frame_count=frames, tr_seconds=inputs.tr,
                     outputs=outputs, volume_ready=True, volume_reused=True,
                     reconstruction_status=reconstruction["status"], reconstruction_reused=True,
                     reconstruction_reuse_kind="verified original provided input" if direct_provided else "owned manifest cache",
                     adapter_preflight_lookup_seconds=lookup_seconds,
                     readonly_inputs_before=before, readonly_inputs_after=after,
                     readonly_input_guards_equal=before == after,
                     frozen_source_guards_equal=source_before == source_after,
                     validation_helper_guard_equal=helper_sha_before == helper_sha_after,
                     demo_runner_guard_equal=runner_sha_before == runner_sha_after,
                     configuration_guard_equal=config_sha_before == config_sha_after,
                     original_t1_identity=metadata["FNIT"]["Geometry"]["OriginalT1Identity"],
                     registration=metadata["FNIT"]["Registration"],
                     registration_details=metadata["FNIT"]["RegistrationDetails"],
                     volume_cold_seconds_included=False, reconstruction_cold_seconds_included=False)
    except BaseException as error:
        (args.output / "failure.private.txt").write_text(traceback.format_exc())
        state.update(status="failed", error_type=type(error).__name__,
                     detail="See preserved private failure log", elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        if monitor is not None:
            monitor.stop.set()
            monitor.thread.join(timeout=10)
        memory_measured = monitor is not None and not monitor.errors and monitor.peak > 0
        state.update(job_tree_peak_gpu_bytes=monitor.peak if memory_measured else None,
                     job_tree_peak_gpu_gb=monitor.peak / 1e9 if memory_measured else None,
                     job_tree_peak_gpu_gib=monitor.peak / 1024**3 if memory_measured else None,
                     gpu_monitor_error_count=len(monitor.errors) if monitor is not None else None,
                     gpu_memory_scope="sampled owned CUDA process tree" if monitor is not None else "CPU execution; GPU memory is not measured",
                     under_20_gb=monitor.peak <= 20_000_000_000 if memory_measured else None)
        validation.write(args.output / "report.public.json", state)


if __name__ == "__main__":
    main()
