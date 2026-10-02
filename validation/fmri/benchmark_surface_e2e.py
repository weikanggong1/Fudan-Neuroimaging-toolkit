"""Benchmark the complete surface API from existing volume and recon-all inputs.

The API estimates both MSMSulc spheres and publishes all surface derivatives.
Recon-all and volume processing are excluded. Optional private captures preserve
the API's actual inputs for audit; they are not independent reference geometry.
"""

import time

_import_started = time.perf_counter()

import argparse
from dataclasses import asdict
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

import nibabel as nib
import numpy as np
import torch

from fnit.fmri import fMRISurface_pipeline
from fnit.fmri.bids import locate_bids_inputs
from fnit.fmri.derivatives import sidecar
import fnit.fmri.surface_pipeline as surface_pipeline
from fnit.msm import _fastpd_native

_helpers_spec = importlib.util.spec_from_file_location(
    "surface_benchmark_helpers", Path(__file__).with_name("benchmark_bids.py")
)
_helpers = importlib.util.module_from_spec(_helpers_spec)
_helpers_spec.loader.exec_module(_helpers)
sha256 = _helpers.sha256
source_hashes = _helpers.source_hashes
_import_seconds = time.perf_counter() - _import_started


def gpu_state():
    """Observe shared device load without changing any other process."""
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        return {"available": False, "exit_code": result.returncode}
    devices = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 5:
            devices.append(dict(zip(
                ("physical_index", "name", "total_memory_mib", "used_memory_mib", "utilization_percent"),
                fields,
            )))
    return {"available": True, "devices": devices}


def capture_file(source, destination):
    """Retain geometry without changing the path used by the numerical API."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copyfile(source, destination)
        return "copy"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bids-root", "derivatives-root", "recon-all", "hcp-assets-dir", "source-root",
                 "report-out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--wb-command", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--serial-hemispheres", action="store_true",
                        help="Run L/R sequentially for a paired execution comparison")
    parser.add_argument("--msm-execution", choices=("optimized", "reference"), default="optimized")
    parser.add_argument("--gpu-memory-limit-gb", type=float, default=20)
    parser.add_argument("--capture-dir", type=Path,
                        help="New protected directory for actual API geometry and MSM inputs")
    args = parser.parse_args()
    if args.threads < 1 or not 0 < args.gpu_memory_limit_gb <= 20:
        parser.error("threads must be positive and gpu-memory-limit-gb must be in (0, 20]")
    if not str(args.device).startswith("cuda"):
        parser.error("this real-data benchmark requires the selected CUDA device")

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    selected = torch.device(args.device)
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, args.gpu_memory_limit_gb * 1e9 /
            torch.cuda.get_device_properties(selected).total_memory), selected,
    )
    torch.cuda.synchronize(selected)
    torch.cuda.reset_peak_memory_stats(selected)
    inputs = locate_bids_inputs(args.bids_root, subject=args.subject)
    if len(inputs.t1w_images) != 1 or nib.load(str(inputs.bold)).shape[3] != 490:
        raise ValueError("benchmark requires one unambiguous real T1w and all 490 BOLD frames")
    input_hashes = {"bold": sha256(inputs.bold), "t1w": sha256(inputs.t1w_images[0]),
                    "sbref": sha256(inputs.sbref) if inputs.sbref else None}
    initial_gpu_state = gpu_state()
    capture_seconds = 0.0
    captured_projection = None
    captured_msm = None
    captured_initial_spheres = []
    capture_methods = {"hardlink": 0, "copy": 0}
    memory_phases = []
    original_reset = torch.cuda.reset_peak_memory_stats
    original_projection = surface_pipeline.run_fmriprep_surface_projection
    original_msm = surface_pipeline.run_msmsulc
    original_prepare_msm = surface_pipeline.prepare_msmsulc_inputs
    if args.capture_dir is not None:
        args.capture_dir.mkdir(parents=True, exist_ok=False, mode=0o700)

    def memory_snapshot(label):
        memory_phases.append({"boundary": label,
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(selected),
                              "peak_reserved_bytes": torch.cuda.max_memory_reserved(selected)})

    def observed_reset(device=None):
        # Preserve any internal reset boundary. Parallel registration resets
        # once in its parent; legacy serial registration resets per hemisphere.
        memory_snapshot("before_internal_reset_" + str(len(memory_phases)))
        return original_reset(device)

    def observed_msm(msm_inputs, output_dir, **kwargs):
        nonlocal capture_seconds, captured_msm
        if args.capture_dir is not None:
            capture_started = time.perf_counter()
            captured_msm = {}
            for hemisphere, entry in msm_inputs.items():
                captured_msm[hemisphere] = {}
                for field, source in asdict(entry).items():
                    if source is None:
                        captured_msm[hemisphere][field] = None
                        continue
                    destination = args.capture_dir / "msm_inputs" / hemisphere / Path(source).name
                    capture_methods[capture_file(source, destination)] += 1
                    captured_msm[hemisphere][field] = str(destination.resolve())
            (args.capture_dir / "msm_inputs.private.json").write_text(
                json.dumps(captured_msm, indent=2) + "\n"
            )
            capture_seconds += time.perf_counter() - capture_started
        return original_msm(msm_inputs, output_dir, **kwargs)

    def observed_prepare_msm(*positional, **keywords):
        nonlocal capture_seconds
        prepared = original_prepare_msm(*positional, **keywords)
        if args.capture_dir is not None:
            capture_started = time.perf_counter()
            bound = inspect.signature(original_prepare_msm).bind(*positional, **keywords)
            for hemisphere, source in zip(("L", "R"), bound.arguments["initial_spheres"]):
                destination = args.capture_dir / "msm_inputs" / hemisphere / "initial_sphere.surf.gii"
                capture_methods[capture_file(source, destination)] += 1
                captured_initial_spheres.append(str(destination.resolve()))
            capture_seconds += time.perf_counter() - capture_started
        return prepared

    def observed_projection(*positional, **keywords):
        nonlocal capture_seconds, captured_projection
        if args.capture_dir is not None:
            capture_started = time.perf_counter()
            bound = inspect.signature(original_projection).bind(*positional, **keywords)
            bound.apply_defaults()
            values = bound.arguments
            captured_projection = {
                "bold_file": str(Path(values["clean_t1w"]).resolve()),
                "bold_std": str(Path(values["clean_mni"]).resolve()),
                "volume_roi": str(Path(values["goodvoxels"]).resolve())
                    if values["goodvoxels"] is not None else None,
                "repetition_time": float(values["tr_seconds"]), "expected_frames": 490,
                "signal": "preproc", "geometry_space": "T1w world RAS",
                "sphere_kind": "estimated_msmsulc",
            }
            field_mapping = {
                "white": "white", "pial": "pial", "midthickness": "midthickness",
                "midthickness_fsLR": "atlas_midthickness", "sphere_reg_fsLR": "registered_sphere",
                "cortex_mask": "native_roi",
            }
            for field in field_mapping:
                captured_projection[field] = []
            for hemisphere, geometry in zip(("L", "R"), (values["left"], values["right"])):
                for field, attribute in field_mapping.items():
                    destination = args.capture_dir / "projection_inputs" / f"{hemisphere}.{field}.gii"
                    capture_methods[capture_file(getattr(geometry, attribute), destination)] += 1
                    captured_projection[field].append(str(destination.resolve()))
            captured_projection["native_rois"] = list(captured_projection["cortex_mask"])
            captured_projection["initial_spheres"] = list(captured_initial_spheres)
            captured_projection["area_surfaces"] = {
                "native": list(captured_projection["midthickness"]),
                "fsLR": list(captured_projection["midthickness_fsLR"]),
            }
            (args.capture_dir / "inputs.private.json").write_text(
                json.dumps(captured_projection, indent=2) + "\n"
            )
            capture_seconds += time.perf_counter() - capture_started
        return original_projection(*positional, **keywords)

    torch.cuda.reset_peak_memory_stats = observed_reset
    surface_pipeline.run_msmsulc = observed_msm
    surface_pipeline.prepare_msmsulc_inputs = observed_prepare_msm
    surface_pipeline.run_fmriprep_surface_projection = observed_projection
    execution_options = {}
    supported_options = inspect.signature(fMRISurface_pipeline).parameters
    if "parallel" in supported_options:
        execution_options["parallel"] = not args.serial_hemispheres
    if "cpu_threads" in supported_options:
        execution_options["cpu_threads"] = args.threads
    try:
        started = time.perf_counter()
        result = fMRISurface_pipeline(
            args.bids_root, args.derivatives_root, subject=args.subject,
            recon_all=args.recon_all, hcp_assets_dir=args.hcp_assets_dir,
            wb_command=args.wb_command, device=args.device, signal="preproc",
            registered_spheres=None, msm_config=None, msm_execution=args.msm_execution,
            **execution_options,
        )
        torch.cuda.synchronize(selected)
        observed_api_seconds = time.perf_counter() - started
        memory_snapshot("after_complete_api")
    finally:
        torch.cuda.reset_peak_memory_stats = original_reset
        surface_pipeline.run_msmsulc = original_msm
        surface_pipeline.prepare_msmsulc_inputs = original_prepare_msm
        surface_pipeline.run_fmriprep_surface_projection = original_projection
    final_gpu_state = gpu_state()

    # All hashes and finite-value checks below are outside the API timer.
    metadata = json.loads(result.metadata.read_text())
    qc = json.loads(result.qc_report.read_text())
    estimated = metadata["FNIT"]["RegisteredSpheres"]
    if (qc.get("MSM") is None or set(estimated) != {"L", "R"}
            or not all(value["EstimatedHere"] is True for value in estimated.values())):
        raise ValueError("complete surface run did not estimate both MSMSulc spheres")
    cifti = nib.load(str(result.dtseries))
    values = np.asarray(cifti.dataobj)
    if values.shape != (490, 91282) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("CIFTI must retain 490 frames, 91,282 grayordinates and finite float32 values")
    if not np.isclose(cifti.header.get_axis(0).step, inputs.tr, rtol=1e-6, atol=1e-7):
        raise ValueError("CIFTI did not retain original TR")
    checks = {"cifti_shape": list(values.shape), "cifti_all_finite": True,
              "cifti_sha256": sha256(result.dtseries), "cifti_dtype": str(values.dtype),
              "tr_seconds": float(cifti.header.get_axis(0).step),
              "varying_grayordinates": int(np.count_nonzero(np.ptp(values, axis=0) > 0)),
              "brain_models": {name: int(model.size) for name, _, model in
                               cifti.header.get_axis(1).iter_structures()}, "hemispheres": {}}
    outputs = [result.left, result.right, result.dtseries, result.metadata,
               sidecar(result.left), sidecar(result.right), result.qc_report]
    for hemisphere, path, sphere in zip(("L", "R"), (result.left, result.right), result.registered_spheres):
        metric = nib.load(str(path))
        if len(metric.darrays) != 490 or any(
            array.data.shape != (32492,) or array.data.dtype != np.float32
            or not np.isfinite(array.data).all() for array in metric.darrays
        ):
            raise ValueError("GIFTI must retain all finite float32 490 x 32,492 values")
        checks["hemispheres"][hemisphere] = {
            "frames": 490, "vertices": 32492, "all_finite": True,
            "sha256": sha256(path), "sphere_sha256": sha256(sphere), "estimated_here": True,
        }
        outputs.extend((sphere, sphere.with_name(sphere.name.removesuffix(".surf.gii") + ".json")))
    if len(set(outputs)) != 11 or not all(path.is_file() for path in outputs):
        raise ValueError("complete surface run must retain all 11 persistent outputs")
    checks["persistent_output_count"] = 11
    checks["qc_sha256"] = sha256(result.qc_report)
    checks["msm_report_present"] = True
    checks["registered_spheres_estimated_here"] = True
    actual_input_hashes = {}
    if captured_projection is not None:
        for field in ("bold_file", "bold_std", "volume_roi", "white", "pial", "midthickness",
                      "midthickness_fsLR", "sphere_reg_fsLR", "cortex_mask"):
            paths = captured_projection[field]
            actual_input_hashes[field] = ([sha256(path) for path in paths] if isinstance(paths, list)
                                          else sha256(paths) if paths is not None else None)
    msm_input_hashes = {hemisphere: {field: sha256(path) if path is not None else None
                                   for field, path in fields.items()}
                        for hemisphere, fields in (captured_msm or {}).items()}
    if args.capture_dir is not None:
        private_outputs = {
            "left": str(result.left.resolve()), "right": str(result.right.resolve()),
            "dtseries": str(result.dtseries.resolve()),
            "registered_spheres": [str(path.resolve()) for path in result.registered_spheres],
            "projection_inputs_json": str((args.capture_dir / "inputs.private.json").resolve()),
            "msm_inputs_json": str((args.capture_dir / "msm_inputs.private.json").resolve()),
            "startpoint_sha256": {"t1w_preproc": actual_input_hashes["bold_file"],
                                  "mni_preproc": actual_input_hashes["bold_std"]},
            "msm_config": metadata["FNIT"]["RegistrationDetails"]["Configuration"],
            "source_revision": args.source_revision,
        }
        (args.capture_dir / "outputs.private.json").write_text(
            json.dumps(private_outputs, indent=2) + "\n"
        )
    native_path = Path(_fastpd_native.__file__)
    sources = source_hashes(args.source_root)
    report = {
        "schema_version": 1, "stage": "surface", "source_revision": args.source_revision,
        "source_sha256": sources, "source_hash_count": len(sources),
        "driver_sha256": sha256(__file__), "native_extension_sha256": sha256(native_path),
        "data": {"kind": "one real UK Biobank run", "subjects": 1,
                 "bold_shape": list(nib.load(str(inputs.bold)).shape),
                 "t1w_shape": list(nib.load(str(inputs.t1w_images[0])).shape)},
        "input_sha256": input_hashes, "actual_projection_inputs_sha256": actual_input_hashes,
        "actual_msm_inputs_sha256": msm_input_hashes, "checks": checks,
        "algorithm": {"signal": "preproc", "registered_spheres": None,
                      "hemisphere_execution": (
                          "parallel" if execution_options.get("parallel") and args.threads > 1 else "serial"
                      ),
                      "parallel_requested": bool(execution_options.get("parallel")),
                      "parallel_api_supported": "parallel" in supported_options,
                      "cpu_thread_budget": args.threads,
                      "msm_execution": args.msm_execution,
                      "registration": metadata["FNIT"]["Registration"],
                      "registration_details": metadata["FNIT"]["RegistrationDetails"],
                      "msm_report": qc["MSM"]["Report"], "coverage": qc["Coverage"]},
        "timing_seconds": {
            "public_api_observed_including_capture_and_output_save": observed_api_seconds,
            "public_api_capture_adjusted_including_output_save": observed_api_seconds - capture_seconds,
            "private_input_capture_overhead": capture_seconds,
            "module_import_initialization_excluded": _import_seconds,
            "pipeline_stages_unmodified": dict(result.timing_seconds),
            "pipeline_internal_total_includes_capture_and_excludes_final_publication": True,
        },
        "memory": {"peak_cuda_allocated_bytes": max(phase["peak_allocated_bytes"] for phase in memory_phases),
                   "peak_cuda_reserved_bytes": max(phase["peak_reserved_bytes"] for phase in memory_phases),
                   "phases_before_each_internal_reset_and_final": memory_phases},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(selected),
                        "visible_cuda_device": str(selected), "cpu_threads": args.threads,
                        "gpu_memory_limit_gb": args.gpu_memory_limit_gb, "tf32": True,
                        "low_precision_enabled": False,
                        "gpu_state_before_api": initial_gpu_state, "gpu_state_after_api": final_gpu_state},
        "observer": {"calculation_modified": False, "capture_enabled": args.capture_dir is not None,
                     "capture_methods": capture_methods,
                     "captured_geometry_is_independent_reference": False},
        "scope": "Complete default surface API: volume/recon-all identity checks, geometry preparation, "
                 "both MSMSulc estimates, sphere-specific area surfaces, 490-frame projection, CIFTI, "
                 "QC and final output publication. Existing volume and recon-all are prerequisites.",
        "limits": ["Recon-all, prior volume processing, installation/build and module import are excluded.",
                   "This execution report establishes contracts; independent official MSM and geometry/projection "
                   "comparisons must be bound to these inputs separately.",
                   "Private capture overhead is reported alongside the unmodified observed API elapsed time.",
                   "A shared-H100 observation; device load is recorded and other jobs are untouched.",
                   "Post-run hashing and numerical validation are outside the API timer."],
        "privacy": "Public reports contain anonymous shapes, scalars and hashes; paths and source images remain private.",
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete_surface_api": True,
                      "api_observed_seconds": observed_api_seconds,
                      "api_capture_adjusted_seconds": observed_api_seconds - capture_seconds,
                      "capture_seconds": capture_seconds,
                      "msmsulc_seconds": result.timing_seconds["msmsulc_preparation_and_registration"],
                      "outputs": checks["persistent_output_count"],
                      "peak_cuda_allocated_bytes": report["memory"]["peak_cuda_allocated_bytes"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
