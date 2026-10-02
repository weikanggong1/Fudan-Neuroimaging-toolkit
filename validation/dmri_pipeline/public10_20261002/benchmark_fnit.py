"""Benchmark one fresh public-data case through the actual FNIT GPU pipeline.

Raw inputs are normalized to FNIT's AP/PA directory contract before this process.
This driver never imports or reads reference-software outputs. The API timer starts
immediately before ``DMRIPipeline.run``; provenance hashing and output checks run
after that timer. Use an external process timer to include Python startup and all
post-run provenance work. Every numerical operation remains the production call.
"""
from __future__ import annotations

import time

SCRIPT_STARTED_MONOTONIC = time.perf_counter()

import argparse
from contextlib import AbstractContextManager
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys


MAP_NAMES = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_provenance(path):
    path = Path(path)
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def public_value(value):
    """Retain scalar QC while removing any filesystem paths in third-party QC."""
    if isinstance(value, dict):
        return {str(key): public_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_value(item) for item in value]
    if isinstance(value, Path):
        return "[private filesystem path]"
    if isinstance(value, str):
        # Numerical QC sometimes describes a source operation using slashes;
        # only actual absolute Unix/Windows filesystem paths are redacted.
        if value.startswith(("/", "~\\", "~/")) or re.match(r"^[A-Za-z]:[\\/]", value):
            return "[private filesystem path]"
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


class StageRecorder(AbstractContextManager):
    """Observe synchronous public calls and preserve peaks across native resets.

    No extra counter reset is introduced inside the pipeline. A child may reset
    CUDA counters before an ancestor returns, so the recorder saves each counter
    window first. Per-call peaks are conservative observations of those windows,
    not isolated incremental allocations; the whole-run maximum is exact for the
    observed PyTorch allocator counters. CUDA context/other processes are excluded.
    """

    def __init__(self, torch_module, device, emit=True):
        self.torch = torch_module
        self.device = device
        self.emit = emit
        self.events = []
        self.stack = []
        self.checkpoints = []
        self.patches = []
        self.peak_allocated = 0
        self.peak_reserved = 0
        self._reset_original = None

    def _snapshot(self, reason):
        allocated = int(self.torch.cuda.max_memory_allocated(self.device))
        reserved = int(self.torch.cuda.max_memory_reserved(self.device))
        current = int(self.torch.cuda.memory_allocated(self.device))
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)
        for row in self.stack:
            row["peak_allocated_bytes"] = max(row["peak_allocated_bytes"], allocated)
            row["peak_reserved_bytes"] = max(row["peak_reserved_bytes"], reserved)
        return allocated, reserved, current

    def _target_reset(self, device):
        if device is None:
            index = self.torch.cuda.current_device()
        else:
            parsed = self.torch.device(device) if not isinstance(device, int) else None
            index = device if isinstance(device, int) else parsed.index
            if index is None:
                index = self.torch.cuda.current_device()
        selected = self.device.index
        if selected is None:
            selected = self.torch.cuda.current_device()
        return index == selected

    def __enter__(self):
        self._reset_original = self.torch.cuda.reset_peak_memory_stats

        @functools.wraps(self._reset_original)
        def reset(device=None):
            if self._target_reset(device):
                self.torch.cuda.synchronize(self.device)
                allocated, reserved, current = self._snapshot("before_component_reset")
                self.checkpoints.append({
                    "event": "before_component_peak_reset",
                    "active_stages": [row["stage"] for row in self.stack],
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "current_allocated_bytes": current,
                })
            return self._reset_original(device)

        self.torch.cuda.reset_peak_memory_stats = reset
        return self

    def patch(self, target, name, label):
        original = getattr(target, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            stage = label(*args, **kwargs) if callable(label) else label
            if stage is None:
                return original(*args, **kwargs)
            self.torch.cuda.synchronize(self.device)
            parent = self.stack[-1]["stage"] if self.stack else None
            row = {"stage": stage, "parent_stage": parent,
                   "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
            self.stack.append(row)
            self._snapshot("stage_entry")
            started = time.perf_counter()
            if self.emit:
                print(json.dumps({"event": "stage_start", "stage": stage,
                                  "parent_stage": parent}), flush=True)
            try:
                result = original(*args, **kwargs)
            except BaseException:
                row["status"] = "failed"
                raise
            else:
                row["status"] = "complete"
                return result
            finally:
                self.torch.cuda.synchronize(self.device)
                self._snapshot("stage_exit")
                row["seconds"] = time.perf_counter() - started
                row["event"] = "stage_end"
                self.events.append(row)
                self.stack.pop()
                if self.emit:
                    print(json.dumps(row), flush=True)

        setattr(target, name, measured)
        self.patches.append((target, name, original))

    def __exit__(self, exception_type, exception, traceback):
        self.torch.cuda.synchronize(self.device)
        self._snapshot("run_exit")
        for target, name, original in reversed(self.patches):
            setattr(target, name, original)
        self.torch.cuda.reset_peak_memory_stats = self._reset_original
        return False


def install_instrumentation(recorder, pipeline, backend, t1, output_dir):
    import fnit.topup.ukb as topup_preparation
    import fnit.eddy.ukb as eddy_preparation
    from fnit.synthstrip import SynthStrip
    from fnit.mmorf import MMORFResult, MMORFWarpPlan, TorchMMORF
    from fnit.fnirt import TorchFNIRT
    from fnit._nib import FNITNifti1Image
    from fnit.dmri_pipeline.tbss import TBSSResult
    from fnit.applywarp import ApplyWarpPlan

    recorder.patch(pipeline, "run_ukb_topup", "topup_and_b0_selection")
    recorder.patch(topup_preparation, "prepare_ukb_topup", "b0_selection_and_pair_save")
    recorder.patch(topup_preparation.TorchTOPUP, "run", "topup_estimation_and_save")
    recorder.patch(pipeline, "prepare_ukb_eddy", "eddy_preparation")
    recorder.patch(pipeline, "_prepare_ap_only", "ap_only_eddy_preparation")
    recorder.patch(eddy_preparation, "_get_synthstrip", "b0_synthstrip_model_resolve_verify_load")
    recorder.patch(pipeline, "_get_synthstrip", "t1_synthstrip_model_resolve_verify_load")
    recorder.patch(SynthStrip, "__init__", "synthstrip_model_constructor")

    def strip_label(model, image, *args, **kwargs):
        return ("t1_synthstrip_preprocessing_inference_native_grid"
                if isinstance(image, (str, os.PathLike)) and t1 is not None
                and Path(image).resolve() == t1.resolve()
                else "b0_synthstrip_preprocessing_inference_native_grid")

    recorder.patch(SynthStrip, "__call__", strip_label)
    recorder.patch(pipeline, "select_shell", "dti_shell_read_selection_and_save")
    for label, cls in (("eddy", pipeline.TorchEDDY),
                       ("dtifit", pipeline.TorchDTIFIT),
                       ("noddi", pipeline.TorchAMICONODDI),
                       ("tbss", pipeline.TorchTBSS)):
        recorder.patch(cls, "__init__", label + "_constructor")
        recorder.patch(cls, "run", label + "_fit_or_registration_and_save")
    recorder.patch(TBSSResult, "save", "tbss_all_outputs_save")
    recorder.patch(TorchFNIRT, "__call__", "tbss_fnirt_registration")

    affine_calls = 0

    def affine_label(*args, **kwargs):
        nonlocal affine_calls
        affine_calls += 1
        if backend == "tbss":
            return "tbss_weighted_fa_flirt"
        return ({1: "mmorf_t1_flirt", 2: "mmorf_fa_flirt"}
                .get(affine_calls, "mmorf_additional_flirt"))

    recorder.patch(pipeline.TorchFLIRT, "__init__", "flirt_constructor")
    recorder.patch(pipeline.TorchFLIRT, "__call__", affine_label)
    recorder.patch(pipeline, "run_mmorf", "mmorf_constructor_registration_and_save")
    recorder.patch(TorchMMORF, "__init__", "mmorf_constructor")
    recorder.patch(TorchMMORF, "run", "mmorf_registration_and_save")
    recorder.patch(MMORFResult, "save", "mmorf_registration_outputs_save")
    recorder.patch(pipeline, "prepare_mmorf_warp", "mmorf_nine_map_sampling_plan")
    recorder.patch(MMORFWarpPlan, "apply", "mmorf_one_map_propagation")
    recorder.patch(ApplyWarpPlan, "apply", "tbss_one_map_propagation")

    def standard_save_label(image, filename, *args, **kwargs):
        path = Path(filename)
        if backend == "mmorf" and path.parent.resolve() == (
                output_dir / "registration" / "standard").resolve():
            return "mmorf_one_standard_map_save"
        return None

    def t1_save_label(image, filename, *args, **kwargs):
        if backend == "mmorf" and Path(filename).name in (
                "t1_brain.nii.gz", "t1_brain_mask.nii.gz"):
            return "t1_synthstrip_brain_or_mask_save"
        return None

    recorder.patch(pipeline.nib, "save", standard_save_label)
    recorder.patch(FNITNifti1Image, "save", t1_save_label)


def input_provenance(args, nib):
    paths = {f"raw_{name}": args.raw_dir / name for name in
             ("AP.nii.gz", "AP.bval", "AP.bvec", "AP.json",
              "PA.nii.gz", "PA.bval", "PA.bvec", "PA.json")}
    paths["t1"] = args.t1
    paths["fa_template"] = args.fa_template
    paths["synthstrip_weights"] = args.synthstrip_weights
    if args.backend == "tbss":
        paths["fa_skeleton"] = args.fa_skeleton
    else:
        paths.update(t1_template=args.t1_template, tensor_template=args.tensor_template)
    result = {}
    for role, path in paths.items():
        if path is None or not path.is_file():
            continue
        entry = file_provenance(path)
        if str(path).endswith((".nii", ".nii.gz")):
            image = nib.load(str(path))
            entry.update(shape=list(image.shape), affine=image.affine.tolist(),
                         voxel_sizes_mm=list(map(float, image.header.get_zooms()[:3])),
                         stored_dtype=str(image.get_data_dtype()))
        elif role in ("raw_AP.json", "raw_PA.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            entry["acquisition_metadata"] = {
                name: metadata[name] for name in
                ("PhaseEncodingDirection", "TotalReadoutTime", "EffectiveEchoSpacing")
                if name in metadata}
        result[role] = entry
    return result


def output_contract(result, args, nib, np):
    rows = {}
    native_reference = nib.load(str(args.raw_dir / "AP.nii.gz"))
    standard_reference = nib.load(str(
        args.fa_template if args.backend == "tbss" else args.t1_template))
    spaces = {"native": (result.native_maps, native_reference),
              "standard": (result.standard_maps, standard_reference)}
    if args.backend == "tbss":
        spaces["skeleton"] = ({name: nib.load(str(args.output_dir / "registration" /
                                                  "skeleton" / f"{name}.nii.gz"))
                              for name in MAP_NAMES}, standard_reference)
    for space, (images, reference) in spaces.items():
        if set(images) != set(MAP_NAMES):
            raise ValueError(f"{space} output map names are incomplete")
        for name in MAP_NAMES:
            image = images[name]
            values = np.asanyarray(image.dataobj)
            row = {"shape": list(values.shape), "stored_dtype": str(image.get_data_dtype()),
                   "all_finite": bool(np.isfinite(values).all()),
                   "shape_matches_reference": values.shape == reference.shape[:3],
                   "affine_matches_reference": bool(np.allclose(
                       image.affine, reference.affine, atol=1e-5, rtol=0))}
            rows[f"{space}/{name}"] = row
    passed = all(row["all_finite"] and row["shape_matches_reference"]
                 and row["affine_matches_reference"] for row in rows.values())
    common_files = ("dmri_pipeline_report.json", "eddy/data.nii.gz",
                    "eddy/nodif_brain_mask.nii.gz", "eddy/data.eddy_rotated_bvecs",
                    "native/dti_tensor.nii.gz")
    registration_files = (
        ("registration/dti_FA_to_MNI_affine.mat", "registration/dti_FA_to_MNI_warp.nii.gz")
        if args.backend == "tbss" else
        ("registration/t1_brain.nii.gz", "registration/t1_brain_mask.nii.gz",
         "registration/t1_to_MNI_affine.mat", "registration/dti_FA_to_MNI_affine.mat",
         "registration/mmorf_warp.nii.gz", "registration/mmorf_jacobian.nii.gz",
         "registration/mmorf_warped_scalar.nii.gz", "registration/mmorf_warped_tensor.nii.gz",
         "registration/mmorf_report.json"))
    required = {name: (args.output_dir / name).is_file()
                for name in (*common_files, *registration_files)}
    return {"passed": passed and all(required.values()), "maps_checked": len(rows),
            "maps": rows, "required_files_exist": required}


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True, help="anonymous identifier, e.g. case01")
    parser.add_argument("--backend", required=True, choices=("tbss", "mmorf"))
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--t1", type=Path, required=True,
                        help="paired original T1; recorded for both branches, used by MMORF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fa-template", type=Path, required=True)
    parser.add_argument("--fa-skeleton", type=Path)
    parser.add_argument("--t1-template", type=Path)
    parser.add_argument("--tensor-template", type=Path)
    parser.add_argument("--synthstrip-weights", type=Path, required=True)
    parser.add_argument("--source-commit", required=True,
                        help="actual supplied source Git commit, not inferred by this driver")
    parser.add_argument("--device", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--eddy-gp-seed", type=int, default=12345)
    parser.add_argument("--memory-limit-bytes", type=int, default=20_000_000_000)
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if not re.fullmatch(r"case[0-9]{2,3}", args.case_id):
        parser.error("--case-id must be an anonymous caseNN identifier")
    if args.threads < 1 or args.memory_limit_bytes < 1:
        parser.error("threads and memory-limit-bytes must be positive")
    required = {"fa-skeleton": args.fa_skeleton} if args.backend == "tbss" else {
        "t1-template": args.t1_template, "tensor-template": args.tensor_template}
    if any(path is None for path in required.values()):
        parser.error("missing branch resource: " + ", ".join(
            name for name, path in required.items() if path is None))
    if args.report.exists():
        parser.error("report already exists; use a new independent benchmark directory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty; benchmark result reuse is forbidden")

    import torch
    import numpy as np
    import nibabel as nib
    import fnit
    import fnit.dmri_pipeline.pipeline as pipeline

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this benchmark requires an available explicit CUDA device")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if args.memory_limit_bytes > properties.total_memory:
        parser.error("requested allocator cap exceeds this GPU's total memory")
    torch.cuda.set_per_process_memory_fraction(
        args.memory_limit_bytes / properties.total_memory, device)
    torch.cuda.reset_peak_memory_stats(device)
    recorder = StageRecorder(torch, device)
    result = None
    error = None
    runner = pipeline.DMRIPipeline(
        device=device, registration_backend=args.backend,
        synthstrip_weights=args.synthstrip_weights, noddi_fit_method="amico",
        bvec_source="rotated", eddy_gp_seed=args.eddy_gp_seed,
    )
    # Install wrappers before the timer; they only call unchanged production
    # functions. Neither raw images nor resource bytes are opened here.
    with recorder:
        install_instrumentation(recorder, pipeline, args.backend, args.t1, args.output_dir)
        torch.cuda.synchronize(device)
        api_started = time.perf_counter()
        startup_seconds = api_started - SCRIPT_STARTED_MONOTONIC
        try:
            result = runner.run(
                args.raw_dir, args.output_dir, fa_template=args.fa_template,
                fa_skeleton=args.fa_skeleton, t1=args.t1,
                t1_template=args.t1_template, tensor_template=args.tensor_template,
                overwrite=False,
            )
        except BaseException as exception:
            error = exception
        finally:
            torch.cuda.synchronize(device)
            api_seconds = time.perf_counter() - api_started
    post_started = time.perf_counter()
    source_root = Path(fnit.__file__).parent
    source_hashes = {str(path.relative_to(source_root)): sha256_file(path)
                     for path in sorted(source_root.rglob("*.py"))}
    report = {
        "schema": "fnit_public10_single_case_v1", "case_id": args.case_id,
        "status": "failed" if error else "complete", "registration_backend": args.backend,
        "source_commit_supplied": args.source_commit,
        "source_python_sha256": source_hashes,
        "benchmark_driver_sha256": sha256_file(__file__),
        "input_and_resource_provenance": input_provenance(args, nib),
        "parameters": {"device": str(device), "threads": args.threads,
                       "eddy_gp_seed": args.eddy_gp_seed, "bvec_source": "rotated",
                       "noddi_fit_method": "amico", "registration_configuration": "production_defaults",
                       "memory_limit_bytes": args.memory_limit_bytes, "overwrite": False,
                       "result_reuse": False, "reference_output_reuse": False},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "nibabel": nib.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(), "gpu_name": properties.name,
                    "gpu_total_memory_bytes": properties.total_memory,
                    "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
                    "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "torch_num_threads": torch.get_num_threads(),
                    "torch_interop_threads": torch.get_num_interop_threads(),
                    "cuda_matmul_tf32_after_run": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32_after_run": torch.backends.cudnn.allow_tf32,
                    "autocast_enabled_after_run": torch.is_autocast_enabled(),
                    "thread_environment": {name: os.environ.get(name) for name in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "timing": {"startup_before_api_seconds": startup_seconds,
                   "api_wall_seconds": api_seconds,
                   "full_process_wall_seconds": None,
                   "full_process_scope": "external launcher timer required: interpreter startup, imports, "
                                         "configuration, API, provenance hashing, output gates and JSON save",
                   "api_scope": "actual DMRIPipeline.run call from raw AP/PA to complete production outputs "
                                "and QC; synchronized before/after; excludes startup and post-run hashing"},
        "stage_events": recorder.events,
        "timing_tree_note": "nested child calls are included in parent calls; do not sum both. "
                            "qc.timings_seconds gives the five complete public pipeline stages; "
                            "dtifit includes shell reading/selection/save; registration includes T1 strip, "
                            "two affines, MMORF construction/full optimization/output save and nine map saves.",
        "memory": {"peak_allocated_bytes": recorder.peak_allocated,
                   "peak_reserved_bytes": recorder.peak_reserved,
                   "scope": "exact maximum observed process PyTorch allocator counter before every "
                            "component reset and public call exit; excludes CUDA context and other processes. "
                            "Per-call event peaks conservatively include existing counter history; they "
                            "are not isolated incremental stage allocations.",
                   "before_reset_checkpoints": recorder.checkpoints},
        "qc": public_value(result.qc) if result is not None else None,
        "failure": {"exception_type": type(error).__name__,
                    "last_completed_or_failed_stage": recorder.events[-1]["stage"] if recorder.events else None}
                   if error else None,
    }
    if result is not None:
        report["output_contract"] = output_contract(result, args, nib, np)
        if not report["output_contract"]["passed"]:
            report["status"] = "output_contract_failed"
    report["timing"]["post_api_provenance_and_output_gates_seconds"] = time.perf_counter() - post_started
    report["timing"]["driver_wall_before_report_save_seconds"] = time.perf_counter() - SCRIPT_STARTED_MONOTONIC
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(public_value(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete" if report["status"] == "complete" else "failed",
                      "case_id": args.case_id, "backend": args.backend,
                      "api_wall_seconds": api_seconds}), flush=True)
    if error:
        raise error
    if report["status"] != "complete":
        raise RuntimeError("benchmark output contract failed")
    return report


if __name__ == "__main__":
    main()
