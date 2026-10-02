#!/usr/bin/env python3
"""Validate actual public resampler routing in a complete real BIDS run.

Run baseline and candidate in separate processes/source trees and fresh output
directories. The private case JSON uses the existing ``volume`` API kwargs.
Only anonymous roles, content hashes and repository-relative callers are
exported. No original software, downloads, anatomical cache or frame reduction
is used. CUDA execution occurs only when the explicit CLI run is requested.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import functools
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch


def _load_helpers():
    path = Path(__file__).with_name("validate_fnirt_pipeline_lossless.py")
    spec = importlib.util.spec_from_file_location("_fnit_lossless_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _array_record(value):
    import numpy as np

    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _image(value):
    import nibabel as nib

    return nib.load(str(value)) if isinstance(value, (str, Path)) else value


def _image_geometry(value):
    image = _image(value)
    return {"shape": list(image.shape), "dtype": str(image.get_data_dtype()),
            "affine": _array_record(image.affine),
            "zooms": [float(v) for v in image.header.get_zooms()],
            "units": list(image.header.get_xyzt_units())}


def route_role(record, expected_frames):
    """Classify a complete-volume operation using its scientific arguments."""
    shape = record["input"]["shape"]
    interpolation = record["interpolation"]
    chain = record["chain"]
    has_pull = chain["pull"] is not None
    has_motion = chain["motion"] is not None
    if len(shape) == 3 and interpolation == "nearest" and has_pull and not has_motion:
        return "mask_mni"
    if len(shape) != 4 or shape[3] != expected_frames or interpolation != "spline":
        return None
    if record["boundary"] == "periodic" and has_pull and not has_motion:
        return "clean_mni"
    if (record["boundary"] == "grid-constant" and has_motion
            and chain["coordinate_precision"] == "fmriprep"):
        return "preproc_mni" if has_pull else "preproc_t1w"
    return None


def routing_gate(records, *, backend, variant, expected_frames):
    """Reject missing/wrong public routes even when image comparisons pass."""
    expected = ("mask_mni", "clean_mni", "preproc_t1w", "preproc_mni")
    failures = []
    counts = Counter()
    for record in records:
        role = route_role(record, expected_frames)
        if role is None:
            continue
        counts[(record["api"], role)] += 1
        if record["output_mask_present"] != (role == "clean_mni"):
            failures.append(f"{record['api']}:{role}:incorrect_output_mask_contract")
        if role in ("mask_mni", "clean_mni") and record["chain"]["coordinate_precision"] != "float64":
            failures.append(f"{record['api']}:{role}:incorrect_coordinate_precision")
        if role == "mask_mni" and record["boundary"] != "grid-constant":
            failures.append(f"{record['api']}:{role}:incorrect_mask_boundary")
        if not record["caller_chain"]:
            failures.append(f"{record['api']}:{role}:no_production_caller")
        if not any(c["module"].startswith("fnit.fmri.") for c in record["caller_chain"]):
            failures.append(f"{record['api']}:{role}:no_volume_caller")
        returned = record["returned"]
        expected_shape = record["chain"]["reference"]["shape"]
        if role != "mask_mni":
            expected_shape = [*expected_shape, expected_frames]
        if returned is None or returned["shape"] != expected_shape:
            failures.append(f"{record['api']}:{role}:incomplete_return")
        if returned is not None and (returned["dtype"] != "float32"
                or returned["affine"] != record["chain"]["reference"]["affine"]):
            failures.append(f"{record['api']}:{role}:incorrect_return_geometry_or_dtype")
        motion = record["chain"]["motion"]
        if motion is not None:
            if motion["shape"] != [expected_frames, 4, 4] or motion["nonidentity_frames"] == 0:
                failures.append(f"{record['api']}:{role}:missing_real_frame_motion")
        pull = record["chain"]["pull"]
        if pull is not None and not pull["nonzero"]:
            failures.append(f"{record['api']}:{role}:zero_pull")
        if record["api"] != "fnit.fmri.normalization.resample_world" and not record["public_chain"]:
            failures.append(f"{record['api']}:{role}:not_WorldTransformChain")
    if variant == "baseline":
        required = ("fnit.fmri.normalization.resample_world",)
    elif backend == "fnirt":
        required = ("fnit.applywarp.TorchApplyWarp.run_world",
                    "fnit.applywarp.TorchApplyWarp.apply_world")
    else:
        required = ("fnit.synthmorph.apply_transform",)
    for api in required:
        for role in expected:
            if counts[(api, role)] != 1:
                failures.append(f"{api}:{role}:expected_one_actual_call")
    if variant == "candidate":
        forbidden = ("fnit.fmri.normalization.resample_world",
                     "fnit.synthmorph.apply_transform" if backend == "fnirt"
                     else "fnit.applywarp.TorchApplyWarp.run_world",
                     "fnit.applywarp.TorchApplyWarp.apply_world" if backend == "synthmorph"
                     else "fnit.synthmorph.apply_transform")
        for (api, role), count in counts.items():
            if api in forbidden and count:
                failures.append(f"{api}:{role}:wrong_final_route")
    return {"passed": not failures, "failures": failures,
            "actual_call_counts": {api: {role: counts[(api, role)] for role in expected}
                                   for api in sorted({api for api, _ in counts})}}


class RouteSpy:
    """Observe real calls while forwarding the original argument objects."""
    def __init__(self, source_root, helpers):
        self.source_root = Path(source_root).resolve()
        self.helpers = helpers
        self.records = []
        self.registrations = Counter()
        self.metadata_seconds = 0.0
        self._pull_records = {}
        self._source_hashes = {}
        self._stack = ExitStack()
        self._public_chain_type = None

    def _callers(self):
        callers = []
        frame = inspect.currentframe().f_back
        try:
            while frame is not None:
                module = frame.f_globals.get("__name__", "")
                if module.startswith("fnit."):
                    path = Path(frame.f_code.co_filename).resolve()
                    try:
                        relative = path.relative_to(self.source_root).as_posix()
                    except ValueError:
                        raise AssertionError("production caller is outside the requested source tree")
                    if relative not in self._source_hashes:
                        self._source_hashes[relative] = self.helpers.sha256(path)
                    callers.append({"module": module, "function": frame.f_code.co_name,
                                    "line": frame.f_lineno, "source_file": relative,
                                    "source_sha256": self._source_hashes[relative]})
                frame = frame.f_back
        finally:
            del frame
        return callers

    def _chain(self, chain):
        import numpy as np

        pull = None
        if chain.pre_affine_pull_ras is not None:
            key = str(chain.pre_affine_pull_ras)
            if key not in self._pull_records:
                image = _image(chain.pre_affine_pull_ras)
                values = np.asanyarray(image.dataobj)
                self._pull_records[key] = {**_image_geometry(image),
                                          "decoded_values": _array_record(values),
                                          "nonzero": bool(np.any(values != 0))}
            pull = self._pull_records[key]
        motion = None
        if chain.motion_pull_world is not None:
            values = np.asarray(chain.motion_pull_world)
            motion = {**_array_record(values),
                      "nonidentity_frames": int(np.count_nonzero(np.any(
                          values != np.eye(4), axis=(1, 2))))}
        return {"reference": _image_geometry(chain.reference),
                "reference_to_source_world": _array_record(chain.reference_to_source_world),
                "pull": pull, "motion": motion,
                "coordinate_precision": chain.coordinate_precision}

    def _observe(self, original, api, *, legacy=False):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            if legacy:
                source = arguments["source"]
                chain = SimpleNamespace(
                    reference=arguments["reference"],
                    reference_to_source_world=arguments["reference_to_source_world"],
                    pre_affine_pull_ras=arguments["pre_affine_pull_ras"],
                    motion_pull_world=arguments["motion_pull_world"],
                    coordinate_precision=arguments["coordinate_precision"],
                )
            else:
                chain = arguments.get("chain", arguments.get("transformation"))
                if self._public_chain_type is None or not isinstance(chain, self._public_chain_type):
                    return original(*args, **kwargs)
                source = arguments.get("input", arguments.get("image"))
            options = {}
            for name, value in arguments.items():
                if signature.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                    options.update(value)
                else:
                    options[name] = value
            record = {"api": api, "public_chain": not legacy,
                      "input": _image_geometry(source), "chain": self._chain(chain),
                      "interpolation": options.get("interpolation", options.get("method", "spline")),
                      "boundary": options.get("boundary", "grid-constant"),
                      "output_mask_present": options.get("output_mask") is not None,
                      "batch_size": options.get("batch_size", options.get("frame_chunk_size", 8)),
                      "spatial_chunk_size": options.get("spatial_chunk_size", 262144),
                      "caller_chain": self._callers(), "returned": None}
            self.records.append(record)
            self.metadata_seconds += time.perf_counter() - started
            returned = original(*args, **kwargs)
            started = time.perf_counter()
            record["returned"] = _image_geometry(returned)
            self.metadata_seconds += time.perf_counter() - started
            return returned

        return wrapped

    def __enter__(self):
        from fnit.fmri import end_to_end, normalization
        from fnit.applywarp import TorchApplyWarp
        from fnit.synthmorph import pipeline as morph

        try:
            from fnit._world_resampling import WorldTransformChain
            self._public_chain_type = WorldTransformChain
        except ImportError:
            pass  # The frozen baseline predates the public chain.
        original = normalization.resample_world
        wrapped = self._observe(original, "fnit.fmri.normalization.resample_world", legacy=True)
        self._stack.enter_context(patch.object(normalization, "resample_world", wrapped))
        if getattr(end_to_end, "resample_world", None) is original:
            self._stack.enter_context(patch.object(end_to_end, "resample_world", wrapped))
        for method in ("run_world", "apply_world"):
            if hasattr(TorchApplyWarp, method):
                original = getattr(TorchApplyWarp, method)
                wrapped = self._observe(original, f"fnit.applywarp.TorchApplyWarp.{method}")
                self._stack.enter_context(patch.object(TorchApplyWarp, method, wrapped))
        original = morph.apply_transform
        wrapped = self._observe(original, "fnit.synthmorph.apply_transform")
        self._stack.enter_context(patch.object(morph, "apply_transform", wrapped))
        # Public imports and dispatcher-bound aliases must point at the same spy.
        import fnit.synthmorph as morph_public
        for module in (morph_public, end_to_end, normalization):
            if getattr(module, "apply_transform", None) is original:
                self._stack.enter_context(patch.object(module, "apply_transform", wrapped))
        original = morph.SynthMorph.__call__

        @functools.wraps(original)
        def registration(*args, **kwargs):
            self.registrations["synthmorph"] += 1
            return original(*args, **kwargs)

        self._stack.enter_context(patch.object(morph.SynthMorph, "__call__", registration))
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


class WholeRunMemory:
    """Preserve actual peaks when a mature component resets its own counter."""
    def __init__(self, cuda):
        self.cuda = cuda
        self.allocated = 0
        self.reserved = 0
        self.reset_calls = 0
        self.metadata_seconds = 0.0

    def collect(self, *, inside_api=False):
        started = time.perf_counter()
        self.allocated = max(self.allocated, self.cuda.max_memory_allocated())
        self.reserved = max(self.reserved, self.cuda.max_memory_reserved())
        if inside_api:
            self.metadata_seconds += time.perf_counter() - started

    def __enter__(self):
        self.collect()
        original = self.cuda.reset_peak_memory_stats

        @functools.wraps(original)
        def reset(*args, **kwargs):
            self.collect(inside_api=True)
            self.reset_calls += 1
            return original(*args, **kwargs)

        self._patch = patch.object(self.cuda, "reset_peak_memory_stats", reset)
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        self.collect()
        return self._patch.__exit__(*exc)


def compare_trees(first, second, *, backend, helpers):
    comparison = helpers.compare_trees(first, second)
    if backend == "synthmorph":
        # SynthMorph has no FNIRT solver trace. All other unmatched science
        # outputs, all image bits/headers, matrices and scalar texts stay gated.
        comparison["missing_outputs"] = [value for value in comparison["missing_outputs"]
                                          if value != "fnirt_solver_trace_missing"]
        comparison["all_equal"] = bool(comparison["outputs"]) and not comparison["missing_outputs"] and all(
            all(value for key, value in record.items() if key.endswith("equal"))
            for record in comparison["outputs"].values())
    return comparison


def verified_inputs(case, backend, helpers):
    from fnit.weights import WEIGHT_FILES, resolve_weights

    summaries = helpers.input_summaries("volume", case)
    names = {"synthstrip_weights": ("synthstrip.1.pt", case.get("synthstrip_weights"))}
    if backend == "synthmorph":
        explicit = case.get("synthmorph_weights")
        if isinstance(explicit, dict):
            explicit = explicit.get("deform")
        names["synthmorph_deform_weights"] = ("synthmorph.deform.3.h5", explicit)
    for role, (name, explicit) in names.items():
        record = helpers.input_file_summary(resolve_weights(name, explicit))
        _, size, digest = WEIGHT_FILES[name]
        if (record["size_bytes"], record["sha256"]) != (size, digest):
            raise AssertionError(f"{role} differs from the fixed resource manifest")
        summaries[role] = {**record, "verified_against_resource_manifest": True}
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("fnirt", "synthmorph"), required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--expected-frames", type=int, default=490)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit-bytes", type=int, default=20_000_000_000)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.expected_frames < 1 or not 0 < args.memory_limit_bytes <= 20_000_000_000:
        parser.error("positive frames/threads and a memory limit at most 20 GB are required")
    import fnit
    import torch

    package = Path(fnit.__file__).resolve().parent
    if package != args.source_root.resolve() / "src/fnit":
        raise AssertionError("imported FNIT does not match --source-root")
    if args.output_dir.exists():
        raise FileExistsError("use a new output directory; no cached execution")
    helpers = _load_helpers()
    case = dict(json.loads(args.case_json.read_text())["volume"])
    for reserved in ("registration_backend", "derivatives_root", "device", "reuse_anatomical"):
        if reserved in case:
            raise ValueError(f"case JSON must not override {reserved}")
    inputs = verified_inputs(case, args.backend, helpers)
    if inputs["bold"]["shape"][3] != args.expected_frames:
        raise AssertionError("the selected raw BIDS input does not contain all expected frames")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(min(
        1, args.memory_limit_bytes / torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    from fnit.fmri import fMRIVolume_pipeline

    with WholeRunMemory(torch.cuda) as memory, helpers.Phases(args.output_dir / "captured") as phases, RouteSpy(args.source_root, helpers) as spy:
        started = time.perf_counter()
        result = fMRIVolume_pipeline(
            **case, derivatives_root=args.output_dir / "outputs",
            registration_backend=args.backend, device="cuda:0", reuse_anatomical=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    gate = routing_gate(spy.records, backend=args.backend, variant=args.variant,
                        expected_frames=args.expected_frames)
    expected_fnirt, expected_morph = ((1, 0) if args.backend == "fnirt" else (0, 1))
    registration_counts = {"fnirt": phases.phases["fnirt"]["calls"],
                           "synthmorph": spy.registrations["synthmorph"]}
    if registration_counts != {"fnirt": expected_fnirt, "synthmorph": expected_morph}:
        gate["passed"] = False
        gate["failures"].append("requested_registration_backend_was_not_actually_executed_once")
    metadata = json.loads(result.metadata.read_text())
    actual_report = metadata["FNIT"]["Report"]
    configuration = actual_report["configuration"]
    cache = actual_report["anatomical_cache"]
    if cache["reused"]:
        gate["passed"] = False
        gate["failures"].append("anatomical_cache_reused")
    images = {role: helpers.image_check(getattr(result, role))
              for role in ("clean_native", "clean_mni", "mask_mni", "t1_brain", "preproc_t1w", "preproc_mni")}
    if any(images[role]["shape"][-1] != args.expected_frames
           for role in ("clean_native", "clean_mni", "preproc_t1w", "preproc_mni")):
        gate["passed"] = False
        gate["failures"].append("saved_output_is_missing_frames")
    peak = memory.allocated
    if peak >= args.memory_limit_bytes:
        gate["passed"] = False
        gate["failures"].append("peak_allocated_memory_is_not_strictly_below_limit")
    stages, attribution = helpers.corrected_pipeline_timings("volume", result.timing_seconds,
                                                             phases.capture_seconds)
    scientific_options = {key: configuration[key] for key in (
        "registration_backend", "fnirt_config", "ica_n_components", "ica_max_iter",
        "aroma_mode", "regress_wm", "regress_csf", "regress_motion", "motion_model",
        "bandpass", "global_signal", "highpass_cutoff_seconds", "slice_timing",
        "slice_time_reference", "batch_size", "motion_iterations", "n_splits",
        "random_state", "bbr_execution", "fnirt_execution", "mni_interpolation",
        "preproc_interpolation", "preproc_coordinate_precision",
    ) if key in configuration}
    report = {"schema_version": 1, "backend": args.backend, "variant": args.variant,
              "real_data": True, "expected_frames": args.expected_frames,
              "input_summaries": inputs, "input_provenance_collection": "before API timer",
              "source_sha256": helpers.source_hashes(args.source_root),
              "validation_tool_sha256": helpers.sha256(Path(__file__)),
              "comparison_helper_sha256": helpers.sha256(Path(helpers.__file__)),
              "actual_public_routes": spy.records, "routing_gate": gate,
              "actual_registration_counts": registration_counts,
              "scientific_options": scientific_options,
              "anatomical_cache_reused": cache["reused"], "outputs": images,
              "timing_seconds": {"api_including_save_excluding_validation_overhead":
                                 elapsed - phases.capture_seconds - spy.metadata_seconds - memory.metadata_seconds,
                                 "validation_fnirt_capture": phases.capture_seconds,
                                 "validation_route_metadata": spy.metadata_seconds,
                                 "validation_peak_metadata": memory.metadata_seconds,
                                 "pipeline_stages_excluding_fnirt_capture": stages,
                                 "capture_attribution": attribution,
                                 "scope": "full public raw-BIDS API and saves; CUDA initialization, input/source audit and output comparisons excluded; route metadata overhead excluded only from API wall; stage clocks remain nested and include route metadata"},
              "memory": {"peak_cuda_allocated_bytes": peak,
                         "peak_cuda_reserved_bytes": memory.reserved,
                         "component_peak_counter_resets": memory.reset_calls,
                         "allocator_limit_bytes": args.memory_limit_bytes},
              "conditions": {"threads": args.threads, "tf32": True, "float16": False,
                             "original_software_called": False, "shared_gpu": True}}
    comparison = None
    if args.compare_to:
        previous = json.loads((args.compare_to / "report.safe.json").read_text())
        if (previous["backend"] != args.backend or previous["input_summaries"] != inputs
                or previous["scientific_options"] != scientific_options):
            raise AssertionError("baseline and candidate must have the same backend and content-identified inputs")
        comparison = compare_trees(args.compare_to, args.output_dir, backend=args.backend, helpers=helpers)
        (args.output_dir / "comparison.safe.json").write_text(json.dumps(comparison, indent=2) + "\n")
        report["strict_scientific_comparison_passed"] = comparison["all_equal"]
    (args.output_dir / "report.safe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"backend": args.backend, "variant": args.variant,
                      "routing_passed": gate["passed"],
                      "scientific_equal": None if comparison is None else comparison["all_equal"],
                      "actual_call_counts": gate["actual_call_counts"]}), flush=True)
    if not gate["passed"] or (comparison is not None and not comparison["all_equal"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
