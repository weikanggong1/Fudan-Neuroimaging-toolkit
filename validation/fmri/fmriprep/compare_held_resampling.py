"""固定 FNIT 变换、同一全帧 minimal BOLD 的组合和单次插值对照。

reference 只在固定 fMRIPrep 容器调用实际安装的 resample_image；candidate
只调用 FNIT。此控制不读取或伪造 Nipype result，不验证变换估计。私密 inputs
manifest 可含路径；公开报告只保存数值、SHA 和经过选择的阶段来源。
"""

import argparse
from contextlib import nullcontext
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import time

import nibabel as nib
import numpy as np

from compare_resampling import compare_pair, image_checks, mapped_path, save_json, sha256


def affine_file(path):
    value = np.loadtxt(path, dtype=np.float64)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], rtol=0, atol=1e-8)):
        raise ValueError("held reference-to-source world matrix must be a finite affine")
    return value


def reference(args):
    import nitransforms as nt
    import fmriprep.interfaces.resampling as official

    if (importlib.metadata.version("fmriprep") != args.reference_version
            or importlib.metadata.version("nitransforms") != "25.1.0"):
        raise ValueError("installed reference versions do not match the fixed container")
    root = args.private_output.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root / "manifest.private.json").exists():
        raise FileExistsError("held-transform reference already exists")
    inputs = json.loads(args.inputs.read_text())
    captured = json.loads(args.capture_report.read_text())
    provenance = json.loads(args.source_provenance.read_text())
    source_path = Path(inputs["minimal_bold"])
    raw_path = Path(inputs["raw_bold"])
    target_path = Path(inputs["mni_reference" if args.space == "MNI152NLin6Asym" else "t1_reference"])
    pipeline_path = Path(inputs["preproc_mni" if args.space == "MNI152NLin6Asym" else "preproc_t1w"])
    raw_json = Path(inputs["raw_bold_json"])
    tr = json.loads(raw_json.read_text()).get("RepetitionTime")
    if not isinstance(tr, (int, float)) or not math.isfinite(tr) or tr <= 0:
        raise ValueError("raw BIDS sidecar must define positive finite TR seconds")
    source, source_record = image_checks(source_path, frames=args.frames,
                                         allow_unknown_time_unit=True, expected_tr_seconds=tr)
    raw, raw_record = image_checks(raw_path, frames=args.frames,
                                   allow_unknown_time_unit=True, expected_tr_seconds=tr)
    target_input = nib.load(str(target_path))
    target_origin = {"kind": "retained_3d_reference", "source_image_sha256": sha256(target_path),
                     "source_image_shape": list(target_input.shape)}
    if target_input.ndim == 4:
        # This explicit held-candidate mode may retain the completed output
        # grid rather than its cleaned-up temporary 3D reference. Reference
        # intensities are unused by resample_image: create a zero-valued 3D
        # grid and record its origin instead of pretending it is an official
        # workflow reference file. The actual-node runner remains unchanged.
        target_origin["kind"] = "3d_grid_from_completed_fnit_preproc"
        target_path = root / "held_target_grid.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(target_input.shape[:3], dtype=np.float32),
                                 target_input.affine), str(target_path))
    elif target_input.ndim != 3:
        raise ValueError("held target must be a 3D reference or completed 4D preproc grid")
    target, target_record = image_checks(target_path)
    _, pipeline_record = image_checks(pipeline_path, frames=args.frames, target=target,
                                       float32=True, expected_tr_seconds=tr)
    if (target_origin["kind"] == "3d_grid_from_completed_fnit_preproc"
            and target_origin["source_image_sha256"] != pipeline_record["sha256"]):
        raise ValueError("a reconstructed target grid must originate from this completed preproc file")
    check_key = "preproc_mni" if args.space == "MNI152NLin6Asym" else "preproc_t1w"
    if (captured.get("stage") != "volume"
            or captured["data"]["bold_shape"][3] != args.frames
            or captured["input_sha256"]["bold"] != raw_record["sha256"]
            or captured["checks"][check_key]["sha256"] != pipeline_record["sha256"]):
        raise ValueError("held inputs do not match the completed candidate volume report")
    minimal_hash = provenance.get("minimal_sha256", provenance.get("source_sha256"))
    if (minimal_hash != source_record["sha256"]
            or provenance.get("raw_sha256") != raw_record["sha256"]
            or provenance.get("frames") != args.frames):
        raise ValueError("minimal-source provenance does not match these full-frame files")
    if provenance.get("kind") == "recreated_slice_timing_stage":
        if (provenance.get("slice_timing_corrected") is not True
                or provenance.get("original_minimal_retained") is not False
                or provenance.get("slice_timing_helper_sha256") !=
                   captured["source_sha256"]["src/fnit/fmri/slice_timing.py"]):
            raise ValueError("recreated STC source must identify the completed candidate's exact helper")
        parameters = provenance.get("parameters", {})
        expected_fraction = captured["algorithm"]["configuration"]["slice_time_reference"]
        if (parameters.get("reference_fraction") != expected_fraction
                or parameters.get("ignore") != 0):
            raise ValueError("recreated STC parameters must match the completed candidate and preserve every frame")
    elif provenance.get("kind") == "raw":
        if source_record["sha256"] != raw_record["sha256"] or provenance.get("slice_timing_corrected") is not False:
            raise ValueError("raw minimal source must be the same uncorrected raw file")
    else:
        raise ValueError("unsupported or unidentified minimal-source provenance")
    if (inputs.get("interpolation") not in ("spline", "cubic-bspline")
            or inputs.get("boundary") != "grid-constant"):
        raise ValueError("held capture must use spline/grid-constant")
    artifacts = {}
    for key, name in (("reference_to_source_world", "held_affine.txt"),
                      ("motion_pull_world", "held_motion.npy")):
        shutil.copyfile(inputs[key], root / name)
        artifacts[key] = name
    affine = affine_file(root / artifacts["reference_to_source_world"])
    motion = np.load(root / artifacts["motion_pull_world"], allow_pickle=False)
    if (motion.shape != (args.frames, 4, 4) or not np.isfinite(motion).all()
            or not np.allclose(motion[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-8)
            or sha256(root / artifacts["motion_pull_world"]) != captured["checks"]["motion_pull_sha256"]):
        raise ValueError("held HMC must match every frame in the completed candidate")
    common = []
    pull_record = None
    target_world = nt.base.SpatialReference.factory(target).ndcoords
    fnit_common = target_world.copy()
    if args.space == "MNI152NLin6Asym":
        name = "held_mni_pull_ras.nii.gz"
        shutil.copyfile(inputs["mni_pull_ras"], root / name)
        artifacts["mni_pull_ras"] = name
        field = nib.load(str(root / name))
        values = np.asarray(field.dataobj)
        if (values.shape != (*target.shape, 3) or not np.isfinite(values).all()
                or not np.allclose(field.affine, target.affine, rtol=0, atol=1e-5)
                or sha256(root / name) != captured["checks"]["mni_pull_sha256"]):
            raise ValueError("held MNI pull must match the completed candidate and target grid")
        common.append(nt.DenseFieldTransform(field, is_deltas=True))
        fnit_common += values.reshape(-1, 3)
        pull_record = {"sha256": sha256(root / name), "dtype": str(values.dtype),
                       "shape": list(values.shape), "all_finite": True}
    fnit_common = nib.affines.apply_affine(affine, fnit_common)
    common.append(nt.Affine(affine))
    common_chain = nt.TransformChain(common)
    target_f4 = target_world.astype("f4")
    official_common = common_chain.map(target_f4)
    difference = official_common - fnit_common
    chain = nt.TransformChain(common + [nt.linear.LinearTransformsMapping(motion)])
    output_path = root / "held_reference.nii.gz"
    started = time.perf_counter()
    output = official.resample_image(
        source=source, target=target, transforms=chain, fieldmap=None, pe_info=None,
        jacobian=False, nthreads=args.threads, output_dtype="f4", order=3,
        mode="grid-constant", cval=0., prefilter=True)
    nib.save(output, str(output_path))
    wall = time.perf_counter() - started
    _, checked = image_checks(output_path, frames=args.frames, target=target,
                               source=source, float32=True, allow_unknown_time_unit=True,
                               expected_tr_seconds=tr)
    artifacts["reference"] = output_path.name
    selected_provenance = {key: provenance[key] for key in
                          ("kind", "raw_sha256", "frames",
                           "slice_timing_corrected", "original_minimal_retained",
                           "slice_timing_helper_sha256", "source_revision") if key in provenance}
    selected_provenance["minimal_sha256"] = minimal_hash
    if provenance.get("parameters"):
        selected_provenance["parameters"] = {
            key: provenance["parameters"][key] for key in
            ("reference_fraction", "ignore", "voxel_batch_size") if key in provenance["parameters"]}
    public = {"schema_version": 1, "mode": "installed_reference_with_held_fnit_transforms",
              "scope": "Full-frame composition/interpolation only; FNIT estimated transforms are held fixed.",
              "actual_official_node_used": False, "space": args.space, "frames": args.frames,
              "reference_version": args.reference_version, "nitransforms_version": "25.1.0",
              "source": source_record, "raw": raw_record, "target": target_record,
              "target_origin": target_origin,
              "completed_candidate_preproc": pipeline_record, "reference": checked,
              "raw_bids_tr_seconds": tr, "raw_bids_json_sha256": sha256(raw_json),
              "minimal_source_provenance": selected_provenance,
              "source_provenance_sha256": sha256(args.source_provenance),
              "capture_report_sha256": sha256(args.capture_report),
              "captured_normalization_sha256": captured["source_sha256"]["src/fnit/fmri/normalization.py"],
              "captured_coordinate_precision": captured["algorithm"]["configuration"].get(
                  "preproc_coordinate_precision", "float64"),
              "affine_sha256": sha256(root / artifacts["reference_to_source_world"]),
              "motion_sha256": sha256(root / artifacts["motion_pull_world"]), "mni_pull": pull_record,
              "order": 3, "mode_boundary": "grid-constant", "cval": 0., "prefilter": True,
              "SDC": False, "dummy_scans": 0, "threads": args.threads,
              "cpu_reference_seconds_including_write": wall,
              "coordinate_precision": {"official_target_world_dtype": str(target_f4.dtype),
                                       "official_common_world_dtype": str(official_common.dtype),
                                       "fnit_target_world_dtype": str(target_world.dtype),
                                       "fnit_common_world_dtype": str(fnit_common.dtype),
                                       "common_mapping_difference_rmse_mm": float(np.sqrt(np.mean(difference**2))),
                                       "common_mapping_difference_max_abs_mm": float(np.abs(difference).max()),
                                       "field_mapping": "Installed DenseFieldTransform: on-grid tolerance1e-3, otherwise cubic constant."},
              "installed_source_sha256": {"fmriprep.interfaces.resampling": sha256(official.__file__),
                                           "nitransforms.DenseFieldTransform": sha256(inspect.getsourcefile(nt.DenseFieldTransform))},
              "privacy": "Only anonymous metrics, selected stage provenance and hashes are public."}
    manifest = {"public": public, "source": str(source_path), "target": str(target_path),
                "pipeline_preproc": str(pipeline_path), "artifacts": artifacts}
    save_json(root / "manifest.private.json", manifest)
    save_json(root / "reference.public.json", public)
    print(json.dumps({"space": args.space, "frames": args.frames, "reference_seconds": wall}))


def candidate(args):
    import torch
    if args.normalization_module is not None:
        if args.coordinate_precision != "fmriprep":
            raise ValueError("a private module override is limited to the explicit precision variant")
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("fnit.fmri.normalization", args.normalization_module)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    from fnit.fmri.normalization import resample_world

    manifest = json.loads(args.manifest.read_text())
    fixed, root = manifest["public"], args.manifest.parent
    source = mapped_path(manifest["source"], args.path_map)
    target = mapped_path(manifest["target"], args.path_map)
    pipeline = mapped_path(manifest["pipeline_preproc"], args.path_map)
    for path, expected in ((source, fixed["source"]), (target, fixed["target"]),
                            (pipeline, fixed["completed_candidate_preproc"])):
        if sha256(path) != expected["sha256"]:
            raise ValueError("held source, target or completed preproc changed")
    current_normalization = sha256(inspect.getsourcefile(resample_world))
    baseline = None
    captured_precision = fixed.get("captured_coordinate_precision", "float64")
    resampler_update = current_normalization != fixed["captured_normalization_sha256"]
    if args.coordinate_precision != captured_precision:
        if args.baseline_control_report is None:
            raise ValueError("a precision variant requires the completed exact baseline control")
        baseline = json.loads(args.baseline_control_report.read_text())
        if (baseline.get("completed_candidate_match") is not True
                or baseline["rerun_vs_completed_candidate"]["all_voxels_all_frames"]["max_abs"] != 0
                or baseline["normalization_sha256"] != fixed["captured_normalization_sha256"]
                or sha256(args.baseline_control_report.parent / "held_candidate.nii.gz") != baseline["candidate"]["sha256"]):
            raise ValueError("precision variant baseline must exactly reproduce the completed frozen volume")
        expected_baseline = {"source_sha256": fixed["source"]["sha256"],
                             "target_sha256": fixed["target"]["sha256"],
                             "affine_sha256": fixed["affine_sha256"],
                             "motion_sha256": fixed["motion_sha256"],
                             "mni_pull_sha256": fixed["mni_pull"]["sha256"] if fixed["mni_pull"] else None}
        if any(baseline.get(key) != value for key, value in expected_baseline.items()):
            raise ValueError("precision variant baseline used different held inputs")
    elif resampler_update:
        if not args.verify_resampler_update or args.completed_match_tolerance != 0:
            raise ValueError("a resampler update requires explicit zero-tolerance comparison to the completed volume")
    artifacts = manifest["artifacts"]
    affine_path = root / artifacts["reference_to_source_world"]
    motion_path = root / artifacts["motion_pull_world"]
    pull_path = root / artifacts["mni_pull_ras"] if "mni_pull_ras" in artifacts else None
    if (sha256(affine_path) != fixed["affine_sha256"] or sha256(motion_path) != fixed["motion_sha256"]
            or pull_path is not None and sha256(pull_path) != fixed["mni_pull"]["sha256"]):
        raise ValueError("held transforms changed")
    if args.output.exists():
        raise FileExistsError("held candidate output already exists")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cuda = args.device.startswith("cuda")
    if cuda:
        total = torch.cuda.get_device_properties(args.device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., args.gpu_memory_gb*1e9/total), args.device)
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)  # This standalone validator owns its process.
    started = time.perf_counter()
    precision_arguments = ({"coordinate_precision": args.coordinate_precision}
                           if args.coordinate_precision == "fmriprep" else {})
    resample_world(source, target, affine_file(affine_path), args.output,
                   pre_affine_pull_ras=pull_path, motion_pull_world=np.load(motion_path, allow_pickle=False),
                   interpolation="spline", boundary="grid-constant", device=args.device,
                   batch_size=args.batch_size, spatial_chunk_size=args.spatial_chunk_size,
                   **precision_arguments)
    if cuda:
        torch.cuda.synchronize(args.device)
    wall = time.perf_counter()-started
    agreement = compare_pair(args.output, pipeline, frames=fixed["frames"],
                              expected_tr_seconds=fixed["raw_bids_tr_seconds"])
    matched = agreement["all_voxels_all_frames"]["max_abs"] <= args.completed_match_tolerance
    inputs_proven = matched if baseline is None else True
    public = {"schema_version": 1, "space": fixed["space"], "frames": fixed["frames"],
              "candidate": agreement["candidate"], "rerun_vs_completed_candidate": agreement,
              "completed_candidate_match": matched, "completed_match_tolerance": args.completed_match_tolerance,
              "source_sha256": fixed["source"]["sha256"], "target_sha256": fixed["target"]["sha256"],
              "affine_sha256": fixed["affine_sha256"], "motion_sha256": fixed["motion_sha256"],
              "mni_pull_sha256": fixed["mni_pull"]["sha256"] if fixed["mni_pull"] else None,
              "normalization_sha256": current_normalization,
              "captured_normalization_sha256": fixed["captured_normalization_sha256"],
              "captured_coordinate_precision": captured_precision,
              "resampler_update_zero_tolerance_control": resampler_update and baseline is None,
              "coordinate_precision": args.coordinate_precision,
              "held_source_control_proven": inputs_proven,
              "baseline_control_report_sha256": sha256(args.baseline_control_report) if baseline is not None else None,
              "variant_note": ("Uses the same held inputs proven by the exact frozen baseline; this precision variant is not a rerun of the completed pipeline."
                               if baseline is not None else
                               "Tests a newer resampler on the completed API's exact inputs at zero tolerance; the original API benchmark remains bound to its captured source."
                               if resampler_update else None),
              "gpu_candidate_seconds_including_io": wall, "device": args.device,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu_name": torch.cuda.get_device_name(args.device) if cuda else None,
              "threads": args.threads, "tf32": True, "batch_size": args.batch_size,
              "spatial_chunk_size": args.spatial_chunk_size,
              "allocator_limit_gb": args.gpu_memory_gb if cuda else None,
              "peak_allocated_gb": torch.cuda.max_memory_allocated(args.device)/1e9 if cuda else None}
    save_json(args.report_out, public)
    print(json.dumps({"frames": fixed["frames"], "seconds": wall, "completed_match": matched}))
    if not inputs_proven:
        raise SystemExit(2)


def compare(args):
    manifest = json.loads(args.manifest.read_text())
    fixed = manifest["public"]
    held = json.loads(args.candidate_report.read_text())
    reference_path = args.manifest.parent / manifest["artifacts"]["reference"]
    if not held.get("held_source_control_proven", held["completed_candidate_match"]):
        raise ValueError("candidate rerun did not match the completed pipeline")
    for key in ("affine_sha256", "motion_sha256"):
        if held[key] != fixed[key]:
            raise ValueError("candidate did not use the held transforms")
    for key in ("source", "target"):
        if held[key+"_sha256"] != fixed[key]["sha256"]:
            raise ValueError("candidate did not use the held images")
    if (held["mni_pull_sha256"] != (fixed["mni_pull"]["sha256"] if fixed["mni_pull"] else None)
            or sha256(args.candidate) != held["candidate"]["sha256"]
            or sha256(reference_path) != fixed["reference"]["sha256"]):
        raise ValueError("held input or result changed")
    agreement = compare_pair(args.candidate, reference_path, frames=fixed["frames"],
                              expected_tr_seconds=fixed["raw_bids_tr_seconds"])
    metric = agreement["all_voxels_all_frames"]
    gates = {"max_abs": metric["max_abs"] <= args.max_abs_tolerance,
             "relative_rmse": metric["relative_rmse"] is not None and metric["relative_rmse"] <= args.relative_rmse_tolerance,
             "r": metric["r"] is not None and metric["r"] >= args.min_correlation}
    public = {"schema_version": 1, "mode": fixed["mode"], "scope": fixed["scope"],
              "actual_official_node_used": False, "space": fixed["space"], "frames": fixed["frames"],
              "reference_version": fixed["reference_version"], "agreement": agreement,
              "source_sha256": fixed["source"]["sha256"], "raw_sha256": fixed["raw"]["sha256"],
              "target_sha256": fixed["target"]["sha256"], "affine_sha256": fixed["affine_sha256"],
              "target_origin": fixed["target_origin"],
              "motion_sha256": fixed["motion_sha256"], "mni_pull": fixed["mni_pull"],
              "minimal_source_provenance": fixed["minimal_source_provenance"],
              "coordinate_precision": fixed["coordinate_precision"],
              "installed_source_sha256": fixed["installed_source_sha256"],
              "cpu_reference_seconds_including_write": fixed["cpu_reference_seconds_including_write"],
              "candidate_runtime": {key: held[key] for key in
                                    ("gpu_candidate_seconds_including_io", "device", "threads", "tf32",
                                     "allocator_limit_gb", "peak_allocated_gb", "normalization_sha256",
                                     "cuda_visible_devices", "gpu_name")},
              "completed_candidate_match": held["completed_candidate_match"],
              "held_source_control_proven": held.get("held_source_control_proven", held["completed_candidate_match"]),
              "coordinate_precision_mode": held.get("coordinate_precision", "float64"),
              "baseline_control_report_sha256": held.get("baseline_control_report_sha256"),
              "normalization_sha256": held["normalization_sha256"],
              "variant_note": held.get("variant_note"),
              "numerical_gates": gates, "passed": all(gates.values()),
              "tolerances": {"max_abs": args.max_abs_tolerance, "relative_rmse": args.relative_rmse_tolerance,
                             "min_correlation": args.min_correlation},
              "limits": ["Holds FNIT estimated transforms; does not test estimation against an official workflow.",
                         "Uses actual installed reference float32 target-coordinate casting; the oracle is not rewritten.",
                         "A recreated STC source, when selected, is identified separately from a retained original intermediate."]}
    save_json(args.report_out, public)
    print(json.dumps({"frames": fixed["frames"], "passed": public["passed"], "agreement": metric}))
    if not public["passed"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference", help="固定容器：读取真实FNITcapture并用实际installed oracle")
    for name in ("inputs", "capture-report", "source-provenance", "private-output"):
        ref.add_argument("--"+name, type=Path, required=True)
    ref.add_argument("--space", choices=("T1w", "MNI152NLin6Asym"), required=True)
    ref.add_argument("--reference-version", choices=("25.2.4", "25.2.5"), default="25.2.4")
    ref.add_argument("--frames", type=int, default=490)
    ref.add_argument("--threads", type=int, default=8)
    ref.set_defaults(run=reference)
    cand = sub.add_parser("candidate", help="FNIT：同source和held变换独立GPU单次插值")
    for name in ("manifest", "output", "report-out"):
        cand.add_argument("--"+name, type=Path, required=True)
    cand.add_argument("--path-map", action="append", default=[])
    cand.add_argument("--device", default="cuda:0")
    cand.add_argument("--threads", type=int, default=8)
    cand.add_argument("--gpu-memory-gb", type=float, default=20.)
    cand.add_argument("--batch-size", type=int, default=4)
    cand.add_argument("--spatial-chunk-size", type=int, default=262144)
    cand.add_argument("--completed-match-tolerance", type=float, default=0.)
    cand.add_argument("--coordinate-precision", choices=("float64", "fmriprep"), default="float64")
    cand.add_argument("--baseline-control-report", type=Path)
    cand.add_argument("--verify-resampler-update", action="store_true",
                      help="同captured精度模式下验证新resampler；必须逐值复现已完成API，容差固定为0")
    cand.add_argument("--normalization-module", type=Path,
                      help="显式precision变体的私密候选模块；不修改已冻结完整pipeline")
    cand.set_defaults(run=candidate)
    comp = sub.add_parser("compare", help="全部帧比较；匿名聚合和明确阈值")
    for name in ("manifest", "candidate", "candidate-report", "report-out"):
        comp.add_argument("--"+name, type=Path, required=True)
    comp.add_argument("--max-abs-tolerance", type=float, required=True)
    comp.add_argument("--relative-rmse-tolerance", type=float, required=True)
    comp.add_argument("--min-correlation", type=float, required=True)
    comp.add_argument("--threads", type=int, default=8)
    comp.set_defaults(run=compare)
    args = parser.parse_args()
    for key in ("frames", "threads", "batch_size", "spatial_chunk_size", "gpu_memory_gb"):
        if hasattr(args, key) and (not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0):
            parser.error(key+" must be positive")
    for key in ("max_abs_tolerance", "relative_rmse_tolerance", "completed_match_tolerance"):
        if hasattr(args, key) and (not math.isfinite(getattr(args, key)) or getattr(args, key) < 0):
            parser.error(key+" must be finite and nonnegative")
    if hasattr(args, "min_correlation") and not -1 <= args.min_correlation <= 1:
        parser.error("min-correlation must be between -1 and 1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    # threadpoolctl is available in the reference container but is not a
    # FNIT runtime dependency. Torch is limited explicitly in candidate();
    # the launcher sets BLAS/OpenMP thread variables before imports.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        thread_context = nullcontext()
    else:
        thread_context = threadpool_limits(limits=args.threads)
    with thread_context:
        args.run(args)


if __name__ == "__main__":
    main()
