"""固定官方变换的全帧单次重采样对照，官方依赖仅用于参照。

export 在固定 fMRIPrep 容器内运行，读取保留的 Nipype result_resample.pklz。
candidate 在 FNIT 环境运行；compare 只需要 NumPy/Nibabel。所有影像、路径和
Nipype 内容保存在 private-output；公开报告只包含匿名指标与 SHA-256。
"""

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def image_checks(path, *, frames=None, target=None, source=None, float32=False,
                 allow_unknown_time_unit=False, expected_tr_seconds=None, cache_full=False):
    image = nib.load(str(path), keep_file_open=True)
    if frames is not None and (image.ndim != 4 or image.shape[3] != frames):
        raise ValueError("image does not contain the complete expected frame count")
    if target is not None:
        if image.shape[:3] != target.shape[:3] or not np.allclose(
                image.affine, target.affine, rtol=0, atol=1e-5):
            raise ValueError("image differs from the fixed target grid")
    if float32 and image.get_data_dtype() != np.dtype("float32"):
        raise ValueError("resampled image must be float32")
    if image.ndim == 4:
        raw_tr = float(image.header.get_zooms()[3])
        if not math.isfinite(raw_tr) or raw_tr <= 0:
            raise ValueError("image has an invalid TR")
        unit = image.header.get_xyzt_units()[1]
        scales = {"sec": 1., "msec": .001, "usec": .000001}
        if unit == "unknown" and allow_unknown_time_unit and expected_tr_seconds is not None:
            tr = raw_tr
        elif unit in scales:
            tr = raw_tr * scales[unit]
        else:
            raise ValueError("image time unit is unknown without an explicit raw BIDS TR")
        if expected_tr_seconds is not None and not np.isclose(tr, expected_tr_seconds, rtol=0, atol=1e-6):
            raise ValueError("image TR differs from the raw BIDS RepetitionTime")
        if source is not None:
            source_unit = source.header.get_xyzt_units()[1]
            source_tr = source.header.get_zooms()[3] * scales.get(source_unit, 1.)
            if not np.isclose(tr, source_tr, rtol=0, atol=1e-6):
                raise ValueError("output changed source TR")
        if cache_full:
            # A complete float32 array lets the paired comparison reuse one
            # gzip read for finite checks and every numerical block.
            if not np.isfinite(image.get_fdata(dtype=np.float32)).all():
                raise ValueError("image contains nonfinite values")
        else:
            for start in range(0, image.shape[3], 4):
                if not np.isfinite(np.asarray(image.dataobj[..., start:start + 4],
                                              dtype=np.float32)).all():
                    raise ValueError("image contains nonfinite values")
    else:
        tr = None
        raw_tr = None
        unit = None
        if not np.isfinite(np.asarray(image.dataobj)).all():
            raise ValueError("image contains nonfinite values")
    return image, {"shape": list(image.shape), "dtype": str(image.get_data_dtype()),
                   "tr_seconds": tr, "time_unit": unit, "all_finite": True,
                   "header_pixdim_tr": raw_tr,
                   "sha256": sha256(path)}


def export_reference(args):
    """Use actual installed reference loaders and interpolation, never FNIT."""
    from nipype.interfaces.base import isdefined
    from nipype.utils.filemanip import loadpkl
    import nitransforms as nt
    import fmriprep.interfaces.resampling as official_resampling
    import fmriprep.utils.transforms as official_transforms

    version = importlib.metadata.version("fmriprep")
    nt_version = importlib.metadata.version("nitransforms")
    if version != args.reference_version or nt_version != "25.1.0":
        raise ValueError("container does not match the fixed reference versions")
    root = args.private_output.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root / "manifest.private.json").exists():
        raise FileExistsError("reference export already exists")
    result = loadpkl(str(args.result))
    inputs = dict(result.inputs)
    def defined(value):
        return value is not None and isdefined(value)
    for name in ("in_file", "ref_file"):
        if not defined(inputs.get(name)):
            raise ValueError("reference node has no defined " + name)
    source_path, target_path = Path(inputs["in_file"]), Path(inputs["ref_file"])
    raw_tr = json.loads(args.raw_bold_json.read_text()).get("RepetitionTime")
    if not isinstance(raw_tr, (int, float)) or not math.isfinite(raw_tr) or raw_tr <= 0:
        raise ValueError("raw BIDS sidecar must define a positive finite RepetitionTime")
    source, source_record = image_checks(source_path, frames=args.expected_frames,
                                        allow_unknown_time_unit=True, expected_tr_seconds=raw_tr,
                                        cache_full=True)
    target, target_record = image_checks(target_path)
    if target.ndim != 3:
        raise ValueError("reference target must be 3D")
    run_record = json.loads(args.reference_run_report.read_text())
    expected_stc = args.expected_stc == "on"
    if (run_record.get("STC") is not expected_stc or run_record.get("SDC") is not False
            or run_record.get("dummy_scans") != 0
            or run_record.get("input_frames") != args.expected_frames):
        raise ValueError("reference run must match explicit STC state, SDC off, zero dummy scans and all frames")
    order = inputs.get("order", 3)
    mode = inputs.get("mode", "grid-constant")
    cval = inputs.get("cval", 0.)
    prefilter = inputs.get("prefilter", True)
    if order != 3 or mode != "grid-constant" or cval != 0 or not prefilter:
        raise ValueError("this candidate gate requires cubic/grid-constant/cval=0/prefilter")
    fieldmap_path = inputs.get("fieldmap")
    if defined(fieldmap_path):
        fieldmap = np.asarray(nib.load(str(fieldmap_path)).dataobj)
        if not np.isfinite(fieldmap).all() or np.any(fieldmap != 0):
            raise ValueError("fixed-transform gate does not implement SDC")
    paths = inputs.get("transforms")
    paths = list(paths) if defined(paths) else []
    inverses = inputs.get("inverse", [False])
    inverses = list(inverses) if defined(inverses) else [False]
    chain = official_transforms.load_transforms(paths, inverses.copy())
    if not isinstance(chain, nt.TransformChain):
        chain = nt.TransformChain([chain])
    if isinstance(chain[-1], nt.linear.LinearTransformsMapping):
        common, motion = list(chain.transforms[:-1]), chain[-1]
        matrices = np.asarray([xfm.matrix for xfm in motion], dtype=np.float64)
    else:
        if any(isinstance(xfm, nt.linear.LinearTransformsMapping) for xfm in chain):
            raise ValueError("official HMC mapping must occur last in pull order")
        common = list(chain.transforms)
        matrices = np.broadcast_to(np.eye(4), (args.expected_frames, 4, 4)).copy()
    if matrices.shape != (args.expected_frames, 4, 4) or not np.isfinite(matrices).all():
        raise ValueError("reference HMC does not contain one affine per input frame")
    if not np.allclose(matrices[:, 3], (0, 0, 0, 1), rtol=0, atol=1e-8):
        raise ValueError("reference HMC is not affine")
    common_chain = nt.TransformChain(common or [nt.Affine()])
    # Match actual resample_image: target points are cast to float32 first.
    target_world_exact = nt.base.SpatialReference.factory(target).ndcoords
    target_world = target_world_exact.astype("f4")
    source_world = common_chain.map(target_world)
    if source_world.shape != target_world.shape or not np.isfinite(source_world).all():
        raise ValueError("reference common pull mapping is invalid")
    # The next official Affine (source world->voxel) casts its input to f4.
    # Preserve that effective input exactly; a float32 displacement cannot
    # generally encode it relative to a different, exact target-world point.
    effective_source_world = source_world.astype(np.float32).astype(np.float64)
    pull = effective_source_world - target_world_exact
    field_path, motion_path = root / "common_pull_ras.nii.gz", root / "motion_pull_world.npy"
    field = nib.Nifti1Image(pull.reshape((*target.shape, 3)), target.affine)
    field.header.set_intent("vector")
    nib.save(field, str(field_path))
    np.save(motion_path, matrices, allow_pickle=False)
    restored_source_world = target_world_exact + pull
    reconstruction_error = restored_source_world - effective_source_world
    effective_f4_roundtrip_mismatches = int(np.count_nonzero(
        restored_source_world.astype(np.float32) != effective_source_world.astype(np.float32),
    ))
    if effective_f4_roundtrip_mismatches:
        raise ValueError("F64 pull export does not exactly reconstruct the official effective f4 coordinates")
    raw_f32_pull = (source_world - target_world_exact).astype(np.float32)
    raw_f32_error = target_world_exact + raw_f32_pull.astype(np.float64) - source_world
    effective_world_cast_error = effective_source_world - source_world
    coordinate_cast = target_world.astype(np.float64) - target_world_exact
    reference_path = root / "reference_resampled.nii.gz"
    reference_source = source
    pe_info = None
    pe_dir, ro_time = inputs.get("pe_dir"), inputs.get("ro_time")
    if defined(pe_dir) and defined(ro_time) and pe_dir and ro_time:
        # Match ResampleSeries preprocessing even when the field is zero:
        # this is an exact axis flip, never an additional interpolation.
        reference_source, axcodes = official_resampling.ensure_positive_cosines(source)
        pe_axis = "ijk".index(pe_dir[0])
        sign = -1 if ((axcodes[pe_axis] in "LPI") ^ pe_dir.endswith("-")) else 1
        pe_info = [(pe_axis, sign * ro_time)] * args.expected_frames
    inverse_reference_affine = np.linalg.inv(reference_source.affine)
    official_common_voxel = nt.Affine(inverse_reference_affine).map(source_world)
    exported_common_voxel = nt.Affine(inverse_reference_affine).map(restored_source_world)
    source_voxel_roundtrip_max_abs = float(np.abs(
        official_common_voxel - exported_common_voxel,
    ).max())
    if source_voxel_roundtrip_max_abs != 0:
        raise ValueError("exported effective world does not preserve the actual source-voxel coordinates")
    started = time.perf_counter()
    output = official_resampling.resample_image(
        source=reference_source, target=target, transforms=chain, fieldmap=None, pe_info=pe_info,
        jacobian=False, nthreads=args.threads, output_dtype="f4", order=order,
        mode=mode, cval=cval, prefilter=prefilter)
    # ResampleSeries inherits target header units; require the preserved time
    # unit to be correct instead of silently rewriting the reference metadata.
    nib.save(output, str(reference_path))
    reference_seconds = time.perf_counter() - started
    _, reference_record = image_checks(reference_path, frames=args.expected_frames,
                                       target=target, source=source, float32=True,
                                       allow_unknown_time_unit=True, expected_tr_seconds=raw_tr,
                                       cache_full=True)
    actual_path = result.outputs.out_file
    actual_image, actual_record = image_checks(actual_path, frames=args.expected_frames,
                                              target=target, source=source, float32=True,
                                              allow_unknown_time_unit=True, expected_tr_seconds=raw_tr,
                                              cache_full=True)
    replay_agreement = compare_pair(reference_path, actual_path, frames=args.expected_frames,
                                    expected_tr_seconds=raw_tr, reference_is_oracle=True)
    if replay_agreement["all_voxels_all_frames"]["max_abs"] != 0:
        raise ValueError("installed-source replay differs from the actual reference node output")
    public = {"schema_version": 1, "space": args.space, "scope": "Fixed official transforms; full-frame interpolation only.",
              "reference_version": version, "nitransforms_version": nt_version,
              "raw_bids_tr_seconds": raw_tr, "raw_bold_json_sha256": sha256(args.raw_bold_json),
              "complete_reference_pipeline_exit_code": run_record.get("exit_code"),
              "complete_reference_pipeline_succeeded": run_record.get("exit_code") == 0,
              "completed_resampling_node": True,
              "STC": expected_stc, "expected_STC": args.expected_stc,
              "SDC": False, "dummy_scans": 0, "frames": args.expected_frames,
              "reference_run_report_sha256": sha256(args.reference_run_report),
              "source_origin": "Actual retained resampling node in_file; its exact SHA is held for FNIT.",
              "reference_replay_threads": args.threads,
              "actual_node_threads": inputs.get("num_threads"),
              "order": order, "mode": mode, "cval": cval, "prefilter": prefilter,
              "source": source_record, "target": target_record,
              "reference_replay": reference_record, "actual_node_output": actual_record,
              "replay_vs_actual_node": replay_agreement,
              "transform_file_sha256": [sha256(path) for path in paths],
              "transform_classes_pull_order": [type(xfm).__name__ for xfm in chain],
              "common_pull_sha256": sha256(field_path), "motion_pull_sha256": sha256(motion_path),
              "reference_result_sha256": sha256(args.result),
              "installed_source_sha256": {"fmriprep.interfaces.resampling": sha256(official_resampling.__file__),
                                          "fmriprep.utils.transforms": sha256(official_transforms.__file__)},
              "pull_export_precision_mm": {"rmse": float(np.sqrt(np.mean(reconstruction_error ** 2))),
                                            "max_abs": float(np.abs(reconstruction_error).max()),
                                            "dtype": "float64",
                                            "effective_f4_roundtrip_component_mismatches": effective_f4_roundtrip_mismatches,
                                            "source_voxel_roundtrip_max_abs": source_voxel_roundtrip_max_abs,
                                            "effective_source_world_f4_cast_max_abs_mm": float(np.abs(effective_world_cast_error).max()),
                                            "raw_common_as_f32_delta_rmse_mm": float(np.sqrt(np.mean(raw_f32_error ** 2))),
                                            "raw_common_as_f32_delta_max_abs_mm": float(np.abs(raw_f32_error).max()),
                                            "target_f4_cast_max_abs": float(np.abs(coordinate_cast).max())},
              "reference_replay_seconds_including_write": reference_seconds,
              "privacy": "Only anonymous aggregate values and hashes are public."}
    manifest = {"public": public, "source": str(source_path), "target": str(target_path),
                "actual_node_output": str(actual_path),
                "artifacts": {"pull": field_path.name, "motion": motion_path.name,
                              "reference": reference_path.name},
                "reference_inputs": {"transforms": [str(path) for path in paths], "inverse": inverses}}
    save_json(root / "manifest.private.json", manifest)
    save_json(root / "export.public.json", public)
    print(json.dumps({"space": args.space, "frames": args.expected_frames,
                      "exported": True, "reference_replay_seconds": reference_seconds}))


def mapped_path(path, maps):
    for item in sorted(maps, key=len, reverse=True):
        old, new = item.split("=", 1)
        if path == old or path.startswith(old.rstrip("/") + "/"):
            return Path(new + path[len(old):])
    return Path(path)


def candidate(args):
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    import torch
    from fnit.fmri.normalization import resample_world

    manifest = json.loads(args.manifest.read_text())
    record, root = manifest["public"], args.manifest.parent
    source_path = args.source or mapped_path(manifest["source"], args.path_map)
    target_path = args.target or mapped_path(manifest["target"], args.path_map)
    for path, expected in ((source_path, record["source"]), (target_path, record["target"])):
        if sha256(path) != expected["sha256"]:
            raise ValueError("candidate source/target differs from the fixed reference input")
    torch.set_num_threads(args.threads)
    source, _ = image_checks(source_path, frames=record["frames"],
                             allow_unknown_time_unit=True,
                             expected_tr_seconds=record.get("raw_bids_tr_seconds"))
    target, _ = image_checks(target_path)
    pull_path, motion_path = (root / manifest["artifacts"][key] for key in ("pull", "motion"))
    if sha256(pull_path) != record["common_pull_sha256"] or sha256(motion_path) != record["motion_pull_sha256"]:
        raise ValueError("exported reference transforms changed")
    motion = np.load(motion_path, allow_pickle=False)
    if args.output.exists():
        raise FileExistsError("candidate output already exists")
    cuda = args.device.startswith("cuda")
    if cuda:
        properties = torch.cuda.get_device_properties(args.device)
        fraction = min(1., args.gpu_memory_gb * 1e9 / properties.total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction, args.device)
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)  # Separate benchmark process owns the reset.
    started = time.perf_counter()
    resample_world(source_path, target_path, np.eye(4), args.output,
                   pre_affine_pull_ras=pull_path, motion_pull_world=motion,
                   interpolation="spline", boundary="grid-constant", device=args.device,
                   coordinate_precision="fmriprep",
                   batch_size=args.batch_size, spatial_chunk_size=args.spatial_chunk_size)
    if cuda:
        torch.cuda.synchronize(args.device)
    wall = time.perf_counter() - started
    _, checked = image_checks(args.output, frames=record["frames"], target=target,
                              source=source, float32=True,
                              expected_tr_seconds=record.get("raw_bids_tr_seconds"))
    if checked["time_unit"] != "sec":
        raise ValueError("candidate output must explicitly declare seconds")
    import inspect
    public = {"schema_version": 1, "space": record["space"], "candidate": checked,
              "frames": record["frames"], "single_interpolation": True,
              "coordinate_precision": "fmriprep", "exported_pull_dtype": "float64",
              "source_sha256": record["source"]["sha256"], "target_sha256": record["target"]["sha256"],
              "common_pull_sha256": record["common_pull_sha256"], "motion_pull_sha256": record["motion_pull_sha256"],
              "candidate_source_sha256": sha256(inspect.getsourcefile(resample_world)),
              "seconds_including_io": wall, "device": args.device,
              "batch_size": args.batch_size, "spatial_chunk_size": args.spatial_chunk_size,
              "threads": args.threads, "allocator_limit_gb": args.gpu_memory_gb if cuda else None,
              "peak_allocated_gb": torch.cuda.max_memory_allocated(args.device) / 1e9 if cuda else None}
    save_json(args.report_out, public)
    print(json.dumps({"space": record["space"], "frames": record["frames"], "seconds": wall}))


class Agreement:
    """Streaming, centered covariance and errors without loading all frames."""
    def __init__(self):
        self.n = 0
        self.mean_x = self.mean_y = 0.
        self.m2x = self.m2y = self.cov = self.squared_error = self.absolute_error = self.max_abs = 0.

    def add(self, x, y):
        x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
        if x.size != y.size:
            raise ValueError("comparison sizes differ")
        n = x.size
        if not n:
            return
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("nonfinite comparison data")
        mx, my = float(x.mean()), float(y.mean())
        xc, yc = x - mx, y - my
        dx, dy, total = mx - self.mean_x, my - self.mean_y, self.n + n
        correction = self.n * n / total
        self.m2x += float(xc @ xc) + dx * dx * correction
        self.m2y += float(yc @ yc) + dy * dy * correction
        self.cov += float(xc @ yc) + dx * dy * correction
        self.mean_x += dx * n / total
        self.mean_y += dy * n / total
        self.n = total
        error = x - y
        self.squared_error += float(error @ error)
        self.absolute_error += float(np.abs(error).sum())
        self.max_abs = max(self.max_abs, float(np.abs(error).max()))

    def result(self):
        if not self.n:
            return {"samples": 0, "r": None, "rmse": None, "relative_rmse": None, "mae": None, "max_abs": None}
        r = self.cov / math.sqrt(self.m2x * self.m2y) if self.m2x > 0 and self.m2y > 0 else None
        rmse = math.sqrt(self.squared_error / self.n)
        reference_std = math.sqrt(self.m2y / self.n)
        return {"samples": self.n, "r": max(-1., min(1., r)) if r is not None else None,
                "rmse": rmse, "relative_rmse": rmse / reference_std if reference_std else None,
                "mae": self.absolute_error / self.n, "max_abs": self.max_abs,
                "reference_std": reference_std}


def compare_pair(candidate_path, reference_path, *, mask=None, frames=490,
                 expected_tr_seconds=None, reference_is_oracle=False):
    reference, reference_record = image_checks(reference_path, frames=frames, float32=True,
                                               allow_unknown_time_unit=expected_tr_seconds is not None,
                                               expected_tr_seconds=expected_tr_seconds, cache_full=True)
    candidate_image, candidate_record = image_checks(candidate_path, frames=frames,
                                                     target=reference, source=reference, float32=True,
                                                     allow_unknown_time_unit=reference_is_oracle,
                                                     expected_tr_seconds=expected_tr_seconds, cache_full=True)
    if not reference_is_oracle and candidate_record["time_unit"] != "sec":
        raise ValueError("candidate output must explicitly declare seconds")
    all_samples, nonzero, masked = Agreement(), Agreement(), Agreement()
    per_frame = []
    candidate_values = candidate_image.get_fdata(dtype=np.float32)
    reference_values = reference.get_fdata(dtype=np.float32)
    for start in range(0, frames, 4):
        x = candidate_values[..., start:start + 4]
        y = reference_values[..., start:start + 4]
        all_samples.add(x, y)
        selected = y != 0
        nonzero.add(x[selected], y[selected])
        if mask is not None:
            masked.add(x[mask], y[mask])
        for index in range(x.shape[3]):
            one = Agreement()
            one.add(x[..., index], y[..., index])
            per_frame.append(one.result())
    result = {"candidate": candidate_record, "reference": reference_record,
              "all_voxels_all_frames": all_samples.result(),
              "reference_nonzero_samples": nonzero.result(), "per_frame": per_frame}
    if mask is not None:
        result["target_mask_all_frames"] = masked.result()
        result["mask_voxels"] = int(mask.sum())
    return result


def compare(args):
    manifest = json.loads(args.manifest.read_text())
    reference_path = args.reference or args.manifest.parent / manifest["artifacts"]["reference"]
    fixed = manifest["public"]
    if sha256(reference_path) != fixed["reference_replay"]["sha256"]:
        raise ValueError("reference replay output changed")
    mask = None
    mask_record = None
    if args.mask:
        target = nib.load(str(reference_path))
        mask_image, mask_record = image_checks(args.mask, target=target)
        if mask_image.ndim != 3:
            raise ValueError("target mask must be 3D")
        mask = np.asarray(mask_image.dataobj) > 0
        if not mask.any():
            raise ValueError("target mask is empty")
    candidate_record = json.loads(args.candidate_report.read_text())
    for key in ("source_sha256", "target_sha256", "common_pull_sha256", "motion_pull_sha256"):
        expected = fixed[key] if key in fixed else fixed[key.removesuffix("_sha256")]["sha256"]
        if candidate_record[key] != expected:
            raise ValueError("candidate did not use the exported fixed inputs/transforms")
    if candidate_record["candidate"]["sha256"] != sha256(args.candidate):
        raise ValueError("candidate output differs from its runtime report")
    paired = compare_pair(args.candidate, reference_path, mask=mask, frames=fixed["frames"],
                           expected_tr_seconds=fixed["raw_bids_tr_seconds"])
    if args.actual_node_output:
        if sha256(args.actual_node_output) != fixed["actual_node_output"]["sha256"]:
            raise ValueError("actual reference node output changed")
        replay_pair = compare_pair(reference_path, args.actual_node_output, frames=fixed["frames"],
                                    expected_tr_seconds=fixed["raw_bids_tr_seconds"], reference_is_oracle=True)
    else:
        replay_pair = None
    metrics = paired["all_voxels_all_frames"]
    gates = {"max_abs": metrics["max_abs"] <= args.max_abs_tolerance,
             "relative_rmse": metrics["relative_rmse"] is not None and metrics["relative_rmse"] <= args.relative_rmse_tolerance,
             "r": metrics["r"] is not None and metrics["r"] >= args.min_correlation}
    public = {"schema_version": 1, "scope": "Identical full-frame source and official transforms; interpolation gate only.",
              "space": fixed["space"], "frames": fixed["frames"], "reference_version": fixed["reference_version"],
              "STC": fixed["STC"], "SDC": False, "order": 3, "mode": "grid-constant",
              "source_sha256": fixed["source"]["sha256"], "target_sha256": fixed["target"]["sha256"],
              "common_pull_sha256": fixed["common_pull_sha256"], "motion_pull_sha256": fixed["motion_pull_sha256"],
              "pull_export_precision_mm": fixed["pull_export_precision_mm"], "agreement": paired,
              "raw_bids_tr_seconds": fixed["raw_bids_tr_seconds"],
              "complete_reference_pipeline_succeeded": fixed["complete_reference_pipeline_succeeded"],
              "candidate_runtime": {key: candidate_record[key] for key in
                                    ("candidate_source_sha256", "seconds_including_io", "device", "threads",
                                     "allocator_limit_gb", "batch_size", "peak_allocated_gb")},
              "mask": mask_record, "replay_vs_actual_node": replay_pair,
              "tolerances": {"max_abs": args.max_abs_tolerance, "relative_rmse": args.relative_rmse_tolerance,
                             "min_correlation": args.min_correlation},
              "numerical_gates": gates, "passed": all(gates.values()),
              "limits": ["Tests interpolation/composition using fixed official transforms, not FNIT registration estimates.",
                         "Reference maps float32 target coordinates; float64 displacements exactly retain the effective float32 common-world coordinates. Bitwise output equality is not assumed.",
                         "Per-frame metrics contain every input frame; no cropped-frame performance extrapolation."],
              "privacy": "No subject IDs, raw paths, image values or transform matrices are public."}
    save_json(args.report_out, public)
    print(json.dumps({"space": fixed["space"], "frames": fixed["frames"],
                      "passed": public["passed"], "agreement": metrics}))
    if not public["passed"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    export = subs.add_parser("export", help="在官方容器读取实际result并导出固定pull/HMC及参考输出")
    export.add_argument("--result", type=Path, required=True)
    export.add_argument("--private-output", type=Path, required=True)
    export.add_argument("--reference-run-report", type=Path, required=True)
    export.add_argument("--raw-bold-json", type=Path, required=True, help="原始BIDS sidecar，提供TR秒")
    export.add_argument("--reference-version", choices=("25.2.4", "25.2.5"), default="25.2.4")
    export.add_argument("--space", choices=("T1w", "MNI152NLin6Asym"), required=True)
    export.add_argument("--expected-frames", type=int, default=490)
    export.add_argument("--expected-stc", choices=("off", "on"), default="off",
                        help="必须与实际reference run report的STC布尔状态相符；不重新做STC")
    export.add_argument("--threads", type=int, default=1,
                        help="actual installed replay的并发线程数；默认串行，另行核对actual node payload")
    export.set_defaults(run=export_reference)
    cand = subs.add_parser("candidate", help="在FNIT环境单次插值相同全帧source")
    cand.add_argument("--manifest", type=Path, required=True)
    cand.add_argument("--source", type=Path, help="宿主机路径；须与reference source逐字节SHA一致")
    cand.add_argument("--target", type=Path, help="宿主机路径；须与reference target逐字节SHA一致")
    cand.add_argument("--path-map", action="append", default=[], help="容器前缀=宿主机前缀；可多次指定")
    cand.add_argument("--output", type=Path, required=True)
    cand.add_argument("--report-out", type=Path, required=True)
    cand.add_argument("--device", default="cuda:0")
    cand.add_argument("--batch-size", type=int, default=4)
    cand.add_argument("--threads", type=int, default=8)
    cand.add_argument("--gpu-memory-gb", type=float, default=20., help="独立验证进程的allocator上限")
    cand.add_argument("--spatial-chunk-size", type=int, default=262144)
    cand.set_defaults(run=candidate)
    comp = subs.add_parser("compare", help="全部帧匿名误差、网格/TR/finite检查及明确阈值门禁")
    comp.add_argument("--manifest", type=Path, required=True)
    comp.add_argument("--candidate", type=Path, required=True)
    comp.add_argument("--candidate-report", type=Path, required=True)
    comp.add_argument("--reference", type=Path, help="复制的参考结果路径；须与export SHA一致")
    comp.add_argument("--actual-node-output", type=Path, help="额外核对reference replay与实际节点输出")
    comp.add_argument("--mask", type=Path)
    comp.add_argument("--report-out", type=Path, required=True)
    comp.add_argument("--max-abs-tolerance", type=float, required=True)
    comp.add_argument("--relative-rmse-tolerance", type=float, required=True)
    comp.add_argument("--min-correlation", type=float, required=True)
    comp.add_argument("--threads", type=int, default=8)
    comp.set_defaults(run=compare)
    args = parser.parse_args()
    for name in ("threads", "expected_frames", "batch_size", "spatial_chunk_size"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(name + " must be positive")
    for name in ("max_abs_tolerance", "relative_rmse_tolerance"):
        if hasattr(args, name) and (not math.isfinite(getattr(args, name)) or getattr(args, name) < 0):
            parser.error(name + " must be finite and nonnegative")
    if hasattr(args, "min_correlation") and not -1 <= args.min_correlation <= 1:
        parser.error("min-correlation must be between -1 and 1")
    if hasattr(args, "gpu_memory_gb") and (not math.isfinite(args.gpu_memory_gb) or args.gpu_memory_gb <= 0):
        parser.error("gpu-memory-gb must be finite and positive")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    # Available in the official container, optional in the isolated FNIT
    # validator environment. Candidate also sets Torch threads explicitly;
    # launchers set BLAS/OpenMP variables before importing NumPy.
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
