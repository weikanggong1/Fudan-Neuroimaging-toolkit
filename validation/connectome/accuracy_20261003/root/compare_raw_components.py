"""只读比较本轮真实 raw-DWI 输出与独立官方链；不运行 GPU 或原软件。

输入：正式 accuracy_configuration.json 的路径和 SHA-256；可选病例和版本。
输出：新目录内 components.json；只比较已完成的真实病例，未完成者明确列出。
完整 DWI 按原帧读取，统计全体素及官方脑 mask；FA 同样保留非有限值。
RMSE/MAE/Pearson 的主统计遇到非有限值为 null，同时提供注明分母的
finite-pair 诊断和逐值非有限计数；不补零、不删帧、不裁体素、不重采样。

示例（变量名完整，SHA 必须取正式冻结配置）：
  python compare_raw_components.py --configuration /absolute/accuracy_configuration.json \
    --configuration-sha256 ACTUAL_SHA256 --case-id sub-CON01 --version candidate \
    --output-dir /absolute/new_comparison_directory

此脚本报告阶段差异，不把不同校正 DWI 上的 FA 差异称作固定输入 DTI 误差，
也不自行定义新的科学验收阈值。依赖仅为现有 Conda 中的 NumPy/nibabel。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

# Reuse the mature, stdlib-only raw selection and completion checks.
REPOSITORY = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY))
from tools import benchmark_connectome_raw_cohort as cohort


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_identity(first, second):
    return (Path(first["path"]).resolve() == Path(second["path"]).resolve()
            and first["sha256"] == second["sha256"])


class Audit:
    """Bind immutable bytes before and after comparison, including conflicts."""

    def __init__(self):
        self.records = {}

    def check(self, record, *, allow_empty_source=False):
        path = Path(record["path"])
        digest = record["sha256"]
        require(path.is_absolute() and path.is_file(), f"missing absolute bound file: {path}")
        require(isinstance(digest, str) and cohort.SHA.fullmatch(digest), f"invalid SHA: {path}")
        key = str(path.resolve())
        previous = self.records.get(key)
        require(previous is None or previous["sha256"] == digest, f"conflicting SHA: {path}")
        if previous is None:
            require(cohort.sha256(path) == digest, f"file bytes changed: {path}")
        size = path.stat().st_size
        # Source inventories can include legitimate empty package markers.
        # They still require the exact frozen SHA before and after. MRI,
        # contracts and saved outputs retain the nonempty producer guard.
        require((size > 0 or allow_empty_source) and record.get("exists", True) is True,
                f"empty/missing producer file: {path}")
        require(record.get("size_bytes", size) == size, f"file size differs: {path}")
        result = {"path": key, "sha256": digest, "size_bytes": size}
        self.records[key] = result
        return result

    def json(self, record):
        actual = self.check(record)
        return json.loads(Path(actual["path"]).read_bytes())

    def snapshot_json(self, path):
        return self.json({"path": str(path), "sha256": cohort.sha256(path)})

    def finish(self):
        for record in self.records.values():
            require(Path(record["path"]).stat().st_size == record["size_bytes"] and
                    cohort.sha256(record["path"]) == record["sha256"],
                    f"bound file changed during comparison: {record['path']}")


def map_identities(records):
    return {str(Path(item["path"]).resolve()): item["sha256"] for item in records}


def resolved_hashes(table, root=None):
    result = {}
    for path, digest in table.items():
        actual = Path(path)
        if not actual.is_absolute():
            require(root is not None, "relative producer path lacks its actual report root")
            actual = Path(root) / actual
        key = str(actual.resolve())
        require(key not in result or result[key] == digest, "conflicting producer paths")
        result[key] = digest
    return result


def commands_completed(commands):
    require(bool(commands) and all(row.get("returncode") == 0 for row in commands),
            "actual official commands are absent or not all successful")


def official_inputs(binding, raw_case, raw_manifest, config, audit):
    """Follow the manifest's actual producer pointers, including CON11's namespace."""
    for record in raw_case["input_files"]:
        audit.check(record)
    reference = audit.json(binding["official_reference_manifest"])
    require(reference.get("state") == "completed" and reference.get("execution_completed") is True
            and reference.get("case_id") == raw_case["case_id"], "official reference not completed/same case")
    planned, completed = reference["commands"], reference["completed_commands"]
    commands_completed(completed)
    require(len(planned) == len(completed) == 198 and
            all(all(done.get(key) == value for key, value in plan.items())
                for plan, done in zip(planned, completed)), "official planned/executed sequence differs")
    raw_binding = reference["raw_case_binding"]
    require(raw_binding["files"] == raw_case["input_files"] and
            raw_binding["manifest_sha256"] == config["raw_manifest"]["sha256"] and
            all(raw_binding[key] == raw_manifest[key] for key in ("dataset", "snapshot", "license")),
            "official canonical raw acquisition differs")
    audit.check({"path": raw_binding["manifest_path"], "sha256": raw_binding["manifest_sha256"]})
    rawprep_record = {"path": raw_binding["official_rawprep_report"],
                      "sha256": raw_binding["official_rawprep_report_sha256"]}
    rawprep = audit.json(rawprep_record)
    require(rawprep == raw_binding["rawprep_producer_identity"] and rawprep.get("completed") is True
            and not rawprep.get("error"), "actual rawprep identity is incomplete or changed")
    commands_completed(rawprep["commands"])
    require(rawprep["subject"] == raw_case["subject"] and rawprep["session"] == raw_case["session"]
            and resolved_hashes(rawprep["input_sha256"]) == map_identities(
                [row for row in raw_case["input_files"] if row["kind"] != "dataset_description"]),
            "actual official rawprep processed another raw subject/session/acquisition")
    source = reference["source"]
    consumer = audit.json(source["official_dwi_contract"])
    require(consumer.get("state") == "completed" and consumer.get("execution_completed") is True
            and consumer.get("case_id") == raw_case["case_id"]
            and consumer.get("scope") == "official_self_produced_raw_dwi_chain",
            "official DWI consumer is incomplete/different")
    require(same_identity(consumer["upstream_report"], rawprep_record),
            "DWI consumer upstream differs from actual reference rawprep")
    require(same_identity({"path": consumer["upstream_official_rawprep_report"],
                           "sha256": consumer["upstream_official_rawprep_report_sha256"]}, rawprep_record),
            "DWI consumer rawprep SHA/path disagree")
    modeling_record = {"path": consumer["modeling_report"], "sha256": consumer["modeling_report_sha256"]}
    modeling = audit.json(modeling_record)
    require(modeling.get("state") == "completed" and modeling.get("execution_completed") is True
            and modeling["subject"] == raw_case["subject"], "actual official modeling incomplete/different")
    commands = consumer["actualcommands"]
    commands_completed(commands)
    require(commands == modeling["actualcommands"], "consumer commands differ from actual modeling report")
    require(Path(modeling["consumer_contract_path"]).resolve() ==
            Path(source["official_dwi_contract"]["path"]).resolve(), "modeling produced another consumer")
    sidecar = consumer["completed_rawprep_sidecar"]
    audit.json(sidecar)
    audit.json({"path": sidecar["verified_path"], "sha256": sidecar["verified_sha256"]})
    files = consumer["files"]
    roles = {"dwi": "official_corrected_dwi", "bvecs": "official_rotated_bvecs",
             "bvals": "official_bvals", "mask": "brain_mask", "fa": "FA"}
    selected = {key: audit.check(files[role]) for key, role in roles.items()}
    produced_raw = resolved_hashes(rawprep["output_sha256"], Path(rawprep_record["path"]).parent)
    for role in ("dwi", "bvecs"):
        require(produced_raw.get(selected[role]["path"]) == selected[role]["sha256"],
                f"official {role} lacks actual successful rawprep output SHA")
    canonical_bvals = [item for item in raw_case["input_files"] if item["kind"] == "bval"]
    require(len(canonical_bvals) == 1 and canonical_bvals[0]["sha256"] == selected["bvals"]["sha256"],
            "official bvals changed the raw frame/shell table")
    gradient = [row for row in commands if row.get("label") == "gradient"]
    fa_command = [row for row in commands if row.get("label") == "FA"]
    tensor_command = [row for row in commands if row.get("label") == "DTI"]
    require(len(gradient) == len(fa_command) == len(tensor_command) == 1,
            "ambiguous official gradient/tensor/FA producer")
    for role in ("dwi", "bvecs", "bvals"):
        require(resolved_hashes(gradient[0]["input_sha256"]).get(selected[role]["path"])
                == selected[role]["sha256"], f"official gradient used another {role}")
    require(resolved_hashes(fa_command[0]["output_sha256"]).get(selected["fa"]["path"])
            == selected["fa"]["sha256"], "FA was not actually produced by the successful official command")
    require(resolved_hashes(fa_command[0]["input_sha256"]).get(selected["mask"]["path"])
            == selected["mask"]["sha256"], "FA command used another brain mask")
    tensor_inputs = resolved_hashes(tensor_command[0]["input_sha256"])
    require(tensor_inputs.get(selected["dwi"]["path"]) == selected["dwi"]["sha256"] and
            tensor_inputs.get(selected["mask"]["path"]) == selected["mask"]["sha256"],
            "official tensor used another corrected DWI/brain mask")
    gradient_output = audit.check(files["official_gradient_mrtrix"])
    require(resolved_hashes(gradient[0]["output_sha256"]).get(gradient_output["path"]) == gradient_output["sha256"]
            and tensor_inputs.get(gradient_output["path"]) == gradient_output["sha256"],
            "official tensor gradient is not the actually produced gradient table")
    tensor_path = str(Path(fa_command[0]["command"][1]).resolve())
    tensor_digest = resolved_hashes(tensor_command[0]["output_sha256"]).get(tensor_path)
    require(tensor_digest is not None and
            resolved_hashes(fa_command[0]["input_sha256"]).get(tensor_path) == tensor_digest,
            "FA did not consume the tensor produced from this corrected acquisition")
    audit.check({"path": tensor_path, "sha256": tensor_digest})
    for role in ("fa", "wm_fod"):
        require(same_identity(source["images"][role], reference["input_readbacks"][role]["source"]),
                f"official {role} reader did not consume the declared original producer")
    require(same_identity(source["images"]["fa"], selected["fa"]), "tracking reference consumed another FA")
    # Ancillary geometry comes from explicit source roles, never a guessed directory.
    selected["wm_fod"] = audit.check(source["images"]["wm_fod"])
    selected["five_tissue"] = audit.check(source["images"]["five_tissue_act"])
    selected["atlases"] = {name: audit.check(profile["image"])
                           for name, profile in source["profiles"].items()}
    return selected, {"reference_manifest": binding["official_reference_manifest"],
                      "DWI_consumer": source["official_dwi_contract"], "rawprep_report": rawprep_record,
                      "modeling_report": modeling_record, "rawprep_timing_policy": rawprep.get("timing_policy"),
                      "official_FA_command": fa_command[0]["command"],
                      "official_gradient_command": gradient[0]["command"]}


def candidate_inputs(config, raw_case, binding, version, state_row, audit):
    case_id = raw_case["case_id"]
    job = Path(config["run_root"]) / version / case_id
    require(Path(state_row["gpu_report"]).resolve() == (job / "gpu_report.json").resolve() and
            Path(state_row["wall_report"]).resolve() == (job / "raw_bids_wall.json").resolve(),
            "completed controller row points outside the explicit planned job")
    gpu = audit.snapshot_json(state_row["gpu_report"])
    wall = audit.snapshot_json(state_row["wall_report"])
    require(gpu.get("status") == "completed" and gpu.get("exit_code") == 0
            and gpu.get("case_id") == case_id and gpu.get("version") == version,
            "candidate GPU producer incomplete or another case/version")
    require(Path(gpu["wall_report"]).resolve() == Path(state_row["wall_report"]).resolve(),
            "GPU worker's actual wall report differs")
    cohort.check_wall_report(wall, config, raw_case)
    cohort.check_selected_inputs(wall, raw_case)
    require(gpu["raw_input_provenance"] == raw_case["input_files"], "candidate original raw inputs differ")
    expected_verification = cohort.verify_inputs(raw_case)
    require(gpu["input_verification"] == gpu["input_verification_after"] == expected_verification,
            "candidate raw pre/post verification differs")
    expected_source = config["declared_source_manifests"][version]
    actual_source = cohort.source_manifest(config["sources"][version])
    require(actual_source == gpu["source_before"] == gpu["source_after"] == expected_source,
            "candidate source bytes differ from frozen before/after inventory")
    require(state_row["actual_source_fingerprint"] == expected_source["source_fingerprint"],
            "controller source identity differs")
    for name, digest in expected_source["source_sha256"].items():
        audit.check({"path": str(Path(expected_source["directory"]) / name), "sha256": digest},
                    allow_empty_source=True)
    audit.check(config["source_archives"][version])
    for key in ("worker_script", "wall_script"):
        audit.check({"path": config[key], "sha256": config[key + "_sha256"]})
    require(gpu["wall_script_sha256"] == config["wall_script_sha256"] and
            wall["provenance"]["benchmark_sha256"] == config["wall_script_sha256"],
            "actual wall tool changed")
    package = Path(config["sources"][version]) / "src/fnit"
    require(Path(wall["provenance"]["fnit_package"]).resolve() == package.resolve(),
            "actual imported FNIT package differs")
    for record in wall["provenance"]["loaded_fnit_files"].values():
        path = Path(record["path"]).resolve()
        relative = str(path.relative_to(Path(config["sources"][version]).resolve()))
        require(expected_source["source_sha256"].get(relative) == record["sha256"],
                f"loaded module bytes differ: {path}")
        audit.check(record, allow_empty_source=True)
    anatomy = binding["anatomy"]
    require(gpu["anatomy"] == gpu["anatomy_after"] == anatomy["files"] and
            cohort.check_anatomy(anatomy["directory"], config["atlases"]) == anatomy["files"]
            and Path(wall["selected_inputs"]["freesurfer_subject_dir"]).resolve() ==
                Path(anatomy["directory"]).resolve(), "supplied official anatomy differs")
    for item in anatomy["files"].values():
        audit.check(item)
    arguments = cohort.cli_command(config, raw_case, job, anatomy_subject=anatomy["directory"])
    require(wall["cli_arguments"] == arguments, "actual CLI parameters differ from formal plan")
    expected_command = [config["gpu_python"], config["wall_script"], "--mode", "wall",
                        "--eddy-gp-seed", str(config["eddy_gp_seed"]), "--report", str(job / "raw_bids_wall.json"),
                        "--gpu-uuid", config["gpu_uuid"], "--result-export-dir", str(job / "returned_result"),
                        "--", *arguments]
    require(gpu["command"] == expected_command, "actual worker argv differs from the completed formal plan")
    selected = {}
    for role in ("dwi", "bvals", "bvecs"):
        selected[role] = audit.check(wall["inputs"]["prepared/" + role])
        require(Path(wall["selected_inputs"][role]).resolve() == Path(selected[role]["path"]),
                f"selected/prepared candidate {role} differ")
        if role in ("dwi", "bvecs"):
            require(Path(selected[role]["path"]).is_relative_to((job / "connectome").resolve()),
                    f"candidate corrected {role} belongs to another run")
    canonical_bval = [item for item in raw_case["input_files"] if item["kind"] == "bval"]
    require(len(canonical_bval) == 1 and selected["bvals"]["sha256"] == canonical_bval[0]["sha256"],
            "candidate bval frames/shells differ from the raw acquisition")
    for item in wall["inputs"].values():
        # Only consumed files are checked; no software execution or reconstruction.
        audit.check(item)
    for role, filename in (("fa", "fa_dwi.nii.gz"), ("mask", "brain_mask_dwi.nii.gz"),
                           ("five_tissue", "five_tissue_dwi_world.nii.gz")):
        selected[role] = audit.check(wall["outputs"]["files"][filename])
        require(Path(selected[role]["path"]).resolve() == (job / "connectome" / filename).resolve(),
                "candidate output belongs to another job")
    exported = wall["post_timing_result_export"]
    require(exported.get("status") == "completed", "same-run FOD export did not complete")
    fod_path = job / "returned_result/wm_fod_normalized.nii.gz"
    require(str(fod_path) in exported["files"], "same-run normalized FOD lacks an explicit saved SHA")
    selected["wm_fod"] = audit.check({"path": str(fod_path), **exported["files"][str(fod_path)]})
    selected["atlases"] = {}
    for name in config["atlases"]:
        key = f"atlases/{name}/atlas_dwi.nii.gz"
        selected["atlases"][name] = audit.check(wall["outputs"]["files"][key])
    return selected, {"gpu_report": audit.records[str(Path(state_row["gpu_report"]).resolve())],
                      "wall_report": audit.records[str(Path(state_row["wall_report"]).resolve())],
                      "source_fingerprint": actual_source["source_fingerprint"],
                      "source_commit": config["source_archives"][version]["git_origin_commit"],
                      "total_runtime_seconds": wall["total_runtime_seconds"],
                      "memory_budget": wall["memory_budget"]}


class Statistics:
    """All values are counted; finite-only diagnostics are explicitly secondary."""

    def __init__(self):
        self.total = self.n = self.unequal = self.nonfinite_mismatch = 0
        self.counts = {name: 0 for name in ("candidate_nan", "candidate_posinf", "candidate_neginf",
                                           "reference_nan", "reference_posinf", "reference_neginf",
                                           "candidate_negative", "reference_negative", "zero_support_mismatch")}
        self.absolute_sum = self.square_sum = self.reference_square_sum = self.signed_sum = 0.0
        self.max_absolute = 0.0
        self.mean_x = self.mean_y = self.m2x = self.m2y = self.covariance = 0.0
        self.above = {str(value): 0 for value in (1e-6, 1e-5, 1e-4, 1e-3)}

    def add(self, candidate, reference):
        import numpy as np
        x, y = np.asarray(candidate, dtype=np.float64).ravel(), np.asarray(reference, dtype=np.float64).ravel()
        require(x.shape == y.shape, "metric arrays differ in shape")
        self.total += x.size
        self.unequal += int(np.count_nonzero(x != y))
        fx, fy = np.isfinite(x), np.isfinite(y)
        same_nonfinite = ((np.isnan(x) & np.isnan(y)) | (np.isposinf(x) & np.isposinf(y))
                          | (np.isneginf(x) & np.isneginf(y)))
        self.nonfinite_mismatch += int(np.count_nonzero(~(fx & fy) & ~same_nonfinite))
        for prefix, values in (("candidate", x), ("reference", y)):
            self.counts[prefix + "_nan"] += int(np.count_nonzero(np.isnan(values)))
            self.counts[prefix + "_posinf"] += int(np.count_nonzero(np.isposinf(values)))
            self.counts[prefix + "_neginf"] += int(np.count_nonzero(np.isneginf(values)))
            self.counts[prefix + "_negative"] += int(np.count_nonzero(values < 0))
        self.counts["zero_support_mismatch"] += int(np.count_nonzero((x == 0) != (y == 0)))
        x, y = x[fx & fy], y[fx & fy]
        if not x.size:
            return
        difference = x - y
        absolute = np.abs(difference)
        self.absolute_sum += float(absolute.sum(dtype=np.float64))
        self.signed_sum += float(difference.sum(dtype=np.float64))
        self.square_sum += float(np.dot(difference, difference))
        self.reference_square_sum += float(np.dot(y, y))
        self.max_absolute = max(self.max_absolute, float(absolute.max()))
        for value in self.above:
            self.above[value] += int(np.count_nonzero(absolute > float(value)))
        # Chan's merge avoids cancellation on large, nearly constant volumes.
        batch_n = x.size
        mx, my = float(x.mean()), float(y.mean())
        cx, cy = x - mx, y - my
        dx, dy = mx - self.mean_x, my - self.mean_y
        merged_n = self.n + batch_n
        correction = self.n * batch_n / merged_n
        self.m2x += float(np.dot(cx, cx)) + dx * dx * correction
        self.m2y += float(np.dot(cy, cy)) + dy * dy * correction
        self.covariance += float(np.dot(cx, cy)) + dx * dy * correction
        self.mean_x += dx * batch_n / merged_n
        self.mean_y += dy * batch_n / merged_n
        self.n = merged_n

    def report(self):
        finite = {"count": int(self.n), "mae": None, "rmse": None, "max_absolute": None,
                  "signed_mean_error": None, "relative_rmse_reference_rms": None, "pearson": None}
        if self.n:
            finite.update(mae=self.absolute_sum / self.n, rmse=math.sqrt(self.square_sum / self.n),
                          max_absolute=self.max_absolute, signed_mean_error=self.signed_sum / self.n,
                          relative_rmse_reference_rms=(math.sqrt(self.square_sum / self.reference_square_sum)
                                                      if self.reference_square_sum > 0 else None),
                          pearson=(max(-1.0, min(1.0, self.covariance / math.sqrt(self.m2x * self.m2y)))
                                   if self.m2x > 0 and self.m2y > 0 else None))
        return {"count_all_values": int(self.total), "finite_pair_count": int(self.n),
                "all_values_finite": self.total == self.n, "unequal_ieee": self.unequal,
                "nonfinite_state_mismatch": self.nonfinite_mismatch, **self.counts,
                "metrics_all_values": {key: value if self.total == self.n else None
                                       for key, value in finite.items() if key != "count"},
                "finite_pair_diagnostic": {**finite, "count_absolute_gt": self.above},
                "finite_pair_scope": "secondary diagnostic only; nonfinite cells remain counted above"}


def geometry(image, *, nonspatial_axis_type=None):
    import numpy as np
    affine = np.asarray(image.affine, dtype=np.float64)
    spacing = np.asarray(image.header.get_zooms(), dtype=np.float64)
    # Tissue/SH channels can carry an undefined nonspatial zoom in the
    # original MRtrix NIfTI header. Preserve that meaning as explicit null
    # metadata, with its axis, rather than modifying the header or data.
    return {"shape": [int(value) for value in image.shape],
            "affine": [[float(value) if np.isfinite(value) else None for value in row] for row in affine],
            "affine_nonfinite_count": int(np.count_nonzero(~np.isfinite(affine))),
            "spacing": [float(value) if np.isfinite(value) else None for value in spacing],
            "nonfinite_spacing_axes": np.flatnonzero(~np.isfinite(spacing)).astype(int).tolist(),
            "undefined_spacing": [{"axis": int(axis),
                                   "axis_type": "spatial" if axis < 3 else (nonspatial_axis_type or "unspecified_nonspatial"),
                                   "stored_header_value": str(float(spacing[axis]))}
                                  for axis in np.flatnonzero(~np.isfinite(spacing))],
            "storage_dtype": str(image.get_data_dtype())}


def geometry_check(candidate, reference, *, spatial_only=False, nonspatial_axis_type=None):
    import numpy as np
    shape_match = (candidate.shape[:3] == reference.shape[:3] if spatial_only else candidate.shape == reference.shape)
    delta = float(np.max(np.abs(np.asarray(candidate.affine) - np.asarray(reference.affine))))
    finite = bool(np.isfinite(candidate.affine).all() and np.isfinite(reference.affine).all())
    # Report header rounding tolerance explicitly; never interpolate or reorient arrays.
    return {"same_grid": bool(shape_match and finite and delta <= 1e-5), "shape_match": bool(shape_match),
            "affine_max_absolute_mm": delta if finite else None, "affine_tolerance_mm": 1e-5,
            "candidate": geometry(candidate, nonspatial_axis_type=nonspatial_axis_type),
            "reference": geometry(reference, nonspatial_axis_type=nonspatial_axis_type),
            "array_policy": "original stored voxel/frame indices; no resampling or canonical reorientation"}


def compare_image(candidate_record, reference_record, mask_record, *, expected_ndim=None):
    import nibabel as nib
    import numpy as np
    before = time.perf_counter()
    candidate = nib.load(candidate_record["path"], keep_file_open=True)
    reference = nib.load(reference_record["path"], keep_file_open=True)
    grid = geometry_check(candidate, reference, nonspatial_axis_type="time_frame" if expected_ndim == 4 else None)
    if not grid["same_grid"]:
        return {"status": "not_comparable_grid", "geometry": grid}
    require(len(candidate.shape) in (3, 4) and (expected_ndim is None or len(candidate.shape) == expected_ndim),
            "image dimensionality differs from its declared scientific role")
    mask_image = nib.load(mask_record["path"], keep_file_open=True)
    mask_grid = geometry_check(reference, mask_image, spatial_only=True)
    if not mask_grid["same_grid"] or len(mask_image.shape) != 3:
        return {"status": "not_comparable_official_mask_grid", "geometry": grid, "mask_geometry": mask_grid}
    mask_values = np.asarray(mask_image.dataobj, dtype=np.float64)
    require(np.isfinite(mask_values).all() and np.isin(mask_values, (0, 1)).all(),
            "official mask is nonfinite or not binary; refuse an invented mask")
    mask = mask_values != 0
    require(mask.any(), "official brain mask is empty")
    whole, brain = Statistics(), Statistics()
    frames = candidate.shape[3] if len(candidate.shape) == 4 else 1
    rows = []
    for frame in range(frames):
        selection = (..., frame) if len(candidate.shape) == 4 else (...,)
        x = np.asarray(candidate.dataobj[selection], dtype=np.float64)
        y = np.asarray(reference.dataobj[selection], dtype=np.float64)
        whole.add(x, y)
        brain.add(x[mask], y[mask])
        per_whole, per_brain = Statistics(), Statistics()
        per_whole.add(x, y)
        per_brain.add(x[mask], y[mask])
        finite = np.isfinite(x) & np.isfinite(y)
        maximum_index = None
        if finite.any():
            errors = np.where(finite, np.abs(x - y), -np.inf)
            maximum_index = [int(value) for value in np.unravel_index(np.argmax(errors), errors.shape)]
        row = {"frame_index": frame, "whole_volume": per_whole.report(), "official_brain_mask": per_brain.report(),
               "max_finite_error_voxel_ijk": maximum_index}
        # Exact FA tail quantiles; DWI uses complete per-frame/whole-volume reductions.
        if expected_ndim == 3:
            for name, selector in (("whole_volume", np.ones(x.shape, dtype=bool)), ("official_brain_mask", mask)):
                errors = np.abs(x[selector] - y[selector])
                row[name]["absolute_error_percentiles_all_values"] = (
                    {str(p): float(np.percentile(errors, p)) for p in (50, 95, 99, 99.9)}
                    if np.isfinite(errors).all() else None)
        rows.append(row)
    return {"status": "compared", "geometry": grid, "mask_geometry": mask_grid,
            "frames_retained": frames, "spatial_voxels_all": int(np.prod(candidate.shape[:3])),
            "official_brain_mask_voxels": int(mask.sum()), "whole_volume": whole.report(),
            "official_brain_mask": brain.report(), "per_frame": rows, "CPU_comparison_seconds": time.perf_counter() - before}


def compare_gradients(candidate, reference, frames):
    import numpy as np
    tables = {}
    for arm, records in (("candidate", candidate), ("reference", reference)):
        bvecs = np.loadtxt(records["bvecs"]["path"], dtype=np.float64, ndmin=2)
        bvals = np.loadtxt(records["bvals"]["path"], dtype=np.float64, ndmin=1).reshape(-1)
        require(bvecs.shape == (3, frames) and bvals.shape == (frames,),
                f"{arm} FSL gradients do not contain exactly every DWI frame")
        tables[arm] = (bvecs.T, bvals)
    x, bx = tables["candidate"]
    y, by = tables["reference"]
    rows = []
    for index in range(frames):
        finite = bool(np.isfinite(x[index]).all() and np.isfinite(y[index]).all()
                      and np.isfinite(bx[index]) and np.isfinite(by[index]))
        nx, ny = np.linalg.norm(x[index]), np.linalg.norm(y[index])
        angle = axial = None
        if finite and nx > 0 and ny > 0:
            cosine = float(np.clip(np.dot(x[index] / nx, y[index] / ny), -1, 1))
            angle, axial = math.degrees(math.acos(cosine)), math.degrees(math.acos(abs(cosine)))
        def number(value):
            return float(value) if np.isfinite(value) else None
        rows.append({"frame_index": index, "finite_all_components_bvals": finite,
                     "candidate_bvec": [number(v) for v in x[index]], "reference_bvec": [number(v) for v in y[index]],
                     "candidate_bval": number(bx[index]), "reference_bval": number(by[index]),
                     "candidate_norm": number(nx), "reference_norm": number(ny),
                     "candidate_exact_zero_vector": bool(np.all(x[index] == 0)),
                     "reference_exact_zero_vector": bool(np.all(y[index] == 0)),
                     "candidate_b0_lt50": bool(np.isfinite(bx[index]) and bx[index] < 50),
                     "reference_b0_lt50": bool(np.isfinite(by[index]) and by[index] < 50),
                     "directed_angle_degrees": angle, "antipodal_equivalent_angle_degrees": axial})
    stats, bval_stats, norm_stats = Statistics(), Statistics(), Statistics()
    stats.add(x, y)
    bval_stats.add(bx, by)
    norm_stats.add(np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1))
    return {"status": "compared", "FSL_shape": [3, frames], "frames_retained": frames,
            "bvec_components": stats.report(), "bvals": bval_stats.report(), "norms": norm_stats.report(),
            "zero_vector_support_mismatch": sum(row["candidate_exact_zero_vector"] != row["reference_exact_zero_vector"] for row in rows),
            "b0_support_lt50_mismatch": sum(row["candidate_b0_lt50"] != row["reference_b0_lt50"] for row in rows),
            "all_frames": rows, "policy": "compare original FSL voxel-frame vectors without normalization or sign alignment; b<50 only labels FNIT's b0 convention"}


def compare_case(config, raw_manifest, raw_case, binding, version, row, audit):
    candidate, candidate_identity = candidate_inputs(config, raw_case, binding, version, row, audit)
    reference, reference_identity = official_inputs(binding, raw_case, raw_manifest, config, audit)
    dwi = compare_image(candidate["dwi"], reference["dwi"], reference["mask"], expected_ndim=4)
    import nibabel as nib
    candidate_frames = nib.load(candidate["dwi"]["path"]).shape[-1]
    reference_frames = nib.load(reference["dwi"]["path"]).shape[-1]
    gradients = (compare_gradients(candidate, reference, candidate_frames) if candidate_frames == reference_frames else
                 {"status": "not_comparable_frame_count", "candidate_frames": candidate_frames, "reference_frames": reference_frames})
    fa = compare_image(candidate["fa"], reference["fa"], reference["mask"], expected_ndim=3)
    ancillary = {}
    for role in ("wm_fod", "five_tissue"):
        a, b = nib.load(candidate[role]["path"]), nib.load(reference[role]["path"])
        grid = geometry_check(a, b, nonspatial_axis_type="tissue_channel" if role == "five_tissue" else "SH_coefficient")
        ancillary[role] = {"status": "same_grid_not_numerically_assessed" if grid["same_grid"] else "not_comparable_grid",
                           "geometry": grid, "candidate_file": candidate[role], "reference_file": reference[role]}
    ancillary["atlases"] = {}
    for name in config["atlases"]:
        a, b = nib.load(candidate["atlases"][name]["path"]), nib.load(reference["atlases"][name]["path"])
        grid = geometry_check(a, b)
        ancillary["atlases"][name] = {"status": "same_grid_not_numerically_assessed" if grid["same_grid"] else "not_comparable_grid",
                                      "geometry": grid, "candidate_file": candidate["atlases"][name], "reference_file": reference["atlases"][name]}
    return {"case_id": raw_case["case_id"], "version": version, "status": "compared",
            "scientific_parity": "not_assessed",
            "candidate_producer": candidate_identity, "official_producer": reference_identity,
            "candidate_files": candidate, "official_files": reference,
            "corrected_dwi": dwi, "rotated_gradients": gradients, "FA": fa, "ancillary_geometry": ancillary,
            "comparison_scope": "independent end-to-end raw acquisitions; downstream FA can differ because corrected DWI/mask/rotated gradients differ; not fixed-input DTI parity"}


def execute(configuration, digest, output_dir, case_ids=None, versions=None):
    before = time.perf_counter()
    audit = Audit()
    config_record = {"path": str(Path(configuration).resolve()), "sha256": digest}
    config = audit.json(config_record)
    bindings, manifest = audit.json(config["input_bindings"]), audit.json(config["raw_manifest"])
    cases = {case["case_id"]: case for case in cohort.validate_manifest(manifest)}
    require(set(bindings["cases"]) == set(cases), "actual input bindings do not cover the canonical cohort")
    selected_cases = set(case_ids or cases)
    selected_versions = set(versions or ("candidate",))
    require(selected_cases <= cases.keys() and selected_versions <= config["sources"].keys(), "unknown case/version requested")
    root = Path(config["run_root"])
    # Output is a new sibling namespace, never inside/above a source or actual run.
    output = Path(output_dir).resolve()
    require(output.is_relative_to(root.resolve().parent) and output != root.resolve().parent,
            "comparison output must stay inside the new accuracy phase namespace")
    protected = [root.resolve(), Path(configuration).resolve().parent, *[Path(p).resolve() for p in config["sources"].values()]]
    protected += [Path(item["path"]).resolve().parent for case in cases.values() for item in case["input_files"]]
    protected += [Path(item["official_reference_manifest"]["path"]).resolve().parent for item in bindings["cases"].values()]
    protected += [Path(item["anatomy"]["directory"]).resolve() for item in bindings["cases"].values()]
    require(not any(output == item or output.is_relative_to(item) or item.is_relative_to(output) for item in protected),
            "comparison output overlaps an immutable input/source/producer namespace")
    require(not output.exists(), "fresh output directory required; prior evidence cannot be overwritten")
    state_path = root / "status.json"
    state = json.loads(state_path.read_bytes())
    require(same_identity(state["configuration"], config_record) and state["execution_order"] == config["execution_order"]
            and state["raw_manifest"] == config["raw_manifest"] and state["input_bindings"] == config["input_bindings"],
            "actual controller is not the explicit frozen accuracy configuration")
    actual_config = audit.snapshot_json(root / "configuration.json")
    require(actual_config == {**config, "frozen_sources": config["declared_source_manifests"]},
            "run configuration differs from the exact declared source/parameter freeze")
    reports, pending = {}, {}
    checked_rows = {}
    planned_pairs = {(item["version"], item["case_id"]) for item in config["execution_order"]}
    require(len(planned_pairs) == len(config["execution_order"]), "duplicate formal case/version execution")
    for version in selected_versions:
        for case_id in selected_cases:
            if (version, case_id) not in planned_pairs:
                pending[f"{version}/{case_id}"] = {"status": "not_scheduled",
                                                    "reason": "requested pair has no explicit formal execution; no arrays compared"}
    for item in config["execution_order"]:
        case_id, version = item["case_id"], item["version"]
        if case_id not in selected_cases or version not in selected_versions:
            continue
        key = f"{version}/{case_id}"
        row = state["cases"].get(key)
        if row is None or row.get("status") != "completed":
            pending[key] = {"status": "not_launched" if row is None else row.get("status"),
                            "reason": "no genuinely completed producer; no arrays compared"}
            continue
        checked_rows[key] = row
        reports[key] = compare_case(config, manifest, cases[case_id], bindings["cases"][case_id], version, row, audit)
    audit.finish()
    current = json.loads(state_path.read_bytes())
    require(all(current["cases"].get(key) == row for key, row in checked_rows.items()),
            "selected completed controller rows changed during comparison")
    result = {"schema_version": 1, "status": "compared_subset" if pending else "requested_completed_cases_compared",
              "scientific_parity": "not_assessed",
              "UTC": datetime.now(timezone.utc).isoformat(), "configuration": config_record,
              "script": {"path": str(Path(__file__).resolve()), "sha256": cohort.sha256(__file__)},
              "scope": "CPU read-only actual completed raw DWI/gradient/FA comparison; no GPU/official commands; no scientific acceptance implied",
              "coverage": {"requested_cases": sorted(selected_cases), "requested_versions": sorted(selected_versions),
                           "actually_compared": sorted(reports), "not_compared": pending},
              "cases": reports, "immutable_files_verified_before_after": list(audit.records.values()),
              "controller_snapshot": {"path": str(state_path), "selected_completed_rows": checked_rows,
                                      "policy": "other cases may progress; selected terminal rows must remain identical"},
              "metric_definitions": {"relative_rmse_reference_rms": "sqrt(sum((candidate-reference)^2)/sum(reference^2)); null when denominator is zero",
                                     "nonfinite": "all nonfinite states counted; primary metrics null; finite-pair diagnostics explicitly named and count retained",
                                     "mask": "the exact bound official modeling brain mask, binary nonzero support; no candidate intersection or error exclusion"},
              "total_CPU_wall_seconds": time.perf_counter() - before}
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    (output / "components.json").write_text(serialized)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--version", action="append", choices=("baseline", "candidate"))
    args = parser.parse_args()
    result = execute(args.configuration, args.configuration_sha256, args.output_dir, args.case_id, args.version)
    print(json.dumps({"status": result["status"], "coverage": result["coverage"], "output": str(args.output_dir / "components.json")}))


if __name__ == "__main__":
    main()
