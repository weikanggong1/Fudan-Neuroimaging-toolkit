"""CPU-only accuracy and timing analysis of a public multi-subject T1 cohort.

Contract: cohort_manifest.json supplies canonical 110-label metadata, frozen
source identity, raw T1 identities and explicit official norm/aseg/wmparc,
native/HR label and soft-volume paths for every planned subject. queue.json
supplies attempted component commands, exit status and actual watcher times.
FNIT outputs live in case_root/fnit_raw and case_root/fnit_stage unless those
directories are explicitly supplied. No fitting or external command is run.

Stage native metrics use that subject's official norm grid; raw native metrics
use its published T1 grid. HR uses official axes/spacing/integer phase with a
union field covering official, raw FNIT and stage FNIT outputs. Subjects are
accuracy samples, never random repeats. Failures remain in the planned count.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

# Reuse the existing CPU comparison implementation; no copied solver modules.
HELPERS = Path(__file__).resolve().parents[1] / "reproducibility_20261002"
sys.path.insert(0, str(HELPERS))
from analyze_repeatability import _grid, audit_group, sha256  # noqa: E402


STRUCTURES = ("brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right")
OFFICIAL_COMPONENTS = ("reconall", "official_brainstem", "official_thalamus", "official_hippo_amygdala")
MODES = ("raw", "stage")
RESOLUTIONS = ("native", "hr")
MEASURES = ("dice", "jaccard", "hard_volume_difference_mm3", "hard_volume_relative_difference",
            "soft_volume_difference_mm3", "soft_volume_relative_difference")


def read_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def finite_json(value, location="json"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite scalar: {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, f"{location}[{index}]")


def path_of(value, base: Path) -> Path:
    value = value["path"] if isinstance(value, dict) else value
    path = Path(value)
    return path if path.is_absolute() else base / path


def identity(value, base: Path) -> dict:
    path = path_of(value, base)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Required nonempty artifact: {path}")
    actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if isinstance(value, dict):
        for key in ("bytes", "sha256"):
            if key in value and value[key] != actual[key]:
                raise ValueError(f"Declared artifact {key} differs: {path}")
    actual["predeclared_identity_verified"] = isinstance(value, dict) and "sha256" in value
    return actual


def geometry(image, path) -> dict:
    affine = np.asarray(image.affine)
    if (len(image.shape) != 3 or any(size <= 0 for size in image.shape)
            or affine.shape != (4, 4) or not np.isfinite(affine).all()
            or abs(np.linalg.det(affine[:3, :3])) <= 0):
        raise ValueError(f"Invalid 3-D image geometry: {path}")
    return {"shape": list(map(int, image.shape)), "affine": affine.tolist(),
            "voxel_volume_mm3": abs(float(np.linalg.det(affine[:3, :3])))}


def scan_image(value, base: Path, *, allowed_labels=None, nonnegative=False) -> dict:
    record = identity(value, base)
    image = nib.load(record["path"])
    grid = geometry(image, record["path"])
    array = np.asarray(image.dataobj)
    finite = np.isfinite(array)
    result = {**record, "geometry": grid, "total_voxels": int(array.size),
              "finite_voxels": int(np.count_nonzero(finite)),
              "positive_finite_voxels": int(np.count_nonzero(finite & (array > 0))),
              "all_voxels_finite": bool(finite.all())}
    if not result["all_voxels_finite"]:
        raise ValueError(f"Nonfinite voxel values: {record['path']}")
    if allowed_labels is not None:
        if not np.equal(array, np.rint(array)).all():
            raise ValueError(f"Noninteger label values: {record['path']}")
        unique = np.unique(array).astype(np.int64)
        outside = unique[~np.isin(unique, [0, *allowed_labels])]
        if outside.size:
            raise ValueError(f"FNIT labels outside declared support in {record['path']}: {outside.tolist()}")
        result.update(all_voxels_integer=True, declared_label_support=True,
                      label_voxels={str(int(label)): int(np.count_nonzero(array == label)) for label in unique})
    elif nonnegative and (array < 0).any():
        raise ValueError(f"Negative voxel values: {record['path']}")
    return result


def audit_source(manifest: dict, base: Path) -> tuple[dict, dict]:
    source = path_of(manifest["source"], base)
    declared = manifest["source"] if isinstance(manifest["source"], dict) else {}
    source_manifest = source / "source_manifest.json"
    record = identity({"path": str(source_manifest), **({"sha256": declared["manifest_sha256"]}
                      if "manifest_sha256" in declared else {})}, base)
    snapshot = read_json(source_manifest)
    entries = snapshot["files"]
    names = [entry["path"] for entry in entries]
    if not entries or len(names) != len(set(names)) or any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError("Frozen source manifest has invalid/duplicate relative paths")
    for entry in entries:
        identity({**entry, "path": str(source / entry["path"])}, base)
    runtime = {entry["path"].removeprefix("src/fnit/"): entry["sha256"] for entry in entries
               if entry["path"].startswith("src/fnit/") and entry["path"].endswith(".py")}
    actual = {str(path.relative_to(source / "src/fnit")) for path in (source / "src/fnit").rglob("*.py")}
    if not runtime or set(runtime) != actual:
        raise ValueError("Frozen manifest must cover every runtime Python module")
    assets = [identity(value, base) for value in manifest.get("fixed_assets", [])]
    return runtime, {"source": str(source), "source_manifest": record, "verified_files": len(entries),
                     "verified_runtime_python_files": len(runtime), "fixed_assets": assets,
                     "manifest_metadata": {key: value for key, value in snapshot.items() if key != "files"}}


def canonical_labels(manifest: dict, base: Path) -> dict:
    labels = manifest.get("canonical_label_metadata")
    if labels is None:
        record = identity(manifest["label_metadata_file"], base)
        value = read_json(Path(record["path"]))
        labels = value.get("labels", value)
    labels = {str(int(key)): value for key, value in labels.items()}
    if len(labels) != 110 or "0" in labels:
        raise ValueError("Canonical metadata must contain exactly 110 nonzero atlas labels")
    for key, entry in labels.items():
        if int(entry["id"]) != int(key) or entry["source"] not in STRUCTURES or not entry["name"]:
            raise ValueError("Invalid canonical label metadata")
        if entry["source"].endswith("right") and not int(key) >= 10000:
            raise ValueError("Right hippo/amygdala IDs must carry the 10000 offset")
    return labels


def family(entry: dict) -> str:
    side = entry.get("hemisphere")
    return entry["parent"] + ("_" + side if entry["source"].startswith("hippo-amygdala") else "")


def successful_component(queue: dict, case_id: str, component: str) -> dict:
    entries = [run for run in queue.get("runs", []) if run["case_id"] == case_id and run["component"] == component]
    success = [entry for entry in entries if entry.get("exit_code") == 0 and entry.get("state") == "completed"
               and entry.get("status", "success") == "success"]
    if len(success) != 1:
        raise ValueError(f"Need one successful {component} for {case_id}; have {len(success)}")
    run = success[0]
    if run.get("state", run.get("status", "completed")) != "completed":
        raise ValueError(f"Successful component has unfinished state: {case_id}/{component}")
    if not run.get("command"):
        raise ValueError(f"Actual command missing: {case_id}/{component}")
    if component.startswith("fnit_") and run.get("phase") != "verified_full_run":
        raise ValueError("FNIT component must complete the full source/input verification gate")
    for record in run.get("artifacts", {}).values():
        if isinstance(record, dict) and "path" in record:
            identity(record, Path("/"))
    return run


def runtime_record(run: dict) -> dict:
    wall = run.get("process_wall_seconds")
    if wall is not None and (not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0):
        raise ValueError("Invalid actual watcher process wall")
    if wall is not None and "started_unix" in run and "finished_unix" in run:
        if abs(run["finished_unix"] - run["started_unix"] - wall) > .2:
            raise ValueError("Watcher process wall and start/finish timestamps disagree")
    return {"component": run["component"], "command": run["command"], "exit_code": run["exit_code"],
            "process_wall_seconds": wall, "started_unix": run.get("started_unix"),
            "finished_unix": run.get("finished_unix"),
            "process_wall_scope": run.get("process_wall_scope", "Not supplied; no timing scope inferred."),
            "timing_available": wall is not None, "gpu": run.get("physical_gpu"),
            "sampled_peak_own_memory_mib": run.get("sampled_peak_own_memory_mib")}


def audit_official(case: dict, base: Path) -> tuple[dict, dict]:
    record = identity(case["official"]["status_file"], base)
    report = read_json(Path(record["path"]))
    finite_json(report, "official_report")
    raw = identity(case["raw_t1"], base)
    if (report["case_id"] != case["id"] or report["state"] != "completed"
            or report["input"] != {key: raw[key] for key in ("path", "bytes", "sha256")}
            or report.get("reference_from_previous_subject_or_timing")):
        raise ValueError("Official report must bind this case's fresh completed public-T1 pipeline")
    software = report["software"]
    if not software["cpu_only"] or "8.2.0" not in software["reconall_version_output"]:
        raise ValueError("Official benchmark must retain the declared FS8.2 CPU run")
    for key in ("executables", "official_python_source", "native_libraries", "atlas_files"):
        if not software[key]:
            raise ValueError("Official source/binary/atlas audit missing")
        for value in software[key]:
            identity(value, base)
    # License content is not read or hashed. The existing official runner records availability only.
    if software["license"].get("content_read_or_copied"):
        raise ValueError("Official benchmark must not copy private license contents")
    components = report["components"]
    if set(components) != set(OFFICIAL_COMPONENTS):
        raise ValueError("Official report must contain recon-all and exactly three subregion commands")
    for name, component in components.items():
        if component["state"] != "completed" or component["exit_code"] != 0 or component["case_id"] != case["id"]:
            raise ValueError(f"Incomplete official component: {name}")
        if component["input_sha256"] != raw["sha256"] or component["threads"] != 4 or not component["cpu_only"]:
            raise ValueError("Official component input/thread/device differs")
        if component.get("record"):
            saved = read_json(path_of(component["record"], base))
            if any(saved.get(key) != value for key, value in component.items() if key != "record"):
                raise ValueError("Official aggregate and saved component record disagree")
        for value in component.get("outputs", {}).values():
            identity(value, base)
        for value in component.get("output_files", []):
            identity(value, base)
        for key in ("log", "completion_marker"):
            if key in component:
                identity(component[key], base)
        runtime_record(component)
    recon = components["reconall"]
    for name, value in (("norm", "norm.mgz"), ("aseg", "aseg.mgz"), ("wmparc", "wmparc.mgz")):
        if path_of(case["official"][name], base) != Path(recon["outputs"][value]["path"]):
            raise ValueError("Manifest stage input differs from newly completed recon-all output")
    for structure in STRUCTURES:
        component = components["official_hippo_amygdala" if structure.startswith("hippo-amygdala") else "official_" + structure]
        saved = {value["path"]: value for value in component["output_files"]}
        specification = case["official"]["subregions"][structure]
        for value in [specification["native"], specification["hr"], *specification["soft_volume_files"]]:
            path = str(path_of(value, base))
            if path not in saved:
                raise ValueError("Manifest official metric artifact missing from completed output identities")
            identity(saved[path], base)
    for key in ("complete_pipeline_wall_seconds", "summed_component_process_wall_seconds"):
        if not math.isfinite(report[key]) or report[key] <= 0:
            raise ValueError("Invalid recorded official end-to-end watcher timing")
    return report, record


def audit_gpu(run: dict, base: Path) -> dict:
    monitor = run.get("gpu_monitor", run.get("monitor"))
    if monitor is None:
        raise ValueError("FNIT own-process GPU monitor identity missing")
    record = identity(monitor, base)
    samples = [json.loads(line) for line in Path(record["path"]).read_text().splitlines() if line.strip()]
    clean = [sample for sample in samples if "sampling_error" not in sample]
    if not clean or not run.get("pid"):
        raise ValueError("No successful own-process GPU samples/PID")
    for sample in clean:
        if "own_pid" in sample and sample["own_pid"] != run["pid"]:
            raise ValueError("Wrong PID in GPU monitor")
        if "processes" in sample:
            memory = max([entry["used_memory_mib"] for entry in sample["processes"] if entry["pid"] == run["pid"]] or [0])
            if memory != sample["own_memory_mib"]:
                raise ValueError("PID-attributed GPU memory disagrees")
        if run.get("physical_gpu") is not None and sample.get("physical_gpu", run["physical_gpu"]) != run["physical_gpu"]:
            raise ValueError("Physical GPU identity changed in monitor")
    peak = max(sample.get("own_memory_mib", 0) for sample in clean)
    if run.get("memory_limit_exceeded") or run.get("own_memory_limit_mib") != 19073 or not 0 < peak <= 19073:
        raise ValueError("FNIT own sampled GPU memory violates 19073 MiB cap")
    if run.get("sampled_peak_own_memory_mib") != peak:
        raise ValueError("Sampled GPU peak differs from runner status")
    return {**record, "samples": len(samples), "sampling_errors": len(samples) - len(clean),
            "sampled_peak_own_memory_mib": peak, "own_memory_limit_mib": 19073,
            "scope": "Own PID sampled memory, not an instantaneous maximum."}


def fnit_output(case: dict, mode: str, base: Path) -> Path:
    explicit = case.get("fnit", {}).get(mode)
    return path_of(explicit, base) if explicit is not None else path_of(case["case_root"], base) / f"fnit_{mode}"


def audit_fnit(case: dict, mode: str, queue: dict, base: Path, canonical: dict, runtime: dict) -> dict:
    case_id = case["id"]
    run = successful_component(queue, case_id, f"fnit_{mode}")
    output = fnit_output(case, mode, base)
    api_path, report_path = output / "api_report.json", output / "report.json"
    api, report = read_json(api_path), read_json(report_path)
    finite_json(api, "api"); finite_json(report, "report")
    for name, path in (("api_report", api_path), ("report", report_path), ("context_identity", output / "context_identity.json")):
        if name in run:
            if path_of(run[name], base) != path:
                raise ValueError("Runner artifact path differs from actual FNIT output")
            identity(run[name], base)
    if api["structures"] != list(STRUCTURES) or api["labels"] != canonical or set(api["volumes"]) != set(canonical):
        raise ValueError("FNIT must be the complete all-structure pipeline with canonical 110 labels")
    if report["source_sha256"] != runtime or report["validation_mode"] == "quick_diagnostic":
        raise ValueError("FNIT runtime source differs or uses incomplete schedules")
    command = run["command"]
    if (command.count("--structures") != 1 or command[command.index("--structures") + 1] != "all"
            or "--quick" in command or command.count("--optimization") != 1
            or command[command.index("--optimization") + 1] != report["optimization"]):
        raise ValueError("Executed FNIT command differs from full-scope API/profile")
    if mode == "raw" and any(value.startswith("--reference-") for value in command):
        raise ValueError("Raw fit must be run independently before loading official scoring references")
    official = case["official"]
    inputs = [case["raw_t1"]] if mode == "raw" else [official[key] for key in ("norm", "aseg", "wmparc")]
    fingerprints = [identity(value, base) for value in inputs]
    expected = {record["path"]: record["sha256"] for record in fingerprints}
    if report["input_sha256"] != expected or Path(report["input"]) != Path(fingerprints[0]["path"]):
        raise ValueError("FNIT actual raw/stage inputs do not match this subject's manifest")
    if "input_sha256" in run and run["input_sha256"] != expected:
        raise ValueError("FNIT runner and report input identities disagree")
    jacobians = api["fit_min_jacobians"]
    if set(jacobians) != set(STRUCTURES) or any(value is None or not math.isfinite(value) or value <= 0 for value in jacobians.values()):
        raise ValueError("Every fitted tetrahedral mesh must have a finite positive minimum Jacobian")
    if jacobians != report["fit_min_jacobians"]:
        raise ValueError("FNIT API/report minimum Jacobians disagree")
    for label, volume in api["volumes"].items():
        if any(not isinstance(volume[key], (int, float)) or not math.isfinite(volume[key]) or volume[key] < 0
               for key in ("soft_volume_mm3", "hard_volume_mm3")):
            raise ValueError(f"Invalid FNIT volume: {label}")
    input_scan = scan_image(inputs[0], base)
    native = scan_image(api["files"]["labels"], base, allowed_labels=list(map(int, canonical)))
    if (native["geometry"]["shape"] != input_scan["geometry"]["shape"]
            or not np.allclose(native["geometry"]["affine"], input_scan["geometry"]["affine"], atol=1e-5, rtol=0)):
        raise ValueError("FNIT native output grid differs from its actual mode input")
    for label, volume in api["volumes"].items():
        measured = native["label_voxels"].get(label, 0) * native["geometry"]["voxel_volume_mm3"]
        if not math.isclose(measured, volume["hard_volume_mm3"], rel_tol=1e-5, abs_tol=1e-4):
            raise ValueError(f"FNIT native hard-volume metadata/count mismatch: {label}")
    arrays = {"labels": native}
    for structure in STRUCTURES:
        allowed = [int(label) for label, entry in canonical.items() if entry["source"] == structure]
        key = f"highres/{structure}"
        arrays[key] = scan_image(api["files"][key], base, allowed_labels=allowed)
    observer_path = output / "context_identity.json"
    context = None
    if observer_path.is_file():
        context = read_json(observer_path); finite_json(context, "context")
        if context.get("solver_options_changed") or context.get("reference_used_by_observer") or not context.get("runtime_source_unchanged", True):
            raise ValueError("Context observer changed solver or used official reference")
        for key in ("runtime_source_sha256_before", "runtime_source_sha256_after"):
            if key in context and context[key] != runtime:
                raise ValueError("Observer observed another runtime source")
    times = runtime_record(run)
    times.update(api_compute_seconds=api["timings"]["compute_seconds"], api_total_seconds=report["api_total_seconds"],
                 output_save_seconds=report["output_save_seconds"], api_step_timings=api["timings"],
                 context_observer_seconds=context.get("observer_seconds") if context else None,
                 context_observer_scope="Measured API/process times retain observer overhead; no subtraction estimate.")
    for key in ("api_compute_seconds", "api_total_seconds", "output_save_seconds"):
        if times[key] is None or not math.isfinite(times[key]) or times[key] < 0:
            raise ValueError(f"FNIT timing missing/nonfinite/negative: {key}")
    return {"api": api, "report": report, "run": run, "record": {
        "output": str(output), "api_report": identity(str(api_path), base), "report": identity(str(report_path), base),
        "input_sha256": expected, "configuration": {key: report.get(key) for key in ("optimization", "torch_version", "cuda_version")},
        "input_grid_and_valid_voxels": input_scan, "arrays": arrays, "gpu_monitor": audit_gpu(run, base),
        "fit_min_jacobians": jacobians,
        "numeric_gates": {"all_saved_native_and_hr_voxels_finite_integer": True, "labels_within_declared_support": True,
                          "all_reported_volumes_finite_nonnegative": True, "native_hard_volumes_match_saved_labels": True,
                          "minimum_fitted_tetrahedral_jacobians_finite_positive": True,
                          "jacobian_scope": "Saved minimum over fitted mesh tetrahedra; no voxelwise Jacobian image was saved/evaluated."},
        "context_identity": context, "context_identity_artifact": identity(str(observer_path), base) if context else None,
        "shared_preprocessing": api["initialization"]["shared_preprocessing"], "timings": times}}


def empty_rows(case: dict, canonical: dict, status: str, error: str, modes=MODES) -> list[dict]:
    return [{"case_id": case["id"], "development_seen": case["development_seen"], "mode": mode,
             "resolution": resolution, "space": f"{mode}_{resolution}", "label": int(label), "name": entry["name"],
             "family": family(entry), "source": entry["source"], "measurement_status": status,
             "hard_status": "not_evaluated", "na_reason": error, **{key: None for key in MEASURES}}
            for mode in modes for resolution in RESOLUTIONS for label, entry in canonical.items()]


def audit_case(case: dict, queue: dict, base: Path, canonical: dict, runtime: dict) -> tuple[dict, list, list]:
    official_report, official_report_record = audit_official(case, base)
    official_runs = official_report["components"]
    reference = case["official"]
    official_inputs = {key: identity(reference[key], base) for key in ("norm", "aseg", "wmparc")}
    raw_identity = identity(case["raw_t1"], base)
    # The actual recon-all command must bind the exact selected published T1.
    command = official_runs["reconall"]["command"]
    if "-i" not in command or Path(command[command.index("-i") + 1]) != Path(raw_identity["path"]):
        raise ValueError("Official recon-all did not use this case's original published T1")
    outputs, mode_failures = {}, {}
    for mode in MODES:
        try:
            outputs[mode] = audit_fnit(case, mode, queue, base, canonical, runtime)
        except Exception as error:
            mode_failures[mode] = f"{type(error).__name__}: {error}"
    if not outputs:
        raise ValueError("No complete valid FNIT mode: " + json.dumps(mode_failures))
    configurations = {mode: output["record"]["configuration"] for mode, output in outputs.items()}
    if len(outputs) == 2 and configurations["raw"] != configurations["stage"]:
        raise ValueError("Raw/stage source environment or optimization profiles differ")
    if ("stage" in outputs and official_runs["reconall"].get("finished_unix") is not None
            and outputs["stage"]["run"].get("started_unix") is not None
            and official_runs["reconall"]["finished_unix"] > outputs["stage"]["run"]["started_unix"]):
        raise ValueError("FNIT stage must follow newly completed recon-all inputs")
    rows = [row for mode, error in mode_failures.items() for row in empty_rows(case, canonical, "mode_failed_or_invalid", error, (mode,))]
    groups, references = [], {}
    for structure in STRUCTURES:
        specification = reference["subregions"][structure]
        # Official raw hemisphere labels remain unshifted on disk. Offset is applied exactly once in comparison.
        scans = {key: scan_image(specification[key], base) for key in RESOLUTIONS}
        for scan in scans.values():
            values = np.asarray(nib.load(scan["path"]).dataobj)
            if not np.equal(values, np.rint(values)).all() or (values < 0).any():
                raise ValueError("Official label maps must contain finite nonnegative integers")
            if structure.endswith("right") and np.isin(values, [int(label) for label, entry in canonical.items() if entry["source"] == structure]).any():
                raise ValueError("Official right labels already carry the FNIT 10000 offset; refusing a second offset")
        volumes = [identity(value, base) for value in specification["soft_volume_files"]]
        references[structure] = {"arrays": scans, "soft_volume_files": volumes}
    families = sorted({family(entry) for entry in canonical.values()})
    for family_name in families:
        labels = sorted(int(label) for label, entry in canonical.items() if family(entry) == family_name)
        names = {str(label): canonical[str(label)]["name"] for label in labels}
        structure = canonical[str(labels[0])]["source"]
        specification = reference["subregions"][structure]
        official_hr = nib.load(references[structure]["arrays"]["hr"]["path"])
        hr_images = [official_hr, *[nib.load(output["api"]["files"][f"highres/{structure}"]) for output in outputs.values()]]
        hr_shape, hr_affine = _grid({"union_like": str(path_of(specification["hr"], base))}, hr_images, base)
        for mode in outputs:
            api = outputs[mode]["api"]
            for resolution in RESOLUTIONS:
                original_grid = {"like": str(path_of(case["raw_t1"] if mode == "raw" else reference["norm"], base))}
                grid = original_grid if resolution == "native" else {"shape": list(hr_shape), "affine": hr_affine.tolist()}
                group = {"id": f"{case['id']}__{family_name}_{mode}_{resolution}", "family": family_name,
                         "space": f"{mode}_{resolution}", "label_ids": labels, "label_names": names, "grid": grid,
                         "official": [{"id": "official", "labels": str(path_of(specification[resolution], base)),
                                       "label_offset": 10000 if structure.endswith("right") else 0,
                                       "soft_volumes_files": [str(path_of(value, base)) for value in specification["soft_volume_files"]],
                                       "provenance": {"case_id": case["id"], "reference_only_after_fit": True}}],
                         "fnit": [{"id": "fnit", "labels": api["files"]["labels" if resolution == "native" else f"highres/{structure}"],
                                   "label_offset": 0, "soft_volumes_mm3": {str(label): api["volumes"][str(label)]["soft_volume_mm3"] for label in labels},
                                   "provenance": {"case_id": case["id"], "mode": mode, "structures": "all"}}]}
                measured = audit_group(group, base)
                pair = next(pair for pair in measured["pairs"] if pair["kind"] == "cross_method")
                ref_hard = sum(value["first_voxels"] for value in pair["regions"]) * measured["grid"]["voxel_volume_mm3"]
                fnit_hard = sum(value["second_voxels"] for value in pair["regions"]) * measured["grid"]["voxel_volume_mm3"]
                ref_soft = sum(value["first_soft_volume_mm3"] for value in pair["regions"] if value["first_soft_volume_mm3"] is not None)
                fnit_soft = sum(value["second_soft_volume_mm3"] for value in pair["regions"] if value["second_soft_volume_mm3"] is not None)
                jaccard = [value["jaccard"] for value in pair["regions"] if value["jaccard"] is not None]
                soft_relative = [value["soft_volume_relative_difference"] for value in pair["regions"] if value["soft_volume_relative_difference"] is not None]
                family_summary = {"case_id": case["id"], "development_seen": case["development_seen"],
                                  "family": family_name, "mode": mode, "resolution": resolution, "space": measured["space"],
                                  "measurement_status": "evaluated", "grid": measured["grid"],
                                  "reference_weighted_dice": pair["first_reference_volume_weighted_label_dice"],
                                  "mean_label_dice": pair["mean_label_dice"], "foreground_dice": pair["foreground_dice"],
                                  "mean_label_jaccard": float(np.mean(jaccard)) if jaccard else None,
                                  "official_hard_volume_mm3": ref_hard, "fnit_hard_volume_mm3": fnit_hard,
                                  "hard_volume_difference_mm3": abs(ref_hard - fnit_hard),
                                  "hard_volume_relative_difference": 2 * abs(ref_hard - fnit_hard) / (ref_hard + fnit_hard) if ref_hard + fnit_hard > 0 else None,
                                  "official_soft_volume_mm3": ref_soft, "fnit_soft_volume_mm3": fnit_soft,
                                  "soft_volume_difference_mm3": abs(ref_soft - fnit_soft),
                                  "soft_volume_relative_difference": 2 * abs(ref_soft - fnit_soft) / (ref_soft + fnit_soft) if ref_soft + fnit_soft > 0 else None,
                                  "mean_roi_soft_relative_difference": float(np.mean(soft_relative)) if soft_relative else None,
                                  "different_voxels": pair["different_voxels"], "evaluated_labels": pair["evaluated_labels"],
                                  "both_empty_labels": pair["both_empty_labels"], "runs": measured["runs"]}
                groups.append(family_summary)
                for values in pair["regions"]:
                    official_count, fnit_count = values["first_voxels"], values["second_voxels"]
                    status = ("both_empty_hard_label" if not official_count and not fnit_count else
                              "official_present_fnit_hard_label_absent" if not fnit_count else
                              "official_hard_label_absent_fnit_present" if not official_count else "evaluated")
                    soft_a, soft_b = values["first_soft_volume_mm3"], values["second_soft_volume_mm3"]
                    if soft_a is None or soft_b is None:
                        raise ValueError(f"Missing official/FNIT soft volume for {names[str(values['label'])]}")
                    voxel = measured["grid"]["voxel_volume_mm3"]
                    rows.append({"case_id": case["id"], "development_seen": case["development_seen"], "mode": mode,
                                 "resolution": resolution, "space": measured["space"], "label": values["label"],
                                 "name": names[str(values["label"])], "family": family_name, "source": structure,
                                 "measurement_status": "evaluated", "hard_status": status,
                                 "na_reason": "Both hard labels empty; Dice/Jaccard undefined." if status == "both_empty_hard_label" else None,
                                 "dice": values["dice"], "jaccard": values["jaccard"],
                                 "official_voxels": official_count, "fnit_voxels": fnit_count,
                                 "intersection_voxels": values["intersection_voxels"],
                                 "union_voxels": official_count + fnit_count - values["intersection_voxels"],
                                 "different_voxels": values["different_voxels"],
                                 "measurement_grid_total_voxels": int(np.prod(measured["grid"]["shape"])),
                                 "mode_input_finite_voxels": outputs[mode]["record"]["input_grid_and_valid_voxels"]["finite_voxels"],
                                 "mode_input_positive_finite_voxels": outputs[mode]["record"]["input_grid_and_valid_voxels"]["positive_finite_voxels"],
                                 "voxel_volume_mm3": voxel,
                                 "official_hard_volume_mm3": official_count * voxel, "fnit_hard_volume_mm3": fnit_count * voxel,
                                 "hard_volume_difference_mm3": values["hard_volume_difference_mm3"],
                                 "hard_volume_relative_difference": values["hard_volume_relative_difference"],
                                 "hard_volume_reference_relative_difference": abs(fnit_count - official_count) / official_count if official_count else None,
                                 "official_soft_volume_mm3": soft_a, "fnit_soft_volume_mm3": soft_b,
                                 "soft_volume_difference_mm3": values["soft_volume_difference_mm3"],
                                 "soft_volume_relative_difference": values["soft_volume_relative_difference"],
                                 "soft_volume_reference_relative_difference": abs(soft_b - soft_a) / soft_a if soft_a > 0 else None,
                                 "soft_na_reason": "Both soft volumes zero." if soft_a + soft_b == 0 else None})
    if len(rows) != 440 or len(groups) != 12 * len(outputs):
        raise ValueError("A subject must retain 440 ROI-space rows and 12 family-space groups per valid mode")
    official_times = {name: runtime_record(run) for name, run in official_runs.items()}
    case_status = queue.get("cases", {}).get(case["id"], {})
    timing = {"fnit": {mode: output["record"]["timings"] for mode, output in outputs.items()}, "official": official_times,
              "official_end_to_end_seconds": official_report["complete_pipeline_wall_seconds"],
              "official_end_to_end_scope": official_report["complete_wall_scope"],
              "case_end_to_end_seconds": case_status.get("case_end_to_end_seconds"),
              "case_end_to_end_scope": case_status.get("case_end_to_end_scope", "Not supplied; no wall time inferred.")}
    return {"case_id": case["id"], "development_seen": case["development_seen"], "status": "completed" if not mode_failures else "partial_modes",
            "raw_t1": raw_identity, "official_inputs": official_inputs, "official_outputs": references,
            "official_report": official_report_record, "official_software": official_report["software"],
            "official_explicit_step_timers": {key: value.get("explicit_stage_timers") for key, value in official_runs.items()},
            "official_shared_load_resources": {key: {name: value.get(name) for name in ("resources_before", "resources_after")} for key, value in official_runs.items()},
            "fnit": {mode: output["record"] for mode, output in outputs.items()}, "mode_failures": mode_failures,
            "timings": timing, "numeric_gates_passed_for_completed_modes": True, "configuration": configurations}, rows, groups


def distribution(values, planned: int) -> dict:
    defined = [float(value) for value in values if value is not None]
    standard_deviation = float(np.std(defined, ddof=1)) if len(defined) >= 2 else None
    return {"planned_subjects": planned, "defined_subjects": len(defined), "missing_or_na_subjects": planned - len(defined),
            "mean": float(np.mean(defined)) if defined else None, "median": float(np.median(defined)) if defined else None,
            "min": min(defined) if defined else None, "max": max(defined) if defined else None,
            "std": standard_deviation, "std_sample": standard_deviation, "std_ddof": 1}


def case_scores(rows: list[dict]) -> list[dict]:
    result = []
    for case_id, space in sorted({(row["case_id"], row["space"]) for row in rows}):
        current = [row for row in rows if row["case_id"] == case_id and row["space"] == space]
        complete = len(current) == 110 and all(row["measurement_status"] == "evaluated" for row in current)
        defined = [row for row in current if row.get("dice") is not None]
        first = sum(row["official_voxels"] for row in current) if complete else None
        second = sum(row["fnit_voxels"] for row in current) if complete else None
        intersections = sum(row["intersection_voxels"] for row in current) if complete else None
        soft_differences = [row["soft_volume_relative_difference"] for row in current
                            if row.get("soft_volume_relative_difference") is not None]
        result.append({"case_id": case_id, "space": space, "mode": current[0]["mode"],
                       "resolution": current[0]["resolution"], "development_seen": current[0]["development_seen"],
                       "measurement_status": "evaluated" if complete else "not_evaluated",
                       "eligible_for_case_ranking": complete and bool(first),
                       "na_reason": None if complete else current[0].get("na_reason"),
                       "total_official_voxels": first, "total_fnit_voxels": second,
                       "total_intersection_voxels": intersections,
                       "reference_weighted_dice": sum(row["official_voxels"] * row["dice"] for row in defined) / first if complete and first else None,
                       "mean_label_dice": float(np.mean([row["dice"] for row in defined])) if complete and defined else None,
                       "mean_label_jaccard": float(np.mean([row["jaccard"] for row in defined])) if complete and defined else None,
                       "micro_label_dice": 2 * intersections / (first + second) if complete and first + second else None,
                       "evaluated_labels": len(defined) if complete else None,
                       "both_empty_labels": 110 - len(defined) if complete else None,
                       "reported_roi_rows": len(current),
                       "mean_roi_soft_relative_difference": float(np.mean(soft_differences)) if complete and soft_differences else None})
    return result


def cohort_summary(name: str, selected: list[dict], case_records: list[dict], rows: list[dict], groups: list[dict], scores: list[dict]) -> dict:
    ids = {case["id"] for case in selected}; planned = len(ids)
    observed = [record for record in case_records if record["case_id"] in ids]
    roi, families, runtime = [], [], []
    for key in sorted({(row["space"], row["label"]) for row in rows}):
        current = [row for row in rows if row["case_id"] in ids and (row["space"], row["label"]) == key]
        evaluated = [row for row in current if row["measurement_status"] == "evaluated"]
        roi.append({"space": key[0], "label": key[1], "name": current[0]["name"], "family": current[0]["family"],
                    "completed_subjects": len(evaluated), "planned_subjects": planned,
                    "hard_status_counts": {status: sum(row["hard_status"] == status for row in current) for status in sorted({row["hard_status"] for row in current})},
                    "measurements": {measure: distribution([row.get(measure) for row in current], planned) for measure in MEASURES}})
    for key in sorted({(row["space"], row["family"]) for row in rows}):
        current = [group for group in groups if group["case_id"] in ids and (group["space"], group["family"]) == key]
        families.append({"space": key[0], "family": key[1], "planned_subjects": planned, "completed_subjects": len(current),
                         "measurements": {measure: distribution([group[measure] for group in current], planned) for measure in
                                          ("reference_weighted_dice", "mean_label_dice", "mean_label_jaccard", "foreground_dice", "different_voxels",
                                           "hard_volume_difference_mm3", "hard_volume_relative_difference", "soft_volume_difference_mm3",
                                           "soft_volume_relative_difference", "mean_roi_soft_relative_difference")}})
    for mode in MODES:
        current = [record["timings"]["fnit"][mode] for record in observed if mode in record.get("timings", {}).get("fnit", {})]
        runtime.append({"component": f"fnit_{mode}", "measurements": {measure: distribution([entry.get(measure) for entry in current], planned)
                       for measure in ("api_compute_seconds", "api_total_seconds", "output_save_seconds", "process_wall_seconds")}})
    for component in OFFICIAL_COMPONENTS:
        current = [record["timings"]["official"][component] for record in observed if component in record.get("timings", {}).get("official", {})]
        runtime.append({"component": component, "measurements": {"process_wall_seconds": distribution([entry.get("process_wall_seconds") for entry in current], planned)}})
    for measure in ("official_end_to_end_seconds", "case_end_to_end_seconds"):
        runtime.append({"component": measure, "measurements": {measure: distribution([record.get("timings", {}).get(measure) for record in observed], planned)}})
    return {"cohort": name, "planned_subjects": planned, "case_ids": sorted(ids),
            "completed_subjects": sum(record["status"] == "completed" for record in observed),
            "failed_or_unavailable_subjects": [record["case_id"] for record in observed if record["status"] != "completed"],
            "case_status": [{key: record.get(key) for key in ("case_id", "status", "error", "development_seen")} for record in observed],
            "roi": roi, "family": families, "runtime": runtime,
            "case_scores": [{"space": space, "measurements": {measure: distribution(
                               [row[measure] for row in scores if row["case_id"] in ids and row["space"] == space], planned)
                               for measure in ("reference_weighted_dice", "mean_label_dice", "mean_label_jaccard", "micro_label_dice", "mean_roi_soft_relative_difference")}}
                            for space in sorted({row["space"] for row in scores})],
            "interpretation": "Subject-level distributions of accuracy/runtime, not repeated-run noise or confidence intervals. NA/failures are retained in planned denominators."}


def figure_candidates(rows: list[dict], case_rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["space"] == "raw_native" and row["measurement_status"] == "evaluated"]
    scores = [{"case_id": row["case_id"], "reference_weighted_dice": row["reference_weighted_dice"]}
              for row in case_rows if row["space"] == "raw_native" and row["eligible_for_case_ranking"]]
    scores.sort(key=lambda item: (item["reference_weighted_dice"], item["case_id"]))
    worst_roi = sorted((row for row in eligible if row["dice"] is not None and row["official_voxels"] > 0),
                       key=lambda row: (row["dice"], -row["official_voxels"], row["label"], row["case_id"]))
    return {"rule": "All completed raw-native subjects; median uses lower central case if even count; worst ROI sorts Dice ascending, official voxel count descending, label ID and case ID. No image appearance criterion.",
            "full_ascending_raw_native_ranking": scores,
            "unrankable_cases": [{"case_id": row["case_id"], "reason": row["na_reason"] or "No official foreground voxels."}
                                 for row in case_rows if row["space"] == "raw_native" and not row["eligible_for_case_ranking"]],
            "median_subject": scores[(len(scores) - 1) // 2] if scores else None,
            "worst_subject": scores[0] if scores else None,
            "worst_roi": {key: worst_roi[0][key] for key in ("case_id", "label", "name", "dice", "official_voxels")} if worst_roi else None}


def merged_queues(paths: list[Path]) -> dict:
    queues, unavailable = [], []
    for path in paths:
        try:
            queue = read_json(path)
        except (FileNotFoundError, json.JSONDecodeError):
            unavailable.append(str(path)); continue
        queues.append((path, queue))
    terminal = {"completed", "failed", "completed_with_failures"}
    states = [queue.get("state", queue.get("status", "not_available")) for _, queue in queues]
    combined = {"state": "completed" if not unavailable and all(state == "completed" for state in states) else
                         "completed_with_failures" if not unavailable and all(state in terminal for state in states) else "waiting",
                "runs": [], "cases": {}, "queue_sources": [], "unavailable_queue_paths": unavailable}
    for path, queue in queues:
        combined["queue_sources"].append({"path": str(path), "sha256": sha256(path),
                                          "state": queue.get("state", queue.get("status")),
                                          "metadata": {key: value for key, value in queue.items() if key not in ("runs", "cases")}})
        combined["runs"].extend(queue.get("runs", []))
        cases = queue.get("cases", {})
        if isinstance(cases, list):
            cases = {case["case_id"]: case for case in cases}
        for case_id, record in cases.items():
            existing = combined["cases"].setdefault(case_id, {})
            for key, value in record.items():
                if key in existing and existing[key] != value:
                    raise ValueError(f"Separate queue case metadata conflict: {case_id}/{key}")
                existing[key] = value
    return combined


def attempt_kind(run: dict) -> str:
    if run.get("status") == "blocked" or run.get("phase") == "failed_official_preprocessing":
        return "setup_dependency_blocked"
    if run.get("exit_code") == 0 and run.get("state") == "completed" and run.get("status", "success") == "success":
        return "successful"
    if run.get("exit_code") is not None:
        return "process_or_output_validation_failure"
    if run.get("state") == "failed" or run.get("status") == "failed":
        return "setup_or_preflight_failure"
    return "not_completed"


def recover_timing(case: dict, queue: dict, base: Path, canonical: dict, runtime: dict) -> dict:
    """Keep verified FNIT runtimes when unavailable reference outputs prevent Dice."""
    fnit, official, errors = {}, {}, {}
    for mode in MODES:
        try:
            fnit[mode] = audit_fnit(case, mode, queue, base, canonical, runtime)["record"]["timings"]
        except Exception as error:
            errors[f"fnit_{mode}"] = f"{type(error).__name__}: {error}"
    try:
        report = read_json(path_of(case["official"]["status_file"], base))
        for name, component in report.get("components", {}).items():
            if component.get("state") == "completed" and component.get("exit_code") == 0:
                official[name] = runtime_record(component)
    except Exception as error:
        errors["official_report"] = f"{type(error).__name__}: {error}"
    return {"fnit": fnit, "official": official, "timing_recovery_errors": errors,
            "official_end_to_end_seconds": None, "case_end_to_end_seconds": None,
            "scope": "Only complete verified FNIT modes or completed official components retained; missing references still prevent accuracy evaluation."}


def audit_queue_metadata(queue: dict, manifest: dict, manifest_path: Path, source_record: dict) -> list[dict]:
    base = manifest_path.parent
    verified = []
    for record in queue["queue_sources"]:
        metadata = record["metadata"]
        if "runtime_python_files" not in metadata:
            continue  # Official aggregate protocol is verified per subject.
        if (metadata["source"]["sha256"] != source_record["source_manifest"]["sha256"]
                or metadata["runtime_python_files"] != source_record["verified_runtime_python_files"]
                or metadata["manifest"]["sha256"] != sha256(manifest_path)
                or metadata["observer_changes_solver_options"] or metadata["reference_used_for_fitting"]
                or metadata["optimization"] != manifest["fnit_configuration"]["optimization"]
                or metadata["threads_per_process"] != manifest["fnit_configuration"]["threads"]
                or metadata["own_memory_limit_mib"] != 19073):
            raise ValueError("FNIT queue source/full-scope/precision/thread/memory protocol differs from manifest")
        artifacts = [identity(metadata[key], base) for key in ("source", "manifest", "run_driver", "observer", "queue_script")]
        assets = [identity(value, base) for value in metadata["assets"]]
        verified.append({"queue_path": record["path"], "artifacts": artifacts, "verified_asset_files": len(assets),
                         "cohort_manifest_sha256": sha256(Path(metadata["manifest"]["path"])),
                         "assets": assets, "physical_gpu_by_mode": metadata["physical_gpu_by_mode"]})
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--queue", action="append", type=Path,
                        help="Repeat for independent FNIT and official queues; their runs are merged read-only.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=86400)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--no-wait", action="store_true", help="Audit current saved outputs and retain unavailable cases.")
    parser.add_argument("--preparation-history", action="append", type=Path,
                        help="Preserved failed setup queues/reports, listed separately from benchmark fit failures.")
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("poll-seconds must be between 1 and 60")
    manifest_path = args.manifest or args.root / "cohort_manifest.json"
    queue_paths = args.queue or [args.root / "fnit_queue.json", args.root / "official_queue.json"]
    output = args.output_dir or args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status_path = output / "analysis_status.json"
    status = {"state": "waiting", "started_unix": started, "analysis_script_sha256": sha256(Path(__file__)),
              "scope": "CPU-only saved real-subject outputs; no fitting, GPU or official-software launch."}
    save_json(status_path, status)
    while True:
        queue = merged_queues(queue_paths)
        if args.no_wait or queue.get("state") in ("completed", "failed", "completed_with_failures") or time.time() - started >= args.timeout_seconds:
            break
        time.sleep(args.poll_seconds)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Only cohort manifest schema_version=1 is supported")
    cases = manifest["cases"]
    ids = [case["id"] for case in cases]
    if len(cases) != 10 or len(set(ids)) != 10 or sum(bool(case["development_seen"]) for case in cases) != 1:
        raise ValueError("Need ten predetermined distinct subject IDs with exactly one development_seen case")
    if len({case.get("subject_key", case["id"]) for case in cases}) != 10:
        raise ValueError("Cohort must contain ten distinct published subjects")
    canonical = canonical_labels(manifest, manifest_path.parent)
    runtime, source_record = audit_source(manifest, manifest_path.parent)
    if len({case["raw_t1"]["sha256"] for case in cases}) != 10:
        raise ValueError("Ten different subjects must not reuse an identical raw T1 byte stream")
    if manifest.get("reference_used_for_raw_fnit_fitting"):
        raise ValueError("Official labels must not affect the raw FNIT fit")
    queue_metadata_audit = audit_queue_metadata(queue, manifest, manifest_path, source_record)
    known = set(ids)
    if any(run["case_id"] not in known for run in queue.get("runs", [])):
        raise ValueError("Queue contains subjects outside the predetermined manifest")
    records, rows, groups = [], [], []
    status.update(state="auditing", planned_subjects=10, queue_state=queue.get("state"))
    save_json(status_path, status)
    for case in cases:
        try:
            record, case_rows, case_groups = audit_case(case, queue, manifest_path.parent, canonical, runtime)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            attempts = [run for run in queue.get("runs", []) if run["case_id"] == case["id"]]
            failed = any(attempt_kind(run) in ("process_or_output_validation_failure", "setup_or_preflight_failure") for run in attempts)
            blocked = any(attempt_kind(run) == "setup_dependency_blocked" for run in attempts)
            case_state = "setup_dependency_blocked" if blocked and not failed else "failed_or_invalid" if failed or not isinstance(error, FileNotFoundError) else "not_available"
            record = {"case_id": case["id"], "development_seen": case["development_seen"], "status": case_state,
                      "error": reason, "numeric_gates_passed": False, "attempts": attempts,
                      "timings": recover_timing(case, queue, manifest_path.parent, canonical, runtime)}
            case_rows, case_groups = empty_rows(case, canonical, case_state, reason), []
        records.append(record); rows.extend(case_rows); groups.extend(case_groups)
        status.update(completed_case_audits=len(records), current_case=case["id"])
        save_json(status_path, status)
        print(f"audited {case['id']}: {record['status']}", flush=True)
    if len(rows) != 4400:
        raise ValueError("Predetermined cohort must retain 4400 ROI-space rows including failed/unavailable cases")
    scores = case_scores(rows)
    preparation = [{"artifact": identity(str(path), manifest_path.parent), "record": read_json(path)}
                   for path in args.preparation_history or []]
    preparation_summary = [{"artifact": entry["artifact"],
                            "case_outcomes": [{"case_id": case_id, "state": case_record.get("state"),
                                               "child_launched": case_record.get("pid") is not None,
                                               "exit_code": case_record.get("exit_code"),
                                               "failure": case_record.get("failure"), "log": case_record.get("log")}
                                              for case_id, case_record in entry["record"].get("cases", {}).items()],
                            "scope": "Archived environment/startup preparation; separate from FNIT algorithm outcomes."}
                           for entry in preparation]
    result = {"schema_version": 1, "analysis_script_sha256": sha256(Path(__file__)),
              "comparison_helper_sha256": sha256(HELPERS / "analyze_repeatability.py"),
              "manifest_sha256": sha256(manifest_path), "queue_sources": queue["queue_sources"],
              "source_audit": source_record, "dataset": manifest.get("dataset"),
              "snapshot": manifest.get("snapshot"), "license": manifest.get("license"), "dataset_doi": manifest.get("dataset_doi"),
              "selection_rule": manifest.get("selection_rule"), "queue_protocol_audit": queue_metadata_audit,
              "planned_subjects": 10, "planned_roi_space_rows": 4400,
              "methods": {"design": "Ten predetermined different subjects, one final FNIT all-structure result per mode; no within-method repeatability inference.",
                  "cohorts": "cohort_all includes ten planned subjects; cohort_new_subjects excludes development_seen and keeps nine planned subjects, including failures.",
                  "raw": "Published T1 as project input, without added project defacing; automatic FNIT preprocessing is part of measured raw pipeline.",
                  "stage": "This subject's complete official recon-all norm/aseg/wmparc; separate conditional accuracy from raw end-to-end.",
                  "native": "Raw uses original published T1 grid; stage uses its actual official norm input grid. Official labels mapped by nearest-neighbor.",
                  "hr": "One official HR axes/spacing/integer phase union grid covering official plus both FNIT modes; no fitted registration or phase.",
                  "support": "Only requested same-family atlas labels count; other labels are background. Official right hippo/amygdala nonzero IDs receive +10000 once.",
                  "na": "Both-empty hard labels have undefined Dice/Jaccard; one-sided absence has zero Dice/Jaccard. Soft relative difference is abs(a-b)/mean(a,b), undefined if both zero.",
                  "failure": "Every planned subject retains 440 rows. Missing/failed/invalid outputs carry NA with reasons, never zero scores or accuracy-based exclusion.",
                  "aggregation": "Macro subject distributions of each ROI and each case-family score; sample standard deviation ddof=1. Planned denominators, valid counts and NA counts are explicit.",
                  "jacobian": "Finite positive saved minimum of each fitted tetrahedral mesh. No voxelwise Jacobian image or unrecorded all-field finite assertion.",
                  "timing": "API compute/save/total and actual runner watcher process wall reported separately; official recon-all, each subregion command and recorded end-to-end times. No synthetic step Dice."},
              "cases": records, "roi_rows": rows, "family_rows": groups, "case_rows": scores,
              "failed_attempts": [run for run in queue.get("runs", []) if attempt_kind(run) == "process_or_output_validation_failure"],
              "setup_dependency_blocked_attempts": [run for run in queue.get("runs", []) if attempt_kind(run) == "setup_dependency_blocked"],
              "setup_or_preflight_failures": [run for run in queue.get("runs", []) if attempt_kind(run) == "setup_or_preflight_failure"],
              "preserved_preparation_history": preparation,
              "run_preparation_summary": preparation_summary,
              "not_completed_attempts": [run for run in queue.get("runs", []) if attempt_kind(run) == "not_completed"],
              "figure_selection": figure_candidates(rows, scores), "elapsed_cpu_and_wait_seconds": time.time() - started}
    result["final_outcome_ready"] = queue["state"] in ("completed", "completed_with_failures") and not result["not_completed_attempts"]
    summary = {key: value for key, value in result.items() if key not in ("roi_rows", "family_rows", "cases")}
    summary["cohort_all"] = cohort_summary("cohort_all", cases, records, rows, groups, scores)
    new = [case for case in cases if not case["development_seen"]]
    summary["cohort_new_subjects"] = cohort_summary("cohort_new_subjects", new, records, rows, groups, scores)
    save_json(output / "cohort_analysis.json", result)
    save_json(output / "cohort_summary.json", summary)
    columns = sorted({key for row in rows for key in row})
    with (output / "cohort_roi.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t"); writer.writeheader(); writer.writerows(rows)
    compact_families = [{key: value for key, value in group.items() if key not in ("runs", "grid")} for group in groups]
    family_columns = sorted({key for row in compact_families for key in row})
    with (output / "cohort_family.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=family_columns, delimiter="\t"); writer.writeheader(); writer.writerows(compact_families)
    with (output / "cohort_case.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in scores for key in row}), delimiter="\t"); writer.writeheader(); writer.writerows(scores)
    artifacts = [output / name for name in ("cohort_analysis.json", "cohort_summary.json", "cohort_roi.tsv", "cohort_family.tsv", "cohort_case.tsv")]
    status.update(state="partial_snapshot" if not result["final_outcome_ready"] else
                  "completed" if all(record["status"] == "completed" for record in records) else "completed_with_failures_or_unavailable",
                  finished_unix=time.time(), completed_subjects=sum(record["status"] == "completed" for record in records),
                  final_outcome_ready=result["final_outcome_ready"],
                  artifacts={path.name: identity(str(path), output) for path in artifacts})
    save_json(status_path, status)
    print(json.dumps({key: status[key] for key in ("state", "planned_subjects", "completed_subjects")}), flush=True)


if __name__ == "__main__":
    main()
