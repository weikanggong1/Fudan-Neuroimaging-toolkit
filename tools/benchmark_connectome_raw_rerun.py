"""Private controlled raw-DWI rerun after recorded cohort setup/tool failures.

Reconstruction is the official raw-T1 reconstruction from this same round.
All DWI stages execute again in a newly claimed namespace. This is neither
production resume nor a pristine cold ten-subject benchmark.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from . import benchmark_connectome_raw_cohort as cohort
    from . import benchmark_connectome_raw_recovery as recovery
else:
    import benchmark_connectome_raw_cohort as cohort
    import benchmark_connectome_raw_recovery as recovery

MODE = "same_round_common_compatibility_raw_dwi_rerun"


def bound_json(binding, description):
    path = Path(binding["path"])
    if path.is_symlink() or cohort.sha256(path) != binding["sha256"]:
        raise ValueError(f"{description} byte identity changed")
    return json.loads(path.read_text())


def origin_state(config):
    if config.get("recovery_mode") != MODE:
        raise ValueError("controlled raw-DWI rerun requires its explicit private mode")
    if cohort.sha256(__file__) != config["rerun_worker_sha256"]:
        raise ValueError("controlled rerun helper byte identity changed")
    origin = config["rerun_origin"]
    state = bound_json({"path": origin["snapshot_path"], "sha256": origin["snapshot_sha256"]}, "original driver snapshot")
    original = state["config"]
    if original.get("pilot") or set(original["sources"]) != {"baseline"}:
        raise ValueError("controlled rerun must originate in the formal baseline cohort")
    if (state.get("fresh_namespace", {}).get("status") != "claimed_fresh_namespace"
            or state.get("fresh_namespace", {}).get("path") != original["run_root"]):
        raise ValueError("original reconstruction namespace was not claimed fresh")
    new_root = Path(config["run_root"]).resolve()
    old_root = Path(original["run_root"]).resolve()
    if new_root.is_relative_to(old_root) or old_root.is_relative_to(new_root):
        raise ValueError("controlled raw-DWI rerun needs a different output namespace")
    for key in ("cpu_host", "gpu_host", "cpu_python", "gpu_python", "anatomy_validation_python", "freesurfer_home",
                "recon_all", "cpu_threads", "gpu_cpu_threads", "gpu_lock", "gpu_uuid", "device", "n_seeds", "seed",
                "eddy_gp_seed", "atlases", "atlas_options", "cuda_visible_devices", "gpu_path_prefix"):
        if config.get(key) != original.get(key):
            raise ValueError(f"controlled rerun changed the original scientific/runtime setting: {key}")
    for key in ("worker_script", "wall_script"):
        if cohort.sha256(original[key]) != original[key + "_sha256"]:
            raise ValueError(f"original frozen {key} changed")
    manifest = Path(original["run_root"]) / "input_manifest.json"
    if cohort.sha256(manifest) != origin["input_manifest_sha256"]:
        raise ValueError("original raw-input manifest changed")
    original_source = cohort.source_manifest(original["sources"]["baseline"])
    if original_source["source_fingerprint"] != original["frozen_sources"]["baseline"]["source_fingerprint"]:
        raise ValueError("original frozen baseline source changed")
    return state


def original_paths(config, case, version="baseline"):
    if version != "baseline":
        raise ValueError("controlled recovery is baseline only; candidate uses ordinary fresh reconstruction")
    state = origin_state(config)
    original = state["config"]
    manifest = json.loads((Path(original["run_root"]) / "input_manifest.json").read_text())
    declared = [item for item in manifest["cases"] if item["case_id"] == case["case_id"]]
    if declared != [case]:
        raise ValueError("controlled rerun case differs from the original raw manifest")
    job = Path(original["run_root"]) / version / case["case_id"]
    subject = recovery.assert_namespace(original, case, version, job)
    return original, job, subject


def validate_original(config, case, version="baseline"):
    original_config, job, subject = original_paths(config, case, version)
    path = job / "recon_report.json"
    original = json.loads(path.read_text())
    recovery.validate_original_recon(original_config, case, version, job, original)
    if cohort.sha256(original_config["recon_all"]) != original["executable_sha256"]:
        raise ValueError("original official reconstruction executable changed")
    if cohort.sha256(Path(original_config["freesurfer_home"]) / "SetUpFreeSurfer.sh") != original["setup_script_sha256"]:
        raise ValueError("original official reconstruction setup changed")
    return original, path, subject


def required_resource_paths(config):
    """Actual current CLI resources, excluding case-specific original anatomy."""
    files = [("synthstrip", Path(config["fnit_weights"]) / "synthstrip.1.pt")]
    options = dict(zip(config["atlas_options"][::2], config["atlas_options"][1::2]))
    tian = [name for name in config["atlases"] if "+tian" in name]
    if tian:
        templates = Path(options["--atlas-templates-dir"])
        registration = options.get("--mni-template") or options.get("--tian-fnirt-coeff")
        if not registration:
            raise ValueError("Tian requires the explicitly configured registration resource")
        files.append(("tian_registration", Path(registration)))
        for scale in sorted({4 if name.endswith("s4") else 1 for name in tian}):
            files += [("tian", templates / f"Tian_Subcortex_S{scale}_3T.nii.gz"),
                      ("tian_labels", templates / f"Tian_Subcortex_S{scale}_3T_label.txt")]
        if "--mni-template" in options:
            weights = Path(options["--synthmorph-weights"])
            files += [("synthmorph", weights / name) for name in ("synthmorph.affine.2.h5", "synthmorph.deform.3.h5")]
        surface_atlases = [name for name in tian if name.startswith(("schaefer", "glasser"))]
        if surface_atlases:
            files += [("fsaverage", Path(options["--fsaverage-dir"]) / "surf" / f"{hemi}.sphere.reg") for hemi in ("lh", "rh")]
        for parcels in (200, 500, 1000):
            if any(name.startswith(f"schaefer{parcels}+") for name in tian):
                files += [("schaefer", templates / f"{hemi}.Schaefer2018_{parcels}Parcels_7Networks_order.annot") for hemi in ("lh", "rh")]
        if any(name.startswith("glasser") for name in tian):
            files.append(("glasser", templates / "Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"))
            surfaces = templates.parent / "surfaces"
            for side in ("L", "R"):
                files += [("glasser_surface", surfaces / f"{side}.sphere.32k_fs_LR.surf.gii"),
                          ("glasser_surface", surfaces / f"fs_{side}-to-fs_LR_fsaverage.{side}_LR.spherical_std.164k_fs_{side}.surf.gii")]
            executable = next((Path(prefix) / "wb_command" for prefix in config["gpu_path_prefix"] if (Path(prefix) / "wb_command").is_file()), None)
            if executable is None:
                raise FileNotFoundError("declared GPU PATH prefixes lack Workbench")
            files.append(("workbench", executable))
            # Official Linux packages expose a shell launcher and a separate
            # ELF executable. Record both rather than hashing only the launcher.
            with executable.open("rb") as stream:
                prefix = stream.read(4096)
            if b'"$directory"/../exe_rh_linux64/wb_command "$@"' in prefix:
                files.append(("workbench_binary", executable.resolve().parent.parent / "exe_rh_linux64/wb_command"))
    files += [("gpu_python", Path(config["gpu_python"])), ("cpu_python", Path(config["cpu_python"])),
              ("anatomy_validation_python", Path(config["anatomy_validation_python"]))]
    return files


def known_resource_identities(config, *, source_version="baseline"):
    """Read pinned project identities as data without importing production."""
    source = Path(config["sources"][source_version])
    tree = ast.parse((source / "src/fnit/weights.py").read_text())
    weights = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "WEIGHT_FILES" for target in node.targets))
    atlas = json.loads((source / "src/fnit/connectome/atlas_manifest.json").read_text())
    atlas = atlas.get("files", atlas.get("assets", atlas))
    return weights, atlas


def build_resources(config, *, source_version="baseline"):
    weights, atlases = known_resource_identities(config, source_version=source_version)
    files = []
    for role, path in required_resource_paths(config):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"required {role} resource missing/empty: {path}")
        actual = cohort.sha256(path)
        if path.name in weights:
            _, size, sha = weights[path.name]
            if path.stat().st_size != size or actual != sha:
                raise ValueError(f"resource differs from pinned official checkpoint: {path}")
        if path.name in atlases and isinstance(atlases[path.name], dict):
            expected = atlases[path.name]
            if actual != expected["sha256"] or path.stat().st_size != expected.get("size_bytes", expected.get("size", path.stat().st_size)):
                raise ValueError(f"atlas differs from pinned project manifest: {path}")
        if role in ("workbench", "workbench_binary", "gpu_python", "cpu_python", "anatomy_validation_python") and not cohort.os.access(path, cohort.os.X_OK):
            raise PermissionError(f"required executable is not executable: {path}")
        files.append({"role": role, "path": str(path), "size_bytes": path.stat().st_size, "sha256": actual})
    programs = []
    for item in files:
        if item["role"] == "workbench":
            result = subprocess.run([item["path"], "-version"], capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError("actual Workbench launcher/version check failed")
            programs.append({"role": "workbench", "argv": [item["path"], "-version"],
                             "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()})
    return {"status": "required_files_read_and_hashed", "files": files, "program_checks": programs,
            "scope": "resource byte/readability preflight; not successful MRI inference or redistribution permission"}


def verify_resources(config):
    manifest = bound_json(config["resources_manifest"], "resource manifest")
    files = manifest.get("files", [])
    expected = {(role, str(path)) for role, path in required_resource_paths(config)}
    if {(item["role"], item["path"]) for item in files} != expected or len(files) != len(expected):
        raise ValueError("resource manifest does not cover the exact required files")
    for item in files:
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or cohort.sha256(path) != item["sha256"]:
            raise ValueError(f"required resource byte identity changed: {path}")
    return manifest


def verify_common_source(config):
    state = origin_state(config)
    identity = bound_json(config["common_identity"], "common compatibility identity")
    if identity.get("performance_optimization") is not False:
        raise ValueError("controlled baseline source must be common compatibility, not an optimization")
    source = cohort.source_manifest(config["sources"]["baseline"])
    if source["source_fingerprint"] != config["frozen_sources"]["baseline"]["source_fingerprint"]:
        raise ValueError("common compatibility source differs from its frozen fingerprint")
    original_hashes = state["config"]["frozen_sources"]["baseline"]["source_sha256"]
    hashes = source["source_sha256"]
    changed = sorted(name for name in set(original_hashes) | set(hashes) if original_hashes.get(name) != hashes.get(name))
    if changed != ["src/fnit/flirt/core.py"] or hashes.get(changed[0]) != identity.get("core_sha256"):
        raise ValueError("common baseline differs from original by more than the declared MGH compatibility repair")
    wall_sha = cohort.sha256(config["wall_script"])
    if wall_sha != config["wall_script_sha256"]:
        raise ValueError("controlled raw-DWI wall runner differs from its frozen byte identity")
    if wall_sha != state["config"]["wall_script_sha256"]:
        change = config.get("wall_change", {})
        if (change.get("original_sha256") != state["config"]["wall_script_sha256"]
                or change.get("new_sha256") != wall_sha
                or change.get("scope") != "common evaluator: mode-isolated diagnostic export, reserved-memory budget and allocator-environment reporting"
                or identity.get("wall_script_sha256", identity.get("wall_sha256")) != wall_sha):
            raise ValueError("new common wall evaluator requires its explicit identity and change scope")
    return {"identity": identity, "identity_binding": config["common_identity"], "changed_files": changed,
            "wall_change": config.get("wall_change"),
            "original_source_fingerprint": state["config"]["frozen_sources"]["baseline"]["source_fingerprint"],
            "common_source_fingerprint": source["source_fingerprint"]}


def assert_new_job(config, case, version, job):
    expected = Path(config["run_root"]) / version / case["case_id"]
    if Path(job) != expected or any(path.is_symlink() for path in (Path(config["run_root"]), expected.parent, expected)):
        raise ValueError("controlled rerun output must remain in its exact new namespace")
    old_job = original_paths(config, case, version)[1]
    if expected.resolve().is_relative_to(old_job.resolve()) or old_job.resolve().is_relative_to(expected.resolve()):
        raise ValueError("new raw-DWI namespace overlaps original case output")


def revalidate_worker(payload):
    config, case, version = payload["config"], payload["case"], payload["version"]
    if cohort.sha256(__file__) != config["rerun_worker_sha256"]:
        raise ValueError("controlled rerun worker byte identity changed")
    verify_common_source(config)
    original, path, subject = validate_original(config, case, version)
    job = Path(config["run_root"]) / version / case["case_id"]
    assert_new_job(config, case, version, job)
    cohort.require_fresh(job)
    started = time.perf_counter()
    report = {"action": "bind_same_round_official_anatomy", "status": "running", "mode": MODE,
              "case_id": case["case_id"], "subject": case["subject"], "version": version, "start_utc": cohort.utc(),
              "identity": cohort.host_identity(), "original_report": {"path": str(path), "sha256": cohort.sha256(path)},
              "original_failure": original["error"], "anatomy_subject_dir": str(subject),
              "recon_all_rerun": False, "raw_dwi_preprocessing_resumed": False,
              "origin_snapshot_sha256": config["rerun_origin"]["snapshot_sha256"],
              "rerun_worker_sha256": config["rerun_worker_sha256"]}
    try:
        report["input_verification"] = cohort.verify_inputs(case)
        anatomy = cohort.check_anatomy(subject, config["atlases"])
        if anatomy != original["anatomy"]:
            raise ValueError("same-round official anatomy changed after original reconstruction")
        report["anatomy"] = anatomy
        keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        old = {key: cohort.os.environ.get(key) for key in keys}
        try:
            cohort.os.environ.update({key: str(config["cpu_threads"]) for key in keys})
            report["anatomy_geometry"] = cohort.validate_anatomy_child(subject, anatomy, config["anatomy_validation_python"])
        finally:
            for key, value in old.items():
                if value is None:
                    cohort.os.environ.pop(key, None)
                else:
                    cohort.os.environ[key] = value
        report["input_verification_after"] = cohort.verify_inputs(case)
        if cohort.check_anatomy(subject, config["atlases"]) != anatomy:
            raise ValueError("anatomy changed during actual array reread")
        recovery.immutable_original(path, report["original_report"]["sha256"])
        recovery.assert_fresh_gpu(job)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    report.update(end_utc=cohort.utc(), revalidation_wall_seconds=time.perf_counter() - started)
    cohort.atomic_json(job / "anatomy_origin.json", report)
    return report


def load_anatomy(config, case, version, job):
    assert_new_job(config, case, version, job)
    verify_common_source(config)
    verify_resources(config)
    original, path, subject = validate_original(config, case, version)
    binding_path = Path(job) / "anatomy_origin.json"
    if binding_path.is_symlink():
        raise ValueError("anatomy binding must not be a symbolic link")
    binding = json.loads(binding_path.read_text())
    checks = {"action": "bind_same_round_official_anatomy", "status": "completed", "mode": MODE,
              "case_id": case["case_id"], "subject": case["subject"], "version": version,
              "anatomy_subject_dir": str(subject), "recon_all_rerun": False, "raw_dwi_preprocessing_resumed": False,
              "original_report": {"path": str(path), "sha256": cohort.sha256(path)}, "original_failure": original["error"],
              "origin_snapshot_sha256": config["rerun_origin"]["snapshot_sha256"], "rerun_worker_sha256": config["rerun_worker_sha256"]}
    for key, expected in checks.items():
        if binding.get(key) != expected:
            raise ValueError(f"same-round anatomy binding changed: {key}")
    anatomy = cohort.check_anatomy(subject, config["atlases"])
    if anatomy != original["anatomy"] or binding.get("anatomy") != anatomy:
        raise ValueError("same-round official anatomy hashes changed")
    inputs = cohort.verify_inputs(case)
    if binding.get("input_verification") != inputs or binding.get("input_verification_after") != inputs:
        raise ValueError("same-round raw inputs changed")
    if binding.get("anatomy_geometry", {}).get("status") != "actual_images_surfaces_annotations_read":
        raise ValueError("same-round official anatomy was not actually read")
    copied = copy.deepcopy(original)
    copied.update(status="completed", anatomy_geometry=binding["anatomy_geometry"],
                  rerun={"mode": MODE, "anatomy_subject_dir": str(subject), "original_report": binding["original_report"],
                         "anatomy_binding": {"path": str(binding_path), "sha256": cohort.sha256(binding_path)},
                         "original_failure": original["error"], "common_identity": config["common_identity"],
                         "resources_manifest": config["resources_manifest"], "original_reconstruction_reused_from_this_round": True,
                         "raw_dwi_preprocessing_resumed": False, "pristine_cold_benchmark": False})
    return copied


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def rerun_timing(record, gpu, head_end_utc, driver_queue, reconstruction_worker_wall_seconds=None):
    head_start = recovery.timestamp(record["original_driver_start_utc"])
    head_end = recovery.timestamp(head_end_utc)
    node_start = recovery.timestamp(record["original_recon_start_utc"])
    node_end = recovery.timestamp(record["original_recon_end_utc"])
    recon = nonnegative(record["original_recon_command_seconds"], "reconstruction duration")
    worker = nonnegative(record.get("original_recon_worker_wall_seconds", reconstruction_worker_wall_seconds), "original CPU worker monotonic wall")
    if node_end < node_start or recon > node_end - node_start + .001 or recon > worker + .001:
        raise ValueError("reconstruction command exceeds its same-node recorded interval")
    queue = nonnegative(driver_queue, "head GPU queue") + nonnegative(gpu.get("gpu_lock_queue_seconds", 0.), "GPU lock queue")
    full = head_end - head_start
    if full < 0 or queue > full + .001:
        raise ValueError("head full elapsed or GPU queue is invalid")
    binding = record["revalidation_report"]
    node_revalidation = recovery.timestamp(binding["end_utc"]) - recovery.timestamp(binding["start_utc"])
    if node_revalidation < 0:
        raise ValueError("anatomy revalidation same-node timestamp order is invalid")
    result = {"head_driver_full_elapsed_utc_seconds": full,
              "head_driver_full_elapsed_utc_excluding_gpu_queue_seconds": full - queue,
              "original_recon_same_node_elapsed_utc_seconds": node_end - node_start,
              "original_recon_command_seconds": recon, "original_recon_worker_monotonic_seconds": worker,
              "driver_start_minus_recon_worker_start_utc_seconds": head_start - node_start,
              "revalidation_same_node_elapsed_utc_seconds": node_revalidation,
              "revalidation_monotonic_seconds": nonnegative(binding["revalidation_wall_seconds"], "anatomy revalidation wall"),
              "gpu_driver_queue_seconds": driver_queue, "gpu_lock_queue_seconds": gpu.get("gpu_lock_queue_seconds", 0.),
              "timestamps_adjusted": False, "pristine_cold_benchmark": False,
              "scope": "head UTC start-to-end interval including prior failures and repair gaps; CPU and GPU command timers are their own host monotonic intervals; observed cross-host start difference includes SSH launch plus clock difference; no timestamps shifted and no stage durations substituted for full wall"}
    if record.get("revalidation_head_start_utc") and record.get("revalidation_head_end_utc"):
        start = recovery.timestamp(record["revalidation_head_start_utc"])
        end = recovery.timestamp(record["revalidation_head_end_utc"])
        result["revalidation_head_elapsed_utc_seconds"] = end - start
        if record.get("original_driver_end_utc"):
            result["prior_failure_to_revalidation_head_gap_utc_seconds"] = start - recovery.timestamp(record["original_driver_end_utc"])
    return result


def collect_gpu_result(record, gpu, driver_queue, head_end_utc):
    """A reporting failure must not overwrite the actual execution failure."""
    record.update(status=gpu["status"], gpu_report=gpu, end_utc=head_end_utc)
    if gpu.get("error"):
        record["error"] = gpu["error"]
    try:
        record["timing"] = rerun_timing(record, gpu, head_end_utc, driver_queue)
    except Exception as error:
        record["timing_error"] = {"type": type(error).__name__, "message": str(error)}
    return record


def atomic_cases_csv(path, records):
    """The controlled report uses its own host-scoped timing column names."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{cohort.uuid.uuid4().hex}.tmp")
    columns = ("version", "case_id", "subject", "status", "original_driver_start_utc", "original_recon_start_utc",
               "original_recon_end_utc", "end_utc", "head_driver_full_elapsed_utc_seconds",
               "head_driver_full_elapsed_utc_excluding_gpu_queue_seconds", "original_recon_command_seconds",
               "original_recon_same_node_elapsed_utc_seconds", "original_recon_worker_monotonic_seconds",
               "driver_start_minus_recon_worker_start_utc_seconds", "prior_failure_to_revalidation_head_gap_utc_seconds",
               "revalidation_head_elapsed_utc_seconds", "revalidation_same_node_elapsed_utc_seconds",
               "revalidation_monotonic_seconds", "gpu_driver_queue_seconds", "gpu_lock_queue_seconds",
               "raw_dwi_cli_total_runtime_seconds", "memory_budget_status", "timing_error")
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for record in records.values():
                row = {key: record.get(key, record.get("timing", {}).get(key)) for key in columns}
                row["raw_dwi_cli_total_runtime_seconds"] = record.get("gpu_report", {}).get("raw_dwi_cli_total_runtime_seconds")
                row["memory_budget_status"] = record.get("gpu_report", {}).get("memory_budget", {}).get("status")
                row["timing_error"] = json.dumps(record["timing_error"]) if record.get("timing_error") else None
                writer.writerow(row)
        cohort.os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def remote(config, host_kind, action, case=None, log_path=None):
    payload = {"action": action, "config": config, "case": case, "version": "baseline"}
    command = [config[host_kind + "_python"], config["rerun_worker_script"], "_worker"]
    result = subprocess.run(cohort.ssh_command(config[host_kind + "_host"], config.get(host_kind + "_port"),
                                              config.get(host_kind + "_control_path"), command),
                            input=json.dumps(payload), capture_output=True, text=True)
    if log_path:
        Path(log_path).write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"controlled {host_kind}/{action} worker exited {result.returncode}; see {log_path}")
    return json.loads(result.stdout)


def run(options):
    report_dir = cohort.require_fresh(options.report_dir)
    original_status = options.original_driver_report_dir / "status.json"
    snapshot_path = report_dir / "origin_driver_snapshot.json"
    snapshot_path.write_bytes(original_status.read_bytes())
    snapshot = json.loads(snapshot_path.read_text())
    config = copy.deepcopy(snapshot["config"])
    manifest_path = Path(config["run_root"]) / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cases = cohort.validate_manifest(manifest, options.pilot, pilot=bool(options.pilot))
    config.update(run_root=str(options.run_root), sources={"baseline": str(options.source_dir)}, pilot=bool(options.pilot),
                  cuda_alloc_conf=options.cuda_alloc_conf,
                  worker_script=str(options.worker_script), worker_script_sha256=cohort.sha256(options.worker_script),
                  recovery_mode=MODE, fnit_weights=str(options.fnit_weights),
                  rerun_worker_script=str(Path(__file__).resolve()), rerun_worker_sha256=cohort.sha256(__file__),
                  common_identity={"path": str(options.common_identity), "sha256": cohort.sha256(options.common_identity)},
                  rerun_origin={"snapshot_path": str(snapshot_path), "snapshot_sha256": cohort.sha256(snapshot_path),
                                "input_manifest_sha256": cohort.sha256(manifest_path)})
    config["frozen_sources"] = {"baseline": cohort.source_manifest(options.source_dir)}
    if options.wall_script:
        config.update(wall_script=str(options.wall_script), wall_script_sha256=cohort.sha256(options.wall_script),
                      wall_change={"original_sha256": snapshot["config"]["wall_script_sha256"],
                                   "new_sha256": cohort.sha256(options.wall_script),
                                   "scope": "common evaluator: mode-isolated diagnostic export, reserved-memory budget and allocator-environment reporting"})
    verify_common_source(config)
    if cohort.sha256(options.worker_script) != cohort.sha256(cohort.__file__):
        raise ValueError("controlled new cohort worker must contain this harness's implementation")
    resources = remote(config, "gpu", "preflight", log_path=report_dir / "preflight.stderr.log")
    resource_path = report_dir / "resources.json"
    cohort.atomic_json(resource_path, resources)
    config["resources_manifest"] = {"path": str(resource_path), "sha256": cohort.sha256(resource_path)}
    state = {"schema_version": 1, "status": "resource_preflight_completed", "start_utc": cohort.utc(), "config": config,
             "requested_cases": len(cases), "scope": "controlled common-compatibility rerun: this round's new raw-T1 official anatomy, new full raw-DWI output; all old failures remain; not pristine cold cohort",
             "scientific_parity": "not_assessed", "speedup": "not_assessed", "cases": {}}
    for case in cases:
        state["cases"]["baseline/" + case["case_id"]] = {"case_id": case["case_id"], "subject": case["subject"], "version": "baseline", "status": "waiting_original_cpu_report"}
    def save():
        cohort.atomic_json(report_dir / "status.json", state)
        atomic_cases_csv(report_dir / "cases.csv", state["cases"])
    save()
    if options.preflight_only:
        return 0
    root = cohort.require_fresh(config["run_root"])
    cohort.atomic_json(root / "cohort_config.json", config)
    cohort.atomic_json(root / "input_manifest.json", manifest)
    state.update(status="running", fresh_namespace={"status": "claimed_fresh_namespace", "path": str(root)})
    started = time.perf_counter()
    dispatched, futures = set(), {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        while True:
            # A sentinel stops future dispatch while retaining real terminal
            # results from already submitted remote jobs.
            if (report_dir / "STOP_DISPATCH").exists():
                state["dispatch_paused"] = True
            current = json.loads(original_status.read_text())
            for case in cases:
                key = "baseline/" + case["case_id"]
                if (report_dir / "STOP_DISPATCH").exists():
                    state["dispatch_paused"] = True
                if key in dispatched or state.get("dispatch_paused"):
                    continue
                # At most one submitted remote GPU job. STOP_DISPATCH must
                # also prevent Python executor waiters from dispatching later.
                if futures:
                    break
                original_config, old_job, _ = original_paths(config, case)
                original_path = old_job / "recon_report.json"
                if not original_path.is_file():
                    continue
                original = json.loads(original_path.read_text())
                if original.get("status") not in ("failed", "completed"):
                    continue
                dispatched.add(key)
                record = state["cases"][key]
                old_case = current.get("cases", {}).get(key, {})
                record.update(original_driver_start_utc=old_case.get("start_utc", original["start_utc"]),
                              original_driver_end_utc=old_case.get("end_utc"), original_recon_start_utc=original["start_utc"],
                              original_recon_end_utc=original.get("end_utc"), original_recon_command_seconds=original.get("recon_command_seconds"),
                              original_recon_worker_wall_seconds=original.get("worker_wall_seconds"),
                              original_report={"path": str(original_path), "sha256": cohort.sha256(original_path)}, original_failure=original.get("error"))
                try:
                    validate_original(config, case)
                    record.update(status="revalidating_same_round_anatomy", revalidation_head_start_utc=cohort.utc())
                    save()
                    repair = remote(config, "cpu", "revalidate", case, report_dir / f"{case['case_id']}-revalidate.stderr.log")
                    record.update(revalidation_report=repair, revalidation_head_end_utc=cohort.utc())
                    if repair.get("status") != "completed":
                        raise RuntimeError(f"same-round anatomy validation failed: {repair.get('error')}")
                    if (report_dir / "STOP_DISPATCH").exists():
                        state["dispatch_paused"] = True
                        record["status"] = "anatomy_validated_dispatch_paused"
                        save()
                        continue
                    ready = time.perf_counter()
                    record["status"] = "gpu_queued"
                    def downstream(case=case, ready=ready):
                        begun = time.perf_counter()
                        if (report_dir / "STOP_DISPATCH").exists():
                            return {"status": "gpu_dispatch_paused", "gpu_lock_queue_seconds": 0.,
                                    "execution_scope": "no remote GPU worker dispatched"}, begun-ready, cohort.utc()
                        result = cohort.remote(config, "gpu", {"action": "gpu", "config": config, "case": case, "version": "baseline"}, report_dir / f"{case['case_id']}-gpu.stderr.log")
                        return result, begun-ready, cohort.utc()
                    futures[pool.submit(downstream)] = key
                except Exception as error:
                    record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
                save()
            for future, key in list(futures.items()):
                if not future.done():
                    continue
                record = state["cases"][key]
                try:
                    gpu, queue, head_end = future.result()
                    collect_gpu_result(record, gpu, queue, head_end)
                except Exception as error:
                    record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
                print(json.dumps({"case": key, "status": record["status"], "error": record.get("error"), "timing_error": record.get("timing_error")}), flush=True)
                del futures[future]
                save()
            if len(dispatched) == len(cases) and not futures or state.get("dispatch_paused") and not futures:
                break
            if time.perf_counter() - started > options.timeout_hours * 3600:
                state.update(timeout_reached=True, dispatch_paused=True)
            time.sleep(options.poll_seconds)
    completed = sum(item["status"] == "completed" for item in state["cases"].values())
    timing_complete = all("timing" in item and "timing_error" not in item for item in state["cases"].values())
    state.update(status="completed_controlled_rerun" if completed == len(cases) else "failed_or_incomplete",
                 completed_cases=completed, timing_complete=timing_complete, comparison_ready=False,
                 end_utc=cohort.utc(), controller_monotonic_seconds=time.perf_counter()-started)
    save()
    return 0 if completed == len(cases) and timing_complete else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "export-csv":
        parser = argparse.ArgumentParser(description="Export a separate timing CSV from the saved controlled-rerun JSON; no job is resumed")
        parser.add_argument("--status-json", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        selected = parser.parse_args(argv[1:])
        if selected.output.exists() or selected.output.is_symlink():
            raise FileExistsError("saved-report export requires a new CSV path")
        state = json.loads(selected.status_json.read_text())
        if state.get("config", {}).get("recovery_mode") != MODE:
            raise ValueError("saved status is not a controlled raw-DWI rerun")
        atomic_cases_csv(selected.output, state["cases"])
        return 0
    if argv and argv[0] == "_worker":
        payload = json.load(sys.stdin)
        if payload["action"] == "preflight":
            verify_common_source(payload["config"])
            result = build_resources(payload["config"])
        elif payload["action"] == "revalidate":
            result = revalidate_worker(payload)
        else:
            raise ValueError("controlled private worker only preflights resources or revalidates anatomy")
        print(json.dumps(result, allow_nan=False), flush=True)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-driver-report-dir", "report-dir", "run-root", "source-dir", "common-identity", "fnit-weights", "worker-script"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pilot", nargs="+", help="explicit diagnostic subset; not ten-case formal execution")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--wall-script", type=Path, help="declared common evaluator update, independently frozen; its SHA must also be in --common-identity")
    parser.add_argument("--cuda-alloc-conf", default="expandable_segments:True",
                        choices=("expandable_segments:True",), help="same explicit allocator configuration is required in the later candidate benchmark")
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--timeout-hours", type=float, default=36.)
    options = parser.parse_args(argv)
    for value in vars(options).values():
        if isinstance(value, Path):
            cohort.absolute_path(str(value), "controlled rerun path")
    if not 1 <= options.poll_seconds <= 60 or not 0 < options.timeout_hours <= 168:
        parser.error("poll must be 1..60 seconds and timeout positive up to 168 hours")
    return run(options)


if __name__ == "__main__":
    raise SystemExit(main())
