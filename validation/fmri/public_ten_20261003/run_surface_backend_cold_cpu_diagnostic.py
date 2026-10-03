"""公开同例的额外 FS8.2 冷重建＋完整 CPU surface；排除已完成 volume。"""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
import traceback

REVISION = "1128bc52c7a0233266e5b8a8d7dc0b382994e676"
RAW_T1 = "2cf6d2f2afc65cb7e87ef68ac95a645387ab5c397f93e080a6b25ac485b810b6"
RAW_BOLD = "24f4c4547182eb8267a455cfbee24593564e96ca97160da58037bbc7bbcbdd4b"
FASTPD = "33c3f4c157897c42f9e5a4e26276248dbe2b3c23948078d92c254eab5896da56"
HELPER = "7cf5ef33c17af14e55a776ec1ff539718552e2c948a6e1575ba0b5f95a940db9"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_path(value):
    """JSON 文件图保留 dataclass 中真实 Path；不把未知对象悄悄转为字符串。"""
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=json_path) + "\n")
    temporary.replace(path)


def hashes(paths):
    return {name: sha256(path) for name, path in paths.items()}


def protect_output(root, target, protected):
    """只允许统一 runs 下的新独立子目录；拒绝保护实体及父/子目录重叠。"""
    target = target.expanduser().resolve()
    runs = (root / "runs").resolve()
    if target == runs or not target.is_relative_to(runs):
        raise ValueError("diagnostic target must be a new child of the indexed FNIT/runs")
    for path in protected:
        protected_path = Path(path).expanduser().resolve()
        if target.is_relative_to(protected_path) or protected_path.is_relative_to(target):
            raise ValueError("diagnostic output overlaps an original protected entity")
    if target.exists() or target.is_symlink():
        raise FileExistsError("diagnostic output must be a fresh directory")


def check_cifti_axes(path, assets, tr):
    """按真实固定 ROI/dseg 核对 21 个结构的顺序、顶点、体素与完整时间轴。"""
    import nibabel as nib
    import numpy as np
    from fnit.fmri.surface_fmriprep import _cifti_assets, _SUBCORTEX

    image = nib.load(str(path))
    if not isinstance(image, nib.Cifti2Image) or image.shape != (180, 91282):
        raise ValueError("saved CIFTI must contain all 180 frames and 91,282 grayordinates")
    series, brain = image.header.get_axis(0), image.header.get_axis(1)
    if (not isinstance(series, nib.cifti2.SeriesAxis) or not isinstance(brain, nib.cifti2.BrainModelAxis)
            or series.unit != "SECOND" or series.start != 0 or len(series) != 180
            or not np.isclose(series.step, tr, rtol=1e-6, atol=1e-7)
            or not np.allclose(series.time, np.arange(180) * tr, rtol=1e-6, atol=1e-7)):
        raise ValueError("saved CIFTI time axis differs from the complete raw run")
    atlas = Path(assets) / "global/templates/standard_mesh_atlases"
    labels, vertices = _cifti_assets(atlas / "L.atlasroi.32k_fs_LR.shape.gii",
        atlas / "R.atlasroi.32k_fs_LR.shape.gii",
        Path(assets) / "fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz")
    if (brain.nvertices != {"CIFTI_STRUCTURE_CORTEX_LEFT": 32492, "CIFTI_STRUCTURE_CORTEX_RIGHT": 32492}
            or brain.volume_shape != labels.shape
            or not np.allclose(brain.affine, labels.affine, rtol=0, atol=1e-5)):
        raise ValueError("CIFTI native fsLR vertex domains or fixed LAS subcortical grid differ")
    expected = [("CIFTI_STRUCTURE_CORTEX_" + hemi, vertices[hemi], None) for hemi in ("LEFT", "RIGHT")]
    values = np.asarray(labels.dataobj)
    for name, value in _SUBCORTEX:
        k, j, i = np.nonzero(values.T == value)
        expected.append(("CIFTI_STRUCTURE_" + name, None, np.stack((i, j, k), axis=1)))
    structures = list(brain.iter_structures())
    if len(structures) != 21:
        raise ValueError("CIFTI must contain exactly two cortex and nineteen subcortical structures")
    counts = {}
    for (name, _, actual), (expected_name, vertex, voxel) in zip(structures, expected):
        if name != expected_name:
            raise ValueError("CIFTI structure order differs from the fixed assets")
        if vertex is not None and (not actual.surface_mask.all() or not np.array_equal(actual.vertex, vertex)):
            raise ValueError("CIFTI cortical scalar indices differ from the official ROI")
        if voxel is not None and (not actual.volume_mask.all() or not np.array_equal(actual.voxel, voxel)):
            raise ValueError("CIFTI subcortical voxel indices differ from the official dseg")
        counts[name] = len(actual)
    if sum(counts[name] for name in counts if "CORTEX" in name) != 59412 or sum(counts[name] for name in counts if "CORTEX" not in name) != 31870:
        raise ValueError("CIFTI cortex/subcortex totals differ from the actual fixed assets")
    return {"brain_model_counts": counts, "exact_fixed_asset_indices": True,
            "series_start": float(series.start), "series_step": float(series.step), "series_unit": series.unit}


def prepare(root, target, source, helper):
    from fnit.fmri import locate_bids_inputs
    from fnit.fmri.derivatives import ensure_derivative_dataset, sidecar
    from fnit.fmri.surface_volume import inspect_surface_volume

    cohort = root / "workspaces/fnit_surface_ten_public_20261003"
    formal = cohort / "candidate_v4/CON08"
    report_path = formal / "report/report.public.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete" or report.get("source_revision") != REVISION:
        raise ValueError("CON08 formal volume producer is not complete at the required source")
    original_config = root / "runs/fmri_surface_backends_20261003/freesurfer-source1128-prepared-v4/config.private.json"
    config = json.loads(original_config.read_text())
    config.update(subject="CON08", device="cpu", cpu_threads=4, parallel=True,
                  derivatives_root=str(target / "derivatives"), auto_volume=False,
                  recon_all_output_dir=str(target / "reconstruction"))
    config["recon_all_options"] = dict(config["recon_all_options"], threads=4)
    protect_output(root, target, [source, cohort / "raw", formal, original_config,
                   config["hcp_assets_dir"], json.loads(original_config.read_text())["derivatives_root"],
                   json.loads(original_config.read_text())["recon_all_output_dir"]])
    inputs = locate_bids_inputs(config["bids_root"], subject="CON08", session="preop", task="rest")
    # Existing task directories are intentionally indexed by a symlink; the
    # mature inspector returns resolved derivative paths.
    original = (formal / "derivatives").resolve()
    volume = inspect_surface_volume(inputs, original, signal="preproc", hcp_assets_dir=config["hcp_assets_dir"],
                                    mni_template=config["volume_options"]["mni_template"])
    if volume.state != "ready" or hashes({"t1": volume.source_t1w, "bold": inputs.bold}) != {"t1": RAW_T1, "bold": RAW_BOLD}:
        raise ValueError("CON08 original volume or exact public raw identity failed")
    manifest_path = cohort / "raw/public_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = [entry for entry in manifest["subjects"] if entry.get("subject") == "CON08"]
    if (manifest.get("license") != "CC0" or manifest.get("complete") is not True or len(entries) != 1
            or entries[0]["T1w"]["sha256"] != RAW_T1 or entries[0]["BOLD"]["sha256"] != RAW_BOLD
            or entries[0]["complete_original_frames"] != 180):
        raise ValueError("the exact raw files are not bound to the completed public CC0 dataset manifest")
    files = list(volume.expected_paths)
    # Keep the original brain sidecar too; the surface read-only handoff has five
    # required paths, plus this anatomical sidecar and dataset description.
    brain_sidecar = sidecar(files[-1])
    if not brain_sidecar.is_file():
        raise ValueError("formal source T1 brain sidecar is absent")
    files += [brain_sidecar, original / "dataset_description.json"]
    if len(files) != 7 or len(set(files)) != 7:
        raise ValueError("ready-volume copy must contain seven distinct original files")
    raw_paths = {"raw/T1w": volume.source_t1w, "raw/BOLD": inputs.bold}
    for key, path in list(raw_paths.items()):
        raw_sidecar = sidecar(path)
        if raw_sidecar.is_file():
            raw_paths[key + "_sidecar"] = raw_sidecar
    paths = {**raw_paths, "formal/report": report_path, "formal/files": formal / "report/files.private.json",
             "public_data_manifest": manifest_path,
             "template_configuration": original_config, "helper/run_fnit_subject.py": source / "validation/fmri/public_ten_20261003/run_fnit_subject.py",
             "runner": Path(__file__).resolve()}
    paths.update({"original_volume/" + path.relative_to(original).as_posix(): path for path in files})
    paths.update({"resources/" + path.relative_to(config["hcp_assets_dir"]).as_posix(): path
                  for path in sorted(Path(config["hcp_assets_dir"]).rglob("*")) if path.is_file()})
    binaries = list((source / "src/fnit/msm").glob("_fastpd_native*.so"))
    if len(binaries) != 1 or sha256(binaries[0]) != FASTPD:
        raise ValueError("independently compiled frozen FastPD identity differs")
    paths["native/fastpd"] = binaries[0]
    paths.update({"native/fastpd_source/" + path.name: path
                  for path in sorted((source / "src/fnit/msm/_fastpd_src").iterdir()) if path.is_file()})
    for label, command in (("workbench", config["wb_command"]),
                           ("recon-all", config["recon_all_options"]["command"]),
                           ("mris_expand", config["recon_all_options"]["mris_expand_command"])):
        selected = shutil.which(str(command))
        if selected is None:
            raise FileNotFoundError(command)
        paths["native/" + label] = Path(selected).resolve(strict=True)
    before = hashes(paths)
    source_before = helper.source_hashes(source)
    target.mkdir(mode=0o700, exist_ok=False)
    tick = time.perf_counter()
    copied = {}
    for path in files:
        relative = path.relative_to(original)
        destination = target / "derivatives" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied[relative.as_posix()] = sha256(destination)
        if copied[relative.as_posix()] != before["original_volume/" + relative.as_posix()]:
            raise ValueError("copied volume differs from the formal original")
    copy_seconds = time.perf_counter() - tick
    description_path = target / "derivatives/dataset_description.json"
    description = json.loads(description_path.read_text())
    description.setdefault("DatasetLinks", {})["raw"] = os.path.relpath(inputs.bids_root, (target / "derivatives").resolve())
    description_path.write_text(json.dumps(description, indent=2) + "\n")
    ensure_derivative_dataset((target / "derivatives").resolve(), inputs.bids_root)
    check = inspect_surface_volume(inputs, config["derivatives_root"], signal="preproc", hcp_assets_dir=config["hcp_assets_dir"],
                                   mni_template=config["volume_options"]["mni_template"])
    if check.state != "ready":
        raise ValueError("owned volume copy failed the full handoff contract")
    paths.update({"copied_volume/" + path.relative_to(target / "derivatives").as_posix(): path
                  for path in (target / "derivatives").rglob("*") if path.is_file()})
    for name, path in paths.items():
        if name.startswith("copied_volume/") and name != "copied_volume/dataset_description.json":
            if sha256(path) != before["original_volume/" + name.removeprefix("copied_volume/")]:
                raise ValueError("relocating DatasetLinks changed MRI or sidecar bytes")
    save(target / "config.private.json", config)
    paths["configuration"] = target / "config.private.json"
    final = hashes(paths)
    if any(final[name] != value for name, value in before.items()) or helper.source_hashes(source) != source_before:
        raise ValueError("original data or source changed while preparing the copy")
    save(target / "binding.private.json", {"paths": {name: str(path) for name, path in paths.items()},
         "hashes_before": final, "source_hashes_before": source_before})
    save(target / "preparation.public.json", {
        "status": "ready_volume_prepared_cold_reconstruction_not_started", "subject": "CON08",
        "source_revision": REVISION, "configuration_sha256": sha256(target / "config.private.json"),
        "runner_sha256": sha256(__file__), "raw_t1w_sha256": RAW_T1, "raw_bold_sha256": RAW_BOLD,
        "copied_file_count": 7, "volume_copy_seconds_excluded": copy_seconds,
        "volume_producer": "formal-v4 CON08, source1128; original files remain read-only",
        "original_and_source_guards_equal": True, "volume_handoff_ready": True,
        "owned_dataset_description_updated_fields": ["DatasetLinks.raw"],
        "native_sha256": {name: value for name, value in final.items() if name.startswith("native/")},
        "scope": "Preparation only; no reconstruction or surface computation has started."})


def execute(target, source, helper, affinity):
    import nibabel as nib
    import numpy as np
    import torch
    from fnit.fmri import fMRISurface_pipeline, locate_bids_inputs
    from fnit.fmri import surface_reconstruction as adapter
    from fnit.fmri.surface_volume import inspect_surface_volume

    config_path = target / "config.private.json"
    config = json.loads(config_path.read_text())
    binding_path = target / "binding.private.json"
    binding_sha = sha256(binding_path)
    binding = json.loads(binding_path.read_text())
    paths = {name: Path(path) for name, path in binding["paths"].items()}
    if (config["device"] != "cpu" or config["cpu_threads"] != 4 or config["auto_volume"] is not False
            or config["recon_all_backend"] != "freesurfer" or config.get("recon_all") is not None):
        raise ValueError("cold CPU diagnostic configuration differs from the declared scope")
    if Path(config["recon_all_output_dir"]).exists():
        raise FileExistsError("cold reconstruction directory must not already exist")
    if hashes(paths) != binding["hashes_before"] or helper.source_hashes(source) != binding["source_hashes_before"]:
        raise ValueError("preparation input/native/source identities changed before launch")
    for label, command in (("workbench", config["wb_command"]),
                           ("recon-all", config["recon_all_options"]["command"]),
                           ("mris_expand", config["recon_all_options"]["mris_expand_command"])):
        selected = shutil.which(str(command))
        if selected is None or Path(selected).resolve(strict=True) != paths["native/" + label]:
            raise ValueError("the current executable selection differs from preparation")
    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU diagnosis must hide all CUDA devices before Python starts")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    inputs = locate_bids_inputs(config["bids_root"], subject="CON08", session="preop", task="rest")
    volume = inspect_surface_volume(inputs, config["derivatives_root"], signal="preproc", hcp_assets_dir=config["hcp_assets_dir"],
                                    mni_template=config["volume_options"]["mni_template"])
    if volume.state != "ready" or nib.load(str(inputs.bold)).shape[3] != 180:
        raise ValueError("the full 180-frame volume is not ready")
    output = target / "report"
    output.mkdir(mode=0o700, exist_ok=False)
    started = time.perf_counter()
    state = {"status": "running", "subject": "CON08", "source_revision": REVISION,
             "backend": "freesurfer", "surface_device": "cpu", "requested_reconstruction_device": "cpu",
             "actual_reconstruction_device": "cpu", "cuda_visible_devices": "", "torch_cuda_available": False,
             "cpu_threads": 4, "cpu_affinity": affinity, "inherited_affinity_cpu_count": len(affinity),
             "pid": os.getpid(), "runner_sha256": sha256(__file__), "configuration_sha256": sha256(config_path),
             "validation_helper_sha256": sha256(source / "validation/fmri/public_ten_20261003/run_fnit_subject.py"),
             "raw_t1w_sha256": RAW_T1, "raw_bold_sha256": RAW_BOLD,
             "cold_reconstruction_preexisting": False, "volume_executed": False,
             "native_sha256_before": {name: value for name, value in binding["hashes_before"].items() if name.startswith("native/")},
             "scope": "Additional same-case FS8.2 cold reconstruction, genuine middle and complete CPU surface API using verified formal-v4 volume. Volume computation, copy/preflight and queue wait are excluded. This result does not replace the ten-case fMRIPrep7.3 comparison or GPU demo measurements."}
    save(output / "report.public.json", state)
    try:
        tick = time.perf_counter()
        result = fMRISurface_pipeline(**config)
        api_seconds = time.perf_counter() - tick
        if result.volume_executed:
            raise ValueError("additional diagnostic unexpectedly recomputed volume")
        metadata = json.loads(result.metadata.read_text())
        recon = metadata["FNIT"]["Reconstruction"]
        if recon.get("status") != "complete" or recon.get("reused", False) is not False:
            raise ValueError("this attempt must complete a cold reconstruction, not reuse one")
        request = recon["request"]
        if request["backend"] != "freesurfer" or request["device"] != "cpu" or request["source_sha256"] != RAW_T1:
            raise ValueError("actual reconstruction request differs from the public same-case identity")
        if not adapter._cached_report(recon, request, Path(result.recon_all)):
            raise ValueError("new owned reconstruction actual closure failed final validation")
        outputs = {"dtseries": helper.image_check(result.dtseries, 180, inputs.tr)}
        outputs["dtseries"].update(check_cifti_axes(result.dtseries, config["hcp_assets_dir"], inputs.tr))
        for hemisphere, path in (("L", result.left), ("R", result.right)):
            image = nib.load(str(path))
            if len(image.darrays) != 180 or any(array.data.shape != (32492,) or not np.isfinite(array.data).all() for array in image.darrays):
                raise ValueError("complete saved cortical run failed frame/vertex/finite checks")
            outputs[hemisphere] = {"shape": [180, 32492], "sha256": sha256(path), "all_finite": True}
        for path in (result.metadata, result.qc_report, *result.registered_spheres):
            if not path.is_file():
                raise ValueError("persistent geometry metadata/QC/sphere is absent")
        save(output / "files.private.json", {"result": dataclasses.asdict(result), "configuration": config})
        state.update(status="complete", full_api_seconds=api_seconds, timing_seconds=result.timing_seconds,
                     reconstruction_timing_seconds=recon["timing"], reconstruction_reused=False,
                     original_t1_identity=metadata["FNIT"]["Geometry"]["OriginalT1Identity"],
                     volume_reused=True, frame_count=180, tr_seconds=inputs.tr, outputs=outputs,
                     reconstruction_commands=[{"program": Path(item["argv"][0]).name,
                         "sha256": item["sha256"], "seconds": item["seconds"]} for item in recon["commands"]],
                     registration=metadata["FNIT"]["Registration"], registration_details=metadata["FNIT"]["RegistrationDetails"])
    except BaseException as error:
        (output / "failure.private.txt").write_text(traceback.format_exc())
        state.update(status="failed", error_type=type(error).__name__, detail="See preserved private failure log")
        raise
    finally:
        after = hashes(paths)
        source_after = helper.source_hashes(source)
        data_equal = after == binding["hashes_before"]
        source_equal = source_after == binding["source_hashes_before"]
        binding_equal = sha256(binding_path) == binding_sha
        state.update(driver_through_saved_output_validation_seconds=time.perf_counter() - started,
                     readonly_input_guards_equal=data_equal, frozen_source_guards_equal=source_equal,
                     binding_guard_equal=binding_equal, readonly_inputs_before=binding["hashes_before"],
                     readonly_inputs_after=after, job_tree_peak_gpu_bytes=None, under_20_gb=None,
                     gpu_memory_scope="CPU execution with hidden CUDA; no GPU memory measurement claimed")
        if not (data_equal and source_equal and binding_equal):
            state.update(status="failed", detail="A data/native/source/configuration guard changed")
        save(output / "report.public.json", state)
        if not (data_equal and source_equal and binding_equal):
            raise RuntimeError("read-only guards failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-root", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or "PYTORCH_NO_CUDA_MEMORY_CACHING" in os.environ:
        raise ValueError("hide CUDA and unset the allocator flag before this CPU process starts")
    affinity = sorted(os.sched_getaffinity(0))[-4:]
    if len(affinity) != 4:
        raise ValueError("the diagnosis requires four available CPU cores")
    os.sched_setaffinity(0, affinity)
    source = args.fnit_root / "workspaces/fnit_surface_ten_public_20261003/source_1128bc52"
    helper_path = source / "validation/fmri/public_ten_20261003/run_fnit_subject.py"
    if sha256(helper_path) != HELPER:
        raise ValueError("frozen validation helper identity changed before importing it")
    import fnit
    if Path(fnit.__file__).resolve() != (source / "src/fnit/__init__.py").resolve():
        raise ValueError("the actual imported package differs from frozen source1128")
    spec = importlib.util.spec_from_file_location("frozen_validation", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if Path(helper.__file__).resolve() != helper_path.resolve() or sha256(helper_path) != HELPER:
        raise ValueError("the actual imported helper is not the verified frozen module")
    if args.prepare:
        prepare(args.fnit_root, args.target, source, helper)
    else:
        execute(args.target, source, helper, affinity)


if __name__ == "__main__":
    main()
