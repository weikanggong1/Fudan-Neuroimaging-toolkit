"""Read-only comparison of published continuous-API reference images.

No FNIT/Torch imports, MRI execution, fitting, resampling or recovered clocks.
The reference-stage timer belongs to the already completed continuous API.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

import nibabel as nib
import numpy as np

MANIFEST_SHA = "40ecbdf2cce9ef4e5fabcce71231347ea5bcdde85ad2d668ef02efecb143c9d1"
METRIC_SHA = "a4a794fbfb26d20cd393521aa19bc510ff1bcbff2f4e5db6e39cbaec8a06b4aa"
RENDERER_SHA = "c9bae0e3d10ead4f433a26cda91307ac6f3cdbdab258c6cef813b02c2193a584"
REFERENCE_SHA = "ac885355a286ff6799aaeafc9735de1d0c0264b8afba55041ea4e94b1ddc3484"
CONTINUOUS_DRIVER_SHA = "7964a43f6c0e9ecf52a3751dfbebad93a1bf742b848c1c4912353d2019bd0a74"
CASES = ("CON01", "CON06")
OWNED_OUTPUT = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_import(path, expected, name):
    if sha(path) != expected:
        raise ValueError("analysis helper SHA mismatch: " + name)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_report(report, case, variant, raw, config_sha, source_sha):
    if (report.get("status") != "scientific_complete" or
            report.get("case_id") != case or report.get("variant") != variant):
        raise ValueError("continuous report is not a matching scientific_complete run")
    for key in ("input_guards_equal", "source_guards_equal"):
        if report.get(key) is not True:
            raise ValueError("continuous report guard failed: " + key)
    if report.get("raw_hashes") != {key: raw[key]["sha256"] for key in ("t1w", "bold")}:
        raise ValueError("continuous report has different raw MRI identities")
    if report.get("manifest_sha256") != config_sha or report.get("source_sha256") != source_sha:
        raise ValueError("continuous report manifest/source SHA mismatch")
    if report.get("driver_sha256") != CONTINUOUS_DRIVER_SHA or report.get("frames") != 180:
        raise ValueError("unbound continuous driver or frame count")


def validate_description(configuration, variant, published_sha, published_bytes):
    if configuration.get("motion_iterations") != [1, 1, 1]:
        raise ValueError("continuous motion iteration setting changed")
    if configuration.get("device") != "cuda:0":
        raise ValueError("continuous execution device differs")
    description = configuration["bold_reference"]
    if description.get("strategy") != variant or configuration.get("bold_reference_strategy") != variant:
        raise ValueError("saved reference strategy differs from requested variant")
    if variant == "middle":
        if description.get("selected_indices") != [90]:
            raise ValueError("legacy middle reference is not frame 90")
        return description
    expected_parameters = {"n_volumes": 40, "zero_dummy_masked": 20, "nonnegative": True,
                           "dummy_scans": None, "motion_correction": True,
                           "stage_iterations": [1, 1, 1], "spatial_chunk_size": 262144}
    if (description.get("status") != "complete" or description.get("input_frames") != 180 or
            description.get("parameters") != expected_parameters or
            description.get("selected_indices") != list(range(20, 40)) or
            description.get("algorithm_dummy_scans") != 0 or description.get("skip_vols") != 0 or
            description.get("discarded_input_frames") != 0 or
            description.get("source_sha256") != REFERENCE_SHA or
            description.get("dtype") != "float32" or description.get("device") != "cuda:0"):
        raise ValueError("published helper description differs from the frozen scientific call")
    motion = description["reference_motion"]
    if (motion.get("backend") != "fnit.TorchMCFLIRT" or
            motion.get("interpolation") != "spline" or motion.get("reference_selected_index") != 0):
        raise ValueError("published reference motion route differs")
    output = description["outputs"]["bold_reference.nii.gz"]
    if output != {"sha256": published_sha, "bytes": published_bytes}:
        raise ValueError("published reference differs from helper output identity")
    return description


def validate_stage_seconds(metadata, description, variant):
    outer = metadata["FNIT"]["TimingSeconds"]["bold_reference"]
    if isinstance(outer, bool) or not isinstance(outer, (int, float)) or not math.isfinite(outer) or outer < 0:
        raise ValueError("reference-stage outer wall is invalid")
    if variant == "robust":
        inner = description["timing_seconds"]["total"]
        if not math.isfinite(inner) or inner < 0 or outer < inner:
            raise ValueError("saved helper total does not fit its enclosing stage wall")
    return outer


def middle_is_raw_frame(path, raw):
    image, bold = nib.load(path), nib.load(raw)
    if bold.shape[-1] != 180 or image.shape != bold.shape[:3]:
        raise ValueError("legacy reference has the wrong native shape")
    if not np.allclose(image.affine, bold.affine, rtol=0, atol=1e-4):
        raise ValueError("legacy reference has a different native physical grid")
    if not np.array_equal(np.asarray(image.dataobj, dtype=np.float32),
                          np.asarray(bold.dataobj[..., 90], dtype=np.float32)):
        raise ValueError("legacy published reference is not the raw middle frame")


def main(argv=None):
    global OWNED_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "source-root", "manifest", "metric-helper", "renderer", "continuous-driver", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if sha(args.manifest) != MANIFEST_SHA or sha(args.continuous_driver) != CONTINUOUS_DRIVER_SHA:
        raise ValueError("immutable manifest/continuous driver differs")
    manifest = json.loads(args.manifest.read_text())
    metric = checked_import(args.metric_helper, METRIC_SHA, "reference_metrics")
    renderer = checked_import(args.renderer, RENDERER_SHA, "reference_renderer")
    if (manifest["dataset"]["id"], manifest["dataset"]["version"], manifest["dataset"]["license"]) != ("ds001226", "5.0.1", "CC0"):
        raise ValueError("bound public dataset/license differs")
    description_path = metric.record_path(manifest["dataset"]["description"])
    if "cc0" not in str(json.loads(description_path.read_text()).get("License", "")).lower():
        raise ValueError("actual dataset description does not state CC0")
    paths = {"producer": Path(__file__), "manifest": args.manifest, "metric_helper": args.metric_helper,
             "renderer": args.renderer, "continuous_driver": args.continuous_driver,
             "dataset_description": description_path}
    before = {key: sha(path) for key, path in paths.items()}
    sources_before = metric.source_hashes(args.source_root)
    if sources_before.get("src/fnit/fmri/reference.py") != REFERENCE_SHA:
        raise ValueError("current reference source differs from the scientific freeze")
    report = {"status": "complete", "scope": "Read-only 3D reference comparison using already saved continuous API outputs; no new FNIT reference execution",
              "metric_scope": "3D spatial Pearson and RMSE/reference RMS on full FOV and reference-nonzero domains; no fitting/resampling/scaling",
              "dataset": {"id": "ds001226", "version": "5.0.1", "license": "CC0", "url": "https://openneuro.org/datasets/ds001226/versions/5.0.1"},
              "reference_source_sha256": REFERENCE_SHA, "producer_sha256": sha(__file__),
              "scientific_equivalence": "not_assessed", "independent_FNIT_reference_API_wall_seconds": None,
              "cases": {}}
    images = {}
    rows = []
    for case in CASES:
        bound = manifest["cases"][case]
        for key, entry in bound["raw"].items():
            paths[case + "/raw/" + key] = metric.record_path(entry)
        original = bound["official"]["bold_reference"]
        for key in ("input", "output", "report"):
            paths[case + "/original_official/" + key] = metric.record_path(original[key])
        images[case] = {"official": paths[case + "/original_official/output"]}
        collected = {}
        for variant in ("robust", "middle"):
            prefix = case + "/continuous/" + variant
            root = args.run_root / "continuous_v1" / case / variant
            report_dir = root / "report"
            local_paths = {"report": report_dir / "report.public.json", "filemap": report_dir / "files.private.json",
                           "sources": report_dir / "source.private.json",
                           "configuration": args.run_root / "continuous_v1/manifests" / (case + "." + variant + ".private.json")}
            run_report = json.loads(local_paths["report"].read_text())
            filemap = json.loads(local_paths["filemap"].read_text())
            config = json.loads(local_paths["configuration"].read_text())
            stored_sources = json.loads(local_paths["sources"].read_text())
            validate_report(run_report, case, variant, bound["raw"], sha(local_paths["configuration"]), sha(local_paths["sources"]))
            if stored_sources != sources_before or config["source_root"] != str(args.source_root):
                raise ValueError("actual current source root/snapshot differs from the completed call")
            if config["case_id"] != case or config["variant"] != variant or config["configuration"]["volume_options"]["bold_reference_strategy"] != variant:
                raise ValueError("continuous configuration belongs to another case/strategy")
            metric.record_path(config["source_manifest"])
            if config["source_manifest"]["sha256"] != sha(local_paths["sources"]):
                raise ValueError("configuration source manifest differs from run snapshot")
            if run_report["tr_seconds"] != bound["tr_seconds"] or config["tr_seconds"] != bound["tr_seconds"]:
                raise ValueError("continuous case TR differs")
            for name, entry in config["input_files"].items():
                local_paths["configured_input/" + name] = metric.record_path(entry)
            for name in bound["raw"]:
                if config["input_files"][name]["sha256"] != bound["raw"][name]["sha256"]:
                    raise ValueError("configured raw MRI/JSON identity differs")
            for name in ("bold_reference", "volume_metadata"):
                image_path = Path(filemap[name]["path"]).resolve(strict=True)
                if root.resolve(strict=True) not in image_path.parents or sha(image_path) != filemap[name]["sha256"]:
                    raise ValueError("published file does not match its owned filemap: " + name)
                local_paths[name] = image_path
            metadata = json.loads(local_paths["volume_metadata"].read_text())
            helper = validate_description(metadata["FNIT"]["Configuration"], variant,
                                          sha(local_paths["bold_reference"]), local_paths["bold_reference"].stat().st_size)
            stage_seconds = validate_stage_seconds(metadata, helper, variant)
            if variant == "middle":
                middle_is_raw_frame(local_paths["bold_reference"], paths[case + "/raw/bold"])
            collected[variant] = {"continuous_reference_stage_wall_seconds": stage_seconds,
                                  "time_boundary": "Within completed continuous API: reference-image generation through helper metadata read; not a new isolated run",
                                  "helper_timing_seconds": helper.get("timing_seconds"),
                                  "selected_indices": helper["selected_indices"],
                                  "reference_sha256": filemap["bold_reference"]["sha256"],
                                  "report_sha256": sha(local_paths["report"]), "filemap_sha256": sha(local_paths["filemap"]),
                                  "volume_metadata_sha256": sha(local_paths["volume_metadata"]),
                                  "configuration_sha256": sha(local_paths["configuration"]),
                                  "continuous_API_wall_seconds_excluding_cold_reconstruction_MSM": run_report["api_seconds"],
                                  "continuous_load_average_before": run_report["pre_api_load_average"],
                                  "requested_surface_cpu_threads": config["configuration"]["cpu_threads"],
                                  "continuous_driver_torch_threads": 8,
                                  "matmul_allow_tf32": metadata["FNIT"]["Configuration"]["matmul_allow_tf32"],
                                  "cudnn_allow_tf32": metadata["FNIT"]["Configuration"]["cudnn_allow_tf32"]}
            images[case][variant] = local_paths["bold_reference"]
            for name, path in local_paths.items():
                paths[prefix + "/" + name] = path
        official_root = args.run_root / "bold_reference_v1" / case / "official_attempt01"
        official_report_path, official_map_path = official_root / "report.public.json", official_root / "files.private.json"
        official_report = json.loads(official_report_path.read_text())
        official_map = json.loads(official_map_path.read_text())
        if official_report.get("status") != "complete" or official_report.get("case_id") != case or official_report.get("phase") != "official":
            raise ValueError("fresh official attempt is not complete for this case")
        if official_report["manifest_sha256"] != MANIFEST_SHA or official_report["driver_sha256"] != METRIC_SHA:
            raise ValueError("fresh official manifest/driver identity differs")
        for flag in ("input_guards_equal", "source_guards_equal", "manifest_driver_guards_equal", "selected_indices_match_original"):
            if official_report.get(flag) is not True:
                raise ValueError("fresh official guard failed: " + flag)
        if official_report["source_sha256_before"] != sources_before or official_report["source_sha256_after"] != sources_before:
            raise ValueError("official attempt FNIT source guard differs from current source")
        official_inputs = {key: bound["raw"][key]["sha256"] for key in bound["raw"]}
        official_inputs.update(original_average=original["output"]["sha256"], original_node_report=original["report"]["sha256"], original_node_input=original["input"]["sha256"])
        if official_report["input_sha256_before"] != official_inputs or official_report["input_sha256_after"] != official_inputs:
            raise ValueError("fresh official actual input identities differ")
        fresh = Path(official_map["reference"]).resolve(strict=True)
        if official_root.resolve(strict=True) not in fresh.parents or sha(fresh) != official_report["outputs"]["reference"]["sha256"]:
            raise ValueError("fresh official output is not bound to its report")
        paths.update({case + "/fresh_official/report": official_report_path,
                      case + "/fresh_official/filemap": official_map_path, case + "/fresh_official/reference": fresh})
        execution = official_report["execution"]
        if (execution["exit_code"] != 0 or execution["actual_cuda_visible_devices"] != "" or
                execution["cpu_threads_requested"] != 4 or execution["installed_source_guards_equal"] is not True or
                execution["installed_source_sha256_before"] != execution["installed_source_sha256_after"] or
                execution["installed_source_sha256_before"] != {key: value["sha256"] for key, value in manifest["tools"]["official_sources"].items()}):
            raise ValueError("fresh official environment/source identity differs")
        for key, path in paths.items():
            if key not in before:
                before[key] = sha(path)
        comparisons = {variant: metric.compare_reference(image, images[case]["official"])
                       for variant, image in {"fresh_official": fresh, "robust": images[case]["robust"], "middle": images[case]["middle"]}.items()}
        if any(values["different_values"] != 0 for values in comparisons["fresh_official"].values()):
            raise ValueError("fresh official reference is not identical to the saved original")
        report["cases"][case] = {"frames": 180, "tr_seconds": bound["tr_seconds"], "raw_sha256": {key: entry["sha256"] for key, entry in bound["raw"].items()},
                                 "original_official_node_wall_seconds": original["runtime"]["duration"], "original_official_host": original["runtime"]["hostname"],
                                 "fresh_official_execution": execution, "fresh_official_report_sha256": sha(official_report_path),
                                 "FNIT_continuous": collected, "comparisons": comparisons}
        for variant, domains in comparisons.items():
            for domain, values in domains.items():
                rows.append({"case": case, "variant": variant, "domain": domain, "spatial_pearson": values["spatial_pearson"],
                             "NRMSE": values["NRMSE"], "RMSE": values["RMSE"], "voxels": values["voxels"]})
    metric.protect_output(args.output, args.run_root, [args.source_root, args.manifest, args.metric_helper.parent, args.renderer.parent,
                                                     args.continuous_driver.parent, Path(__file__).parent, description_path,
                                                     args.run_root / "continuous_v1", args.run_root / "bold_reference_v1"])
    args.output.mkdir(parents=True, exist_ok=False)
    OWNED_OUTPUT = args.output
    report["figure_display"] = renderer.render(images, args.output / "reference_images.png")
    after = {key: sha(path) for key, path in paths.items()}
    sources_after = metric.source_hashes(args.source_root)
    if before != after or sources_before != sources_after:
        raise ValueError("original input/source/helper changed during read-only posthoc")
    report.update(input_sha256_before=before, input_sha256_after=after, input_guards_equal=True, source_guards_equal=True,
                  source_sha256_before=sources_before, source_sha256_after=sources_after,
                  posthoc_comparison_and_render_seconds=time.perf_counter() - started,
                  figure_sha256=sha(args.output / "reference_images.png"))
    with (args.output / "metrics.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "summary.public.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "files.private.json").write_text(json.dumps({key: str(path) for key, path in paths.items()}, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "posthoc_seconds": report["posthoc_comparison_and_render_seconds"],
                      "report_sha256": sha(args.output / "summary.public.json"), "figure_sha256": report["figure_sha256"],
                      "cases": {case: report["cases"][case]["comparisons"] for case in CASES}}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OWNED_OUTPUT is not None:
            (OWNED_OUTPUT / "failure.private.txt").write_text(traceback.format_exc())
            (OWNED_OUTPUT / "summary.public.json").write_text(json.dumps({"status": "failed", "scope": "posthoc analysis only; original MRI/clock unchanged",
                                                                         "error_type": type(error).__name__, "producer_sha256": sha(__file__)}, indent=2) + "\n")
        raise
