"""Real same-input CPU surface operators, isolated from reconstruction and GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import nibabel as nib
import numpy as np


SCHEMA = "fnit.surface_sampler_same_input.v1"
FIELDS = ("white", "pial", "midthickness", "registered_sphere", "native_roi",
          "atlas_sphere", "atlas_midthickness", "atlas_roi")
STAGES = ("native", "native_dilated", "native_masked", "atlas", "32k")
GENERATED_FIELDS = ("white", "pial", "midthickness", "native_roi", "atlas_midthickness")
REFERENCE_SOURCE_SHA256 = {
    "fmriprep/workflows/bold/resampling.py":
        "aa7c6f3b7da700ce745504dd8402805a8970f26fde7c43fff2936d8bf3a0db03",
    "niworkflows/interfaces/cifti.py":
        "1c243989bdb1ff9ffdc040b48932dfe408cd23427681a79fb004fc9ccec162b1",
    "niworkflows/interfaces/nibabel.py":
        "93b2975c9c4a11fd5607726c0f7c277c780d8c226ba092f19b1d6d8e63120dfc",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def input_specs(manifest):
    if manifest.get("schema") != SCHEMA or manifest.get("data_kind") != "real_mri":
        raise ValueError("Manifest must declare the real-MRI same-input schema")
    case = manifest.get("case_id")
    if not isinstance(case, str) or not case or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for c in case):
        raise ValueError("case_id must be a public identifier")
    inputs = manifest["inputs"]
    required = ("raw_bold", "raw_t1w", "t1w_bold", "mni_bold", "left_label",
                "right_label", "hcp_dseg")
    specs = {key: inputs[key] for key in required}
    if inputs.get("goodvoxels") is not None:
        specs["goodvoxels"] = inputs["goodvoxels"]
    if manifest.get("sampling_grid") is not None:
        for key in ("fixed_t1w", "moving_bold", "fov_mask"):
            specs["sampling_grid." + key] = manifest["sampling_grid"][key]
    if manifest.get("preparation") is not None:
        for key, spec in manifest["preparation"]["input_files"].items():
            specs["preparation." + key] = spec
    for hemi in ("L", "R"):
        for key in FIELDS:
            if manifest.get("preparation") is not None and key in GENERATED_FIELDS:
                continue
            specs[hemi + "." + key] = manifest["hemispheres"][hemi][key]
    for key, spec in specs.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            raise ValueError("Missing path for " + key)
        expected = spec.get("sha256", "")
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("Invalid SHA-256 for " + key)
    return specs


def require_fresh_output(output, protected_paths, protected_roots=()):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("A fresh output directory is required")
    for path in protected_paths:
        path = Path(path).expanduser().resolve()
        if path == output or output in path.parents:
            raise ValueError("Output would contain or replace a protected input")
    for root in protected_roots:
        root = Path(root).expanduser().resolve()
        if root == output or root in output.parents or output in root.parents:
            raise ValueError("Output overlaps an existing protected root")
    return output


def verify_specs(specs):
    hashes = {}
    for key, spec in specs.items():
        value = sha256(spec["path"])
        if value != spec["sha256"]:
            raise ValueError("Input SHA-256 mismatch: " + key)
        hashes[key] = value
    return hashes


def validate_time_axes(manifest, paths):
    count = manifest.get("expected_frames")
    tr = manifest.get("tr_seconds")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("expected_frames must be a positive integer")
    if isinstance(tr, bool) or not isinstance(tr, (int, float)) or not math.isfinite(tr) or tr <= 0:
        raise ValueError("tr_seconds must be positive and finite")
    for key in ("raw_bold", "t1w_bold", "mni_bold"):
        image = nib.load(paths[key], mmap=False)
        if image.ndim != 4 or image.shape[3] != count:
            raise ValueError(key + " does not retain the declared complete frame axis")
        unit = image.header.get_xyzt_units()[1]
        if unit not in ("sec", "msec", "usec"):
            raise ValueError(key + " has unspecified time units")
        actual = float(image.header.get_zooms()[3]) * {"sec": 1, "msec": .001, "usec": .000001}[unit]
        if not np.isclose(actual, tr, rtol=1e-6, atol=1e-7):
            raise ValueError(key + " does not have the declared original TR")


def load_snapshot():
    return {
        "host": platform.node(), "platform": platform.platform(),
        "load_average": list(os.getloadavg()), "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cuda_initialized": bool("torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()),
    }


def run_command(command, *, threads, cwd=None):
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(threads)
    env["CUDA_VISIBLE_DEVICES"] = ""
    started = time.perf_counter()
    result = subprocess.run(list(map(str, command)), check=True, capture_output=True,
                            text=True, env=env, cwd=cwd)
    return result.stdout, time.perf_counter() - started


def reference_prefix(manifest, threads):
    reference = manifest["reference"]
    prefix = reference["container_prefix"]
    if (not isinstance(prefix, list) or not prefix or not all(isinstance(x, str) for x in prefix)
            or "--nv" in prefix or "--cleanenv" not in prefix):
        raise ValueError("Reference must be a clean CPU-only container exec prefix")
    container = str(Path(reference["container"]["path"]).expanduser().resolve())
    if str(Path(prefix[-1]).expanduser().resolve()) != container:
        raise ValueError("The reference prefix must end with its SHA-bound SIF")
    env = ("CUDA_VISIBLE_DEVICES=,OMP_NUM_THREADS=" + str(threads) +
           ",OPENBLAS_NUM_THREADS=" + str(threads) + ",MKL_NUM_THREADS=" + str(threads))
    return prefix[:-1] + ["--env", env, prefix[-1]]


def reference_metadata(prefix, workbench, threads):
    program = """
import hashlib,importlib.metadata,importlib.util,json,pathlib,shutil,subprocess,sys
p=pathlib.Path(shutil.which(sys.argv[1])).resolve()
sources={}
for package,rel in [('fmriprep','workflows/bold/resampling.py'),('niworkflows','interfaces/cifti.py'),('niworkflows','interfaces/nibabel.py')]:
 root=pathlib.Path(importlib.util.find_spec(package).origin).parent
 sources[package+'/'+rel]=hashlib.sha256((root/rel).read_bytes()).hexdigest()
print(json.dumps({'binary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'version':subprocess.run([str(p),'-version'],capture_output=True,text=True,check=True).stdout.strip(),'package_versions':{p:importlib.metadata.version(p) for p in ('fmriprep','niworkflows')},'source_sha256':sources}))
"""
    stdout, _ = run_command(prefix + ["python", "-c", program, workbench], threads=threads)
    result = json.loads(stdout)
    if result["package_versions"] != {"fmriprep": "25.2.4", "niworkflows": "1.14.4"}:
        raise ValueError("Reference package versions changed")
    if result["source_sha256"] != REFERENCE_SOURCE_SHA256:
        raise ValueError("Reference operator sources changed")
    return result


def original_projection(prefix, workbench, paths, hemis, output, tr, threads):
    """Original Workbench operators and original NiWorkflows CIFTI assembly."""
    output.mkdir()
    timings = {}
    for hemi in ("L", "R"):
        h = hemis[hemi]
        native, dilated, masked, atlas, final = [output / f"{hemi}.{s}.func.gii" for s in STAGES]
        operations = [
            ("ribbon", ["-volume-to-surface-mapping", paths["t1w_bold"], h["midthickness"], native,
                        "-ribbon-constrained", h["white"], h["pial"]]),
            ("dilate", ["-metric-dilate", native, h["midthickness"], 10, dilated, "-nearest"]),
            ("native_mask", ["-metric-mask", dilated, h["native_roi"], masked]),
            ("resample", ["-metric-resample", masked, h["registered_sphere"], h["atlas_sphere"],
                          "ADAP_BARY_AREA", atlas, "-area-surfs", h["midthickness"],
                          h["atlas_midthickness"], "-current-roi", h["native_roi"]]),
            ("atlas_mask", ["-metric-mask", atlas, h["atlas_roi"], final]),
        ]
        if paths.get("goodvoxels") is not None:
            operations[0][1].extend(["-volume-roi", paths["goodvoxels"]])
        for stage, arguments in operations:
            _, timings[hemi + "_" + stage] = run_command(prefix + [workbench, *arguments], threads=threads)
    # Use the installed original function; fixed explicit assets prevent network fetching.
    cifti_program = """
from pathlib import Path
import sys
from niworkflows.interfaces.cifti import _create_cifti_image
p=_create_cifti_image(sys.argv[1],sys.argv[2],(sys.argv[3],sys.argv[4]),(sys.argv[5],sys.argv[6]),float(sys.argv[7]))
p.replace(Path.cwd()/'space-fsLR_den-91k_bold.dtseries.nii')
"""
    _, timings["cifti"] = run_command(
        prefix + ["python", "-c", cifti_program, paths["mni_bold"], paths["hcp_dseg"],
                  output / "L.32k.func.gii", output / "R.32k.func.gii", paths["left_label"],
                  paths["right_label"], str(tr)], threads=threads, cwd=output,
    )
    return timings


def prepare_inputs(manifest, paths, output, workbench, threads):
    """Recreate only disposable geometry/ROI preparation from frozen own recon."""
    from fnit.fmri.surface_prepare import prepare_fmriprep_surface_inputs
    preparation = manifest["preparation"]
    if preparation.get("recon_source") != "FNIT_own_frozen_reconstruction":
        raise ValueError("Only declared own frozen recon may generate diagnostic sampler inputs")
    subject = Path(preparation["subject_dir"]).resolve()
    assets = Path(preparation["hcp_assets_dir"]).resolve()
    required = [subject / "mri" / "orig.mgz"]
    for hemi in ("lh", "rh"):
        required.extend(subject / "surf" / f"{hemi}.{name}" for name in ("white", "pial", "sphere.reg", "thickness"))
        mid = subject / "surf" / f"{hemi}.midthickness"
        required.append(mid if mid.exists() else subject / "surf" / f"{hemi}.graymid")
    mesh = assets / "global/templates/standard_mesh_atlases"
    for h in ("L", "R"):
        required.extend([mesh / f"fs_{h}/fsaverage.{h}.sphere.164k_fs_{h}.surf.gii",
                         mesh / f"fs_{h}/fs_{h}-to-fs_LR_fsaverage.{h}_LR.spherical_std.164k_fs_{h}.surf.gii"])
    bound = {Path(spec["path"]).resolve() for spec in preparation["input_files"].values()}
    if any(path.resolve() not in bound for path in required):
        raise ValueError("Preparation inputs/resources must all be SHA-bound")
    started = time.perf_counter()
    prepared = prepare_fmriprep_surface_inputs(
        subject, assets, output / "prepared", wb_command=workbench,
        fsnative_to_t1w=None, parallel=False, cpu_threads=threads)
    for index, h in enumerate(("L", "R")):
        geometry = prepared.geometry.left if h == "L" else prepared.geometry.right
        for key in ("white", "pial", "midthickness"):
            paths[h + "." + key] = str(getattr(geometry, key))
        paths[h + ".native_roi"] = str(prepared.individual_rois[index])
        area = output / "prepared" / f"{h}.midthickness.32k_fsLR.surf.gii"
        run_command([workbench, "-surface-resample", geometry.midthickness,
                     paths[h + ".registered_sphere"], paths[h + ".atlas_sphere"],
                     "BARYCENTRIC", area], threads=threads)
        paths[h + ".atlas_midthickness"] = str(area)
    seconds = time.perf_counter() - started
    hashes = {h + "." + key: sha256(paths[h + "." + key]) for h in ("L", "R") for key in GENERATED_FIELDS}
    return seconds, hashes


def array_difference(candidate, reference):
    if candidate.shape != reference.shape:
        raise ValueError("Compared output shapes differ")
    a, b = np.asarray(candidate), np.asarray(reference)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Compared outputs contain nonfinite values")
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {"shape": list(a.shape), "candidate_dtype": str(a.dtype), "reference_dtype": str(b.dtype),
            "different_values": int(np.count_nonzero(delta)),
            "maximum_absolute_error": float(np.abs(delta).max(initial=0)),
            "root_mean_squared_error": float(np.sqrt(np.mean(delta * delta))),
            "absolute_tolerance": 1e-6, "within_declared_absolute_tolerance": bool(np.all(np.abs(delta) <= 1e-6))}


def compare_files(candidate, reference):
    # GIFTI loaders do not accept the NIfTI-specific keep_file_open option.
    # These published files persist throughout analysis, so generic load is safe.
    a = nib.load(candidate)
    b = nib.load(reference)
    if type(a) != type(b):
        raise ValueError("Compared output image classes differ")
    if isinstance(a, nib.GiftiImage):
        av = np.stack([d.data for d in a.darrays], axis=0)
        bv = np.stack([d.data for d in b.darrays], axis=0)
    else:
        if isinstance(a, nib.Cifti2Image):
            if any(not bool(a.header.get_axis(i) == b.header.get_axis(i)) for i in (0, 1)):
                raise ValueError("Compared CIFTI axes differ")
        elif a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-6, rtol=0):
            raise ValueError("Compared sampling-reference grids differ")
        av, bv = np.asarray(a.dataobj), np.asarray(b.dataobj)
    return array_difference(av, bv)


def aggregate(manifest_path, output_root, threads=4):
    if isinstance(threads, bool) or not isinstance(threads, int) or threads != 4:
        raise ValueError("This declared CPU protocol requires total threads=4")
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    specs = input_specs(manifest)
    wb = Path(shutil.which(manifest["fnit_workbench"]["path"]) or manifest["fnit_workbench"]["path"]).resolve()
    guards = dict(specs)
    guards["FNIT_workbench"] = {"path": str(wb), "sha256": manifest["fnit_workbench"]["sha256"]}
    guards["reference_container"] = manifest["reference"]["container"]
    for key, path in (("manifest", manifest_path), ("driver", Path(__file__).resolve())):
        guards[key] = {"path": str(path), "sha256": sha256(path)}
    from fnit.fmri import surface_fmriprep, sampling_reference, surface, assets_setup, surface_prepare
    from fnit import _hemisphere_parallel
    from fnit.fmri.surface import SurfaceHemisphere
    for key, module in (("surface_fmriprep_source", surface_fmriprep), ("sampling_reference_source", sampling_reference),
                        ("surface_contract_source", surface), ("assets_source", assets_setup),
                        ("hemisphere_parallel_source", _hemisphere_parallel), ("surface_prepare_source", surface_prepare)):
        guards[key] = {"path": module.__file__, "sha256": sha256(module.__file__)}
    output = require_fresh_output(output_root, [spec["path"] for spec in guards.values()],
                                  manifest.get("protected_roots", []))
    hashes_before = verify_specs(guards)
    paths = {key: str(Path(spec["path"]).expanduser().resolve()) for key, spec in specs.items()}
    validate_time_axes(manifest, paths)
    prefix = reference_prefix(manifest, threads)
    reference_wb = manifest["reference"].get("workbench", "wb_command")
    reference_before = reference_metadata(prefix, reference_wb, threads)
    if reference_before["binary_sha256"] != manifest["reference"]["workbench_sha256"]:
        raise ValueError("Reference Workbench binary changed")
    version, _ = run_command([wb, "-version"], threads=threads)
    output.mkdir(parents=True)
    write_json(output / "manifest.snapshot.private.json", manifest)
    report = {
        "schema": SCHEMA, "case_id": manifest["case_id"], "data_kind": "real_mri",
        "status": "running", "scope": "isolated_same_input_surface_operator_diagnostic",
        "expected_frames": manifest["expected_frames"], "tr_seconds": manifest["tr_seconds"],
        "scientific_equivalence": "not_assessed", "cpu_threads": threads,
        "parallel_hemispheres": False, "goodvoxels_supplied": paths.get("goodvoxels") is not None,
        "input_sha256_before": hashes_before, "input_sha256_after": None,
        "input_guards_equal": False, "load_before": load_snapshot(),
        "programs": {"FNIT": {"binary_sha256": hashes_before["FNIT_workbench"], "version": version.strip()},
                     "reference": reference_before},
        "same_workbench_binary": hashes_before["FNIT_workbench"] == reference_before["binary_sha256"],
        "timing_boundaries": {"FNIT_sampler_api": "existing projection API including validation, Workbench, CIFTI, QC and publication",
                              "reference_operators": "serial original Workbench subprocesses and original NiWorkflows CIFTI subprocess; includes container-launch overhead",
                              "excluded": "reconstruction, MSM solver, volume preprocessing, SHA guards and differential analysis"},
    }
    write_json(output / "report.public.json", report)
    try:
        if manifest.get("preparation") is not None:
            seconds, prepared_hashes = prepare_inputs(manifest, paths, output, wb, threads)
            report["preparation_seconds_excluded_from_sampler"] = seconds
            report["preparation_derived_input_sha256_before"] = prepared_hashes
            report["preparation_scope"] = "new CPU geometry/ROI/individual-area preparation from own frozen recon; no MRI reconstruction or MSM solver rerun"
            write_json(output / "prepared_inputs.private.json", {key: {"path": paths[key], "sha256": value} for key, value in prepared_hashes.items()})
        hemis = {h: {k: paths[h + "." + k] for k in FIELDS} for h in ("L", "R")}
        if manifest.get("sampling_grid") is not None:
            grid = {key: paths["sampling_grid." + key] for key in ("fixed_t1w", "moving_bold", "fov_mask")}
            sampling_reference.native_bold_sampling_reference(
                grid["fixed_t1w"], grid["moving_bold"], grid["fov_mask"], output / "FNIT.sampling_reference.nii.gz")
            program = "from niworkflows.interfaces.nibabel import _gen_reference;import sys;_gen_reference(sys.argv[1],sys.argv[2],fov_mask=sys.argv[3],out_file=sys.argv[4])"
            run_command(prefix + ["python", "-c", program, grid["fixed_t1w"], grid["moving_bold"],
                                  grid["fov_mask"], output / "reference.sampling_reference.nii.gz"], threads=threads)
            report["sampling_grid_difference"] = compare_files(output / "FNIT.sampling_reference.nii.gz", output / "reference.sampling_reference.nii.gz")
        started = time.perf_counter()
        result = surface_fmriprep.run_fmriprep_surface_projection(
            clean_t1w=paths["t1w_bold"], clean_mni=paths["mni_bold"],
            left=SurfaceHemisphere(**hemis["L"]), right=SurfaceHemisphere(**hemis["R"]),
            left_label=paths["left_label"], right_label=paths["right_label"], hcp_dseg=paths["hcp_dseg"],
            output_dir=output / "FNIT", tr_seconds=manifest["tr_seconds"], goodvoxels=paths.get("goodvoxels"),
            wb_command=wb, parallel=False, cpu_threads=threads,
        )
        report["FNIT_sampler_api_seconds"] = time.perf_counter() - started
        report["FNIT_operator_seconds"] = result.timing_seconds
        report["load_between_roles"] = load_snapshot()
        started = time.perf_counter()
        report["reference_operator_seconds"] = original_projection(
            prefix, reference_wb, paths, hemis, output / "reference", manifest["tr_seconds"], threads)
        report["reference_operators_seconds"] = time.perf_counter() - started
        names = [f"{h}.{stage}.func.gii" for h in ("L", "R") for stage in STAGES]
        names.append("space-fsLR_den-91k_bold.dtseries.nii")
        report["differences"] = {name: compare_files(output / "FNIT" / name, output / "reference" / name) for name in names}
        report["output_sha256"] = {role: {name: sha256(output / role / name) for name in names} for role in ("FNIT", "reference")}
        report["status"] = "operator_comparison_complete"
    except Exception as error:
        report["status"] = "validation_failed"
        report["error_type"] = type(error).__name__
        (output / "failure.private.txt").write_text(str(error), encoding="utf-8")
    finally:
        try:
            reference_after = reference_metadata(prefix, reference_wb, threads)
        except Exception as error:
            reference_after = None
            report["reference_end_guard_error_type"] = type(error).__name__
        # Keep actual after hashes even when they differ from the requested SHA.
        after = {}
        for key, spec in guards.items():
            try:
                after[key] = sha256(spec["path"])
            except OSError:
                after[key] = None
        report["input_sha256_after"] = after
        report["input_guards_equal"] = after == hashes_before and reference_after == reference_before
        if report.get("preparation_derived_input_sha256_before") is not None:
            prepared_after = {}
            for key in report["preparation_derived_input_sha256_before"]:
                try:
                    prepared_after[key] = sha256(paths[key])
                except OSError:
                    prepared_after[key] = None
            report["preparation_derived_input_sha256_after"] = prepared_after
            report["input_guards_equal"] &= prepared_after == report["preparation_derived_input_sha256_before"]
        report["load_after"] = load_snapshot()
        if not report["input_guards_equal"]:
            report["status"] = "input_changed_during_diagnostic"
        write_json(output / "report.public.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = aggregate(args.manifest, args.output_root, args.threads)
    print(json.dumps({"status": report["status"], "case_id": report["case_id"],
                      "same_workbench_binary": report["same_workbench_binary"]}))
    return 0 if report["status"] == "operator_comparison_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
