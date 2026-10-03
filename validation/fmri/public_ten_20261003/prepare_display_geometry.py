#!/usr/bin/env python3
"""从正式 CON01 实际重建与保存的 MSM 球面准备只供脑图的 fsLR32k middle。

CLI: --manifest PRIVATE.json --output-root NEW_DIRECTORY
私有 manifest：cohort_id='formal-vN'（N>=3）, case_id='CON01', candidate_root,
source_root/source_revision, candidate={report,files,source_inventory} 各为 path/sha256,
raw_t1w, registered_spheres/atlas_spheres 的 left/right, workbench 均为 path/sha256。
调用成熟 prepare_t1w_surface_geometry；原生 middle 来自实际 graymid/midthickness，
Workbench BARYCENTRIC 保持固定 atlas 顶点顺序。准备时间不计入生产 benchmark。
路径、原始/派生网格及命令只保留私有；公开 JSON 仅含公共病例、方法和来源 SHA。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import nibabel as nib
import numpy as np

from compare_subject import checked_file, read_json, sha256, write_json
from render_cohort import mesh


ATLAS_SHA256 = {
    "left": "1846b053f870405466776d004d714cc1da0cec7361761c65a864782dd09f30a8",
    "right": "1a898433a9f1070e4e0435d4db966776ecc3291bfcd444efc7910aeffb91561c",
}


def gifti_surface(path):
    image = nib.load(str(path))
    if not isinstance(image, nib.GiftiImage):
        raise ValueError("surface must be GIFTI")
    coordinates = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
    triangles = image.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
    if len(coordinates) != 1 or len(triangles) != 1:
        raise ValueError("surface needs one coordinate and one triangle array")
    points, faces = np.asarray(coordinates[0].data), np.asarray(triangles[0].data)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(points)):
        raise ValueError("surface contains invalid coordinates or triangles")
    return points, faces


def source_inventory(root, records):
    expected = {str(path.relative_to(root)): sha256(path) for path in sorted((root / "src/fnit").rglob("*.py"))}
    if not expected or expected != records:
        raise ValueError("display helper source differs from the actual benchmark source inventory")
    return expected


def prepare(manifest, output):
    started = time.perf_counter()
    cohort = manifest.get("cohort_id")
    match = re.fullmatch(r"formal-v([0-9]+)", cohort or "")
    if manifest.get("case_id") != "CON01" or match is None or int(match.group(1)) < 3:
        raise ValueError("final ten-case display geometry requires explicitly bound formal cohort CON01; diagnostic geometry is separate")
    candidate_name = "candidate_v" + match.group(1)
    cpu_threads = manifest.get("cpu_threads", 4)
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise ValueError("display CPU thread budget must be a positive integer")
    watched, inputs = {}, {}

    def bind(name, entry):
        path, digest = checked_file(entry)
        watched[path] = digest; inputs[name] = digest
        return path

    candidate = manifest["candidate"]
    report_path = bind("candidate_report", candidate["report"])
    files_path = bind("candidate_files", candidate["files"])
    source_path = bind("candidate_source_inventory", candidate["source_inventory"])
    candidate_root = Path(manifest["candidate_root"]).resolve()
    if (candidate_root.name != candidate_name or not report_path.is_relative_to(candidate_root)
            or not files_path.is_relative_to(candidate_root) or not source_path.is_relative_to(candidate_root)
            or files_path.parent != report_path.parent or source_path.parent != report_path.parent):
        raise ValueError("display inputs must belong to the explicit formal candidate root")
    report, files = read_json(report_path), read_json(files_path)
    if (report.get("status") != "complete" or report.get("subject") != "CON01"
            or report.get("backend") != "fnit" or report.get("volume_executed") is not True
            or report.get("source_unchanged_during_run") is not True
            or report.get("source_revision") != manifest["source_revision"]
            or report.get("source_sha256") != watched[source_path]):
        raise ValueError("display geometry needs the actual completed fresh formal FNIT source/run")
    raw_t1 = bind("raw_t1w", manifest["raw_t1w"])
    if report.get("input_sha256", {}).get("t1w") != watched[raw_t1]:
        raise ValueError("display raw T1w differs from the formal run")
    source_root = Path(manifest["source_root"]).resolve()
    before_source = source_inventory(source_root, read_json(source_path))
    source_digest = hashlib.sha256(json.dumps(before_source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata_path = Path(files["metadata"]).resolve()
    watched[metadata_path] = sha256(metadata_path); inputs["surface_metadata"] = watched[metadata_path]
    metadata = read_json(metadata_path)["FNIT"]
    if (metadata.get("Signal") != "preproc"
            or metadata.get("RegistrationDetails", {}).get("Method") != "FNIT MSMSulc-HOCR-FastPD"):
        raise ValueError("display source must be the formal full preproc pipeline")
    final_cifti = Path(files["dtseries"]).resolve()
    watched[final_cifti] = sha256(final_cifti); inputs["formal_final_dtseries"] = watched[final_cifti]
    if report.get("output_checks", {}).get("dtseries", {}).get("sha256") != watched[final_cifti]:
        raise ValueError("display source final CIFTI differs from its completed run")
    subject = Path(files["recon_all"]).resolve()
    for name in ("mri/orig.mgz", "mri/orig/001.mgz", "surf/lh.white", "surf/lh.pial", "surf/rh.white", "surf/rh.pial"):
        path = subject / name
        watched[path] = sha256(path); inputs["recon_" + name.replace("/", ".")] = watched[path]
    middles = {}
    for hemisphere, fs in (("left", "lh"), ("right", "rh")):
        path = next((subject / "surf" / f"{fs}.{suffix}" for suffix in ("midthickness", "graymid")
                     if (subject / "surf" / f"{fs}.{suffix}").is_file()), None)
        if path is None:
            raise FileNotFoundError("formal reconstruction has no actual middle surface")
        watched[path] = sha256(path); inputs[hemisphere + "_native_middle"] = watched[path]
        middles[hemisphere] = path
    spheres, atlas = {}, {}
    for hemisphere, short in (("left", "L"), ("right", "R")):
        spheres[hemisphere] = bind(hemisphere + "_saved_msmsulc_sphere", manifest["registered_spheres"][hemisphere])
        saved = metadata.get("RegisteredSpheres", {}).get(short, {})
        if saved.get("EstimatedHere") is not True or saved.get("SHA256") != watched[spheres[hemisphere]]:
            raise ValueError("registered sphere is not the actual estimated-and-saved formal pipeline output")
        atlas[hemisphere] = bind(hemisphere + "_fixed_32k_atlas_sphere", manifest["atlas_spheres"][hemisphere])
        if watched[atlas[hemisphere]] != ATLAS_SHA256[hemisphere]:
            raise ValueError("display atlas sphere differs from the fixed original HCP asset")
    executable = bind("workbench_program", manifest["workbench"])
    if not os.access(executable, os.X_OK):
        raise ValueError("bound Workbench program must be executable")
    watched[Path(__file__).resolve()] = sha256(__file__)
    watched[Path(__file__).with_name("render_cohort.py").resolve()] = sha256(Path(__file__).with_name("render_cohort.py"))
    sys.path.insert(0, str(source_root / "src"))
    from fnit.fmri import surface_prepare
    from fnit.fmri.surface_pipeline import _matching_original_t1
    from fnit._hemisphere_parallel import workbench_environment
    if Path(surface_prepare.__file__).resolve() != source_root / "src/fnit/fmri/surface_prepare.py":
        raise ValueError("display geometry imported a different FNIT source tree")
    _matching_original_t1(subject, raw_t1, None)
    geometry = surface_prepare.prepare_t1w_surface_geometry(
        subject, output / "native.private", parallel=True, cpu_threads=cpu_threads,
    )
    original = nib.load(str(subject / "mri/orig.mgz"))
    header_record = {"shape": [int(size) for size in original.shape], "scanner_ras_affine": original.affine.tolist(),
                     "vox2ras_tkr": original.header.get_vox2ras_tkr().tolist(), "coordinate_unit": "mm"}
    header_digest = hashlib.sha256(json.dumps(header_record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    commands, outputs, meshes = [], {}, {}
    for hemisphere, pair in (("left", geometry.left), ("right", geometry.right)):
        if pair.midthickness_source.resolve() != middles[hemisphere]:
            raise ValueError("mature geometry helper selected a different native middle")
        native_points, native_faces = gifti_surface(pair.midthickness)
        sphere_points, sphere_faces = gifti_surface(spheres[hemisphere])
        _, atlas_faces = gifti_surface(atlas[hemisphere])
        if native_points.shape != sphere_points.shape or not np.array_equal(native_faces, sphere_faces):
            raise ValueError("saved MSM sphere does not preserve the actual native middle vertex/triangle order")
        destination = output / f"{hemisphere}.CON01.{cohort}.midthickness.32k_fsLR.private.surf.gii"
        command = [str(executable), "-surface-resample", str(pair.midthickness), str(spheres[hemisphere]),
                   str(atlas[hemisphere]), "BARYCENTRIC", str(destination)]
        command_start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, env=workbench_environment(cpu_threads), check=True)
        command_record = {"hemisphere": hemisphere, "argv": command, "seconds": time.perf_counter() - command_start,
                          "stdout": result.stdout, "stderr": result.stderr}
        commands.append(command_record)
        entry = {"path": str(destination), "sha256": sha256(destination), "space": "fsLR32k", "kind": "midthickness"}
        (_, generated_faces), _, _ = mesh(entry)
        if not np.array_equal(generated_faces, atlas_faces):
            raise ValueError("display resampling changed fixed atlas triangle/vertex order")
        outputs[hemisphere] = {key: entry[key] for key in ("sha256", "space", "kind")}
        meshes[hemisphere] = entry
    if any(sha256(path) != digest for path, digest in watched.items()) or source_inventory(source_root, read_json(source_path)) != before_source:
        raise RuntimeError("display input, program or actual source changed during preparation")
    write_json(output / "commands.private.json", {"commands": commands, "orig_header": header_record})
    provenance = {"schema_version": 1, "status": "complete", "cohort_id": cohort, "case_id": "CON01",
                  "source_revision": manifest["source_revision"], "source_inventory_sha256": source_digest,
                  "method": "actual CON01 native graymid/midthickness converted from tkRAS to source T1w scanner-RAS by mature FNIT prepare_t1w_surface_geometry, then Workbench BARYCENTRIC using its final saved MSMSulc sphere and fixed original fsLR32k atlas sphere",
                  "inputs_sha256": inputs, "outputs": outputs, "orig_header_sha256": header_digest,
                  "inputs_source_programs_unchanged": True, "cpu_threads": cpu_threads,
                  "commands_private_sha256": sha256(output / "commands.private.json"),
                  "prepare_script_sha256": sha256(__file__),
                  "display_usage": "one real CON01 display geometry shared across ten independent case-specific CIFTI scalar maps; not ten subject-specific displayed cortical shapes",
                  "preparation_wall_seconds_excluded_from_benchmark": time.perf_counter() - started,
                  "publication": "provenance numbers/hashes and brain PNG only; raw or derived surface arrays, paths and commands remain private"}
    write_json(output / "display_geometry.public.json", provenance)
    write_json(output / "render_geometry.private.json", {"cohort_id": cohort, "cortical_meshes": meshes,
               "display_geometry": {"case_id": "CON01", "provenance": {"path": str(output / "display_geometry.public.json"),
                                                                         "sha256": sha256(output / "display_geometry.public.json")}}})
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {}
    try:
        manifest_digest = sha256(args.manifest)
        manifest = read_json(args.manifest)
        report = prepare(manifest, args.output_root)
        if sha256(args.manifest) != manifest_digest:
            raise RuntimeError("display private manifest changed during preparation")
        report["private_manifest_sha256"] = manifest_digest
    except Exception as error:
        (args.output_root / "failure.private.txt").write_text(traceback.format_exc())
        report = {"schema_version": 1, "status": "failed", "case_id": "CON01",
                  "prepare_script_sha256": sha256(__file__), "failure": {"type": type(error).__name__, "details": "failure.private.txt"}}
        if re.fullmatch(r"formal-v[0-9]+", manifest.get("cohort_id", "")):
            report["cohort_id"] = manifest["cohort_id"]
    write_json(args.output_root / "display_geometry.public.json", report)
    # Final manifest must bind the final public provenance bytes, including private-config SHA.
    if report["status"] == "complete":
        binding = read_json(args.output_root / "render_geometry.private.json")
        binding["display_geometry"]["provenance"]["sha256"] = sha256(args.output_root / "display_geometry.public.json")
        write_json(args.output_root / "render_geometry.private.json", binding)
    print(json.dumps({key: report.get(key) for key in ("status", "case_id", "cohort_id", "prepare_script_sha256")}), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
