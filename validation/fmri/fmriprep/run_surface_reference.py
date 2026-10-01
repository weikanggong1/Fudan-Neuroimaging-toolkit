"""独立重跑原版 surface-only 链；仅用于基准，不属于 FNIT 运行时。

私有输入 JSON 包含 recon_all（已有同源 subject 目录）、hcp_assets_dir、
newmsm_env（含 bin/newmsm 和 lib）、t1w_bold、mni_bold、repetition_time、
expected_frames，以及可选正向 scanner-RAS fsnative_to_t1w 文本矩阵。
所有路径均为 host 绝对路径。输入 BOLD 为完整 preproc，已有 graymid 或
midthickness 属于排除的 recon-all 输入；不会重新重建、校正或清理 volume。
独立参考的 FreeSurfer 转换通过 --fs-license 读取已有许可文件；不复制或
记录许可内容，也不在公开报告保存其路径或校验值。

一个连续 worker 墙钟包括原版 FreeSurfer/sMRIPrep 几何与形态转换、
Workbench ROI/FS→fsLR 初始化/球面 affine 准备、官方 newMSM 双侧配准、
球面对应的 32k 面积表面、原版 fMRIPrep 投影/CIFTI、QC 和最终文件保存。
SIF 校验、解释器启动及包导入在此边界外，container_wall_seconds 另列。
--msm-threads 1 是严格精度参照；8 线程仅为速度观察，不充当确定性参照。
本脚本不导入 FNIT，不读取 FNIT 已准备的几何、ROI 或注册球面。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


IMAGE_SHA256 = "8e32238619053c1f9d1739b26f4afd72df809d914f5a5771707bf5da4b1d0f39"
PATH_FIELDS = ("recon_all", "hcp_assets_dir", "newmsm_env", "t1w_bold", "mni_bold", "fsnative_to_t1w")
SCOPE = (
    "Complete independent surface-only chain from existing same-source recon-all "
    "geometry and complete T1w/MNI preproc BOLD: official FreeSurfer/sMRIPrep "
    "conversion, cortical ROI, FS-to-fsLR sphere initialization, spherical affine "
    "preparation, official newMSM, sphere-specific 32k area surfaces, installed "
    "fMRIPrep projection and CIFTI, output QC and final data/report saving. "
    "Excludes recon-all reconstruction, all prerequisite volume processing, "
    "container verification/startup and package imports."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_inputs(path):
    value = json.loads(Path(path).read_text())
    for field in PATH_FIELDS:
        item = value.get(field)
        if field == "fsnative_to_t1w" and item is None:
            value[field] = None
            continue
        if not isinstance(item, str) or not Path(item).is_absolute():
            raise ValueError(f"{field} must be an absolute file or directory path")
        expected = Path(item).is_dir() if field in PATH_FIELDS[:3] else Path(item).is_file()
        if not expected:
            raise FileNotFoundError(f"missing {field} input")
    frames, tr = value.get("expected_frames"), value.get("repetition_time")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError("expected_frames must be a positive integer")
    if isinstance(tr, bool) or not isinstance(tr, (int, float)) or not (0 < tr < float("inf")):
        raise ValueError("repetition_time must be positive finite seconds")
    subject = Path(value["recon_all"])
    for hemi in ("lh", "rh"):
        for name in ("white", "pial", "sphere", "sphere.reg", "thickness", "sulc"):
            if not (subject / "surf" / f"{hemi}.{name}").is_file():
                raise FileNotFoundError(f"recon_all lacks {hemi}.{name}")
        if not any((subject / "surf" / f"{hemi}.{name}").is_file()
                   for name in ("midthickness", "graymid")):
            raise FileNotFoundError(f"recon_all lacks {hemi}.midthickness or graymid")
    if not (Path(value["newmsm_env"]) / "bin/newmsm").is_file():
        raise FileNotFoundError("newmsm_env lacks bin/newmsm")
    return value


def worker(args):
    # All reference package imports finish before the algorithm wall clock.
    import numpy as np
    from nipype.interfaces.freesurfer import Info, MRIsConvert
    from smriprep.interfaces.freesurfer import MRIsConvertData
    from smriprep.interfaces.surf import NormalizeSurf
    import smriprep.workflows.surfaces as official_surfaces
    import run_projection_reference as projection

    inputs = read_inputs(args.inputs_json)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("surface reference output must be new")
    work = args.work_root
    work.mkdir(parents=True, exist_ok=False)
    subject, assets, msm_env = (Path(inputs[field]) for field in PATH_FIELDS[:3])
    mesh = assets / "global/templates/standard_mesh_atlases"
    wb = shutil.which("wb_command")
    if wb is None or sha256(wb) != projection.WORKBENCH_SHA256:
        raise ValueError("reference Workbench differs from the inspected 25.2.4 image")
    expected = json.loads((Path(__file__).parent / "installed_sources.public.json").read_text())
    if sha256(official_surfaces.__file__) != expected["smriprep/workflows/surfaces.py"]:
        raise ValueError("installed sMRIPrep surface workflow differs from the fixed source")
    transform = None
    if inputs["fsnative_to_t1w"] is not None:
        transform = np.loadtxt(inputs["fsnative_to_t1w"])
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
                or abs(np.linalg.det(transform[:3, :3])) < 1e-12):
            raise ValueError("fsnative_to_t1w needs a finite invertible forward world affine")
    stage_seconds, commands = {}, []
    prepared = {field: [] for field in projection.SURFACE_FIELDS}
    sources, registration_inputs, registered, initial_spheres = {}, {}, [], []
    total_started = time.perf_counter()

    def command(arguments, *, environment=None):
        commands.append([str(item) for item in arguments])
        with (work / "commands.private.log").open("a") as log:
            completed = subprocess.run(arguments, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"reference command exited {completed.returncode}; inspect private log")

    def wb_command(*arguments):
        command([wb, *map(str, arguments)])

    for hemi, fs_hemi in (("L", "lh"), ("R", "rh")):
        start = time.perf_counter()
        hemisphere = work / hemi
        hemisphere.mkdir()
        sources[hemi] = {}
        for surface in ("white", "pial", "midthickness"):
            source = subject / "surf" / f"{fs_hemi}.{surface}"
            if surface == "midthickness" and not source.is_file():
                source = subject / "surf" / f"{fs_hemi}.graymid"
            sources[hemi][surface] = {"name": source.name, "sha256": sha256(source)}
            converted = hemisphere / f"{fs_hemi}.{surface}.scanner.surf.gii"
            result = MRIsConvert(in_file=str(source), out_file=str(converted),
                                 to_scanner=True).run(cwd=str(hemisphere))
            normalized_dir = hemisphere / f"normalized_{surface}"
            normalized_dir.mkdir()
            normalized = Path(NormalizeSurf(in_file=result.outputs.converted).run(
                cwd=str(normalized_dir)).outputs.out_file)
            if transform is not None:
                transformed = hemisphere / f"{fs_hemi}.{surface}.T1w.surf.gii"
                wb_command("-surface-apply-affine", normalized, inputs["fsnative_to_t1w"], transformed)
                normalized = transformed
            prepared[surface].append(str(normalized))

        sphere_files = {}
        for name in ("sphere", "sphere.reg"):
            source = subject / "surf" / f"{fs_hemi}.{name}"
            sources[hemi][name] = {"name": source.name, "sha256": sha256(source)}
            converted = hemisphere / f"{fs_hemi}.{name}.surf.gii"
            result = MRIsConvert(in_file=str(source), out_file=str(converted),
                                 to_scanner=False).run(cwd=str(hemisphere))
            sphere_files[name] = Path(result.outputs.converted)

        morphs = {}
        for name in ("thickness", "sulc"):
            source = subject / "surf" / f"{fs_hemi}.{name}"
            sources[hemi][name] = {"name": source.name, "sha256": sha256(source)}
            converted = hemisphere / f"{fs_hemi}.{name}.shape.gii"
            result = MRIsConvertData(scalarcurv_file=str(source), out_file=str(converted)).run(cwd=str(hemisphere))
            morphs[name] = Path(result.outputs.converted)

        initial = hemisphere / "sphere.FS_to_fsLR.surf.gii"
        wb_command("-surface-sphere-project-unproject", sphere_files["sphere.reg"],
                   mesh / f"fs_{hemi}/fsaverage.{hemi}.sphere.164k_fs_{hemi}.surf.gii",
                   mesh / f"fs_{hemi}/fs_{hemi}-to-fs_LR_fsaverage.{hemi}_LR.spherical_std.164k_fs_{hemi}.surf.gii",
                   initial)
        initial_spheres.append(str(initial))
        raw_roi, filled, native_roi = (hemisphere / f"roi.{name}.shape.gii"
                                      for name in ("thickness", "filled", "individual"))
        wb_command("-metric-math", "abs(x) > 0", raw_roi, "-var", "x", morphs["thickness"])
        wb_command("-metric-fill-holes", prepared["midthickness"][-1], raw_roi, filled)
        wb_command("-metric-remove-islands", prepared["midthickness"][-1], filled, native_roi)
        prepared["cortex_mask"].append(str(native_roi))
        affine = hemisphere / "sphere_rot.mat"
        unscaled, rotated = hemisphere / "sphere_rot.unscaled.surf.gii", hemisphere / "sphere_rot.surf.gii"
        wb_command("-surface-affine-regression", sphere_files["sphere"], initial, affine)
        wb_command("-surface-apply-affine", sphere_files["sphere"], affine, unscaled)
        wb_command("-surface-modify-sphere", unscaled, 100, rotated)
        sulc = hemisphere / "sulc.inverted.shape.gii"
        wb_command("-metric-math", "-x", sulc, "-var", "x", morphs["sulc"])
        registration_inputs[hemi] = {
            "rotated_sphere": str(rotated), "native_sulc": str(sulc),
            "reference_sphere": str(mesh / f"fsaverage.{hemi}_LR.spherical_std.164k_fs_LR.surf.gii"),
            "reference_sulc": str(mesh / f"{hemi}.refsulc.164k_fs_LR.shape.gii"),
        }
        stage_seconds[f"{hemi}_geometry_roi_sphere_preparation"] = time.perf_counter() - start

    config_source = assets / "MSMConfig/MSMSulcStrainFinalconf"
    config_file = work / "MSMSulc.reference.conf"
    config_file.write_text(config_source.read_text() + f"\n--numthreads={args.msm_threads}\n")
    msm_binary = msm_env / "bin/newmsm"
    msm_environment = {**os.environ, "LD_LIBRARY_PATH": str(msm_env / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
                       "OMP_NUM_THREADS": str(args.msm_threads), "MKL_NUM_THREADS": str(args.msm_threads),
                       "OPENBLAS_NUM_THREADS": str(args.msm_threads)}
    for hemi in ("L", "R"):
        start = time.perf_counter()
        hemisphere = work / hemi
        fields = registration_inputs[hemi]
        prefix = hemisphere / "msm."
        command([str(msm_binary), "--inmesh=" + fields["rotated_sphere"],
                 "--refmesh=" + fields["reference_sphere"], "--indata=" + fields["native_sulc"],
                 "--refdata=" + fields["reference_sulc"], "--conf=" + str(config_file), "--out=" + str(prefix)],
                environment=msm_environment)
        sphere = Path(str(prefix) + "sphere.reg.surf.gii")
        if not sphere.is_file():
            raise RuntimeError("official newMSM did not create the registered sphere")
        prepared["sphere_reg_fsLR"].append(str(sphere))
        registered.append(sphere)
        stage_seconds[f"{hemi}_official_msmsulc"] = time.perf_counter() - start
        start = time.perf_counter()
        area = hemisphere / "midthickness.32k_fsLR.surf.gii"
        wb_command("-surface-resample", prepared["midthickness"]["LR".index(hemi)], sphere,
                   mesh / f"{hemi}.sphere.32k_fs_LR.surf.gii", "BARYCENTRIC", area)
        prepared["midthickness_fsLR"].append(str(area))
        stage_seconds[f"{hemi}_area_surface"] = time.perf_counter() - start

    fixed_inputs = {
        "bold_file": inputs["t1w_bold"], "bold_std": inputs["mni_bold"], "volume_roi": None,
        **prepared, "repetition_time": inputs["repetition_time"], "expected_frames": inputs["expected_frames"],
        "signal": "preproc", "geometry_space": "T1w world RAS", "sphere_kind": "estimated_msmsulc",
    }
    fixed_path = work / "projection_inputs.private.json"
    write_json(fixed_path, fixed_inputs)
    projection_output = work / "projection_output"
    start = time.perf_counter()
    projection.worker(argparse.Namespace(inputs_json=fixed_path, work_root=work / "projection_workflow",
                                         output_root=projection_output, threads=args.threads))
    stage_seconds["official_projection_cifti_qc_and_staging"] = time.perf_counter() - start
    record = json.loads((projection_output / "run.public.json").read_text())
    args.output_root.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    for name in ("hemi-L_space-fsLR_den-32k_bold.func.gii", "hemi-R_space-fsLR_den-32k_bold.func.gii",
                 "space-fsLR_den-91k_bold.dtseries.nii", "space-fsLR_den-91k_bold.json"):
        shutil.copyfile(projection_output / name, args.output_root / name)
    for hemi, sphere in zip(("L", "R"), registered):
        shutil.copyfile(sphere, args.output_root / f"hemi-{hemi}_sphere.MSMSulc.surf.gii")
    write_json(work / "outputs_manifest.private.json", {
        "left": str(args.output_root / "hemi-L_space-fsLR_den-32k_bold.func.gii"),
        "right": str(args.output_root / "hemi-R_space-fsLR_den-32k_bold.func.gii"),
        "dtseries": str(args.output_root / "space-fsLR_den-91k_bold.dtseries.nii"),
        "registered_spheres": [str(args.output_root / f"hemi-{hemi}_sphere.MSMSulc.surf.gii")
                               for hemi in ("L", "R")],
        "projection_inputs_json": str(fixed_path),
        "white": prepared["white"], "pial": prepared["pial"], "midthickness": prepared["midthickness"],
        "native_rois": prepared["cortex_mask"], "initial_spheres": initial_spheres,
        "area_surfaces": {"native": prepared["midthickness"], "fsLR": prepared["midthickness_fsLR"]},
        "msm_inputs": registration_inputs,
        "startpoint_sha256": {"t1w_preproc": record["projection_inputs_sha256"]["bold_file"],
                              "mni_preproc": record["projection_inputs_sha256"]["bold_std"]},
        "msm_config": {"source_sha256": sha256(config_source), "effective_sha256": sha256(config_file),
                       "numthreads": args.msm_threads, "configuration_text": config_file.read_text()},
    })
    write_json(work / "commands.private.json", {"commands": commands})
    record.update({
        "scope": SCOPE, "registration_estimated_here": True,
        "reference_msm_threads": args.msm_threads, "strict_singlethread_accuracy_reference": args.msm_threads == 1,
        "reference_newmsm_sha256": sha256(msm_binary), "hcp_config_source_sha256": sha256(config_source),
        "reference_freesurfer_conversion_version": Info.version(),
        "effective_msm_config_sha256": sha256(config_file), "surface_only_script_sha256": sha256(__file__),
        "recon_geometry_sources": sources,
        "msm_prepared_inputs_sha256": {hemi: {field: sha256(value) for field, value in fields.items()}
                                       for hemi, fields in registration_inputs.items()},
        "registered_spheres_sha256": {hemi: sha256(path) for hemi, path in zip(("L", "R"), registered)},
        "fsnative_to_t1w": transform.tolist() if transform is not None else None,
        "stage_seconds": stage_seconds,
    })
    report_path = args.output_root / "run.public.json"
    write_json(report_path, record)
    stage_seconds["final_publication_and_report_save"] = time.perf_counter() - start
    # The first complete report write is inside this recorded boundary. Only
    # appending the measured duration to that already-saved report follows it.
    record["surface_only_wall_seconds_including_output_save"] = time.perf_counter() - total_started
    record["timing_endpoint"] = "all final data, spheres and complete QC report saved; adding this duration excluded"
    write_json(report_path, record)
    print(json.dumps({"validation_complete": True, "msm_threads": args.msm_threads,
                      "surface_only_wall_seconds_including_output_save": record["surface_only_wall_seconds_including_output_save"]}), flush=True)


def launch(args):
    inputs = read_inputs(args.inputs_json)
    if sha256(args.container_image) != IMAGE_SHA256:
        raise ValueError("container image differs from the fixed fMRIPrep 25.2.4 SIF")
    engine = shutil.which(str(args.singularity))
    if engine is None:
        raise FileNotFoundError("Singularity executable not found")
    output, work = args.output_root.resolve(), args.work_root.resolve()
    if output.exists() or output.is_symlink() or work.exists() or work.is_symlink():
        raise FileExistsError("reference output and work directories must both be new")
    work.mkdir(parents=True)
    home = work / "runtime_home"
    home.mkdir()
    mapped, bindings = dict(inputs), []
    for field in PATH_FIELDS:
        value = inputs[field]
        if value is None:
            continue
        source = Path(value).resolve()
        target = f"/reference-inputs/{field}"
        if source.is_file():
            target += "/" + source.name
            bindings.append((source.parent, str(Path(target).parent)))
        else:
            bindings.append((source, target))
        mapped[field] = target
    write_json(work / "inputs.private.json", mapped)
    command = [engine, "exec", "--cleanenv", "--home", f"{home}:/home/reference",
               "-B", f"{work}:/reference-work", "-B", f"{Path(__file__).resolve().parent}:/reference-script:ro"]
    for source, target in bindings:
        if any(character in str(source) for character in (",", ":")):
            raise ValueError("input directory cannot contain Singularity bind separators")
        command += ["-B", f"{source}:{target}:ro"]
    cache = args.templateflow_dir.resolve()
    if not cache.is_dir():
        raise FileNotFoundError("TemplateFlow cache directory not found")
    license_file = args.fs_license.resolve()
    if not license_file.is_file():
        raise FileNotFoundError("existing FreeSurfer license file not found")
    command += ["-B", f"{license_file}:/reference-license.txt:ro", "-B", f"{cache}:/reference-templateflow", str(args.container_image.resolve()),
                "python", "/reference-script/run_surface_reference.py", "--worker",
                "--inputs-json", "/reference-work/inputs.private.json", "--output-root", "/reference-work/final_output",
                "--work-root", "/reference-work/worker", "--threads", str(args.threads), "--msm-threads", str(args.msm_threads)]
    environment = os.environ.copy()
    for prefix in ("SINGULARITYENV_", "APPTAINERENV_"):
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment[prefix + name] = str(args.threads)
        environment[prefix + "TEMPLATEFLOW_HOME"] = "/reference-templateflow"
        environment[prefix + "FS_LICENSE"] = "/reference-license.txt"
        environment[prefix + "NIPYPE_NO_ET"] = "1"
    write_json(work / "command.private.json", {"command": command, "scope": SCOPE})
    start = time.perf_counter()
    with (work / "workflow.private.log").open("w") as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if completed.returncode:
        raise RuntimeError(f"surface reference exited {completed.returncode}; inspect private work log")
    staging = work / "final_output"
    record = json.loads((staging / "run.public.json").read_text())
    if record.get("validation_complete") is not True:
        raise ValueError("surface reference did not complete its checks")
    record.update({"sif_sha256": IMAGE_SHA256, "container_wall_seconds": elapsed})
    write_json(staging / "run.public.json", record)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("output appeared during execution")
    os.replace(staging, output)
    # Translate only our container paths for offline host comparisons. Source
    # paths stay in this private manifest, never in the public report.
    def host_paths(value):
        if isinstance(value, dict):
            return {key: host_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [host_paths(item) for item in value]
        if isinstance(value, str):
            if value.startswith("/reference-work/final_output/"):
                return str(output / value.removeprefix("/reference-work/final_output/"))
            if value.startswith("/reference-work/"):
                return str(work / value.removeprefix("/reference-work/"))
            for source, target in bindings:
                if value == target or value.startswith(target + "/"):
                    return str(source / value.removeprefix(target).lstrip("/"))
        return value
    manifest = host_paths(json.loads((work / "worker/outputs_manifest.private.json").read_text()))
    fixed_inputs = host_paths(json.loads((work / "worker/projection_inputs.private.json").read_text()))
    # Volume paths are read-only input binds rather than worker products.
    fixed_inputs["bold_file"], fixed_inputs["bold_std"] = inputs["t1w_bold"], inputs["mni_bold"]
    projection_manifest = work / "projection_inputs.host.private.json"
    write_json(projection_manifest, fixed_inputs)
    manifest["projection_inputs_json"] = str(projection_manifest)
    write_json(work / "outputs_manifest.private.json", manifest)
    print(json.dumps({"validation_complete": True, "reference_msm_threads": args.msm_threads,
                      "surface_only_wall_seconds_including_output_save": record["surface_only_wall_seconds_including_output_save"],
                      "container_wall_seconds": elapsed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--container-image", type=Path)
    parser.add_argument("--templateflow-dir", type=Path)
    parser.add_argument("--fs-license", type=Path, help="仅参考转换读取已有许可文件，不公开内容或校验值")
    parser.add_argument("--singularity", default="singularity")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--msm-threads", type=int, choices=(1, 8), default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.worker:
        worker(args)
    else:
        if args.container_image is None or args.templateflow_dir is None or args.fs_license is None:
            parser.error("container-image, templateflow-dir and fs-license are required outside the container")
        launch(args)


if __name__ == "__main__":
    main()
