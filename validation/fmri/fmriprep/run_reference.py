"""运行固定版本官方 fMRIPrep 参照；不属于 FNIT 候选运行时。

输入为同一原始 BIDS 与已经完成的 FreeSurfer subjects 目录。保留工作目录，
以便逐阶段读取 HMC、BBR、T1→MNI、STC/minimal BOLD、球面及 ribbon 结果。
本参照默认关闭 STC 和 SDC，保留全部帧；显式 --slice-timing 可开启 STC。
默认要求既有几何不变，也可明确允许官方更新独立重建副本；不提供 FS 许可。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np


IMAGE_DIGEST = "sha256:15cbf8dcd17440d26ff5e80e9f7313f1cb3c54f13673f1ec4aed4465e8e12d77"
IMAGE_SHA256 = {
    "25.2.4": "8e32238619053c1f9d1739b26f4afd72df809d914f5a5771707bf5da4b1d0f39",
    "25.2.5": "b275c12447a12d4e0ef4df04336f8f3640c7dd4ec1743dea0f45b439c7141601",
}
MSM_CONFIGURATION_IN_IMAGE = (
    "/app/.pixi/envs/fmriprep/lib/python3.12/site-packages/"
    "smriprep/data/msm/MSMSulcStrainFinalconf"
)


def strict_msm_launcher_script(threads):
    """newMSM accepts numthreads only in its configuration, not on its CLI."""
    return (
        "#!/app/.pixi/envs/fmriprep/bin/python\n"
        "import os, sys\n"
        "arguments = sys.argv[1:]\n"
        f"expected = {MSM_CONFIGURATION_IN_IMAGE!r}\n"
        "replacement = '/reference-tools/MSMSulcStrainFinalconf.strict'\n"
        "found = False\n"
        "for index, argument in enumerate(arguments):\n"
        "    if argument.startswith('--conf='):\n"
        "        if argument[7:] != expected: raise ValueError('unexpected MSM configuration')\n"
        "        arguments[index] = '--conf=' + replacement\n"
        "        found = True\n"
        "    elif argument == '--conf':\n"
        "        if arguments[index + 1] != expected: raise ValueError('unexpected MSM configuration')\n"
        "        arguments[index + 1] = replacement\n"
        "        found = True\n"
        "if not found: raise ValueError('original workflow omitted its MSM configuration')\n"
        f"os.environ.update(OMP_NUM_THREADS={str(threads)!r}, MKL_NUM_THREADS={str(threads)!r}, "
        f"OPENBLAS_NUM_THREADS={str(threads)!r})\n"
        "os.environ['LD_LIBRARY_PATH'] = '/reference-newmsm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')\n"
        "binary = '/reference-newmsm/bin/newmsm'\n"
        "os.execve(binary, [binary, *arguments], os.environ)\n"
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def surface_hashes(subject):
    return {f"{hemisphere}.{name}": sha256(subject / "surf" / f"{hemisphere}.{name}")
            for hemisphere in ("lh", "rh")
            for name in ("white", "pial", "sphere", "sphere.reg", "sulc", "thickness", "graymid")}


def output_checks(output, expected_frames, expected_tr, *, volume_only=False):
    checks = []
    for path in sorted(output.glob("**/*bold.nii.gz")):
        image = nib.load(str(path), keep_file_open=True)
        if image.ndim != 4:
            raise ValueError("official BOLD output must be 4D")
        all_finite = True
        for start in range(0, image.shape[3], 8):
            values = np.asarray(image.dataobj[..., start:start + 8], dtype=np.float32)
            all_finite = all_finite and bool(np.isfinite(values).all())
            del values
        space = path.name.split("_space-", 1)[1].split("_", 1)[0] if "_space-" in path.name else "native"
        checks.append({"kind": space,
                       "shape": list(image.shape), "all_finite": all_finite,
                       "dtype": str(image.get_data_dtype()),
                       "tr_seconds": float(image.header.get_zooms()[3]),
                       "time_unit": image.header.get_xyzt_units()[1], "sha256": sha256(path)})
    for path in sorted(output.glob("**/*.dtseries.nii")):
        image = nib.load(str(path))
        values = np.asarray(image.dataobj, dtype=np.float32)
        checks.append({"kind": "CIFTI", "shape": list(image.shape),
                       "all_finite": bool(np.isfinite(values).all()),
                       "tr_seconds": float(image.header.get_axis(0).step),
                       "brain_models": {name: int(model.size) for name, _, model
                                        in image.header.get_axis(1).iter_structures()},
                       "sha256": sha256(path)})
        del values
    required_spaces = {"T1w", "MNI152NLin6Asym"}
    if not required_spaces.issubset({item["kind"] for item in checks}):
        raise ValueError("official reference did not produce both T1w and MNI BOLD")
    if not volume_only and not any(item["kind"] == "CIFTI" for item in checks):
        raise ValueError("official reference did not produce a CIFTI")
    if not all(item["all_finite"] for item in checks):
        raise ValueError("nonfinite official reference output")
    if any((item["shape"][0] if item["kind"] == "CIFTI" else item["shape"][3]) != expected_frames
           for item in checks):
        raise ValueError("official reference changed the input frame count")
    if any(not np.isclose(item["tr_seconds"], expected_tr, atol=1e-6, rtol=0)
           for item in checks):
        raise ValueError("official reference changed the input repetition time")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, required=True, help="同一病例原始 BIDS")
    parser.add_argument("--output-root", type=Path, required=True, help="不存在的新派生目录")
    parser.add_argument("--work-root", type=Path, required=True, help="不存在的新工作目录；运行后保留")
    parser.add_argument("--subjects-dir", type=Path, required=True, help="已有完成的 FreeSurfer subjects")
    parser.add_argument("--participant-label", required=True, help="不含 sub- 前缀的 BIDS 被试 ID")
    parser.add_argument("--container-image", type=Path, required=True, help="已核对大小、SHA-256 和内部版本的官方 SIF")
    parser.add_argument("--reference-version", choices=sorted(IMAGE_SHA256), default="25.2.4",
                        help="固定参考版本；默认使用者提供的 25.2.4")
    parser.add_argument("--fs-license", type=Path, required=True, help="使用者已有合法许可；不读取或公开内容")
    parser.add_argument("--singularity", default="singularity", help="Singularity 可执行文件")
    parser.add_argument("--threads", type=int, default=8, help="全流程和单进程线程额度")
    parser.add_argument("--mem-mb", type=int, default=19000, help="Nipype 调度内存额度；不是系统硬限制")
    parser.add_argument("--image-digest", help="25.2.5 下载时核对的官方 OCI 摘要；25.2.4 未提供该信息")
    parser.add_argument("--templateflow-cache", type=Path, help="原站资源缓存；明确记录复用")
    parser.add_argument("--allow-official-geometry-updates", action="store_true",
                        help="允许原工作流更新独立私有重建副本；记录全部前后哈希，最终几何可能与候选不同")
    parser.add_argument("--slice-timing", action="store_true", help="显式开启 STC；默认关闭，与 FNIT 默认一致")
    parser.add_argument("--volume-only", action="store_true", help="只生成 T1w/MNI 体积 BOLD；不运行表面配准、MSM 或 CIFTI")
    parser.add_argument("--newmsm-env", type=Path,
                        help="参考专用：显式使用含bin/newmsm和lib的官方newMSM环境，非镜像默认MSM")
    parser.add_argument("--msm-threads", type=int, choices=(1, 8), default=1,
                        help="仅显式newMSM子进程线程数；其余原工作流仍使用threads额度")
    args = parser.parse_args()
    bids = args.bids_root.resolve()
    output = args.output_root.resolve()
    work = args.work_root.resolve()
    subjects = args.subjects_dir.resolve()
    image = args.container_image.resolve()
    license_file = args.fs_license.resolve()
    if args.threads < 1 or args.mem_mb < 1:
        parser.error("threads and mem-mb must be positive")
    if args.image_digest and (args.reference_version != "25.2.5" or args.image_digest != IMAGE_DIGEST):
        raise ValueError("unexpected OCI digest for the requested reference version")
    for path in (bids, subjects, image, license_file):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (output, work):
        if path.exists():
            raise FileExistsError(path)
    subject = subjects / f"sub-{args.participant_label.removeprefix('sub-')}"
    if not (subject / "scripts/recon-all.done").is_file():
        raise ValueError("reference requires an already completed matching reconstruction")
    raw_bold = sorted((bids / subject.name).glob("**/*_bold.nii.gz"))
    if len(raw_bold) != 1:
        raise ValueError("reference requires one unambiguous raw BOLD run")
    raw_image = nib.load(str(raw_bold[0]))
    if raw_image.ndim != 4:
        raise ValueError("reference requires a 4D raw BOLD")
    expected_frames = raw_image.shape[3]
    raw_sidecar = raw_bold[0].with_name(raw_bold[0].name.removesuffix(".nii.gz") + ".json")
    expected_tr = float(json.loads(raw_sidecar.read_text())["RepetitionTime"])
    if not np.isfinite(expected_tr) or expected_tr <= 0:
        raise ValueError("raw BIDS repetition time must be finite and positive")
    before = surface_hashes(subject)
    raw_t1 = list((bids / subject.name / "anat").glob("*_T1w.nii.gz"))
    if len(raw_t1) != 1:
        raise ValueError("reference requires one matching T1w input")
    raw_t1_before = sha256(raw_t1[0])
    raw_sbref = sorted((bids / subject.name).glob("**/*_sbref.nii.gz"))
    if len(raw_sbref) > 1:
        raise ValueError("reference requires at most one unambiguous SBRef")
    raw_files = {"bold": raw_bold[0], "t1w": raw_t1[0],
                 "sbref": raw_sbref[0] if raw_sbref else None}
    raw_input_sha256 = {name: sha256(path) if path is not None else None
                        for name, path in raw_files.items()}
    raw_sidecars = {name: path.with_name(path.name.removesuffix(".nii.gz") + ".json")
                    if path is not None else None for name, path in raw_files.items()}
    raw_sidecar_sha256 = {name: sha256(path) if path is not None and path.is_file() else None
                          for name, path in raw_sidecars.items()}
    image_sha256 = sha256(image)
    if image_sha256 != IMAGE_SHA256[args.reference_version]:
        raise ValueError("container image SHA-256 differs from the verified reference")
    engine = shutil.which(args.singularity)
    if engine is None:
        raise FileNotFoundError(args.singularity)
    version = subprocess.run([engine, "exec", "--cleanenv", str(image), "fmriprep", "--version"],
                             capture_output=True, text=True, check=True).stdout.strip()
    if version.split()[-1].removeprefix("v") != args.reference_version:
        raise ValueError(f"unexpected fMRIPrep image version: {version}")
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    home = work / "runtime_home"
    home.mkdir()
    msm_bind = []
    msm_override = None
    if args.newmsm_env is not None:
        if args.volume_only:
            raise ValueError("newmsm-env cannot be used with a volume-only reference")
        msm_environment = args.newmsm_env.resolve()
        binary = msm_environment / "bin/newmsm"
        libraries = msm_environment / "lib"
        if not binary.is_file() or not libraries.is_dir():
            raise ValueError("newmsm-env needs bin/newmsm and lib")
        tools = work / "reference_tools"
        tools.mkdir()
        config_result = subprocess.run(
            [engine, "exec", "--cleanenv", str(image), "python", "-c",
             f"from pathlib import Path; print(Path({MSM_CONFIGURATION_IN_IMAGE!r}).read_text(), end='')"],
            capture_output=True, text=True, check=True)
        source_config_text = config_result.stdout
        effective_config = tools / "MSMSulcStrainFinalconf.strict"
        effective_config.write_text(source_config_text + f"\n--numthreads={args.msm_threads}\n")
        launcher = tools / "msm"
        launcher.write_text(strict_msm_launcher_script(args.msm_threads))
        launcher.chmod(0o755)
        msm_bind = ["-B", f"{msm_environment}:/reference-newmsm:ro",
                    "-B", f"{tools}:/reference-tools:ro"]
        msm_override = {
            "kind": "fMRIPrep 25.2.4 with explicitly selected official newMSM",
            "uses_container_default_msm_binary": False,
            "official_newmsm_binary_sha256": sha256(binary),
            "official_newmsm_binary_size_bytes": binary.stat().st_size,
            "thread_launcher_sha256": sha256(launcher),
            "source_msm_config_sha256": hashlib.sha256(source_config_text.encode()).hexdigest(),
            "effective_msm_config_sha256": sha256(effective_config),
            "effective_msm_configuration_text": effective_config.read_text(),
            "numthreads_supplied_in_configuration_only": True,
            "newmsm_threads": args.msm_threads,
            "only_msm_child_thread_environment_changed": True,
            "scientific_arguments_from_original_smriprep_workflow": True,
        }
    templateflow_bind = []
    if args.templateflow_cache is not None:
        if not args.templateflow_cache.is_dir():
            raise FileNotFoundError("official TemplateFlow cache missing")
        templateflow_bind = ["-B", f"{args.templateflow_cache.resolve()}:/reference-templateflow"]
    (work / "initial_geometry.private.json").write_text(json.dumps(before, indent=2))
    command = [engine, "run", "--cleanenv", "--home", f"{home}:/home/reference",
               "-B", f"{bids}:/input:ro", "-B", f"{output}:/output",
               "-B", f"{work}:/work", "-B", f"{subjects}:/fs_subjects",
               "-B", f"{license_file}:/fs_license.txt:ro", *templateflow_bind, *msm_bind, str(image),
               "/input", "/output", "participant", "--participant-label", args.participant_label,
               "--fs-subjects-dir", "/fs_subjects", "--fs-license-file", "/fs_license.txt",
               "--nprocs", str(args.threads), "--omp-nthreads", str(args.threads),
               "--mem-mb", str(args.mem_mb), "--work-dir", "/work/processing",
               "--ignore", "fieldmaps", *([] if args.slice_timing else ["slicetiming"]),
               "--dummy-scans", "0", "--slice-time-ref", "0.5",
               "--output-spaces", "T1w", "MNI152NLin6Asym:res-2",
               *(["--no-msm"] if args.volume_only else ["fsnative", "--cifti-output", "91k", "--msm"]),
               "--random-seed", "0",
               "--notrack", "--skip-bids-validation"]
    environment = os.environ.copy()
    if msm_override is not None:
        environment["SINGULARITYENV_PREPEND_PATH"] = "/reference-tools"
    if args.templateflow_cache is not None:
        environment["SINGULARITYENV_TEMPLATEFLOW_HOME"] = "/reference-templateflow"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[f"SINGULARITYENV_{name}"] = str(args.threads)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if name in environment:
            environment[f"SINGULARITYENV_{name}"] = environment[name]
    start = time.perf_counter()
    with (work / "pipeline.private.log").open("w") as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - start
    after = surface_hashes(subject)
    record = {"schema_version": 1, "fmriprep_version": args.reference_version,
              "image_repository": "nipreps/fmriprep", "image_digest": args.image_digest,
              "sif_sha256": image_sha256, "sif_bytes": image.stat().st_size,
              "exit_code": completed.returncode, "wall_seconds": wall,
              "threads": args.threads, "scheduler_memory_mb": args.mem_mb,
              "uses_gpu": False, "STC": args.slice_timing, "SDC": False, "dummy_scans": 0,
              "volume_only": args.volume_only, "surface_registration_requested": not args.volume_only,
              "run_msmsulc": not args.volume_only,
              "msm_binary_selection": msm_override or {
                  "kind": "unmodified container default MSM", "uses_container_default_msm_binary": True},
              "cifti_requested": not args.volume_only, "input_repetition_time_seconds": expected_tr,
              "raw_bold_sha256": sha256(raw_bold[0]), "input_frames": expected_frames,
              "raw_sbref_sha256": raw_input_sha256["sbref"],
              "raw_input_sha256": raw_input_sha256,
              "raw_bids_sidecar_sha256": raw_sidecar_sha256,
              "raw_inputs_unchanged": raw_input_sha256 == {
                  name: sha256(path) if path is not None else None for name, path in raw_files.items()},
              "raw_sidecars_unchanged": raw_sidecar_sha256 == {
                  name: sha256(path) if path is not None and path.is_file() else None
                  for name, path in raw_sidecars.items()},
              "existing_reconstruction_geometry_unchanged": before == after,
              "geometry_sha256": after, "initial_geometry_sha256": before,
              "changed_geometry_files": [k for k in before if before[k] != after[k]],
              "official_geometry_updates_permitted": args.allow_official_geometry_updates,
              "raw_t1_sha256": raw_t1_before, "raw_t1_unchanged": sha256(raw_t1[0]) == raw_t1_before,
              "reference_runner_sha256": sha256(Path(__file__)),
              "templateflow_cache_reused": args.templateflow_cache is not None,
              "scope": ("Independent raw-BIDS fMRIPrep volume-only reference (T1w and MNI152NLin6Asym 2 mm BOLD) from existing completed same-source FS7 geometry; its prior reconstruction is excluded. No MSM, surface registration or CIFTI is requested."
                        if args.volume_only else
                        "Independent continuous raw-BIDS fMRIPrep T1w/MNI preproc and fsLR32k/91k CIFTI reference from existing completed same-source FS7 geometry; its prior reconstruction is excluded. Explicit binary/thread selection is recorded separately."),
              "limits": ["Official reference only; not an FNIT runtime dependency.",
                         "Starts from existing reconstruction and raw BIDS; no denoising matched to FNIT is applied.",
                         "Scheduler memory is not a system hard limit; shared-host timing is a single observation.",
                         "Full working directory retained for fixed-transform stage comparisons.",
                         "BIDS validation skipped for this separately checked single-run reference input."]}
    private = {**record, "command": command, "bids_root": str(bids),
               "subjects_dir": str(subjects), "output_root": str(output), "work_root": str(work)}
    (work / "run.private.json").write_text(json.dumps(private, indent=2) + "\n")
    if completed.returncode:
        raise RuntimeError(f"official reference exited {completed.returncode}; inspect private log")
    if before != after and not args.allow_official_geometry_updates:
        raise RuntimeError("official reference changed the supplied precomputed geometry")
    if not record["raw_t1_unchanged"]:
        raise RuntimeError("original BIDS T1 identity changed")
    if not record["raw_inputs_unchanged"] or not record["raw_sidecars_unchanged"]:
        raise RuntimeError("original raw BIDS images or sidecars changed")
    record["outputs"] = output_checks(output, expected_frames, expected_tr, volume_only=args.volume_only)
    record["validation_complete"] = True
    (work / "run.public.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"exit_code": completed.returncode, "wall_seconds": wall,
                      "output_checks": len(record["outputs"]),
                      "preexisting_geometry_unchanged": before == after}, indent=2))


if __name__ == "__main__":
    main()
