"""运行固定 fMRIPrep 25.2.4 的表面投影和 CIFTI 工作流，仅用于验证。

在 host 读取私有输入 JSON，并用已核对的 SIF 启动本脚本的容器 worker。
影像、white/pial/实际 midthickness、面积表面、ROI 和注册球面均由调用方
固定；此脚本不估计配准，不验证几何准备，也不是从原始 BIDS 开始的
官方完整流程。它不属于 FNIT 运行时，不导入 FNIT。

输入 JSON 使用原版 inputnode 字段名：bold_file、bold_std、white、pial、
midthickness、midthickness_fsLR、sphere_reg_fsLR、cortex_mask、volume_roi。
各表面字段按 [L, R] 排列，volume_roi 可为 null。另需 repetition_time、
expected_frames、signal="preproc"、geometry_space="T1w world RAS" 和
sphere_kind（initialization_control / estimated_msmsulc / provided_registration）。
所有文件路径必须是 host 上的绝对路径。私密工作目录保留实际命令与日志；
公开报告只含字段名、校验值、尺寸和运行范围。
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np


IMAGE_SHA256 = "8e32238619053c1f9d1739b26f4afd72df809d914f5a5771707bf5da4b1d0f39"
WORKBENCH_SHA256 = "e5e34c7f44fd057d9bb38592a0c173887bbaf0058040fc9cfaa92b69fe01fe52"
SURFACE_FIELDS = (
    "white", "pial", "midthickness", "midthickness_fsLR",
    "sphere_reg_fsLR", "cortex_mask",
)
PATH_FIELDS = ("bold_file", "bold_std", "volume_roi") + SURFACE_FIELDS
SCOPE = (
    "Fixed-input projection and assembly only: installed fMRIPrep 25.2.4 "
    "init_bold_fsLR_resampling_wf followed by init_bold_grayords_wf. "
    "Supplied volumes, white/pial/actual midthickness, cortex masks, area surfaces "
    "and registration spheres are held common. No geometry preparation, MSM "
    "estimation, motion, BBR or T1-to-MNI estimation is validated by this run. "
    "This is not an independent raw-BIDS fMRIPrep end-to-end comparison."
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
    inputs = json.loads(Path(path).read_text())
    if inputs.get("signal") != "preproc" or inputs.get("geometry_space") != "T1w world RAS":
        raise ValueError("reference requires explicit preproc signal and T1w world-RAS geometry")
    if inputs.get("sphere_kind") not in (
        "initialization_control", "estimated_msmsulc", "provided_registration",
    ):
        raise ValueError("sphere_kind must describe the actual supplied registration")
    tr = inputs.get("repetition_time")
    frames = inputs.get("expected_frames")
    if isinstance(tr, bool) or not isinstance(tr, (int, float)) or not np.isfinite(tr) or tr <= 0:
        raise ValueError("repetition_time must be finite and positive, in seconds")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError("expected_frames must be a positive integer; no frames are discarded")
    for field in PATH_FIELDS:
        if field == "volume_roi" and inputs.get(field) is None:
            inputs[field] = None
            continue
        values = inputs.get(field)
        if field in SURFACE_FIELDS:
            if not isinstance(values, list) or len(values) != 2:
                raise ValueError(f"{field} must be [left, right]")
        else:
            values = [values]
        for value in values:
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"{field} must contain absolute file paths")
            if not Path(value).is_file():
                raise FileNotFoundError(f"missing {field} input")
    return inputs


def input_hashes(inputs):
    return {
        field: [sha256(path) for path in inputs[field]] if field in SURFACE_FIELDS
        else sha256(inputs[field]) if inputs[field] is not None else None
        for field in PATH_FIELDS
    }


def image_check(path, name, frames, tr):
    image = nib.load(str(path), keep_file_open=True)
    if image.ndim != 4 or image.shape[3] != frames:
        raise ValueError(f"{name} must retain all expected frames")
    if (not np.isfinite(image.affine).all()
            or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12
            or image.header.get_xyzt_units()[0] != "mm"):
        raise ValueError(f"{name} needs a finite invertible millimeter affine")
    unit = image.header.get_xyzt_units()[1]
    if unit not in ("sec", "msec", "usec"):
        raise ValueError(f"{name} TR must have an explicit time unit")
    seconds = float(image.header.get_zooms()[3]) * {"sec": 1, "msec": .001, "usec": 1e-6}[unit]
    if not np.isfinite(seconds) or seconds <= 0 or not np.isclose(seconds, tr, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{name} header TR does not match the supplied original TR")
    for start in range(0, frames, 8):
        if not np.isfinite(np.asarray(image.dataobj[..., start:start + 8])).all():
            raise ValueError(f"{name} contains nonfinite values")
    return image


def hemisphere_check(image, hemi, name):
    expected = {"L": "CORTEXLEFT", "R": "CORTEXRIGHT"}[hemi]
    other = {"L": "CORTEXRIGHT", "R": "CORTEXLEFT"}[hemi]
    for meta in [image.meta] + [array.meta for array in image.darrays]:
        value = str(meta.get("AnatomicalStructurePrimary", "")).upper().replace("_", "")
        if value == other or (value.startswith("CORTEX") and value != expected):
            raise ValueError(f"{name} has conflicting hemisphere metadata")


def mesh_check(path, hemi, name, *, vertices=None, sphere=False):
    image = nib.load(str(path))
    if not isinstance(image, nib.GiftiImage):
        raise ValueError(f"{name} must be GIFTI")
    hemisphere_check(image, hemi, name)
    points = [array.data for array in image.darrays if array.intent == 1008]
    faces = [array.data for array in image.darrays if array.intent == 1009]
    if len(points) != 1 or len(faces) != 1:
        raise ValueError(f"{name} must have one POINTSET and one TRIANGLE array")
    points, faces = np.asarray(points[0]), np.asarray(faces[0])
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all() or (vertices is not None and len(points) != vertices)):
        raise ValueError(f"{name} has invalid vertices")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or (faces < 0).any() or (faces >= len(points)).any()
            or np.any(np.sort(faces, axis=1)[:, 1:] == np.sort(faces, axis=1)[:, :-1])):
        raise ValueError(f"{name} has invalid topology")
    if sphere and not np.allclose(np.linalg.norm(points, axis=1), 100, rtol=0, atol=.05):
        raise ValueError(f"{name} must be centered on the 100-mm sphere")
    return points, faces


def roi_check(path, hemi, name, vertices):
    image = nib.load(str(path))
    if not isinstance(image, nib.GiftiImage) or len(image.darrays) != 1:
        raise ValueError(f"{name} must be a single GIFTI metric")
    hemisphere_check(image, hemi, name)
    values = np.asarray(image.darrays[0].data)
    if (values.shape != (vertices,) or not np.isfinite(values).all()
            or (values < 0).any() or not (values > 0).any()):
        raise ValueError(f"{name} must contain finite nonnegative vertex inclusion values")
    return values


def validate_common_geometry(inputs):
    counts = {}
    for index, hemi in enumerate(("L", "R")):
        points, faces = mesh_check(inputs["midthickness"][index], hemi, f"{hemi} midthickness")
        for field in ("white", "pial", "sphere_reg_fsLR"):
            _, current_faces = mesh_check(
                inputs[field][index], hemi, f"{hemi} {field}", vertices=len(points),
                sphere=field == "sphere_reg_fsLR",
            )
            if not np.array_equal(current_faces, faces):
                raise ValueError(f"{hemi} {field} does not share native topology")
        roi_check(inputs["cortex_mask"][index], hemi, f"{hemi} cortex_mask", len(points))
        _, target_faces = mesh_check(
            inputs["midthickness_fsLR"][index], hemi, f"{hemi} midthickness_fsLR", vertices=32492,
        )
        counts[hemi] = {"native_vertices": len(points), "native_faces": len(faces),
                        "fsLR_vertices": 32492, "fsLR_faces": len(target_faces)}
    return counts


def template_checks(surface_workflow, inputs):
    from niworkflows.interfaces.cifti import CIFTI_STRUCT_WITH_LABELS, _prepare_cifti

    select = surface_workflow.get_node("select_surfaces")
    spheres = list(select.inputs.template_sphere)
    rois = list(select.inputs.template_roi)
    if len(spheres) != 2 or len(rois) != 2:
        raise ValueError("installed workflow did not select two fsLR template spheres and ROIs")
    labels, dseg, metadata = _prepare_cifti("91k")
    paths = {"surface_spheres": spheres, "surface_rois": rois,
             "cifti_surface_labels": labels, "cifti_volume_dseg": [dseg]}
    counts = {}
    for index, hemi in enumerate(("L", "R")):
        _, target_faces = mesh_check(spheres[index], hemi, f"{hemi} template sphere",
                                    vertices=32492, sphere=True)
        _, supplied_faces = mesh_check(inputs["midthickness_fsLR"][index], hemi,
                                      f"{hemi} target area", vertices=32492)
        if not np.array_equal(target_faces, supplied_faces):
            raise ValueError(f"{hemi} target area and actual template sphere have different topology")
        roi = roi_check(rois[index], hemi, f"{hemi} installed sMRIPrep atlas ROI", 32492)
        label = roi_check(labels[index], hemi, f"{hemi} TemplateFlow cortical label", 32492)
        if not np.array_equal(roi > 0, label > 0):
            raise ValueError(f"{hemi} projection ROI and CIFTI ROI include different vertices")
        counts[hemi] = int(np.count_nonzero(label))
    label_image = nib.as_closest_canonical(nib.load(str(dseg)))
    bold = nib.as_closest_canonical(nib.load(str(inputs["bold_std"])))
    values = np.asanyarray(label_image.dataobj)
    if (label_image.ndim != 3 or not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all() or (values < 0).any()
            or not np.allclose(nib.affines.voxel_sizes(label_image.affine), 2, rtol=0, atol=1e-5)
            or bold.shape[:3] != label_image.shape
            or not np.allclose(bold.affine, label_image.affine, rtol=0, atol=1e-4)):
        raise ValueError("fixed MNI BOLD must share the actual TemplateFlow HCP-dseg 2-mm grid")
    subcounts = {name: int(np.count_nonzero(np.isin(values, labels)))
                 for name, labels in CIFTI_STRUCT_WITH_LABELS.items() if labels is not None}
    if (len(subcounts) != 19 or any(v == 0 for v in subcounts.values())
            or sum(subcounts.values()) != 31870 or counts != {"L": 29696, "R": 29716}):
        raise ValueError("actual reference templates do not provide the fixed 91,282 grayordinates")
    records = {kind: [{"name": Path(path).name, "sha256": sha256(path)} for path in values]
               for kind, values in paths.items()}
    return paths, records, subcounts, metadata


def check_metric(path, hemi, frames):
    image = nib.load(str(path))
    hemisphere_check(image, hemi, f"{hemi} output BOLD")
    if len(image.darrays) != frames:
        raise ValueError("reference surface frame count changed")
    for array in image.darrays:
        if array.data.shape != (32492,) or not np.isfinite(array.data).all():
            raise ValueError("reference surface output has an invalid frame")
    return {"shape": [frames, 32492], "all_finite": True, "sha256": sha256(path)}


def worker(args):
    from fmriprep import config
    from fmriprep.workflows.bold import resampling
    from niworkflows.interfaces import cifti
    from nipype import config as nipype_config
    from nipype.pipeline import engine as pe

    if importlib.metadata.version("fmriprep") != "25.2.4":
        raise ValueError("reference worker must run in the verified fMRIPrep 25.2.4 image")
    packages = {name: importlib.metadata.version(name) for name in (
        "fmriprep", "smriprep", "niworkflows", "nibabel", "templateflow", "nipype", "numpy",
    )}
    if packages["niworkflows"] != "1.14.4" or packages["smriprep"] != "0.19.2":
        raise ValueError("reference dependency versions differ from the pinned image")
    workbench = shutil.which("wb_command")
    if workbench is None or sha256(workbench) != WORKBENCH_SHA256:
        raise ValueError("reference Workbench binary differs from the recorded fixed image")
    workbench_version = subprocess.run([workbench, "-version"], check=True,
                                      capture_output=True, text=True).stdout.strip()
    sources = {"fmriprep/workflows/bold/resampling.py": sha256(resampling.__file__),
               "niworkflows/interfaces/cifti.py": sha256(cifti.__file__)}
    expected = json.loads((Path(__file__).parent / "installed_sources.public.json").read_text())
    if any(expected[name] != value for name, value in sources.items()):
        raise ValueError("installed projection/assembly source differs from the recorded source")
    inputs = read_inputs(args.inputs_json)
    before = input_hashes(inputs)
    frames, tr = inputs["expected_frames"], float(inputs["repetition_time"])
    t1w = image_check(inputs["bold_file"], "T1w preproc", frames, tr)
    mni = image_check(inputs["bold_std"], "MNI preproc", frames, tr)
    if inputs["volume_roi"] is not None:
        roi = nib.load(inputs["volume_roi"])
        values = np.asanyarray(roi.dataobj)
        if (roi.ndim != 3 or roi.shape != t1w.shape[:3]
                or not np.allclose(roi.affine, t1w.affine, rtol=0, atol=1e-4)
                or not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any()):
            raise ValueError("volume_roi must be a nonempty nonnegative finite mask on the T1w BOLD grid")
    geometry = validate_common_geometry(inputs)
    config.execution.notrack = True
    args.work_root.mkdir(parents=True, exist_ok=True)
    nipype_config.update_config({"execution": {"crashfile_format": "txt",
                                              "crashdump_dir": str(args.work_root)}})
    memory = max(.1, np.prod(t1w.shape) * 4 / 1e9)
    projection = resampling.init_bold_fsLR_resampling_wf(
        grayord_density="91k", omp_nthreads=args.threads, mem_gb=memory,
    )
    declared = set(projection.get_node("inputnode").inputs.copyable_trait_names())
    official_fields = {"bold_file", "volume_roi", *SURFACE_FIELDS}
    if declared != official_fields:
        raise ValueError("installed fsLR inputnode fields differ from the inspected 25.2.4 workflow")
    for field in official_fields:
        if inputs[field] is not None:
            setattr(projection.inputs.inputnode, field, inputs[field])
    paths, templates_before, subcounts, expected_metadata = template_checks(projection, inputs)
    grayords = resampling.init_bold_grayords_wf(
        grayord_density="91k", mem_gb=max(.1, np.prod(mni.shape) * 4 / 1e9), repetition_time=tr,
    )
    if set(grayords.get_node("inputnode").inputs.copyable_trait_names()) != {"bold_std", "bold_fsLR"}:
        raise ValueError("installed grayords inputnode fields differ from the inspected workflow")
    grayords.inputs.inputnode.bold_std = inputs["bold_std"]
    workflow = pe.Workflow(name="fixed_surface_projection_reference", base_dir=str(args.work_root))
    workflow.connect(projection, "outputnode.bold_fsLR", grayords, "inputnode.bold_fsLR")
    start = time.perf_counter()
    graph = workflow.run(plugin="Linear")
    wall = time.perf_counter() - start
    cifti_nodes = [node for node in graph.nodes if node.name == "gen_cifti"]
    masked_nodes = [node for node in graph.nodes if node.name == "mask_fsLR"]
    if len(cifti_nodes) != 1 or len(masked_nodes) != 2:
        raise ValueError("actual execution graph did not produce both hemispheres and one CIFTI")
    surface_outputs = {}
    for node in masked_nodes:
        matches = [index for index, path in enumerate(paths["surface_rois"])
                   if Path(node.inputs.mask).resolve() == Path(path).resolve()]
        if len(matches) != 1:
            raise ValueError("cannot identify output hemisphere from its actual atlas ROI")
        hemi = ("L", "R")[matches[0]]
        if hemi in surface_outputs:
            raise ValueError("execution graph repeated one hemisphere")
        surface_outputs[hemi] = Path(node.result.outputs.out_file)
    result = cifti_nodes[0].result.outputs
    image = nib.load(str(result.out_file))
    axis = image.header.get_axis(0)
    if (image.shape != (frames, 91282) or axis.size != frames or axis.start != 0
            or axis.unit != "SECOND" or not np.isclose(axis.step, tr, rtol=0, atol=1e-12)):
        raise ValueError("reference CIFTI does not have the expected complete time axis")
    brainmodels = {name: int(model.size) for name, _, model in image.header.get_axis(1).iter_structures()}
    required_models = {"CIFTI_STRUCTURE_CORTEX_LEFT": 29696, "CIFTI_STRUCTURE_CORTEX_RIGHT": 29716,
                       **subcounts}
    if brainmodels != required_models:
        raise ValueError("reference CIFTI brain models differ from the actual 91k templates")
    for start in range(0, frames, 32):
        if not np.isfinite(np.asarray(image.dataobj[start:start + 32])).all():
            raise ValueError("reference CIFTI contains nonfinite values")
    metadata = json.loads(Path(result.out_metadata).read_text())
    # CIFTI XML metadata values are strings; nibabel stringifies nested dicts.
    embedded_expected = {key: str(value) for key, value in expected_metadata.items()}
    if metadata != expected_metadata or dict(image.header.matrix.metadata) != embedded_expected:
        raise ValueError("reference CIFTI embedded and sidecar metadata do not match the installed generator")
    checks = {hemi: check_metric(path, hemi, frames) for hemi, path in surface_outputs.items()}
    if input_hashes(inputs) != before:
        raise ValueError("fixed input files changed during reference execution")
    templates_after = {kind: [{"name": Path(path).name, "sha256": sha256(path)} for path in values]
                       for kind, values in paths.items()}
    if templates_after != templates_before:
        raise ValueError("actual reference templates changed during execution")
    args.output_root.mkdir(parents=True, exist_ok=False)
    for hemi, path in surface_outputs.items():
        shutil.copyfile(path, args.output_root / f"hemi-{hemi}_space-fsLR_den-32k_bold.func.gii")
    shutil.copyfile(result.out_file, args.output_root / "space-fsLR_den-91k_bold.dtseries.nii")
    shutil.copyfile(result.out_metadata, args.output_root / "space-fsLR_den-91k_bold.json")
    write_json(args.output_root / "run.public.json", {
        "schema_version": 1, "scope": SCOPE, "validation_complete": True,
        "packages": packages, "installed_source_sha256": sources,
        "reference_workbench": {"sha256": WORKBENCH_SHA256, "version": workbench_version},
        "reference_script_sha256": sha256(__file__), "signal": inputs["signal"],
        "sphere_kind": inputs["sphere_kind"], "registration_estimated_here": False,
        "projection_inputs_sha256": before, "geometry": geometry,
        "volume_inputs": {"T1w": {"shape": list(t1w.shape), "affine": t1w.affine.tolist()},
                          "MNI": {"shape": list(mni.shape), "affine": mni.affine.tolist()}},
        "actual_template_resources": templates_before, "templates_unchanged": True,
        "input_files_unchanged": True, "workflow_wall_seconds": wall,
        "threads": args.threads, "execution_plugin": "Linear", "uses_gpu": False,
        "projection_output_checks": checks,
        "cifti_output_checks": {"shape": list(image.shape), "all_finite": True,
                                 "brain_models": brainmodels, "start_seconds": axis.start,
                                 "tr_seconds": axis.step, "sha256": sha256(result.out_file)},
        "cifti_metadata": metadata,
    })


def launch(args):
    inputs = read_inputs(args.inputs_json)
    image = args.container_image.resolve()
    if sha256(image) != IMAGE_SHA256:
        raise ValueError("container SHA-256 must match the fixed primary fMRIPrep 25.2.4 image")
    engine = shutil.which(str(args.singularity))
    if engine is None:
        raise FileNotFoundError("Singularity executable not found")
    output, work = args.output_root.resolve(), args.work_root.resolve()
    if output.exists() or output.is_symlink() or work.exists() or work.is_symlink():
        raise FileExistsError("reference output and work directories must both be new")
    work.mkdir(parents=True)
    home = work / "runtime_home"
    home.mkdir()
    mapped = dict(inputs)
    parents = {}
    for field in PATH_FIELDS:
        if inputs[field] is None:
            continue
        values = inputs[field] if field in SURFACE_FIELDS else [inputs[field]]
        translated = []
        for value in values:
            path = Path(value).resolve()
            if path.parent not in parents:
                parents[path.parent] = f"/reference-inputs/d{len(parents):02d}"
            translated.append(parents[path.parent] + "/" + path.name)
        mapped[field] = translated if field in SURFACE_FIELDS else translated[0]
    write_json(work / "inputs.private.json", mapped)
    command = [engine, "exec", "--cleanenv", "--home", f"{home}:/home/reference",
               "-B", f"{work}:/reference-work", "-B", f"{Path(__file__).resolve().parent}:/reference-script:ro"]
    for source, destination in parents.items():
        if any(character in str(source) for character in (",", ":")):
            raise ValueError("input directory cannot contain Singularity bind separators")
        command += ["-B", f"{source}:{destination}:ro"]
    environment = os.environ.copy()
    for prefix in ("SINGULARITYENV_", "APPTAINERENV_"):
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment[prefix + name] = str(args.threads)
        environment[prefix + "NIPYPE_NO_ET"] = "1"
        if args.templateflow_dir is not None:
            environment[prefix + "TEMPLATEFLOW_HOME"] = "/reference-templateflow"
    if args.templateflow_dir is not None:
        cache = args.templateflow_dir.resolve()
        if not cache.is_dir():
            raise FileNotFoundError("TemplateFlow cache directory not found")
        command += ["-B", f"{cache}:/reference-templateflow"]
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if name in environment:
            environment["SINGULARITYENV_" + name] = environment[name]
    command += [str(image), "python", "/reference-script/run_projection_reference.py", "--worker",
                "--inputs-json", "/reference-work/inputs.private.json",
                "--output-root", "/reference-work/result_staging", "--work-root", "/reference-work/workflow",
                "--threads", str(args.threads)]
    write_json(work / "command.private.json", {"command": command, "scope": SCOPE})
    start = time.perf_counter()
    with (work / "workflow.private.log").open("w") as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - start
    if completed.returncode:
        raise RuntimeError(f"reference workflow exited {completed.returncode}; inspect private work log")
    staging = work / "result_staging"
    record = json.loads((staging / "run.public.json").read_text())
    if record.get("validation_complete") is not True:
        raise ValueError("container did not complete the reference checks")
    record.update({"sif_sha256": IMAGE_SHA256, "container_wall_seconds": wall})
    write_json(staging / "run.public.json", record)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("output appeared during execution; refusing to replace it")
    os.replace(staging, output)
    print(json.dumps({"validation_complete": True, "frames": inputs["expected_frames"],
                      "grayordinates": 91282, "workflow_wall_seconds": record["workflow_wall_seconds"],
                      "scope": SCOPE}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--container-image", type=Path)
    parser.add_argument("--singularity", default="singularity")
    parser.add_argument("--templateflow-dir", type=Path, help="可选原站 TemplateFlow 缓存；不替换 installed ROI")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.worker:
        worker(args)
    else:
        if args.container_image is None:
            parser.error("container-image is required outside the reference container")
        launch(args)


if __name__ == "__main__":
    main()
