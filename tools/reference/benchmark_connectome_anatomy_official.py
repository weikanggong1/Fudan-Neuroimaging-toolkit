"""隔离官方 FS-only/SynthMorph 解剖参照；不导入 FNIT，不伪造 tracking PT。

prepare: 本轮 fresh recon-all → 官方 5TT/GMWMI 与八套 T1 atlas。
recover-prepare: 只读验证旧失败报告中已成功的官方 SynthMorph 三命令，续未运行步骤。
recover-atlas: 绑定 unknown=0 导致的原脚本失败，复用成功前缀并续 atlas。
complete: 已验证 prepare + 官方自产 DWI contract → FLIRT 与 DWI atlas。
仅 CPU；官方 GPU SynthMorph 必须另行实现受授权的设备/锁/显存监测。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

PROFILES = {
    "aparc+tian-s1": ("aparc", 1),
    "aparc.a2009s+tian-s1": ("aparc.a2009s", 1),
    "glasser+tian-s1": ("Glasser", 1),
    "glasser+tian-s4": ("Glasser", 4),
    "schaefer200+tian-s1": ("Schaefer200", 1),
    "schaefer500+tian-s4": ("Schaefer500", 4),
    "schaefer1000+tian-s4": ("Schaefer1000", 4),
}
WEIGHTS = {
    "synthmorph.affine.2.h5": (51455312, "1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6"),
    "synthmorph.deform.3.h5": (3508630424, "95b367cd30788cc647e4704b650642fc1d70d7e419c20c04f1ba1b2902bc6536"),
}
FIELDS = ("index", "original_label", "hemisphere", "name")
REFERENCE_PYTHON_PROBE = (
    "import sys,json,nibabel,numpy,scipy,pandas; "
    "print(json.dumps({'python':sys.version,'executable':sys.executable,"
    "'nibabel':nibabel.__version__,'numpy':numpy.__version__,"
    "'scipy':scipy.__version__,'pandas':pandas.__version__}))"
)

# Official SynthMorph writes an MGH warp intent (version 0x301).  nibabel's
# regular MGH image reader correctly refuses it; read with the installed
# official Surfa runtime instead of reinterpreting or modifying the header.
WARP_READBACK = """
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import numpy as np
import surfa as sf
warp = sf.load_warp(sys.argv[1])
def geometry(value):
    return {'shape': list(map(int, value.shape)),
            'affine': value.vox2world.matrix.tolist()}
result = {'shape': list(map(int, warp.shape)), 'dtype': str(warp.data.dtype),
          'format': int(warp.format), 'nonfinite_count': int((~np.isfinite(warp.data)).sum()),
          'source': geometry(warp.source), 'target': geometry(warp.target),
          'python': platform.python_version(), 'python_executable': sys.executable,
          'packages': {name: metadata.version(name) for name in
                       ('tensorflow', 'surfa', 'voxelmorph', 'neurite', 'numpy')}}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, allow_nan=False) + '\\n')
"""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_file(record):
    path = Path(record["path"]).resolve()
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(path)
    if record.get("size_bytes") is not None and path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"size changed: {path}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"SHA changed: {path}")
    return path


def read_bound_json(record):
    return json.loads(verify_file(record).read_text())


def image_record(path, *, labels=False):
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    result = {**file_record(path), "shape": list(map(int, image.shape)),
              "dtype": str(data.dtype), "affine": image.affine.tolist(),
              "spacing": list(map(float, image.header.get_zooms()[:3])),
              "nonfinite_count": int((~np.isfinite(data)).sum()),
              "foreground_voxels": int(np.count_nonzero(data))}
    if not np.isfinite(image.affine).all():
        raise ValueError(f"nonfinite affine: {path}")
    if result["nonfinite_count"]:
        raise ValueError(f"nonfinite image values: {path}")
    if labels:
        if not np.isfinite(data).all() or (data < 0).any() or not np.equal(data, np.round(data)).all():
            raise ValueError(f"invalid labels: {path}")
        result["labels"] = np.unique(data).astype(int).tolist()
    return result


def verify_anatomy(config):
    report = read_bound_json(config["anatomy_report"])
    if report.get("status") != "completed" or report.get("case_id") != config["case_id"]:
        raise ValueError("fresh anatomy report is incomplete or belongs to another case")
    original = read_bound_json(report["original_report"]) if report.get("original_report") else report
    raw = verify_file(config["raw_t1w"])
    if original.get("exit_code") != 0:
        raise ValueError("official recon-all did not exit successfully")
    command = original.get("command", [])
    if "-i" not in command or Path(command[command.index("-i") + 1]).resolve() != raw:
        raise ValueError("official recon-all does not bind the original T1")
    rows = original.get("raw_input_provenance", [])
    if not any(row.get("kind") == "raw_t1w" and Path(row["path"]).resolve() == raw
               and row["sha256"] == config["raw_t1w"]["sha256"] for row in rows):
        raise ValueError("original T1 SHA absent from fresh reconstruction provenance")
    subject = Path(config["subject_dir"]).resolve()
    expected_subject = (Path(command[command.index("-sd") + 1]) / command[command.index("-s") + 1]).resolve()
    if subject != expected_subject:
        raise ValueError("subject directory differs from actual recon-all command")
    files = report.get("anatomy", {})
    required = ["mri/brain.mgz", "mri/aparc+aseg.mgz", "mri/ribbon.mgz", "scripts/recon-all.done"]
    required += [f"surf/{hemi}.{name}" for hemi in ("lh", "rh") for name in ("white", "pial", "sphere.reg")]
    required += [f"label/{hemi}.{name}.annot" for hemi in ("lh", "rh") for name in ("aparc", "aparc.a2009s")]
    for name in required:
        record = files.get(name)
        if record is None or verify_file(record) != (subject / name).resolve():
            raise ValueError(f"unbound anatomy input: {name}")
    return {"report": config["anatomy_report"], "original_report": report.get("original_report"),
            "raw_t1w": config["raw_t1w"], "subject_dir": str(subject), "files": files,
            "official_recon_command_seconds": original.get("recon_command_seconds"),
            "recon_all_rerun": False}


def verify_dwi_contract(record, case_id):
    contract = read_bound_json(record)
    if (contract.get("schema_version") != 1 or contract.get("case_id") != case_id
            or contract.get("scope") != "official_self_produced_raw_dwi_chain"
            or contract.get("state") != "completed"):
        raise ValueError("independent official raw-DWI contract required")
    report = read_bound_json(contract["upstream_report"])
    if not (report.get("execution_completed") is True or report.get("state") == "completed"
            or report.get("status") == "completed"):
        raise ValueError("official upstream report is incomplete")
    if not report.get("commands") and not report.get("completed_commands"):
        raise ValueError("upstream report has no actual official commands")
    paths = {name: verify_file(contract["files"][name]) for name in
             ("corrected_dwi", "mean_b0", "mean_b0_brain", "brain_mask")}
    grid = nib.load(str(paths["corrected_dwi"]))
    if len(grid.shape) != 4:
        raise ValueError("official corrected DWI must be 4D")
    for name in ("mean_b0", "mean_b0_brain", "brain_mask"):
        image = nib.load(str(paths[name]))
        if image.shape != grid.shape[:3] or not np.allclose(image.affine, grid.affine, atol=1e-5, rtol=0):
            raise ValueError(f"{name}: official output is not on the corrected DWI grid")
    return contract, paths


def dwi_consumption_origin(record, contract):
    """Bind the complete source bytes and only the four images this tool consumes.

    MRtrix vector images can declare NaN spacing for their nonspatial component
    axis. Such unused metadata belongs to the original bound contract; copying
    it into this tool's strict JSON report is unnecessary and prevents saving.
    No source metadata or MRI is changed by this selective provenance record.
    """
    report = read_bound_json(contract["upstream_report"])
    commands = report.get("commands") or report.get("completed_commands")
    files = {}
    for name in ("corrected_dwi", "mean_b0", "mean_b0_brain", "brain_mask"):
        source = contract["files"][name]
        files[name] = {"path": source["path"], "sha256": source["sha256"],
                       "size_bytes": source.get("size_bytes", Path(source["path"]).stat().st_size)}
    return {
        "contract": record,
        "binding": {key: contract[key] for key in ("schema_version", "case_id", "scope", "state")}
            | {"upstream_report": contract["upstream_report"], "files": files},
        "upstream_execution_proof": {
            "report": contract["upstream_report"], "state": report.get("state"),
            "status": report.get("status"), "execution_completed": report.get("execution_completed"),
            "completed_by_verified_contract_rule": True,
            "actual_recorded_commands": len(commands),
            "all_recorded_returncodes_zero": all(item.get("returncode") == 0 for item in commands)},
        "recording_scope": "actual consumed image identities and verified source execution proof only; "
            "the unchanged complete upstream contract, including unused metadata, is bound by its original bytes SHA"
    }


def write_nodes(path, rows):
    if [int(row["index"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("node indices must be contiguous")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_nodes(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return [{**row, "index": int(row["index"]), "original_label": int(row["original_label"])} for row in rows]


def verify_canonical_lut(nodes, freesurfer_lut, mrtrix_lut):
    fs_names = {}
    for line in Path(freesurfer_lut).read_text().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit() and len(parts) >= 2:
            fs_names[int(parts[0])] = parts[1]
    names = {}
    for line in Path(mrtrix_lut).read_text().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit() and len(parts) >= 3:
            names.setdefault(int(parts[0]), set()).add(parts[2])
    if len(nodes) != 84 or [row["index"] for row in nodes] != list(range(1, 85)):
        raise ValueError("canonical fs-aparc nodes must have 84 consecutive rows")
    for row in nodes:
        official_name = fs_names.get(row["original_label"])
        if official_name not in names.get(row["index"], set()):
            raise ValueError(f"node {row['index']} differs from official fs_default/FreeSurfer LUT")


def combine_native(cortical_path, tian_path, output_path, cortical_rows, tian_names):
    cortical, tian = nib.load(str(cortical_path)), nib.load(str(tian_path))
    if cortical.shape != tian.shape or not np.allclose(cortical.affine, tian.affine, atol=1e-5, rtol=0):
        raise ValueError("native cortical/Tian grids differ")
    a, b = np.asanyarray(cortical.dataobj), np.asanyarray(tian.dataobj)
    k = len(cortical_rows)
    for values, maximum in ((a, k), (b, len(tian_names))):
        if not np.isfinite(values).all() or (values < 0).any() or (values > maximum).any() or not np.equal(values, np.round(values)).all():
            raise ValueError("native atlas labels are outside the node table")
    # This is the declared UKB cortical-precedence formula, not a registration port.
    combined = np.where(a > 0, a, np.where(b > 0, b + k, 0)).astype(np.int32)
    nib.save(nib.Nifti1Image(combined, cortical.affine), str(output_path))
    rows = [dict(row) for row in cortical_rows]
    rows += [{"index": k + i, "original_label": i,
              "hemisphere": "R" if name.endswith("-rh") else "L" if name.endswith("-lh") else "",
              "name": name} for i, name in enumerate(tian_names, 1)]
    return rows


def synthmorph_commands(config, directory):
    fs = Path(config["freesurfer_home"])
    program = str(fs / "bin/mri_synthmorph")
    atlas = Path(config["upstream_root"]) / "data/templates/atlases"
    warp = directory / "mni_to_t1.mgz"
    register = [program, "register", "-m", "joint", "-j", str(config["threads"]),
                "-w", str(fs / "models/synthmorph.affine.2.h5"),
                "-w", str(fs / "models/synthmorph.deform.3.h5"),
                "-t", str(warp), config["mni_template"], str(Path(config["subject_dir"]) / "mri/brain.mgz")]
    return [("synthmorph_register", register), *[(f"tian_s{scale}_apply", [program, "apply",
        "-m", "nearest", "-t", "int16", str(warp),
        str(atlas / f"Tian_Subcortex_S{scale}_3T.nii.gz"),
        str(directory / f"tian_s{scale}_t1.nii.gz")]) for scale in (1, 4)]]


def annotation_background_input(source, target):
    """A private input copy fixes the original converter's missing unknown=0 key.

    The original annotation, color table, names and positive ROI indices remain
    unchanged. Only vertices whose label is the proven unknown index 0 become
    the existing background -1. Actual readback and source SHA prove that scope.
    """
    started = time.perf_counter()
    original_record = file_record(source)
    labels, ctab, names = nib.freesurfer.read_annot(str(source))
    count = int(np.count_nonzero(labels == 0))
    result = {"source": original_record, "vertices": int(labels.size),
              "unknown_zero_vertices": count, "negative_background_vertices": int((labels == -1).sum()),
              "unknown_name": names[0].decode("utf-8"), "unknown_ctab": ctab[0].tolist(),
              "positive_indices": np.unique(labels[labels > 0]).astype(int).tolist(),
              "applied": count > 0, "rule": "proven unknown index 0 -> existing background -1"}
    if not count:
        result.update(output=original_record, seconds=time.perf_counter() - started)
        return Path(source), result
    if names[0].decode("utf-8").lower() != "unknown":
        raise ValueError("annotation index 0 is not the declared unknown background")
    if (labels < -1).any() or (labels >= len(names)).any():
        raise ValueError("annotation contains an invalid source index")
    target = Path(target)
    if target.exists() or target.resolve() == Path(source).resolve():
        raise ValueError("background adapter requires a new private annotation")
    target.parent.mkdir(parents=True, exist_ok=True)
    adapted = labels.copy()
    adapted[labels == 0] = -1
    nib.freesurfer.write_annot(str(target), adapted, ctab.copy(), names, fill_ctab=False)
    read_labels, read_ctab, read_names = nib.freesurfer.read_annot(str(target))
    if (not np.array_equal(read_labels, adapted) or not np.array_equal(read_ctab, ctab)
            or read_names != names or not np.array_equal(read_labels[labels > 0], labels[labels > 0])
            or not np.array_equal(read_labels < 1, labels < 1)):
        raise ValueError("annotation background adapter changed a positive ROI, color table or names")
    verify_file(original_record)
    result.update(output=file_record(target), ctab_names_equal=True, positive_indices_equal=True,
                  background_vertex_mask_equal=True, surface_files_changed=False,
                  seconds=time.perf_counter() - started)
    return target, result


def prepare_prefix_commands(config, output):
    """The successful official prefix before the native annotation converter."""
    fs, mr, subject = (Path(config[key]) for key in ("freesurfer_home", "mrtrix_bin", "subject_dir"))
    sm = output / "synthmorph"
    fs_default = mr.parent / "share/mrtrix3/labelconvert/fs_default.txt"
    return [
        ("reference_python_runtime", [config["python"], "-c", REFERENCE_PYTHON_PROBE]),
        ("mrtrix_runtime_version", [mr / "mrconvert", "-version"]),
        *synthmorph_commands(config, sm),
        ("official_warp_readback", [fs / "bin/fspython", "-c", WARP_READBACK,
                                  sm / "mni_to_t1.mgz", sm / "warp_metadata.json"]),
        ("5ttgen", [mr / "5ttgen", "freesurfer", subject / "mri/aparc+aseg.mgz", output / "five_tissue_t1.nii.gz",
                    "-nocrop", "-sgm_amyg_hipp", "-nthreads", "8"]),
        ("5tt2gmwmi", [mr / "5tt2gmwmi", output / "five_tissue_t1.nii.gz", output / "gmwmi_t1.nii.gz", "-nthreads", "8"]),
        ("fs_aparc_labelconvert", [mr / "labelconvert", subject / "mri/aparc+aseg.mgz", fs / "FreeSurferColorLUT.txt",
            fs_default, output / "atlases/fs-aparc/atlas_t1.nii.gz", "-nthreads", "8"]),
    ]


def verified_atlas_recovery(report_record, config, identity):
    """Reuse only the measured successful prefix of the original unknown=0 failure."""
    prior = read_bound_json(report_record)
    if (prior.get("case_id") != config["case_id"] or prior.get("mode") != "prepare"
            or prior.get("state") != "failed" or prior.get("execution_completed") is not False
            or prior.get("preflight") != identity):
        raise ValueError("same-input failed official prepare required for atlas recovery")
    root = Path(report_record["path"]).resolve().parent
    expected = prepare_prefix_commands(config, root)
    commands = prior.get("commands", [])
    if len(commands) != len(expected) + 1:
        raise ValueError("atlas recovery requires exactly the successful official prefix and its failed converter")
    for actual, (stage, argv) in zip(commands[:-1], expected):
        if (actual.get("stage") != stage or actual.get("returncode") != 0
                or actual.get("argv") != list(map(str, argv))):
            raise ValueError("prior official prefix argv or successful status differs")
        verify_file(actual["program"])
    last = commands[-1]
    subject = Path(config["subject_dir"])
    converter = root / "original_wrapper/scripts/python/convert_native_annot.py"
    native = root / "original_wrapper/data/temporary/subjects/public_0/atlases"
    argv = [config["python"], converter, subject / "label/lh.aparc.annot", subject / "label/rh.aparc.annot",
            native / "lh.native.aparc.annot", native / "rh.native.aparc.annot"]
    if (last.get("stage") != "aparc_convert" or last.get("returncode") == 0
            or last.get("argv") != list(map(str, argv))):
        raise ValueError("atlas recovery is limited to the measured original native aparc converter failure")
    verify_file(last["program"])
    failure_log = file_record(last["log"])
    if "KeyError: 0" not in Path(last["log"]).read_text():
        raise ValueError("original converter did not fail on the unknown index 0")
    files = {name: prior["outputs"][name] for name in
             ("tian_s1_t1", "tian_s4_t1", "mni_to_t1", "five_tissue_t1", "gmwmi_t1", "atlas:fs-aparc")}
    for record in files.values():
        verify_file(record)
        if record.get("metadata_sidecar"):
            verify_file(record["metadata_sidecar"])
    nodes_path = root / "atlases/fs-aparc/nodes.tsv"
    if read_nodes(nodes_path) != read_nodes(config["canonical_nodes84"]):
        raise ValueError("prior fs-aparc nodes differ from canonical nodes")
    return {"report": report_record, "original_source_commit": prior["source_commit"],
            "original_script_sha256": prior["script_sha256"], "original_successful_commands": commands[:-1],
            "original_failed_command": last, "original_failure_log": failure_log,
            "failure_log_sha_observation_scope": "first recorded at recovery entry; original failed log preserved",
            "original_successful_command_seconds": sum(row["seconds_inclusive"] for row in commands[:-1]),
            "original_failed_command_seconds": last["seconds_inclusive"], "files": files,
            "nodes84": file_record(nodes_path), "nodes_sha_observation_scope": "first bound at recovery entry; canonical rows verified"}


def verified_synthmorph_recovery(report_record, config, identity):
    """Bind successful official commands without rewriting their failed report."""
    prior = read_bound_json(report_record)
    supplemental = {"runtime_libraries", "official_auxiliary_files"}
    prior_identity = {k: v for k, v in prior.get("preflight", {}).items() if k not in supplemental}
    expected_identity = {k: v for k, v in identity.items() if k not in supplemental}
    if (prior.get("case_id") != config["case_id"] or prior.get("mode") != "prepare"
            or prior.get("state") != "failed" or prior.get("execution_completed") is not False
            or prior_identity != expected_identity):
        raise ValueError("same-input failed official prepare required for staged recovery")
    if prior.get("error", {}).get("type") != "HeaderDataError":
        raise ValueError("recovery only accepts the documented warp image-readback failure")
    source = Path(report_record["path"]).resolve().parent / "synthmorph"
    expected = synthmorph_commands(config, source)
    commands = prior.get("commands", [])
    if len(commands) != 3:
        raise ValueError("exactly three successful official SynthMorph commands required")
    for actual, (stage, argv) in zip(commands, expected):
        if actual.get("stage") != stage or actual.get("returncode") != 0 or actual.get("argv") != list(map(str, argv)):
            raise ValueError("prior official SynthMorph command does not match the actual contract")
        verify_file(actual["program"])
    files = {}
    for scale in (1, 4):
        bound = prior["outputs"][f"tian_s{scale}_t1"]
        if verify_file(bound) != (source / f"tian_s{scale}_t1.nii.gz").resolve():
            raise ValueError("prior Tian output directory mismatch")
        files[f"tian_s{scale}_t1.nii.gz"] = bound
    # The former image reader failed before recording the warp. Its SHA is first
    # observed in this recovery, not retroactively attributed to the old run.
    files["mni_to_t1.mgz"] = file_record(source / "mni_to_t1.mgz")
    return {"report": report_record, "original_source_commit": prior["source_commit"],
            "original_script_sha256": prior["script_sha256"], "original_commands": commands,
            "original_failed_state": prior["state"], "files": files,
            "runtime_libraries_for_continuation": identity.get("runtime_libraries", []),
            "prior_runtime_libraries": prior.get("preflight", {}).get("runtime_libraries", []),
            "continuation_auxiliary_inputs": identity.get("official_auxiliary_files", {}),
            "warp_sha_observation_scope": "first bound at recovery entry; prior three official commands exited zero",
            "original_command_seconds": sum(row["seconds_inclusive"] for row in commands)}


class Runner:
    def __init__(self, config, output, mode):
        self.config, self.output = config, output
        fs = Path(config["freesurfer_home"])
        self.environment = {**os.environ, "FREESURFER_HOME": str(fs),
            "FS_LICENSE": config["fs_license"], "FSLDIR": str(Path(config["fsl_bin"]).parent),
            "FSLOUTPUTTYPE": "NIFTI_GZ", "CUDA_VISIBLE_DEVICES": "",
            "SUBJECTS_DIR": str(output / "subjects"), "PYTHONNOUSERSITE": "1"}
        self.environment["PATH"] = os.pathsep.join([str(Path(config["python"]).parent),
            config["mrtrix_bin"], config["fsl_bin"], str(fs / "bin"), self.environment.get("PATH", "")])
        directories = config.get("runtime_library_dirs", [])
        if directories:
            self.environment["LD_LIBRARY_PATH"] = os.pathsep.join([*directories,
                *[x for x in self.environment.get("LD_LIBRARY_PATH", "").split(os.pathsep) if x]])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
            self.environment[name] = str(config["threads"])
        self.report = {"schema_version": 1, "case_id": config["case_id"],
            "profile": "fnit-native-official-reference", "mode": mode, "state": "running",
            "execution_completed": False, "connectome_completed": False,
            "script_sha256": sha256(__file__), "source_commit": config["tool_commit"],
            "config": config, "commands": [], "outputs": {}, "cpu_host": platform.node(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": config["threads"],
            "precision": "installed official implementations and default joint extent256/steps7; no -g",
            "gpu": {"used": False, "allocated": None, "reserved": None},
            "timing_scope": "tool entry, verification, official commands, native combination/readback/output hashing; Python import/startup excluded; no recon-all rerun"}
        self.report["runtime_library_dirs"] = directories

    def save(self):
        temporary = self.output / "reference_anatomy.json.tmp"
        temporary.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output / "reference_anatomy.json")

    def run(self, stage, argv):
        index = len(self.report["commands"])
        log = self.output / "logs" / f"{index:03d}_{stage}.log"
        timer = self.output / "logs" / f"{index:03d}_{stage}.time"
        executable = Path(argv[0]).resolve()
        record = {"stage": stage, "argv": list(map(str, argv)), "program": file_record(executable), "log": str(log)}
        started = time.perf_counter()
        with log.open("w") as stream:
            result = subprocess.run(["/usr/bin/time", "-f", "wall_seconds=%e peak_rss_kib=%M", "-o", str(timer),
                                     *map(str, argv)], env=self.environment, stdout=stream, stderr=subprocess.STDOUT)
        record.update(seconds_inclusive=time.perf_counter() - started, returncode=result.returncode,
                      time_file=str(timer), time_text=timer.read_text() if timer.exists() else None)
        self.report["commands"].append(record)
        self.save()
        if result.returncode:
            raise RuntimeError(f"official stage {stage} failed: {log}")

    def output_image(self, name, path, labels=False):
        self.report["outputs"][name] = image_record(path, labels=labels)
        self.save()

    def output_warp(self, name, path):
        metadata_path = self.output / "synthmorph" / "warp_metadata.json"
        self.run("official_warp_readback", [Path(self.config["freesurfer_home"]) / "bin/fspython",
                 "-c", WARP_READBACK, path, metadata_path])
        metadata = json.loads(metadata_path.read_text())
        fixed = nib.load(str(Path(self.config["subject_dir"]) / "mri/brain.mgz"))
        moving = nib.load(self.config["mni_template"])
        if metadata["nonfinite_count"] or metadata["shape"] != [*fixed.shape[:3], 3]:
            raise ValueError("official warp is nonfinite or has the wrong target grid")
        for actual, expected in ((metadata["source"], moving), (metadata["target"], fixed)):
            if actual["shape"] != list(expected.shape[:3]) or not np.allclose(actual["affine"], expected.affine, atol=1e-5, rtol=0):
                raise ValueError("official warp embedded geometry differs from the actual source/target")
        self.report["outputs"][name] = {**file_record(path), **metadata,
                                        "metadata_sidecar": file_record(metadata_path),
                                        "reader": "installed official Surfa sf.load_warp; warp intent 0x301"}
        self.save()


def preflight(config):
    if config.get("schema_version") != 1 or config.get("profile") != "fnit-native" or config.get("threads") != 8:
        raise ValueError("schema1, fnit-native, 8 CPU threads required")
    if not config.get("tool_commit"):
        raise ValueError("explicit frozen tool commit required")
    if not Path(config["fs_license"]).is_file():
        raise FileNotFoundError("FreeSurfer license path missing")
    # Never open or hash FS_LICENSE, including when caller puts it in the asset list.
    license_path = Path(config["fs_license"]).resolve()
    assets = {}
    for record in config["assets"]:
        if Path(record["path"]).resolve() == license_path:
            raise ValueError("license contents must not be hashed or published")
        if not record.get("license") or not record.get("source") or not record.get("license_url"):
            raise ValueError("asset license/source declaration required")
        assets[str(verify_file(record))] = record
    fs, upstream = Path(config["freesurfer_home"]), Path(config["upstream_root"])
    required = [Path(config["mni_template"]), Path(config["canonical_nodes84"])]
    required += [fs / "models" / name for name in WEIGHTS]
    required += [Path(config["fsaverage_dir"]) / "surf" / f"{h}.sphere.reg" for h in ("lh", "rh")]
    for scale in (1, 4):
        required += [upstream / "data/templates/atlases" / f"Tian_Subcortex_S{scale}_3T{suffix}" for suffix in (".nii.gz", "_label.txt")]
    for count in (200, 500, 1000):
        required += [upstream / "data/templates/atlases" / f"{h}.Schaefer2018_{count}Parcels_7Networks_order.annot" for h in ("lh", "rh")]
    required += [upstream / "data/templates/atlases" / "Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"]
    for side in ("L", "R"):
        required += [upstream / "data/templates/surfaces" / f"{side}.sphere.32k_fs_LR.surf.gii",
                     upstream / "data/templates/surfaces" / f"fs_{side}-to-fs_LR_fsaverage.{side}_LR.spherical_std.164k_fs_{side}.surf.gii"]
    for path in required:
        if str(path.resolve()) not in assets:
            raise ValueError(f"required asset lacks hash/license binding: {path}")
    for name, (size, expected) in WEIGHTS.items():
        asset = assets[str((fs / "models" / name).resolve())]
        if asset["sha256"] != expected or asset["size_bytes"] != size:
            raise ValueError("official weights differ from fixed FNIT resource identity")
    runtime_libraries = []
    for directory in config.get("runtime_library_dirs", []):
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        library = directory / "libstdc++.so.6"
        record = next((r for r in config.get("runtime_libraries", [])
                       if Path(r["path"]).resolve() == library.resolve()), None)
        if record is None:
            raise ValueError("explicit runtime library directory requires bound libstdc++.so.6 SHA")
        verify_file(record)
        runtime_libraries.append(record)
    scripts = {}
    for name in ("convert_native_annot.py", "convert_schaefer_annot.py", "convert_labels_gii_to_annot.py", "map_surface_label_to_volume.py"):
        record = config["upstream_scripts"][name]
        path = verify_file(record)
        if path != (upstream / "scripts/python" / name).resolve():
            raise ValueError("upstream script directory mismatch")
        scripts[name] = record
    anatomy = verify_anatomy(config)
    for program in [Path(config["python"]), Path(config["workbench_command"]),
                    *[Path(config["mrtrix_bin"]) / name for name in ("5ttgen", "5tt2gmwmi", "labelconvert", "mrtransform", "transformconvert")],
                    Path(config["fsl_bin"]) / "flirt", fs / "bin/mri_synthmorph", fs / "bin/mri_surf2surf", fs / "bin/mri_convert"]:
        if not program.is_file() or not os.access(program, os.X_OK):
            raise FileNotFoundError(program)
    verify_canonical_lut(read_nodes(config["canonical_nodes84"]), fs / "FreeSurferColorLUT.txt",
                         Path(config["mrtrix_bin"]).parent / "share/mrtrix3/labelconvert/fs_default.txt")
    auxiliary = [fs / "FreeSurferColorLUT.txt", Path(config["mrtrix_bin"]).parent / "share/mrtrix3/labelconvert/fs_default.txt",
                 Path(config["mrtrix_bin"]).parent / "lib/libmrtrix.so"]
    auxiliary += [Path(config["fsaverage_dir"]) / f"surf/{h}.orig" for h in ("lh", "rh")]
    return {"anatomy": anatomy, "assets": assets, "upstream_scripts": scripts,
            "official_auxiliary_files": {str(p.resolve()): file_record(p) for p in auxiliary},
            "runtime_libraries": runtime_libraries,
            "freesurfer_build_stamp": file_record(fs / "build-stamp.txt"),
            "freesurfer_version": (fs / "build-stamp.txt").read_text().strip(),
            "official_synthmorph_source": file_record(fs / "python/scripts/mri_synthmorph"),
            "official_fspython_wrapper": file_record(fs / "bin/fspython"),
            "official_synthmorph_modules": {str(p.relative_to(fs)): file_record(p)
                 for p in sorted((fs / "python/packages/synthmorph").glob("*.py"))}}


def cortex_nodes(converted_left, converted_right, subject, name):
    left = nib.freesurfer.read_annot(str(converted_left))
    right = nib.freesurfer.read_annot(str(converted_right))
    if left[2] != right[2]:
        raise ValueError("converted cortical annotation tables differ")
    left_ids = set(np.unique(left[0]).tolist())
    names = [x.decode("utf-8") for x in left[2][1:]]
    rows = [{"index": i, "original_label": i, "hemisphere": "L" if i in left_ids else "R", "name": value}
            for i, value in enumerate(names, 1)]
    if name in ("aparc", "aparc.a2009s"):
        # Original native tables preserve each hemisphere's original annotation index.
        originals = [nib.freesurfer.read_annot(str(subject / f"label/{h}.{name}.annot"))[2] for h in ("lh", "rh")]
        for row in rows:
            h = 0 if row["hemisphere"] == "L" else 1
            prefix = "left_" if h == 0 else "right_"
            row["original_label"] = [x.decode() for x in originals[h]].index(row["name"][len(prefix):])
    return rows


def prepare(runner, recovered_synthmorph=None, recovered_prefix=None):
    c, out = runner.config, runner.output
    fs, mr, upstream = Path(c["freesurfer_home"]), Path(c["mrtrix_bin"]), Path(c["upstream_root"])
    subject = Path(c["subject_dir"])
    runner.run("reference_python_runtime", [c["python"], "-c", REFERENCE_PYTHON_PROBE])
    # Fail on a missing host runtime before doing a several-minute registration.
    runner.run("mrtrix_runtime_version", [mr / "mrconvert", "-version"])
    private = out / "original_wrapper"
    (private / "data/temporary/subjects/public_0/atlases").mkdir(parents=True)
    (private / "reference_subjects/public_0").mkdir(parents=True)
    (private / "reference_subjects/public_0/FreeSurfer").symlink_to(subject, target_is_directory=True)
    (private / "scripts").symlink_to(upstream / "scripts", target_is_directory=True)
    (private / "data/templates").symlink_to(upstream / "data/templates", target_is_directory=True)
    subjects = out / "subjects"
    subjects.mkdir()
    (subjects / "fsaverage").symlink_to(c["fsaverage_dir"], target_is_directory=True)
    (subjects / subject.name).symlink_to(subject, target_is_directory=True)
    native = private / "data/temporary/subjects/public_0/atlases"
    sm = out / "synthmorph"
    sm.mkdir()
    if recovered_prefix is not None:
        runner.report["reused_official_prefix_origin"] = recovered_prefix
        runner.report["timing_scope"] += "; staged atlas recovery reuses separately timed successful official prefix; includes private background adapter; not continuous cold"
        for source_name, target_name in (("tian_s1_t1", "tian_s1_t1.nii.gz"), ("tian_s4_t1", "tian_s4_t1.nii.gz"),
                                          ("mni_to_t1", "mni_to_t1.mgz")):
            record = recovered_prefix["files"][source_name]
            shutil.copyfile(verify_file(record), sm / target_name)
            verify_file(record)
            if sha256(sm / target_name) != record["sha256"]:
                raise ValueError("recovered official prefix output changed while copying")
        runner.save()
    elif recovered_synthmorph is None:
        for name, argv in synthmorph_commands(c, sm):
            runner.run(name, argv)
    else:
        runner.report["reused_synthmorph_origin"] = recovered_synthmorph
        runner.report["timing_scope"] += "; staged recovery reuses separately timed successful official SynthMorph; not continuous cold"
        for name, record in recovered_synthmorph["files"].items():
            source = verify_file(record)
            shutil.copyfile(source, sm / name)
            verify_file(record)
            if sha256(sm / name) != record["sha256"]:
                raise ValueError("recovered official output changed while copying")
        runner.save()
    for scale in (1, 4):
        runner.output_image(f"tian_s{scale}_t1", sm / f"tian_s{scale}_t1.nii.gz", labels=True)
    runner.output_warp("mni_to_t1", sm / "mni_to_t1.mgz")
    if recovered_prefix is None:
        runner.run("5ttgen", [mr / "5ttgen", "freesurfer", subject / "mri/aparc+aseg.mgz", out / "five_tissue_t1.nii.gz", "-nocrop", "-sgm_amyg_hipp", "-nthreads", "8"])
        runner.run("5tt2gmwmi", [mr / "5tt2gmwmi", out / "five_tissue_t1.nii.gz", out / "gmwmi_t1.nii.gz", "-nthreads", "8"])
    else:
        for name in ("five_tissue_t1", "gmwmi_t1"):
            record = recovered_prefix["files"][name]
            shutil.copyfile(verify_file(record), out / f"{name}.nii.gz")
            if sha256(out / f"{name}.nii.gz") != record["sha256"]:
                raise ValueError("recovered official tissue output changed while copying")
    for name in ("five_tissue_t1", "gmwmi_t1"):
        runner.output_image(name, out / f"{name}.nii.gz")
    fs_default = mr.parent / "share/mrtrix3/labelconvert/fs_default.txt"
    runner.report["official_luts"] = {"fs_default": file_record(fs_default), "freesurfer": file_record(fs / "FreeSurferColorLUT.txt")}
    atlas_root = out / "atlases"
    atlas_root.mkdir()
    target = atlas_root / "fs-aparc"
    target.mkdir()
    if recovered_prefix is None:
        runner.run("fs_aparc_labelconvert", [mr / "labelconvert", subject / "mri/aparc+aseg.mgz", fs / "FreeSurferColorLUT.txt", fs_default, target / "atlas_t1.nii.gz", "-nthreads", "8"])
    else:
        record = recovered_prefix["files"]["atlas:fs-aparc"]
        shutil.copyfile(verify_file(record), target / "atlas_t1.nii.gz")
        if sha256(target / "atlas_t1.nii.gz") != record["sha256"]:
            raise ValueError("recovered official native atlas changed while copying")
    rows84 = read_nodes(c["canonical_nodes84"])
    if len(rows84) != 84:
        raise ValueError("canonical fs-aparc nodes must have 84 rows")
    write_nodes(target / "nodes.tsv", rows84)
    runner.output_image("atlas:fs-aparc", target / "atlas_t1.nii.gz", labels=True)
    cortical = {}
    for name in ("aparc", "aparc.a2009s", "Schaefer200", "Schaefer500", "Schaefer1000", "Glasser"):
        left, right = (native / f"{h}.native.{name}.annot" for h in ("lh", "rh"))
        script = private / "scripts/python"
        if name in ("aparc", "aparc.a2009s"):
            inputs = []
            for h in ("lh", "rh"):
                source = subject / f"label/{h}.{name}.annot"
                supplied, binding = annotation_background_input(source, out / "annotation_background_inputs" / f"{h}.{name}.annot")
                runner.report.setdefault("annotation_background_compatibility", {})[f"{h}.{name}"] = binding
                inputs.append(supplied)
            runner.save()
            runner.run(f"{name}_convert", [c["python"], script / "convert_native_annot.py", *inputs, left, right])
        else:
            scratch = out / name
            scratch.mkdir()
            for h, side, cortex in (("lh", "L", "CORTEX_LEFT"), ("rh", "R", "CORTEX_RIGHT")):
                if name == "Glasser":
                    t = upstream / "data/templates"
                    label32, label164 = scratch / f"{h}.32k.label.gii", scratch / f"{h}.164k.label.gii"
                    runner.run(f"Glasser_{h}_separate", [c["workbench_command"], "-cifti-separate", t / "atlases/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii", "COLUMN", "-label", cortex, label32])
                    runner.run(f"Glasser_{h}_resample", [c["workbench_command"], "-label-resample", label32, t / f"surfaces/{side}.sphere.32k_fs_LR.surf.gii", t / f"surfaces/fs_{side}-to-fs_LR_fsaverage.{side}_LR.spherical_std.164k_fs_{side}.surf.gii", "BARYCENTRIC", label164])
                    runner.run(f"Glasser_{h}_convert", [c["python"], script / "convert_labels_gii_to_annot.py", label164, scratch / f"{h}.fsaverage.annot"])
            if name.startswith("Schaefer"):
                count = int(name[8:])
                t = upstream / "data/templates/atlases"
                runner.run(f"{name}_convert", [c["python"], script / "convert_schaefer_annot.py", t / f"lh.Schaefer2018_{count}Parcels_7Networks_order.annot", t / f"rh.Schaefer2018_{count}Parcels_7Networks_order.annot", scratch / "lh.fsaverage.annot", scratch / "rh.fsaverage.annot"])
            for h, destination in (("lh", left), ("rh", right)):
                runner.run(f"{name}_{h}_surf2surf", [fs / "bin/mri_surf2surf", "--srcsubject", "fsaverage", "--trgsubject", subject.name, "--hemi", h, "--sval-annot", scratch / f"{h}.fsaverage.annot", "--tval", destination])
        runner.run(f"{name}_volume", [c["python"], script / "map_surface_label_to_volume.py", private, private / "reference_subjects", "public", "0", name])
        volume = native / f"native.{name}.nii.gz"
        cortical[name] = (volume, cortex_nodes(left, right, subject, name))
        runner.output_image(f"cortical:{name}", volume, labels=True)
    for profile, (name, scale) in PROFILES.items():
        target = atlas_root / profile
        target.mkdir()
        names = (upstream / f"data/templates/atlases/Tian_Subcortex_S{scale}_3T_label.txt").read_text().splitlines()
        volume, cortex_rows = cortical[name]
        rows = combine_native(volume, sm / f"tian_s{scale}_t1.nii.gz", target / "atlas_t1.nii.gz", cortex_rows, names)
        write_nodes(target / "nodes.tsv", rows)
        runner.output_image(f"atlas:{profile}", target / "atlas_t1.nii.gz", labels=True)
    for profile in ("fs-aparc", *PROFILES):
        runner.report["outputs"][f"nodes:{profile}"] = file_record(atlas_root / profile / "nodes.tsv")
    runner.report["state"] = "official_structural_reference_completed"
    runner.report["execution_completed"] = True
    runner.report["full_raw_connectome"] = False
    if recovered_prefix is not None:
        verify_file(recovered_prefix["report"])
        verify_file(recovered_prefix["original_failure_log"])
        for record in recovered_prefix["files"].values():
            verify_file(record)


def complete(runner, prepared_record, dwi_record):
    c, out = runner.config, runner.output
    prepared = read_bound_json(prepared_record)
    if not prepared.get("execution_completed") or prepared.get("state") != "official_structural_reference_completed" or prepared.get("case_id") != c["case_id"]:
        raise ValueError("verified same-case official prepare report required")
    if prepared["preflight"]["anatomy"] != runner.report["preflight"]["anatomy"]:
        raise ValueError("prepared structural anatomy identity differs")
    for record in prepared["outputs"].values():
        verify_file(record)
    contract, paths = verify_dwi_contract(dwi_record, c["case_id"])
    runner.report["prepared_origin"] = prepared_record
    runner.report["official_dwi_origin"] = dwi_consumption_origin(dwi_record, contract)
    mr, fs = Path(c["mrtrix_bin"]), Path(c["freesurfer_home"])
    subject = Path(c["subject_dir"])
    runner.run("brain_to_nifti", [fs / "bin/mri_convert", subject / "mri/brain.mgz", out / "brain.nii.gz"])
    runner.run("flirt", [Path(c["fsl_bin"]) / "flirt", "-in", paths["mean_b0_brain"], "-ref", out / "brain.nii.gz", "-cost", "normmi", "-dof", "6", "-omat", out / "dwi_to_t1_fsl.txt"])
    runner.run("transformconvert", [mr / "transformconvert", out / "dwi_to_t1_fsl.txt", paths["mean_b0_brain"], out / "brain.nii.gz", "flirt_import", out / "dwi_to_t1_mrtrix.txt", "-nthreads", "8"])
    for name in ("five_tissue", "gmwmi"):
        source = prepared["outputs"][f"{name}_t1"]["path"]
        runner.run(f"{name}_world", [mr / "mrtransform", source, out / f"{name}_dwi_world.nii.gz", "-linear", out / "dwi_to_t1_mrtrix.txt", "-inverse", "-nthreads", "8"])
        runner.output_image(name, out / f"{name}_dwi_world.nii.gz")
    (out / "atlases").mkdir()
    for profile in ("fs-aparc", *PROFILES):
        target = out / "atlases" / profile
        target.mkdir()
        source = prepared["outputs"][f"atlas:{profile}"]["path"]
        runner.run(f"atlas_{profile}_nn", [mr / "mrtransform", source, target / "atlas_dwi.nii.gz", "-linear", out / "dwi_to_t1_mrtrix.txt", "-inverse", "-template", paths["mean_b0"], "-interp", "nearest", "-datatype", "uint32", "-nthreads", "8"])
        shutil.copyfile(prepared["outputs"][f"nodes:{profile}"]["path"], target / "nodes.tsv")
        rows = read_nodes(target / "nodes.tsv")
        record = image_record(target / "atlas_dwi.nii.gz", labels=True)
        reference = nib.load(str(paths["mean_b0"]))
        if (record["shape"] != list(reference.shape) or max(record["labels"]) > len(rows)
                or not np.allclose(record["affine"], reference.affine, atol=1e-5, rtol=0)):
            raise ValueError("official atlas output grid/labels inconsistent")
        runner.report["outputs"][f"atlas:{profile}"] = record
        runner.report["outputs"][f"nodes:{profile}"] = file_record(target / "nodes.tsv")
    for name in ("dwi_to_t1_fsl", "dwi_to_t1_mrtrix"):
        runner.report["outputs"][name] = file_record(out / f"{name}.txt")
    verify_file(prepared_record)
    for record in prepared["outputs"].values():
        verify_file(record)
    verify_dwi_contract(dwi_record, c["case_id"])
    runner.report["state"] = "official_anatomy_and_dwi_atlas_completed"
    runner.report["execution_completed"] = True
    runner.report["full_raw_connectome"] = False


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "recover-prepare", "recover-atlas", "complete"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh directory; failures remain intact")
    parser.add_argument("--prepared-report", type=Path)
    parser.add_argument("--official-dwi-contract", type=Path)
    parser.add_argument("--successful-synthmorph-report", type=Path,
                        help="recover-prepare only: preserved failed prepare after three successful official SynthMorph commands")
    parser.add_argument("--failed-atlas-report", type=Path,
                        help="recover-atlas only: preserved original native annotation unknown index 0 failure")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (args.output.exists() or (args.mode == "complete" and (not args.prepared_report or not args.official_dwi_contract))
            or (args.mode == "recover-prepare" and not args.successful_synthmorph_report)
            or (args.mode != "recover-prepare" and args.successful_synthmorph_report)
            or (args.mode == "recover-atlas" and not args.failed_atlas_report)
            or (args.mode != "recover-atlas" and args.failed_atlas_report)):
        parser.error("fresh output required; complete needs both contracts; recovery needs its explicit preserved report")
    config = json.loads(args.config.read_text())
    identity = preflight(config)
    recovery = (verified_synthmorph_recovery(file_record(args.successful_synthmorph_report), config, identity)
                if args.mode == "recover-prepare" else None)
    atlas_recovery = (verified_atlas_recovery(file_record(args.failed_atlas_report), config, identity)
                      if args.mode == "recover-atlas" else None)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "preflight": identity,
                          "synthmorph_commands": synthmorph_commands(config, args.output / "synthmorph")}, default=str, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    runner = Runner(config, args.output, args.mode)
    runner.report["preflight"] = identity
    runner.report["config_sha256"] = sha256(args.config)
    runner.save()
    try:
        if args.mode in ("prepare", "recover-prepare", "recover-atlas"):
            prepare(runner, recovery, atlas_recovery)
        else:
            complete(runner, file_record(args.prepared_report), file_record(args.official_dwi_contract))
        # Detect any input/source/asset change rather than blessing an old hash.
        if preflight(config) != identity:
            raise ValueError("input/asset/source identity changed during official reference")
    except Exception as error:
        runner.report.update(state="failed", execution_completed=False,
                             error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        runner.report["total_wall_seconds"] = time.perf_counter() - started
        runner.save()
    print(json.dumps({"state": runner.report["state"], "report": str(args.output / "reference_anatomy.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
