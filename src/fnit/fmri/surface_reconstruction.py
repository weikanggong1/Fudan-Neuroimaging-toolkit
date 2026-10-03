"""为 fMRI surface 准备只读来源或自产的 recon-all 与真实 middle 表面。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
import zipfile

import nibabel as nib
from nibabel.freesurfer import io as fsio
import numpy as np


_OWNER = "fnit.fmri.surface_reconstruction"
_MANIFEST = "fnit-surface-reconstruction.json"
_FILES = ("mri/orig.mgz", "mri/orig/001.mgz") + tuple(
    f"surf/{hemi}.{name}" for hemi in ("lh", "rh")
    for name in ("white", "pial", "sphere", "sphere.reg", "thickness", "sulc"))
_FNIT_OPTIONS = {"threads", "weights_dir", "assets_dir", "native_bin_dir", "profile_stages",
                 "cuda_allocator_cache", "hemisphere_workers", "native_optimizations"}
_FS_OPTIONS = {"threads", "command", "recon_all_command", "fs_license"}
_COMMON_OPTIONS = {"mris_expand_command", "native_bin_dir", "fs_license", "fsnative_to_t1w"}
_FNIT_BIN_NAMES = ("fnit_n4_itk", "mri_em_register", "mri_segment", "mri_edit_wm_with_aseg",
                   "mris_fix_topology_fnit", "mris_remove_intersection", "mris_inflate",
                   "mris_place_surface", "mris_place_surface_white_fast", "mrisp_paint",
                   "mris_curvature_stats", "mri_label2vol", "mri_warp_convert",
                   "mri_ca_register", "mri_convert", "mris_expand")


@dataclass(frozen=True)
class ReconstructionResult:
    """subject_dir 为可直接给 surface preparation 使用的目录；source_t1w 始终为原件。"""

    subject_dir: Path
    backend: str
    reused: bool
    source_t1w: Path
    metadata: dict


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_manifest(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _middle(subject, hemi):
    return next((subject / "surf" / f"{hemi}.{name}" for name in ("midthickness", "graymid")
                 if (subject / "surf" / f"{hemi}.{name}").is_file()), None)


def _subject_files(subject, *, require_middle):
    files = list(_FILES)
    for name in files:
        if not (subject / name).is_file():
            raise FileNotFoundError(subject / name)
    for hemi in ("lh", "rh"):
        middle = _middle(subject, hemi)
        if middle is not None:
            files.append(middle.relative_to(subject).as_posix())
        elif require_middle:
            raise FileNotFoundError(f"{subject}/surf/{hemi}.midthickness or {hemi}.graymid")
    return tuple(files)


def _validate_subject(subject, *, require_middle=True):
    """验证 surface 所需 closure；完整 FNIT 的 138 项检查仍由成熟 API 执行。"""
    files = _subject_files(subject, require_middle=require_middle)
    orig = nib.load(str(subject / "mri/orig.mgz"))
    if (not isinstance(orig, nib.MGHImage) or orig.ndim != 3
            or not np.isfinite(orig.affine).all()
            or abs(np.linalg.det(orig.affine[:3, :3])) < 1e-10):
        raise ValueError("recon-all orig.mgz must have finite, invertible 3D MGH geometry")
    for hemi in ("lh", "rh"):
        white, faces = fsio.read_geometry(str(subject / "surf" / f"{hemi}.white"))
        if (white.ndim != 2 or white.shape[1] != 3 or not len(white)
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or faces.min() < 0 or faces.max() >= len(white)):
            raise ValueError(f"{hemi} white geometry is invalid")
        geometries = [subject / "surf" / f"{hemi}.{name}"
                      for name in ("white", "pial", "sphere", "sphere.reg")]
        if _middle(subject, hemi) is not None:
            geometries.append(_middle(subject, hemi))
        for path in geometries:
            vertices, triangles = fsio.read_geometry(str(path))
            if (vertices.shape != white.shape or not np.isfinite(vertices).all()
                    or not np.array_equal(triangles, faces)):
                raise ValueError(f"{path.name} must preserve white vertex and face order")
        for name in ("thickness", "sulc"):
            values = fsio.read_morph_data(str(subject / "surf" / f"{hemi}.{name}"))
            if values.shape != (len(white),) or not np.isfinite(values).all():
                raise ValueError(f"{hemi}.{name} must contain one finite value per vertex")
    return files


def _validate_identity(subject, source, affine=None):
    scanner = nib.as_closest_canonical(nib.load(str(subject / "mri/orig/001.mgz")))
    raw = nib.as_closest_canonical(nib.load(str(source)))
    if scanner.ndim != 3 or raw.ndim != 3:
        raise ValueError("source T1w and recon-all orig/001.mgz must be 3D")
    if affine is not None:
        matrix = np.loadtxt(affine) if isinstance(affine, (str, Path)) else np.asarray(affine)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1])
                or abs(np.linalg.det(matrix[:3, :3])) < 1e-10):
            raise ValueError("fsnative_to_t1w must be a finite invertible 4x4 world affine")
        return {"OriginalT1Identity": "explicit fsnative-to-T1w world affine",
                "FsnativeToT1wWorldAffine": matrix.tolist()}
    if scanner.shape != raw.shape or not np.allclose(scanner.affine, raw.affine, rtol=0, atol=1e-4):
        raise ValueError("recon-all original T1 and source T1w have different grids; provide fsnative_to_t1w")
    original, source_values = (np.asarray(image.dataobj, dtype=np.float32) for image in (scanner, raw))
    if (not np.isfinite(original).all() or not np.isfinite(source_values).all()
            or not np.allclose(original, source_values, rtol=1e-5, atol=1e-3)):
        raise ValueError("recon-all original T1 does not match source T1w voxel content")
    return {"OriginalT1Identity": "matching source image grid and voxel content",
            "FsnativeToT1wWorldAffine": np.eye(4).tolist()}


def _zip_members(source):
    """只展开一例的必需文件；所有 ZIP 成员先检查路径、重复和符号链接。"""
    with zipfile.ZipFile(source) as archive:
        members = {}
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if (not info.filename or not path.parts or "\\" in info.filename or "\x00" in info.filename
                    or path.is_absolute() or ".." in path.parts or ":" in path.parts[0]
                    or info.filename in members
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f"unsafe or duplicate recon-all ZIP member: {info.filename!r}")
            members[info.filename] = info
        prefixes = [name[:-len("mri/orig.mgz")] for name in members if name.endswith("mri/orig.mgz")]
        candidates = [prefix for prefix in prefixes if all(prefix + name in members for name in _FILES)]
        if len(candidates) != 1:
            raise ValueError("recon-all ZIP must contain exactly one complete surface input subject")
        prefix = candidates[0]
        selected = {name: prefix + name for name in _FILES}
        for hemi in ("lh", "rh"):
            for name in ("midthickness", "graymid"):
                relative = f"surf/{hemi}.{name}"
                if prefix + relative in members:
                    selected[relative] = prefix + relative
                    break
        if any(members[name].is_dir() or members[name].file_size > 2 * 1024**3
               for name in selected.values()) or sum(members[name].file_size for name in selected.values()) > 8 * 1024**3:
            raise ValueError("recon-all ZIP surface inputs exceed the size limit or contain directories")
    return selected


def _extract_zip(source, subject, members):
    with zipfile.ZipFile(source) as archive:
        for relative, member in members.items():
            output = subject / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as reader, output.open("xb") as writer:
                shutil.copyfileobj(reader, writer)


def _prepare_fnit_t1(source, output):
    """仅改变文件表示及必要头字段，保持原始网格和 float32 体素值；不重采样。"""
    image = nib.load(str(source))
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)) or image.ndim != 3:
        raise ValueError("FNIT surface reconstruction expects a 3D NIfTI T1w")
    values = np.asarray(image.dataobj, dtype=np.float32)
    if (not np.isfinite(values).all() or not np.isfinite(image.affine).all()
            or abs(np.linalg.det(image.affine[:3, :3])) < 1e-10):
        raise ValueError("T1w must contain finite voxels and invertible finite geometry")
    units = image.header.get_xyzt_units()
    if units[0] not in ("mm", "unknown"):
        raise ValueError("source T1w must use millimeter geometry for the fMRI pipeline")
    legal = (isinstance(image, nib.Nifti1Image)
             and image.get_data_dtype().newbyteorder("=") == np.dtype("float32")
             and image.header["sform_code"] != 0
             and image.dataobj.slope == 1 and image.dataobj.inter == 0
             and units[0] == "mm" and units[1] in ("sec", "msec", "usec"))
    if legal:
        return source, {"converted": False, "source_units": list(units)}
    converted = nib.Nifti1Image(values, image.affine)
    converted.set_sform(image.affine, code=1)
    converted.set_qform(image.affine, code=1)
    converted.header.set_xyzt_units("mm", "sec")
    converted.header.set_slope_inter(1, 0)
    nib.save(converted, str(output))
    return output, {"converted": True, "source_units": list(units),
                    "representation": "NIfTI-1 float32, same voxel grid and float32 values, mm sform",
                    "unknown_spatial_units": "retain source affine as millimeters" if units[0] == "unknown" else None}


def _native_dir(options):
    return Path(options.get("native_bin_dir") or os.environ.get("FNIT_RECON_ALL_BIN_DIR")
                or Path(sys.prefix) / "bin").expanduser().resolve()


def _executable(value):
    executable = shutil.which(str(value))
    if executable is None:
        raise FileNotFoundError(f"executable not found: {value}")
    return Path(executable).resolve()


def _fnit_resources(options):
    from fnit.weights import resolve_weights
    from fnit.recon_all.assets import configured_dir, cache_dir

    weights = (Path(options["weights_dir"]).expanduser().resolve() if options.get("weights_dir")
               else resolve_weights("synthstrip.1.pt").resolve().parent)
    assets = Path(options.get("assets_dir") or os.environ.get("FNIT_ASSETS")
                  or configured_dir() or cache_dir() / "recon_all_assets").expanduser().resolve()
    if not weights.is_dir() or not assets.is_dir():
        raise FileNotFoundError("configure existing FNIT recon-all weights_dir and assets_dir before execution")
    return weights, assets


def _producer_fingerprint(backend):
    sources = [Path(__file__)]
    if backend == "fnit":
        package = Path(__file__).parent.parent
        sources.extend(sorted((package / "recon_all").rglob("*.py")))
        sources.append(package / "weights.py")
    package = Path(__file__).parent.parent
    return {path.relative_to(package).as_posix(): _sha256(path) for path in sources}


def _fnit_resource_fingerprint(weights, assets):
    """逐文件核对固定清单及真实字节；相同资源路径的覆盖不能复用旧结果。"""
    from fnit.weights import MODEL_FILES, WEIGHT_FILES
    from fnit.recon_all.assets import CORE_ASSETS, ASSET_FILES

    fingerprint = {}
    for label, root, names, registry in (
            ("weights", weights, MODEL_FILES["recon-all"], WEIGHT_FILES),
            ("assets", assets, CORE_ASSETS, ASSET_FILES)):
        entries = {}
        for name in names:
            path = root / name
            entry = registry[name]
            size, expected = entry[1:3] if label == "weights" else entry[:2]
            if not path.is_file() or path.stat().st_size != size:
                raise FileNotFoundError(f"missing or changed recon-all {label}: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(f"recon-all {label} SHA-256 differs from fixed manifest: {path}")
            entries[name] = {"size": size, "sha256": actual}
        fingerprint[label] = entries
    return fingerprint


def _run_fnit(**kwargs):
    from fnit.recon_all.native_free import run_recon_all_python
    return run_recon_all_python(**kwargs)


def _run_command(argv, *, cwd, env, log):
    with log.open("w") as stream:
        subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def _command_record(binary, argv, seconds, log):
    return {"argv": argv, "binary": str(binary), "sha256": _sha256(binary),
            "seconds": seconds, "log": str(log)}


def _complete_middle(subject, binary, environment, output_root, records):
    for hemi in ("lh", "rh"):
        if _middle(subject, hemi) is not None:
            continue
        # mris_expand resolves the sibling pial/thickness/sphere from the white path.
        argv = [str(binary), "-thickness", str(subject / "surf" / f"{hemi}.white"),
                "0.5", str(subject / "surf" / f"{hemi}.graymid")]
        log = output_root / f"mris-expand-{hemi}.log"
        started = time.perf_counter()
        record = _command_record(binary, argv, None, log)
        records.append(record)
        try:
            _run_command(argv, cwd=subject / "surf", env=environment, log=log)
        finally:
            record["seconds"] = time.perf_counter() - started


def _file_checksums(subject, files):
    return {name: {"size": (subject / name).stat().st_size, "sha256": _sha256(subject / name)}
            for name in files}


def _native_records(report):
    """保留成熟 API 报告中实际选择的 native 程序，供复用时核对二进制。"""
    records = {}

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and (key == "binary" or key.endswith("_binary")):
                    path = Path(item)
                    if path.is_file() and os.access(path, os.X_OK):
                        records[str(path.resolve())] = {"binary": str(path.resolve()), "sha256": _sha256(path)}
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(report)
    return list(records.values())


def _cached_report(manifest, request, subject):
    if manifest.get("status") != "complete" or manifest.get("request") != request:
        return False
    try:
        files = _validate_subject(subject)
        checksums = manifest["files"]
        if not set(files).issubset(checksums):
            return False
        for name, entry in checksums.items():
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                return False
            path = subject / name
            if path.is_symlink() or path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
                return False
        for entry in manifest.get("commands", []) + manifest.get("native_binaries", []):
            binary = Path(entry["binary"])
            if not binary.is_file() or _sha256(binary) != entry["sha256"]:
                return False
    except Exception:
        return False
    return True


def prepare_surface_reconstruction(source_t1w, work_dir, *, recon_all=None, backend=None,
                                   output_dir=None, device="cuda:0", options=None):
    """准备 recon-all：已有目录/ZIP、FNIT 全重建或显式官方 FreeSurfer。

    source_t1w 是 volume 元数据对应的原始 3D T1w；work_dir 为工作目录。
    recon_all 只表示已有输入，生成目录由 output_dir 指定（默认 work_dir/reconstruction）。
    backend=None 时，有 recon_all 选择 provided，否则选择 fnit；显式接受
    provided/fnit/freesurfer。生成物放在 output_dir/subject。options 对 FNIT 支持
    weights_dir/assets_dir 及完整 API 的线程、native、profiling、allocator 参数；
    freesurfer 支持 command（别名 recon_all_command）、threads、fs_license。
    mris_expand_command 可明确指定独立程序，否则使用当前 FNIT Conda native；
    仅显式 freesurfer 模式可从其官方程序目录发现 mris_expand。fsnative_to_t1w
    供已有来源显式指定 world affine；调用方必须继续传给下游表面坐标转换。
    原始输入不修改；缺 middle 的 provided 输入复制必需文件到自产目录再扩展。
    仅本函数 manifest、来源 SHA、参数及输出完整性/校验和均通过时复用自产目录。
    无资源下载或自动安装；命令通过 argv 执行，失败报告不会标记 complete。
    """
    started = time.perf_counter()
    source = Path(source_t1w).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    selected = backend or ("provided" if recon_all is not None else "fnit")
    if selected not in ("fnit", "freesurfer", "provided"):
        raise ValueError("reconstruction backend must be fnit, freesurfer or provided")
    if (selected == "provided") != (recon_all is not None):
        raise ValueError("recon_all is an existing input and requires backend=provided")
    options = dict(options or {})
    allowed = _COMMON_OPTIONS | (_FNIT_OPTIONS if selected == "fnit" else _FS_OPTIONS if selected == "freesurfer" else set())
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"unsupported {selected} reconstruction options: {sorted(unknown)}")
    threads = options.get("threads", 4)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("reconstruction threads must be a positive integer")
    if selected != "provided" and options.get("fsnative_to_t1w") is not None:
        raise ValueError("fsnative_to_t1w is only for provided recon-all inputs")
    request = {"backend": selected, "source_t1w": str(source), "source_sha256": _sha256(source),
               "device": str(device), "options": _json_value(options),
               "producer": _producer_fingerprint(selected)}
    # Snapshot an external affine's values, not merely its filename, for cache identity.
    affine = options.get("fsnative_to_t1w")
    if isinstance(affine, (str, Path)):
        request["options"]["fsnative_to_t1w"] = np.loadtxt(affine).tolist()
    input_subject, members = None, None
    if selected == "provided":
        existing = Path(recon_all).expanduser().resolve()
        if existing.is_file() and existing.suffix.lower() == ".zip":
            members = _zip_members(existing)
            request["provided"] = {"path": str(existing), "sha256": _sha256(existing)}
        elif existing.is_dir():
            input_subject = existing / "FreeSurfer" if (existing / "FreeSurfer").is_dir() else existing
            files = _validate_subject(input_subject, require_middle=False)
            identity = _validate_identity(input_subject, source, affine)
            request["provided"] = {"path": str(input_subject), "files": _file_checksums(input_subject, files)}
            if all(_middle(input_subject, hemi) is not None for hemi in ("lh", "rh")):
                metadata = {"owner": _OWNER, "status": "complete", "request": request,
                            "subject_dir": str(input_subject), "identity": identity,
                            "commands": [], "timing": {"whole_seconds": time.perf_counter() - started}}
                return ReconstructionResult(input_subject, selected, True, source, metadata)
        else:
            raise FileNotFoundError(existing)
    effective = {}
    if selected == "fnit":
        weights, assets = _fnit_resources(options)
        effective.update(weights_dir=str(weights), assets_dir=str(assets), native_bin_dir=str(_native_dir(options)))
        effective["resources"] = _fnit_resource_fingerprint(weights, assets)
        effective["native_binaries"] = {name: _sha256(_native_dir(options) / name)
            for name in _FNIT_BIN_NAMES if (_native_dir(options) / name).is_file()}
    fs_command = None
    if selected == "freesurfer":
        if options.get("command") and options.get("recon_all_command"):
            raise ValueError("specify command or recon_all_command, not both")
        fs_command = _executable(options.get("command") or options.get("recon_all_command") or "recon-all")
        effective.update(command=str(fs_command), command_sha256=_sha256(fs_command))
    needs_middle = (selected != "provided" or
                    (members is not None and any(not any(f"surf/{hemi}.{name}" in members
                     for name in ("midthickness", "graymid")) for hemi in ("lh", "rh"))) or
                    (input_subject is not None and any(_middle(input_subject, hemi) is None
                                                      for hemi in ("lh", "rh"))))
    expand = None
    if needs_middle:
        if options.get("mris_expand_command"):
            expand = _executable(options["mris_expand_command"])
        elif selected == "freesurfer":
            native_expand = _native_dir(options) / "mris_expand"
            expand = _executable(native_expand if native_expand.is_file() else fs_command.parent / "mris_expand")
        else:
            expand = _executable(_native_dir(options) / "mris_expand")
        effective.update(mris_expand=str(expand), mris_expand_sha256=_sha256(expand))
    request["effective"] = effective
    root = Path(output_dir or Path(work_dir) / "reconstruction").expanduser().resolve()
    subject = root / "subject"
    if selected == "provided" and existing.is_relative_to(subject):
        raise ValueError("provided recon-all inputs must be outside the generated subject directory")
    if input_subject is not None and (root.is_relative_to(input_subject) or input_subject.is_relative_to(subject)):
        raise ValueError("generated output must be separate from the read-only provided recon-all directory")
    if source.is_relative_to(subject) or source == root / "source_T1w.nii.gz":
        raise ValueError("source T1w must be separate from generated reconstruction files")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / _MANIFEST
    previous = None
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text())
        except (ValueError, OSError) as error:
            raise FileExistsError(f"invalid existing reconstruction manifest: {manifest_path}") from error
        if (previous.get("owner") != _OWNER or previous.get("schema") != 1
                or previous.get("subject_dir") != str(subject)):
            raise FileExistsError(f"output is not owned by the reconstruction adapter: {root}")
    elif subject.exists() and (subject.is_symlink() or any(subject.iterdir())):
        raise FileExistsError(f"existing subject directory has no FNIT adapter manifest: {subject}")
    if previous and _cached_report(previous, request, subject):
        _validate_identity(subject, source, affine)
        metadata = dict(previous, reused=True)
        metadata["timing"] = dict(previous["timing"], cache_lookup_seconds=time.perf_counter() - started)
        return ReconstructionResult(subject, selected, True, source, metadata)
    # A dedicated lock prevents simultaneous writers from deleting each other's owned output.
    lock = root / ".fnit-surface-reconstruction.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    metadata = {"owner": _OWNER, "schema": 1, "status": "running", "subject_dir": str(subject),
                "request": request, "commands": [], "timing": {}}
    try:
        if subject.is_symlink():
            raise ValueError("generated subject directory cannot be a symlink")
        if subject.exists():
            shutil.rmtree(subject)
        _write_manifest(manifest_path, metadata)
        environment = dict(os.environ)
        environment["OMP_NUM_THREADS"] = str(threads)
        if options.get("fs_license"):
            license_path = Path(options["fs_license"]).expanduser().resolve()
            if not license_path.is_file():
                raise FileNotFoundError(license_path)
            environment["FS_LICENSE"] = str(license_path)
        if selected == "provided":
            tick = time.perf_counter()
            if members is not None:
                _extract_zip(existing, subject, members)
            else:
                for name in request["provided"]["files"]:
                    output = subject / name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(input_subject / name, output)
            metadata["timing"]["input_copy_seconds"] = time.perf_counter() - tick
        elif selected == "fnit":
            tick = time.perf_counter()
            t1, conversion = _prepare_fnit_t1(source, root / "source_T1w.nii.gz")
            metadata["input_preparation"] = conversion
            metadata["timing"]["input_preparation_seconds"] = time.perf_counter() - tick
            kwargs = {key: value for key, value in options.items() if key in _FNIT_OPTIONS}
            kwargs.update(t1=t1, subject_dir=subject, weights_dir=weights, assets_dir=assets,
                          native_bin_dir=Path(effective["native_bin_dir"]), device=device)
            if options.get("fs_license"):
                # The mature API reads FS_LICENSE in its native subprocesses. Avoid changing
                # process-global environment here: the caller must configure it before FNIT.
                if os.environ.get("FS_LICENSE") != environment["FS_LICENSE"]:
                    raise ValueError("FNIT fs_license must already equal the process FS_LICENSE environment")
            tick = time.perf_counter()
            metadata["python_api"] = {"function": "fnit.recon_all.native_free.run_recon_all_python",
                                      "arguments": _json_value(kwargs)}
            metadata["reconstruction"] = _run_fnit(**kwargs)
            metadata["native_binaries"] = _native_records(metadata["reconstruction"])
            metadata["timing"]["reconstruction_seconds"] = time.perf_counter() - tick
            if metadata["reconstruction"].get("status") != "complete":
                raise RuntimeError("FNIT recon-all did not report a complete reconstruction")
        else:
            argv = [str(fs_command), "-all", "-i", str(source), "-s", subject.name,
                    "-sd", str(root), "-parallel", "-openmp", str(threads)]
            environment["SUBJECTS_DIR"] = str(root)
            environment["FREESURFER_HOME"] = str(fs_command.parent.parent)
            fs_home = fs_command.parent.parent
            environment["PATH"] = os.pathsep.join(dict.fromkeys(
                [str(fs_command.parent), str(fs_home / "bin"), str(fs_home / "mni/bin")]
                + environment.get("PATH", "").split(os.pathsep)))
            log = root / "recon-all.log"
            tick = time.perf_counter()
            record = _command_record(fs_command, argv, None, log)
            metadata["commands"].append(record)
            try:
                _run_command(argv, cwd=root, env=environment, log=log)
            finally:
                record["seconds"] = time.perf_counter() - tick
            seconds = record["seconds"]
            metadata["timing"]["reconstruction_seconds"] = seconds
        tick = time.perf_counter()
        _validate_subject(subject, require_middle=False)
        metadata["identity"] = _validate_identity(subject, source, affine)
        metadata["timing"]["pre_middle_validation_seconds"] = time.perf_counter() - tick
        if selected == "fnit":
            environment["FREESURFER_HOME"] = str(assets)
        metadata["runtime_environment"] = {name: environment[name] for name in
            ("FS_LICENSE", "FREESURFER_HOME", "OMP_NUM_THREADS") if name in environment}
        tick = time.perf_counter()
        _complete_middle(subject, expand, environment, root, metadata["commands"])
        metadata["timing"]["middle_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter()
        files = _validate_subject(subject)
        reported_outputs = metadata.get("reconstruction", {}).get("outputs", {})
        extra_files = []
        for value in reported_outputs.values():
            path = Path(value).resolve()
            if not path.is_relative_to(subject):
                raise ValueError("FNIT reconstruction reported an output outside its owned subject directory")
            extra_files.append(path.relative_to(subject).as_posix())
        files = tuple(sorted(set(files) | set(extra_files)))
        metadata["files"] = _file_checksums(subject, files)
        if _sha256(source) != request["source_sha256"]:
            raise RuntimeError("source T1w changed during reconstruction")
        if selected == "provided":
            if members is not None:
                unchanged = _sha256(existing) == request["provided"]["sha256"]
            else:
                unchanged = _file_checksums(input_subject, request["provided"]["files"]) == request["provided"]["files"]
            if not unchanged:
                raise RuntimeError("provided recon-all inputs changed during preparation")
        metadata["timing"]["final_validation_seconds"] = time.perf_counter() - tick
        metadata["timing"]["whole_seconds"] = time.perf_counter() - started
        metadata["status"] = "complete"
        _write_manifest(manifest_path, metadata)
        return ReconstructionResult(subject, selected, False, source, metadata)
    except Exception as error:
        metadata.update(status="failed", error=repr(error))
        metadata["timing"]["whole_seconds"] = time.perf_counter() - started
        try:
            _write_manifest(manifest_path, metadata)
        except OSError as report_error:
            if hasattr(error, "add_note"):
                error.add_note(f"could not write failed reconstruction report: {report_error!r}")
        raise
    finally:
        lock.unlink(missing_ok=True)
