"""Read-only volume handoff inspection before fMRI surface processing."""

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import nibabel as nib
import numpy as np

from .derivatives import fmri_derivative_paths, sidecar


@dataclass(frozen=True)
class SurfaceVolumeStatus:
    """Surface 所需 volume 输入的只读检查结果。

    ``ready`` 表示所选 preproc/clean 分支通过检查；``missing`` 表示该
    run 的全部 volume 文件和 sidecar 均不存在；``partial`` 表示已有
    run 输出，但所需文件不完整；``invalid`` 表示来源、数据或空间无效，
    或完全未做 volume 时不能唯一选择 T1w。仅 ``missing`` 可自动运行
    volume。``source_t1w`` 保留 BIDS 的逻辑路径，``expected_paths`` 列出
    所选分支的必要文件，``reasons`` 为失败原因，ready 时为空。
    """

    state: Literal["ready", "missing", "partial", "invalid"]
    source_t1w: Path | None
    reasons: tuple[str, ...]
    expected_paths: tuple[Path, ...]

    def __post_init__(self):
        if self.state not in ("ready", "missing", "partial", "invalid"):
            raise ValueError("unknown surface volume state")


def _present(path):
    # Dangling links and directories also reserve a publication destination.
    return path.exists() or path.is_symlink()


def _run_paths(paths):
    images = (paths.preproc_mni, paths.preproc_t1w, paths.clean_mni,
              paths.clean_native, paths.mask_mni, paths.mni_pull)
    return (*images, *(sidecar(path) for path in images), paths.bbr_matrix,
            paths.bbr_matrix.with_suffix(".json"), paths.motion_pull,
            paths.motion_pull.with_suffix(".json"))


def _expected(paths, signal, source_t1):
    mni = paths.preproc_mni if signal == "preproc" else paths.clean_mni
    native = paths.preproc_t1w if signal == "preproc" else paths.clean_native
    required = (mni, sidecar(mni), native, sidecar(native))
    if source_t1 is not None:
        required += (paths.t1_brain,)
    if signal == "clean":
        required += (paths.bbr_matrix,)
    return required


def _metadata(path):
    details = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(details, dict) or not isinstance(details.get("FNIT"), dict):
        raise ValueError(f"volume metadata must contain an FNIT object: {path}")
    return details


def _recorded_t1(inputs, metadata):
    label = metadata["FNIT"].get("SourceT1w")
    if not isinstance(label, str) or not label:
        raise ValueError("volume derivative does not identify its source T1w; rerun volume")
    relative = Path(label)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("SourceT1w must be a relative BIDS source path without '..'")
    recorded = inputs.bids_root / relative
    logical = [path for path in inputs.t1w_images if path == recorded]
    matches = logical or [path for path in inputs.t1w_images
                          if path.resolve() == recorded.resolve()]
    if not matches:
        raise ValueError("volume derivative T1w does not belong to this BIDS subject")
    if len(matches) != 1:
        raise ValueError("volume derivative SourceT1w is ambiguous; record its exact BIDS path")
    return matches[0]


def _dataset_source(root, inputs):
    """Inspect an existing dataset declaration without creating or repairing it."""
    description = root / "dataset_description.json"
    if not _present(description):
        return
    details = json.loads(description.read_text(encoding="utf-8"))
    if not isinstance(details, dict) or details.get("DatasetType") != "derivative":
        raise ValueError(f"output is not a BIDS Derivatives dataset: {description}")
    links = details.get("DatasetLinks", {})
    if not isinstance(links, dict):
        raise ValueError("derivative DatasetLinks must be an object")
    raw = links.get("raw")
    if raw is not None:
        if not isinstance(raw, str) or (root / raw).resolve() != inputs.bids_root.resolve():
            raise ValueError("derivative dataset has a different BIDS raw source")
    elif not any(isinstance(item, dict) and item.get("Name") == "fudan-neuroimaging-toolkit"
                 for item in details.get("GeneratedBy", [])):
        raise ValueError("output is not an FNIT derivative dataset")


def _template(metadata, requested):
    details = metadata["FNIT"]
    digest = details.get("StandardTemplateSHA256", "")
    if (details.get("StandardSpace") != "MNI152NLin6Asym"
            or details.get("StandardTemplateIdentity") != "TemplateFlow:MNI152NLin6Asym:res-02"
            or not isinstance(digest, str) or len(digest) != 64
            or any(character.lower() not in "0123456789abcdef" for character in digest)):
        raise ValueError("volume derivative lacks verified MNI152NLin6Asym template identity; rerun volume")
    if requested is None:
        return None
    # This checks voxel content against the two allowed fixed templates, then
    # checks the actual file bytes against what this volume run recorded.
    from .end_to_end import _standard_template_identity

    requested = Path(requested).expanduser().resolve()
    actual = _standard_template_identity(requested)
    if actual["StandardTemplateSHA256"].lower() != digest.lower():
        raise ValueError("volume StandardTemplateSHA256 differs from the supplied mni_template file")
    return nib.load(str(requested))


def _finite(image, name):
    if image.ndim == 4:
        # Decode bounded time chunks instead of materializing a full run.
        for start in range(0, image.shape[3], 8):
            if not np.isfinite(np.asanyarray(image.dataobj[..., start:start + 8])).all():
                raise ValueError(f"{name} contains nonfinite values in frames {start}:{min(start + 8, image.shape[3])}")
    elif not np.isfinite(np.asanyarray(image.dataobj)).all():
        raise ValueError(f"{name} contains nonfinite values")


def inspect_surface_volume(
    inputs, derivatives_root: str | Path, *, signal: str = "preproc",
    t1w_image: str | Path | None = None, mni_template: str | Path | None = None,
    hcp_assets_dir: str | Path | None = None, check_finite: bool = True,
) -> SurfaceVolumeStatus:
    """Check one located BIDS run's volume inputs without writing any files.

    ``inputs`` is the result of ``locate_bids_inputs``. Existing metadata
    selects its logical ``SourceT1w``; an explicit ``t1w_image`` must agree.
    With no run outputs, T1 selection follows volume's explicit/unique rule.
    Only files required by ``signal`` are needed for ``ready``; other volume
    outputs still prevent a half-finished run from being classified missing.

    ``mni_template`` verifies fixed template voxel content and the recorded
    file SHA-256. Without that file, only the declaration/hash format and
    fixed grid are checked; no actual template hash match is claimed.
    ``hcp_assets_dir`` additionally validates the official CIFTI ROI/dseg
    assets and compares the physical MNI/dseg grid. With neither reference
    supplied, the known MNI152NLin6Asym 2-mm LAS grid is checked directly.
    ``check_finite=False`` skips numerical finite scans only, not provenance,
    dimensions, TR or geometry. Invalid options raise; invalid input data
    are returned as ``invalid``. No option grants volume overwrite permission.
    """
    if signal not in ("preproc", "clean"):
        raise ValueError("signal must be 'preproc' or 'clean'")
    if not isinstance(check_finite, bool):
        raise TypeError("check_finite must be bool")
    # Functional naming does not depend on T1. A placeholder here avoids
    # selecting the first anatomical candidate merely to inspect run paths.
    probe = fmri_derivative_paths(inputs, Path("T1w.nii.gz"), derivatives_root, signal=signal)
    source_t1 = None
    expected = _expected(probe, signal, source_t1)
    try:
        from .end_to_end import _select_t1
        from . import surface_pipeline as surface

        _dataset_source(probe.root, inputs)
        present = any(_present(path) for path in _run_paths(probe))
        if not present:
            source_t1 = _select_t1(inputs, t1w_image)
            paths = fmri_derivative_paths(inputs, source_t1, probe.root, signal=signal)
            return SurfaceVolumeStatus("missing", source_t1,
                                       ("no volume outputs exist for the selected BIDS run",),
                                       _expected(paths, signal, source_t1))

        selected = ((probe.preproc_mni, probe.preproc_t1w) if signal == "preproc"
                    else (probe.clean_mni, probe.clean_native))
        other = ((probe.clean_mni, probe.clean_native) if signal == "preproc"
                 else (probe.preproc_mni, probe.preproc_t1w))
        metadata_file = next((sidecar(path) for path in (*selected, *other)
                              if sidecar(path).is_file()), None)
        if metadata_file is not None:
            metadata = _metadata(metadata_file)
            source_t1 = _recorded_t1(inputs, metadata)
            if t1w_image is not None and _select_t1(inputs, t1w_image) != source_t1:
                raise ValueError("t1w_image differs from the existing volume SourceT1w")
            surface._validate_run_metadata(metadata, inputs, source_t1)
        elif t1w_image is not None or len(inputs.t1w_images) == 1:
            source_t1 = _select_t1(inputs, t1w_image)
        paths = (fmri_derivative_paths(inputs, source_t1, probe.root, signal=signal)
                 if source_t1 is not None else probe)
        expected = _expected(paths, signal, source_t1)
        missing = tuple(f"required volume input is missing or not a file: {path}"
                        for path in expected if not path.is_file())
        if source_t1 is None:
            missing += ("volume source T1w is unknown; restore its sidecar or supply t1w_image",)
        if missing:
            return SurfaceVolumeStatus("partial", source_t1, missing, expected)

        mni_path, native_path = selected
        metadata = _metadata(sidecar(mni_path))
        native_metadata = _metadata(sidecar(native_path))
        source_label = source_t1.relative_to(inputs.bids_root).as_posix()
        validator = (surface._validate_preproc_metadata if signal == "preproc"
                     else surface._validate_volume_metadata)
        for details in (metadata, native_metadata):
            validator(details, inputs, source_t1)
            if details["FNIT"].get("SourceT1w") != source_label:
                raise ValueError("volume derivative does not identify the selected logical source T1w")
        template = _template(metadata, mni_template)
        mni = nib.load(str(mni_path), keep_file_open=True)
        native = nib.load(str(native_path), keep_file_open=True)
        brain = nib.load(str(paths.t1_brain), keep_file_open=True)
        raw = nib.load(str(inputs.bold))
        surface._tr_seconds(mni, "MNI BOLD", inputs.tr)
        surface._tr_seconds(native, "T1w BOLD" if signal == "preproc" else "native BOLD", inputs.tr)
        if raw.ndim != 4 or mni.shape[3] != native.shape[3] or native.shape[3] != raw.shape[3]:
            raise ValueError("surface inputs and raw BOLD must have equal frame counts")
        surface._millimeter_affine(native, "T1w BOLD" if signal == "preproc" else "native BOLD")
        surface._millimeter_affine(brain, "source T1w brain")
        if brain.ndim != 3 or any(size < 1 for size in brain.shape):
            raise ValueError("source T1w brain must be a nonempty 3D image")
        raw_t1 = nib.as_closest_canonical(nib.load(str(source_t1)))
        canonical_brain = nib.as_closest_canonical(brain)
        if (raw_t1.ndim != 3 or raw_t1.shape != canonical_brain.shape
                or not np.allclose(raw_t1.affine, canonical_brain.affine, rtol=0, atol=1e-4)):
            raise ValueError("source T1w brain grid differs from the selected raw T1w")
        brain_sidecar = sidecar(paths.t1_brain)
        if _present(brain_sidecar):
            brain_metadata = json.loads(brain_sidecar.read_text(encoding="utf-8"))
            if (not isinstance(brain_metadata, dict)
                    or f"bids:raw:{source_label}" not in brain_metadata.get("Sources", [])):
                raise ValueError("T1w brain derivative does not identify the selected source T1w")

        if hcp_assets_dir is not None:
            from .assets_setup import MESH

            assets = Path(hcp_assets_dir).expanduser().resolve()
            labels, _ = surface._cifti_assets(
                assets / MESH / "L.atlasroi.32k_fs_LR.shape.gii",
                assets / MESH / "R.atlasroi.32k_fs_LR.shape.gii",
                assets / "fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz",
            )
            surface._mni_grid(surface._las_grid(mni), labels)
        if template is not None:
            surface._mni_grid(surface._las_grid(mni), surface._las_grid(template))
        elif hcp_assets_dir is None:
            fixed_grid = SimpleNamespace(shape=(91, 109, 91), affine=np.array([
                [-2., 0., 0., 90.], [0., 2., 0., -126.],
                [0., 0., 2., -72.], [0., 0., 0., 1.],
            ]))
            surface._mni_grid(surface._las_grid(mni), fixed_grid)

        if signal == "preproc":
            if (native_metadata.get("Resolution") != "native BOLD resolution"
                    or native_metadata.get("SpatialReference") != f"bids:raw:{source_label}"):
                raise ValueError("preprocessed T1w BOLD lacks its native-resolution source-T1w identity")
            reference = nib.as_closest_canonical(nib.load(str(inputs.sbref or inputs.bold)))
            native_zooms = np.round(reference.header.get_zooms()[:3], 3)
            if not np.allclose(nib.affines.voxel_sizes(native.affine), native_zooms,
                               rtol=1e-6, atol=1e-5):
                raise ValueError("preprocessed T1w BOLD does not use the native BOLD voxel sizes")
        else:
            surface._validate_native_bold(native, inputs)
            matrix = np.loadtxt(paths.bbr_matrix)
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or abs(np.linalg.det(matrix[:3, :3])) < 1e-10
                    or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-6)):
                raise ValueError("volume BBR matrix must be a finite invertible 4-by-4 affine")
        if check_finite:
            for image, name in ((mni, "MNI BOLD"), (native, "T1w BOLD"), (brain, "source T1w brain")):
                _finite(image, name)
        return SurfaceVolumeStatus("ready", source_t1, (), expected)
    except (ValueError, TypeError, KeyError, AttributeError, OSError, EOFError,
            nib.filebasedimages.ImageFileError) as error:
        return SurfaceVolumeStatus("invalid", source_t1, (str(error),), expected)
