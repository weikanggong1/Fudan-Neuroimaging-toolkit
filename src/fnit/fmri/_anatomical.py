"""Verified, subject/session-local cache of run-independent anatomical work."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from ..fast import TorchFAST
from ..weights import resolve_weights
from .normalization import T1MNIResult, _completed_time, register_t1_to_mni


_ARTIFACTS = (
    "T1_brain.nii.gz", "T1_mask.nii.gz", "T1_pve_wm.nii.gz",
    "T1_pve_csf.nii.gz", "T1_wmseg.nii.gz", "MNI_brain.nii.gz",
    "MNI_mask.nii.gz", "T1_to_MNI152_2mm_affine.mat",
    "MNI152_2mm_to_T1_pull_ras.nii.gz",
)
_PHASES = ("t1_synthstrip", "template_preparation", "fast", "t1_to_mni_affine",
           "t1_to_mni_nonlinear", "warp_conversion")


@dataclass(frozen=True)
class AnatomicalResult:
    directory: Path
    registration: T1MNIResult
    timing_seconds: dict[str, float]
    reused: bool
    fingerprint: str | None

    def path(self, name):
        return self.directory / name


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(data, reference, path):
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    header.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image(data, reference.affine, header), path)


def _implementation_hash():
    package = Path(__file__).resolve().parents[1]
    files = [Path(__file__), Path(__file__).with_name("normalization.py")]
    files.extend(package / name for name in ("_nib.py", "_transforms.py", "weights.py"))
    for component in ("flirt", "fnirt", "fast", "synthstrip", "synthmorph"):
        files.extend(sorted((package / component).glob("*.py")))
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(package).as_posix().encode())
        digest.update(bytes.fromhex(_sha256(file)))
    return digest.hexdigest()


def _fingerprint(t1w, template, template_mask, strip, backend, morph_weights,
                 fnirt_config, device, fnirt_execution):
    weights = {"synthstrip": _sha256(strip.model_path)}
    if backend == "synthmorph":
        explicit = morph_weights.get("deform") if isinstance(morph_weights, dict) else morph_weights
        weights["synthmorph"] = _sha256(resolve_weights("synthmorph.deform.3.h5", explicit))
    config = asdict(fnirt_config) if fnirt_config is not None else None
    identity = {
        "schema": 1, "t1": _sha256(t1w), "template": _sha256(template),
        "template_mask": _sha256(template_mask) if template_mask is not None else None,
        "weights": weights, "backend": backend, "fnirt_config": config,
        "fnirt_execution": fnirt_execution,
        "implementation": _implementation_hash(), "device_type": torch.device(device).type,
        "torch": torch.__version__, "numpy": np.__version__, "nibabel": nib.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": ({"name": torch.cuda.get_device_name(device),
                 "capability": torch.cuda.get_device_capability(device)}
                if torch.device(device).type == "cuda" else None),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@contextmanager
def _lock(directory):
    """Serialize producers; both Unix and Windows use standard-library locks."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as stream:
        try:
            import fcntl
        except ImportError:
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"\0")
                stream.flush()
            while True:
                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_verified(directory, fingerprint):
    try:
        metadata = json.loads((directory / "manifest.json").read_text())
        if metadata["fingerprint"] != fingerprint or set(metadata["sha256"]) != set(_ARTIFACTS):
            return None
        if any(_sha256(directory / name) != metadata["sha256"][name] for name in _ARTIFACTS):
            return None
        matrix = np.asarray(metadata["moving_to_fixed_world"], dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            return None
        return metadata
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _result(directory, metadata, timing, reused):
    return AnatomicalResult(
        directory=directory,
        registration=T1MNIResult(
            affine=directory / "T1_to_MNI152_2mm_affine.mat",
            pull_ras=directory / "MNI152_2mm_to_T1_pull_ras.nii.gz",
            backend=metadata["backend"],
            moving_to_fixed_world=np.asarray(metadata["moving_to_fixed_world"]),
            qc=metadata["qc"],
            timing_seconds={key: timing[key] for key in _PHASES[-3:]},
        ),
        timing_seconds=timing, reused=reused, fingerprint=metadata["fingerprint"],
    )


def _produce(directory, t1w, template_path, template_mask_path, strip, backend,
             morph_weights, fnirt_config, device, fnirt_execution):
    directory.mkdir(parents=True, exist_ok=True)
    timing = {}
    started = _completed_time(device)
    extracted = strip(t1w)
    source = nib.load(str(t1w))
    _save(np.asarray(extracted.image.data, dtype=np.float32), source, directory / "T1_brain.nii.gz")
    _save(np.asarray(extracted.mask.data > 0, dtype=np.uint8), source, directory / "T1_mask.nii.gz")
    finished = _completed_time(device)
    timing["t1_synthstrip"] = finished - started
    started = finished
    template = nib.load(str(template_path))
    if template_mask_path is None:
        extracted = strip(template_path)
        mask = np.asarray(extracted.mask.data) > 0
        brain = np.asarray(extracted.image.data, dtype=np.float32)
    else:
        supplied = nib.load(str(template_mask_path))
        if supplied.shape != template.shape or not np.allclose(supplied.affine, template.affine, atol=1e-4):
            raise ValueError("mni_brain_mask must match mni_template")
        mask = np.asarray(supplied.dataobj) > 0
        brain = np.asarray(template.dataobj, dtype=np.float32) * mask
    if not mask.any():
        raise ValueError("MNI brain mask must be nonempty")
    _save(mask.astype(np.uint8), template, directory / "MNI_mask.nii.gz")
    _save(brain, template, directory / "MNI_brain.nii.gz")
    finished = _completed_time(device)
    timing["template_preparation"] = finished - started
    started = finished
    tissues = TorchFAST(device=device)(directory / "T1_brain.nii.gz", mask=directory / "T1_mask.nii.gz")
    tissues.pve_wm.save(directory / "T1_pve_wm.nii.gz")
    tissues.pve_csf.save(directory / "T1_pve_csf.nii.gz")
    wm = np.asarray(tissues.pve_wm.data) >= .5
    if not wm.any():
        raise ValueError("FAST produced an empty WM segmentation")
    _save(wm.astype(np.uint8), source, directory / "T1_wmseg.nii.gz")
    timing["fast"] = _completed_time(device) - started
    registration = register_t1_to_mni(
        directory / "T1_brain.nii.gz", directory / "MNI_brain.nii.gz", directory,
        backend=backend, synthmorph_weights=morph_weights,
        reference_mask=directory / "MNI_mask.nii.gz", fnirt_config=fnirt_config, device=device,
        fnirt_execution=fnirt_execution,
    )
    timing.update(registration.timing_seconds)
    metadata = {"backend": backend, "moving_to_fixed_world": registration.moving_to_fixed_world.tolist(),
                "qc": registration.qc, "creation_timing_seconds": timing.copy()}
    return metadata, timing


def prepare_anatomical(t1w, template, *, template_mask, strip, backend, morph_weights,
                       fnirt_config, device, work_dir, cache_dir, reuse=True,
                       fnirt_execution="optimized"):
    """Reuse only a complete content- and implementation-matched cache."""
    started = time.perf_counter()
    if not reuse:
        directory = Path(work_dir) / "anatomical"
        metadata, timing = _produce(directory, t1w, template, template_mask, strip, backend,
                                    morph_weights, fnirt_config, device, fnirt_execution)
        metadata["fingerprint"] = None
        timing["anatomical_cache_lookup"] = 0.0
        return _result(directory, metadata, timing, False)
    fingerprint = _fingerprint(t1w, template, template_mask, strip, backend, morph_weights,
                               fnirt_config, device, fnirt_execution)
    directory = Path(cache_dir) / fingerprint
    with _lock(directory):
        metadata = _read_verified(directory, fingerprint)
        if metadata is not None:
            if torch.device(device).type == "cuda":
                # A cold run passes through FAST's CUDA policy. Preserve the
                # same downstream convolution policy when its work is reused.
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            timing = {name: 0.0 for name in _PHASES}
            timing["anatomical_cache_lookup"] = time.perf_counter() - started
            return _result(directory, metadata, timing, True)
        lookup = time.perf_counter() - started
        # The manifest is published last. Interrupted work is never a cache hit.
        (directory / "manifest.json").unlink(missing_ok=True)
        metadata, timing = _produce(directory, t1w, template, template_mask, strip, backend,
                                    morph_weights, fnirt_config, device, fnirt_execution)
        metadata["fingerprint"] = fingerprint
        metadata["sha256"] = {name: _sha256(directory / name) for name in _ARTIFACTS}
        temporary = directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(directory / "manifest.json")
        timing["anatomical_cache_lookup"] = lookup
        return _result(directory, metadata, timing, False)
