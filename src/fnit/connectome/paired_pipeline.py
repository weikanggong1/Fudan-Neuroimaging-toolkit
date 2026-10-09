"""Build rectangular structural connectomes from a shared whole-brain tractogram."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence
import hashlib
import math

import nibabel as nib
import numpy as np
import torch

from .checkpoints import CheckpointStore, fingerprint_paths, tensor_fingerprint
from .freesurfer_subject import ConnectomeNode
from .template_inputs import (
    PreparedTemplate, TemplatePair, prepare_template, template_dependency_paths,
    preflight_template_pairs, validate_readonly_subject_outputs,
)
from .paired_assignment import build_pair_connectomes

PAIR_NUMERICAL_REVISION = "paired-templates-native-tracking-20261009-v2"
MATRIX_NAMES = ("count", "sift2_fbc", "mean_length", "mean_fa")


@dataclass
class PairResult:
    """Four K_first×K_second matrices and the separate row/column node tables."""

    name: str
    matrices: dict[str, torch.Tensor]
    first: PreparedTemplate
    second: PreparedTemplate
    cache_status: str


def _json_value(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _transform_fingerprint(value):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return fingerprint_paths({"mni_to_t1_transform": value})
    if not hasattr(value, "dataobj"):
        raise TypeError("mni_to_t1_transform must be a dense SynthMorph warp or file")
    return {
        "field": tensor_fingerprint(np.asanyarray(value.dataobj)),
        "source": {"shape": list(value.source.shape),
                   "affine": np.asarray(value.source.affine).tolist()},
        "target": {"shape": list(value.target.shape),
                   "affine": np.asarray(value.target.affine).tolist()},
    }


def _reference_geometry(path):
    image = nib.load(str(path))
    if (len(image.shape) != 3 or not np.isfinite(image.affine).all()
            or abs(np.linalg.det(image.affine[:3, :3])) < 1e-10):
        raise ValueError("MNI T1 reference must be a 3D image with a valid affine")
    return {"shape": list(image.shape), "affine": image.affine.tolist()}


def _validate_mni_target(transform, reference):
    if isinstance(transform, (str, Path)):
        image = nib.load(str(transform))
        shape, affine = image.shape[:3], image.affine
    else:
        shape, affine = transform.target.shape, transform.target.affine
    if (tuple(shape) != tuple(reference["shape"])
            or not np.allclose(affine, reference["affine"], rtol=0, atol=1e-5)):
        raise ValueError("MNI-to-T1 transform target must match t1_reference_path")


def _mni_transform(store, *, t1_reference_path, mni_template, weights, device):
    """Cache the unchanged FNIT SynthMorph MNI→T1 dense RAS-mm field."""
    from .._transforms import DenseWarp
    from ..synthmorph import SynthMorph
    from ..weights import resolve_weights

    if t1_reference_path is None or mni_template is None:
        raise ValueError("MNI volume templates require mni_template and a T1 reference")
    filenames = {"affine": "synthmorph.affine.2.h5", "deform": "synthmorph.deform.3.h5"}
    paths = {kind: resolve_weights(filename, weights.get(kind) if isinstance(weights, dict)
                                  else weights) for kind, filename in filenames.items()}
    inputs = fingerprint_paths({"t1": t1_reference_path, "mni": mni_template,
                                **{f"weights_{k}": p for k, p in paths.items()}})
    def unchanged():
        if fingerprint_paths({"t1": t1_reference_path, "mni": mni_template,
                              **{f"weights_{k}": p for k, p in paths.items()}}) != inputs:
            raise RuntimeError("MNI registration inputs changed during computation/restoration")
    key = store.make_key("mni_transform", inputs=inputs, parameters={
        "model": "joint", "extent": 256, "hyper": 0.5, "steps": 7,
        "transform_only": True, "compute_inverse": False,
    }) if store else None
    loaded = store.load("mni_transform", key, device="cpu") if store else None
    mni, t1 = nib.load(str(mni_template)), nib.load(str(t1_reference_path))
    if loaded is not None:
        arrays, metadata = loaded
        if (set(arrays) == {"field"} and arrays["field"].shape == (*t1.shape[:3], 3)
                and arrays["field"].dtype == torch.float32
                and metadata.get("source_affine") == mni.affine.tolist()
                and metadata.get("target_affine") == t1.affine.tolist()):
            transform = DenseWarp(arrays["field"].numpy(), source=mni, target=t1)
            unchanged()
            return transform, key
        store.events.append({"stage": "mni_transform", "key": key,
                             "status": "miss", "reason": "invalid_warp_schema"})
    registrar = SynthMorph(weights=paths, device=str(device), model="joint")
    transform = registrar(moving=mni, fixed=t1, transform_only=True,
                          compute_inverse=False).transform
    unchanged()
    if store:
        store.publish("mni_transform", key,
                      arrays={"field": np.asanyarray(transform.dataobj)},
                      metadata={"source_affine": mni.affine.tolist(),
                                "target_affine": t1.affine.tolist()},
                      before_publish=unchanged)
    return transform, key


def build_template_pairs(
    result,
    pairs: Sequence[TemplatePair | Mapping],
    *,
    subject_dir: str | Path | None = None,
    t1_reference_path: str | Path | None = None,
    mni_to_t1_transform=None,
    mni_template: str | Path | None = None,
    synthmorph_weights=None,
    checkpoint_dir: str | Path | None = None,
    overwrite: bool = False,
    assignment_radius: float = 4.0,
    device: str | torch.device | None = None,
) -> dict[str, PairResult]:
    """Reuse tracks for arbitrary surface/volume pairs; never seed from an atlas.

    Every pair supplies independent row/column labels. Streamlines match in
    either orientation; duplicate contributions to the same cell are removed.
    Changing a template invalidates its mapping and the affected matrices.
    Changing radius invalidates assignment, while retaining both mappings.
    """
    pairs = preflight_template_pairs(pairs, subject_dir=subject_dir)
    validate_readonly_subject_outputs(subject_dir, checkpoint_dir=checkpoint_dir)
    if not pairs or not all(isinstance(pair, TemplatePair) for pair in pairs):
        raise ValueError("template_pairs must contain one or more TemplatePair objects")
    if len({pair.name for pair in pairs}) != len(pairs):
        raise ValueError("template pair names must be distinct")
    if isinstance(assignment_radius, bool) or not math.isfinite(assignment_radius) or assignment_radius <= 0:
        raise ValueError("assignment_radius must be finite and positive")
    device = torch.device(device or result.dwi_affine.device)
    from .pipeline import _checkpoint_policy
    policy = _checkpoint_policy(device)
    package = Path(__file__).parents[1]
    relative_sources = (
        "connectome/paired_pipeline.py", "connectome/template_inputs.py",
        "connectome/paired_assignment.py", "connectome/assignment.py",
        "connectome/anatomy.py", "connectome/atlas_builder.py", "connectome/atlas_surface.py",
        "synthmorph/pipeline.py", "synthmorph/spatial.py", "synthmorph/models.py",
        "_transforms.py", "_world_resampling.py", "_sampling_plan.py",
    )
    source = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
              for name in relative_sources}
    def unchanged_worker():
        if (_checkpoint_policy(device) != policy or source != {
                name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                for name in relative_sources}):
            raise RuntimeError("template source or numerical policy changed during computation")
    store = (CheckpointStore(Path(checkpoint_dir) / "pairs", numerical_revision=PAIR_NUMERICAL_REVISION,
                             device_policy=policy, source_fingerprint=source, overwrite=overwrite)
             if checkpoint_dir is not None else None)
    shared = {"affine": tensor_fingerprint(result.dwi_affine),
              "transform": tensor_fingerprint(result.dwi_to_t1_world),
              "shape": list(result.fa.shape)}
    tracks = {"endpoints": tensor_fingerprint(result.tractogram.endpoints),
              "lengths": tensor_fingerprint(result.tractogram.lengths_mm),
              "fa": tensor_fingerprint(result.tractogram.mean_fa),
              "weights": tensor_fingerprint(result.sift2_weights)}
    if any(spec.space == "mni" for pair in pairs for spec in (pair.first, pair.second)):
        if mni_to_t1_transform is None:
            mni_to_t1_transform, _ = _mni_transform(
                store, t1_reference_path=t1_reference_path, mni_template=mni_template,
                weights=synthmorph_weights, device=device)
        unchanged_worker()
    mapped = {}
    mni_guards = []

    def unchanged_pair_inputs():
        unchanged_worker()
        for guard in mni_guards:
            guard()

    def prepare(spec):
        paths = template_dependency_paths(spec, subject_dir=subject_dir)
        reference = None
        if spec.space == "mni":
            if t1_reference_path is None:
                raise ValueError("MNI template requires t1_reference_path")
            paths = (*paths, Path(t1_reference_path))
            reference = _reference_geometry(t1_reference_path)
            _validate_mni_target(mni_to_t1_transform, reference)
        manifest = fingerprint_paths({str(i): path for i, path in enumerate(paths)})
        parameters = {"spec": _json_value(asdict(spec)), **shared,
                      "mni_transform": _transform_fingerprint(mni_to_t1_transform)
                      if spec.space == "mni" else None,
                      **({"t1_reference_geometry": reference} if reference is not None else {})}
        key = (store.make_key("template", inputs=manifest, parameters=parameters) if store
               else hashlib.sha256(repr((manifest, parameters)).encode()).hexdigest())
        if key in mapped:
            return mapped[key], key
        loaded = store.load("template", key, device=device) if store else None
        if loaded is not None:
            arrays, metadata = loaded
            try:
                nodes = tuple(ConnectomeNode(**node) for node in metadata["nodes"])
                if (set(arrays) != {"labels", "affine"}
                        or arrays["labels"].shape != result.fa.shape
                        or arrays["labels"].dtype != torch.int32
                        or not torch.equal(arrays["affine"], result.dwi_affine)
                        or not nodes or tuple(node.index for node in nodes) != tuple(range(1, len(nodes) + 1))
                        or bool(((arrays["labels"] < 0) | (arrays["labels"] > len(nodes))).any())):
                    raise ValueError("invalid template schema")
                template = PreparedTemplate(spec, arrays["labels"], arrays["affine"], nodes)
            except (KeyError, TypeError, ValueError):
                loaded = None
                store.events.append({"stage": "template", "key": key, "status": "miss",
                                     "reason": "invalid_template_schema"})
        def unchanged_template():
            unchanged_worker()
            if (fingerprint_paths({str(i): path for i, path in enumerate(paths)}) != manifest
                    or (_transform_fingerprint(mni_to_t1_transform) if spec.space == "mni" else None)
                    != parameters["mni_transform"]):
                raise RuntimeError("template inputs changed during mapping")
            if reference is not None:
                if _reference_geometry(t1_reference_path) != reference:
                    raise RuntimeError("MNI T1 reference geometry changed during mapping")
                _validate_mni_target(mni_to_t1_transform, reference)
        if reference is not None:
            mni_guards.append(unchanged_template)
        if loaded is None:
            template = prepare_template(
                spec, subject_dir=subject_dir, dwi_shape=tuple(result.fa.shape),
                dwi_affine=result.dwi_affine, dwi_to_t1_world=result.dwi_to_t1_world,
                t1_reference_path=t1_reference_path,
                mni_to_t1_transform=mni_to_t1_transform, device=str(device))
            if store:
                store.publish("template", key,
                              arrays={"labels": template.labels, "affine": template.affine},
                              metadata={"nodes": [asdict(node) for node in template.nodes]},
                              before_publish=unchanged_template)
        else:
            unchanged_template()
        mapped[key] = template
        return template, key

    output = {}
    for pair in pairs:
        first, first_key = prepare(pair.first)
        second, second_key = prepare(pair.second)
        parameters = {"radius_mm": float(assignment_radius),
                      "same_template": pair.first == pair.second}
        key = store.make_key("matrix", inputs=tracks, parameters=parameters,
                             dependencies={"first": first_key, "second": second_key}) if store else None
        loaded = store.load("matrix", key, device=device) if store else None
        if loaded is not None:
            arrays, metadata = loaded
            if (set(arrays) != set(MATRIX_NAMES)
                    or metadata != {"rows": len(first.nodes), "columns": len(second.nodes)}
                    or any(a.shape != (len(first.nodes), len(second.nodes)) for a in arrays.values())
                    or arrays["count"].dtype != torch.int64
                    or any(arrays[n].dtype != torch.float32 for n in MATRIX_NAMES if n != "count")):
                loaded = None
                store.events.append({"stage": "matrix", "key": key, "status": "miss",
                                     "reason": "invalid_matrix_schema"})
        if loaded is not None:
            matrices, _ = loaded
            status = "skipped"
        else:
            matrices = build_pair_connectomes(
                result.tractogram.endpoints, first.labels, first.affine,
                second.labels, second.affine, weights=result.sift2_weights,
                lengths=result.tractogram.lengths_mm, fa=result.tractogram.mean_fa,
                first_node_count=len(first.nodes), second_node_count=len(second.nodes),
                radius=float(assignment_radius), same_template=pair.first == pair.second)
            if store:
                store.publish("matrix", key, arrays=matrices,
                              metadata={"rows": len(first.nodes), "columns": len(second.nodes)},
                              before_publish=unchanged_pair_inputs)
            status = "completed"
        output[pair.name] = PairResult(pair.name, matrices, first, second, status)
    unchanged_pair_inputs()
    if result.cache_status is not None:
        result.cache_status["template_pairs"] = {name: value.cache_status for name, value in output.items()}
        result.cache_status["pair_events"] = store.events if store else []
    return output
