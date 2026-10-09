"""CLI orchestration for user-provided template pairs."""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

from .paired_pipeline import MATRIX_NAMES
from .template_inputs import (TemplatePair, TemplateSpec, preflight_template_pairs,
                              validate_readonly_subject_outputs)
from .input_protection import (recon_resource_paths, template_readonly_paths,
                               validate_input_output_paths)


def load_template_pairs(path: str | Path, *, readonly_paths: list[Path] | None = None
                        ) -> tuple[TemplatePair, ...]:
    """Read a JSON list or {'pairs': [...]} with paths relative to that JSON."""
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        if set(raw) != {"pairs"}:
            raise ValueError("template pair JSON object must contain only 'pairs'")
        raw = raw["pairs"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("template pair JSON must contain a nonempty list")
    pairs = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"name", "first", "second"}:
            raise ValueError("each template pair must contain name, first and second")
        specs = []
        for side in ("first", "second"):
            value = dict(item[side])
            for field in ("volume_path", "left_path", "right_path", "fsaverage_dir", "nodes_tsv"):
                if value.get(field) is not None:
                    candidate = Path(value[field]).expanduser()
                    candidate = path.parent / candidate if not candidate.is_absolute() else candidate
                    if readonly_paths is not None:
                        # Keep the user-visible alias for protection, while the
                        # existing TemplateSpec/caching contract stays resolved.
                        readonly_paths.append(candidate.absolute())
                    value[field] = str(candidate.resolve())
            specs.append(TemplateSpec(**value))
        pairs.append(TemplatePair(item["name"], *specs))
    if len({pair.name for pair in pairs}) != len(pairs):
        raise ValueError("template pair names must be distinct")
    return tuple(pairs)


def load_recon_options(path: str | Path | None) -> dict | None:
    """Read inline JSON or a file, resolving resource paths from its location."""
    if path is None:
        return None
    if isinstance(path, str) and path.lstrip().startswith("{"):
        value = json.loads(path)
        base = Path.cwd()
    else:
        path = Path(path).expanduser().resolve()
        value = json.loads(path.read_text())
        base = path.parent
    if not isinstance(value, dict):
        raise ValueError("recon-options JSON must contain an object")
    for field in ("weights_dir", "assets_dir", "native_bin_dir", "executable", "freesurfer_home"):
        if value.get(field) is not None:
            if field == "executable" and str(value[field]) == "recon-all":
                continue
            candidate = Path(value[field]).expanduser()
            value[field] = str((base / candidate).resolve()
                               if not candidate.is_absolute() else candidate.resolve())
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix
    descriptor, temporary = tempfile.mkstemp(prefix=".fnit-", suffix=suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_nodes(path, nodes):
    def write(temporary):
        with temporary.open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(("index", "original_label", "hemisphere", "name"))
            writer.writerows((n.index, n.original_label, n.hemisphere, n.name) for n in nodes)
    _atomic(path, write)


def run_paired_connectome(args):
    """Prepare raw inputs or use corrected DWI, then persist selected pair results."""
    from .bids import prepare_bids_connectome
    from .pipeline import UKBConnectome_pipeline
    from .template_inputs import template_dependency_paths

    explicit_template_paths = []
    pairs = preflight_template_pairs(load_template_pairs(args.template_pairs,
                                                        readonly_paths=explicit_template_paths),
                                     subject_dir=args.freesurfer_subject_dir)
    validate_readonly_subject_outputs(
        args.freesurfer_subject_dir, output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir or Path(args.output_dir) / "checkpoints")
    if not math.isfinite(args.assignment_radius) or args.assignment_radius <= 0:
        raise ValueError("--assignment-radius must be finite and positive")
    if tuple(args.atlas) != ("fs-aparc",):
        raise ValueError("--template-pairs selects its own templates; use --atlas in a separate named-atlas invocation")
    for pair in pairs:
        for spec in (pair.first, pair.second):
            for field in ("volume_path", "left_path", "right_path", "nodes_tsv"):
                path = getattr(spec, field)
                if path is not None and not path.is_file():
                    raise FileNotFoundError(path)
    if any(spec.space == "mni" for pair in pairs for spec in (pair.first, pair.second)) and (
        args.mni_to_t1_transform is None and args.mni_template is None
    ):
        raise ValueError("MNI template pairs require --mni-template or --mni-to-t1-transform")
    if args.download_atlases:
        raise ValueError("--download-atlases applies to named atlases; pair templates are user-provided")
    if args.atlas_dwi or args.t1_segmentation:
        raise ValueError("template pairs use a recon-all subject; do not also supply --atlas-dwi/--t1-segmentation")

    # Protect inputs and validate managed final outputs before any BIDS/recon
    # preparation can write. The BIDS adapter additionally checks its own stage
    # namespaces against these inputs and its selected raw/T1 sources.
    output = Path(args.output_dir).expanduser().resolve()
    source_paths = set(template_readonly_paths(pairs, args.freesurfer_subject_dir))
    source_paths.update(explicit_template_paths)
    source_paths.add(Path(args.template_pairs))
    for name in ("dwi", "bvals", "bvecs", "t1", "corrected_dwi", "rotated_bvecs",
                 "brain_mask", "response_mask", "fod_mask", "normalise_mask", "fa_map",
                 "dwi_to_t1_world", "mni_template", "mni_to_t1_transform", "recon_options",
                 "synthmorph_weights"):
        path = getattr(args, name, None)
        if path is not None:
            if name == "recon_options" and isinstance(path, str) and path.lstrip().startswith("{"):
                continue
            source_paths.add(Path(path))
    recon_options = load_recon_options(args.recon_options)
    if recon_options:
        source_paths.update(recon_resource_paths(recon_options))
    output_paths = []
    for pair in pairs:
        folder = output / "pairs" / pair.name
        output_paths.extend(folder / f"connectome_{name}.csv" for name in MATRIX_NAMES)
        output_paths.extend(folder / name for name in (
            "rows.tsv", "columns.tsv", "first_atlas_dwi.nii.gz", "second_atlas_dwi.nii.gz", "pair.json"))
    validate_readonly_subject_outputs(args.freesurfer_subject_dir,
                                     output_paths=tuple(output_paths))
    manifest = output / "pairs_run_state.json"
    checkpoint = Path(args.checkpoint_dir or output / "checkpoints")
    validate_input_output_paths(
        source_paths, output_paths=(*output_paths, manifest),
        reserved_directories=(checkpoint / "shared", checkpoint / "pairs"))
    managed = {}
    if manifest.is_file():
        try:
            state = json.loads(manifest.read_text())
            managed = state.get("owned_outputs", state.get("outputs", {}))
        except (ValueError, OSError):
            pass
    for path in (*output_paths, manifest):
        if path.exists() and not args.overwrite:
            if path == manifest:
                if not managed:
                    raise FileExistsError(f"invalid managed output manifest: {manifest}")
            elif managed.get(str(path.relative_to(output))) != _sha256(path):
                raise FileExistsError(f"output is not an unchanged FNIT-managed file: {path}; use --overwrite")
    preparation = None
    if args.bids_root:
        if not args.subject or any(v is not None for v in (args.dwi, args.bvals, args.bvecs)):
            raise ValueError("provide --bids-root/--subject without explicit --dwi/--bvals/--bvecs")
        preparation = prepare_bids_connectome(
            args.bids_root, args.output_dir, subject=args.subject, session=args.session,
            run=args.run, acquisition=args.acquisition, direction=args.direction,
            t1=args.t1, freesurfer_subject_dir=args.freesurfer_subject_dir,
            corrected_dwi=args.corrected_dwi, rotated_bvecs=args.rotated_bvecs,
            device=args.device, overwrite=args.overwrite, eddy_gp_seed=args.eddy_gp_seed,
            recon_backend=args.recon_backend, recon_options=recon_options,
            readonly_inputs=tuple(source_paths),
            planned_output_paths=(*output_paths, manifest))
        dwi, bvals, bvecs = preparation.dwi, preparation.bvals, preparation.bvecs
        subject = preparation.freesurfer_subject_dir
    else:
        if any(v is None for v in (args.dwi, args.bvals, args.bvecs, args.freesurfer_subject_dir)):
            raise ValueError("explicit template pair mode requires --dwi/--bvals/--bvecs/--freesurfer-subject-dir")
        if args.corrected_dwi or args.rotated_bvecs or args.t1 or args.subject:
            raise ValueError("raw selection options require --bids-root")
        if args.recon_backend not in ("auto", "provided") or args.recon_options:
            raise ValueError("executing recon-all requires raw BIDS mode; explicit mode consumes a supplied subject")
        dwi, bvals, bvecs, subject = args.dwi, args.bvals, args.bvecs, args.freesurfer_subject_dir

    # Subject-dependent surface checks are repeated once generated anatomy exists.
    pairs = preflight_template_pairs(pairs, subject_dir=subject)
    source_paths.update(Path(p) for p in (dwi, bvals, bvecs))
    source_paths.update(Path(p) for pair in pairs for spec in (pair.first, pair.second)
                        for p in template_dependency_paths(spec, subject_dir=subject))
    validate_readonly_subject_outputs(subject,
                                     output_paths=tuple(output_paths))
    validate_input_output_paths(source_paths, output_paths=(*output_paths, manifest))

    transform = None
    if args.dwi_to_t1_world:
        transform = np.loadtxt(args.dwi_to_t1_world, delimiter="," if
                               Path(args.dwi_to_t1_world).suffix == ".csv" else None)
        if transform.shape != (4, 4):
            raise ValueError("--dwi-to-t1-world must contain a 4x4 matrix")
    result = UKBConnectome_pipeline(device=args.device)(
        dwi, bvals, bvecs, freesurfer_subject_dir=subject,
        atlas=tuple(args.atlas), atlas_templates_dir=args.atlas_templates_dir,
        fsaverage_dir=args.fsaverage_dir, mni_template=args.mni_template,
        synthmorph_weights=args.synthmorph_weights, tian_fnirt_coeff=args.tian_fnirt_coeff,
        brain_mask=args.brain_mask, shell_bvals=args.shell_bvals,
        response_mask=args.response_mask, fod_mask=args.fod_mask,
        normalise_mask=args.normalise_mask, fa_map=args.fa_map,
        dwi_to_t1_world=transform, n_seeds=args.n_seeds, seed=args.seed,
        tracking_threads=args.tracking_threads, checkpoint_dir=checkpoint, overwrite=args.overwrite,
        template_pairs=pairs, assignment_radius=args.assignment_radius,
        mni_to_t1_transform=args.mni_to_t1_transform)
    if preparation is not None:
        result.preparation_stages = preparation.stages
    for name, pair_result in result.pair_results.items():
        folder = output / "pairs" / name
        for kind in MATRIX_NAMES:
            array = pair_result.matrices[kind].detach().cpu().numpy()
            _atomic(folder / f"connectome_{kind}.csv", lambda p, a=array, k=kind:
                    np.savetxt(p, a, delimiter=",", fmt="%d" if k == "count" else "%.9g"))
        _write_nodes(folder / "rows.tsv", pair_result.first.nodes)
        _write_nodes(folder / "columns.tsv", pair_result.second.nodes)
        for side, prepared in (("first", pair_result.first), ("second", pair_result.second)):
            image = nib.Nifti1Image(prepared.labels.detach().cpu().numpy().astype(np.int32),
                                   prepared.affine.detach().cpu().numpy())
            _atomic(folder / f"{side}_atlas_dwi.nii.gz", lambda p, image=image: nib.save(image, p))
        pair_metadata = {"name": name, "first": asdict(pair_result.first.spec),
                         "second": asdict(pair_result.second.spec), "radius_mm": args.assignment_radius,
                         "shape": [len(pair_result.first.nodes), len(pair_result.second.nodes)],
                         "tracking": "whole-brain ACT; bidirectional endpoint assignment",
                         "cache_status": pair_result.cache_status}
        _atomic(folder / "pair.json", lambda p: p.write_text(
            json.dumps(pair_metadata, ensure_ascii=False, indent=2, default=str) + "\n"))
        print(folder)
    current_outputs = {str(p.relative_to(output)): _sha256(p) for p in output_paths}
    state = {"cache_status": result.cache_status, "preparation_stages": result.preparation_stages,
             "seed_attempts": result.tractogram.seeds_attempted,
             "accepted_streamlines": len(result.tractogram.paths),
             "outputs": current_outputs, "owned_outputs": {**managed, **current_outputs}}
    _atomic(manifest, lambda p: p.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n"))
    print("cache_status=" + json.dumps(result.cache_status, ensure_ascii=False))
    print(manifest)
    return result
