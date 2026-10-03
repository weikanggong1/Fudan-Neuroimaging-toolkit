"""Check read-only inputs before connectome preparation creates any outputs."""
from __future__ import annotations

from pathlib import Path


def template_readonly_paths(pairs, subject_dir=None) -> tuple[Path, ...]:
    """Collect explicit pair inputs, including deferred native-surface sources."""
    from .template_inputs import template_dependency_paths
    paths = []
    for pair in pairs or ():
        for spec in (pair.first, pair.second):
            if spec.kind == "volume" or subject_dir is not None:
                paths.extend(template_dependency_paths(spec, subject_dir))
            else:
                paths.extend(value for value in (spec.left_path, spec.right_path, spec.nodes_tsv)
                             if value is not None)
                if spec.space == "fsaverage":
                    paths.extend(spec.fsaverage_dir / f"surf/{hemi}.sphere.reg"
                                 for hemi, annotation in (("lh", spec.left_path), ("rh", spec.right_path))
                                 if annotation is not None)
    return tuple(dict.fromkeys(Path(path) for path in paths))


def recon_resource_paths(options) -> tuple[Path, ...]:
    """Protect immutable execution resources, allowing ordinary FS subjects.

    FreeSurfer installations also contain writable subjects directories; the
    entire home must not be treated as a read-only weight/resource directory.
    Protect the startup files, installation resources and fsaverage inputs.
    """
    paths = [Path(options[name]) for name in
             ("weights_dir", "assets_dir", "native_bin_dir", "executable")
             if options.get(name) is not None]
    if options.get("freesurfer_home") is not None:
        home = Path(options["freesurfer_home"])
        paths.extend(home / name for name in (
            "SetUpFreeSurfer.sh", "FreeSurferEnv.sh", "build-stamp.txt", "bin",
            "average", "lib", "etc", "python", "models", "mni", "trctrain")
            if (home / name).exists())
        paths.extend((home / "subjects").glob("fsaverage*"))
    return tuple(dict.fromkeys(paths))


def validate_input_output_paths(inputs, *, output_paths=(), reserved_directories=(),
                                inplace_output_paths=()) -> None:
    """Reject file aliases and inputs inside active write namespaces.

    Inputs may be files or resource directories. Both lexical paths and their
    resolved paths are checked: an existing output symlink must not escape into
    an input, and unlinking an input symlink is also forbidden. Callers reserve
    only stages that can write in this invocation. A supplied corrected DWI
    therefore remains usable inside an existing preprocessing output directory.
    In-place writers also check inode identity to catch external hardlink aliases.
    Do not include unlink-then-create staging or atomic-replacement targets in
    inplace_output_paths: replacing those aliases leaves the source inode intact.
    """
    def forms(value):
        path = Path(value).expanduser().absolute()
        return (path, path.resolve())
    protected = tuple(forms(path) for path in inputs if path is not None)
    for target in (*output_paths, *inplace_output_paths):
        targets = forms(target)
        for original in protected:
            if any(output == source or (source.is_dir() and output.is_relative_to(source))
                   for output in targets for source in original):
                raise ValueError(f"output would overwrite a read-only input: {target}")
    for target in inplace_output_paths:
        path = forms(target)[0]
        if path.exists():
            for original in protected:
                source = original[0]
                if source.exists() and path.samefile(source):
                    raise ValueError(f"output would overwrite a read-only input via a hardlink: {target}")
    for directory in reserved_directories:
        directories = forms(directory)
        for original in protected:
            if any(source == root or source.is_relative_to(root)
                   or (source.is_dir() and root.is_relative_to(source))
                   for source in original for root in directories):
                raise ValueError(f"preparation output namespace overlaps a read-only input: {directory}")


def validate_bids_preparation_inputs(selected, output_dir, *, corrected_dwi=None,
                                     rotated_bvecs=None, recon_backend="provided",
                                     readonly_inputs=(), planned_output_paths=()) -> None:
    """Protect selected BIDS/T1 sources before reconstruction or DWI staging."""
    root = Path(output_dir).expanduser().resolve()
    sources = [selected.image, selected.bval, selected.bvec, selected.reverse,
               selected.reverse_bval, selected.t1w, corrected_dwi, rotated_bvecs,
               *readonly_inputs]
    namespaces = []
    inplace_outputs = []
    if corrected_dwi is None:
        namespaces.extend((root / "preproc/raw", root / "preproc/eddy"))
        # stage_bids_dwi unlinks AP/PA aliases before recreating them. Its
        # manifest, preparation sidecars and numerical result writers instead
        # truncate existing files, so their hardlink aliases need protection.
        inplace_outputs.extend(root / "preproc/raw" / name
                               for name in ("bids_selection.json",))
        eddy_names = ["eddy_index.txt", "nodif_brain_mask.nii.gz",
                      "nodif_brain_mask_report.json", "data.nii.gz"]
        eddy_names.extend("data." + suffix for suffix in (
            "eddy_rotated_bvecs", "eddy_parameters", "eddy_movement_rms",
            "eddy_restricted_movement_rms", "eddy_outlier_map",
            "eddy_outlier_n_stdev_map", "eddy_outlier_n_sqr_stdev_map",
            "eddy_outlier_report", "eddy_qc.json"))
        if selected.reverse is None:
            eddy_names.append("acqparams.txt")
        inplace_outputs.extend(root / "preproc/eddy" / name for name in eddy_names)
        if selected.reverse is not None:
            namespaces.append(root / "preproc/topup")
            inplace_outputs.extend(root / "preproc/topup" / name for name in (
                "B0_AP_PA.nii.gz", "acqparams.txt", "pair_geometry.json",
                "fieldmap_out_fieldcoef.nii.gz", "fieldmap_out_movpar.txt",
                "fieldmap_fout.nii.gz", "fieldmap_iout.nii.gz",
                "fieldmap_jacout_01.nii.gz", "fieldmap_jacout_02.nii.gz"))
    if recon_backend != "provided":
        namespaces.extend((root / "anatomy/state", root / "anatomy" / recon_backend))
    validate_input_output_paths(sources, output_paths=planned_output_paths,
                                reserved_directories=namespaces,
                                inplace_output_paths=inplace_outputs)
