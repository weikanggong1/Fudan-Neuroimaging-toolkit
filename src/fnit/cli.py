"""Command-line entry points for single-image inference."""
import argparse
import csv
import os
import sys
from pathlib import Path
import uuid

from . import __version__


def _atomic_save(volume, path):
    path = Path(path)
    suffix = next((value for value in ('.nii.gz', '.nii', '.mgz', '.npz')
                   if path.name.endswith(value)), path.suffix)
    temporary = path.with_name(
        f'.{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}{suffix}')
    try:
        volume.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _wmh_suffix(path):
    name = Path(path).name
    for suffix in ('.nii.gz', '.nii', '.mgz'):
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix
    raise ValueError('WMH-SynthSeg supports .nii, .nii.gz and .mgz images')


def _run_wmh(args):
    from .wmh_synthseg import LABEL_IDS, LABEL_NAMES, WMHSynthSeg

    source, target = Path(args.i), Path(args.o)
    if not source.is_file():
        raise FileNotFoundError(source)
    _wmh_suffix(source)
    _wmh_suffix(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.csv_vols:
        Path(args.csv_vols).parent.mkdir(parents=True, exist_ok=True)
    model = WMHSynthSeg(weights=args.weights, device=args.device, threads=args.threads)
    rows = []
    result = model(source, crop=args.crop,
                   save_lesion_probabilities=args.save_lesion_probabilities)
    result.segmentation.save(target)
    print(target)
    if args.save_lesion_probabilities:
        stem, suffix = _wmh_suffix(target)
        probability = target.with_name(f'{stem}.lesion_probs{suffix}')
        result.lesion_probability.save(probability)
        print(probability)
    if args.csv_vols:
        import numpy as np
        volumes = result.volumes_mm3
        ordered = np.asarray([volumes[label] for label in LABEL_IDS], dtype=np.float32)
        rows.append([str(target), str(np.sum(ordered[1:])),
                     *(str(value) for value in ordered[1:])])
    if args.csv_vols:
        with Path(args.csv_vols).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Input-file', 'Intracranial-volume',
                             *(f'{name}({label})' for label, name in zip(LABEL_IDS, LABEL_NAMES)
                               if label != 0)])
            writer.writerows(rows)
        print(args.csv_vols)


def _run_synthseg(args):
    source, target = Path(args.i), Path(args.o)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.parc:
        if args.color_lut:
            raise ValueError("--color-lut with --parc is not implemented")
        import torch
        torch.set_num_threads(args.threads)
        from .synthseg_parc import SynthSegPlus
        result = SynthSegPlus(weights=args.weights, parc_weights=args.parc_weights,
                              device=args.device)(source, keep_geometry=args.keep_geometry,
                                                  volumes=bool(args.csv_vols))
        result.combined.save(target)
        if args.csv_vols:
            result.write_volumes_csv(source, args.csv_vols)
            print(args.csv_vols)
        if args.parc_out:
            Path(args.parc_out).parent.mkdir(parents=True, exist_ok=True)
            result.cortical_parcellation.save(args.parc_out)
    else:
        if args.parc_out or args.parc_weights:
            raise ValueError("--parc-out and --parc-weights require --parc")
        from .synthseg_parc import SynthSeg
        result = SynthSeg(weights=args.weights, device=args.device, threads=args.threads)(
            source, keep_geometry=args.keep_geometry, color_lut=args.color_lut)
        result.segmentation.save(target)
        if args.csv_vols:
            result.write_volumes_csv(source, args.csv_vols)
            print(args.csv_vols)
    print(target)


def _run_subregions(args):
    import json
    from dataclasses import asdict
    import nibabel as nib
    import numpy as np
    from .gems import segment_subregions
    selected = "all" if not args.structure or args.structure == ["all"] else args.structure
    result = segment_subregions(
        args.i, args.atlas_root, structures=selected,
        coarse_segmentation=args.coarse_segmentation, synthseg_weights=args.synthseg_weights,
        cortical_parcellation=args.cortical_parcellation, wmparc=args.wmparc,
        synthseg_parc_weights=args.synthseg_parc_weights,
        auto_initialize=not args.no_auto_initialize, device=args.device,
        em_iterations=args.em_iterations, deform_iterations=args.deform_iterations)
    Path(args.o).parent.mkdir(parents=True, exist_ok=True)
    result.labels.save(args.o)
    print(args.o)
    output = Path(args.output_dir) if args.output_dir else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
        result.labels.save(output / "subregions_native.nii.gz")
        metadata = result.label_metadata or {}
        with (output / "labels.tsv").open("w") as stream:
            stream.write("label_id\tname\tparent\tfamily\themisphere\tsource\n")
            for identifier, name in result.label_table.items():
                if identifier in metadata:
                    label = metadata[identifier]
                    stream.write(f"{label.id}\t{label.name}\t{label.parent}\t{label.family}\t{label.hemisphere or ''}\t{label.source}\n")
                elif identifier:
                    stream.write(f"{identifier}\t{name}\t\t\t\t\n")
        with (output / "volumes.tsv").open("w") as stream:
            stream.write("label_id\tname\tparent\themisphere\thard_volume_mm3\tsoft_volume_mm3\n")
            for identifier, value in (result.volumes or {}).items():
                label = metadata[identifier]
                stream.write(f"{identifier}\t{label.name}\t{label.parent}\t{label.hemisphere or ''}\t"
                             f"{value['hard_volume_mm3']:.6f}\t{value['soft_volume_mm3']:.6f}\n")
    if args.save_highres:
        highres = (output or Path(args.o).parent) / "highres"
        highres.mkdir(parents=True, exist_ok=True)
        for name, fit in result.structure_results.items():
            stem = name.replace('-', '_')
            labels = fit.highres_labels or nib.Nifti1Image(
                fit.labels.detach().cpu().numpy().astype(np.int32), fit.affine)
            nib.save(labels, highres / f"{stem}.nii.gz")
            posterior = fit.posterior.detach().cpu().numpy().astype(np.float32)
            nib.save(nib.Nifti1Image(np.moveaxis(posterior, 0, -1), fit.affine),
                     highres / f"{stem}_posterior.nii.gz")
    if args.report_json or output:
        report = Path(args.report_json) if args.report_json else output / "report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"input": str(args.i), "output": str(args.o),
                                      "structures": list(result.structure_results),
                                      "labels": {str(k): asdict(v) for k, v in
                                                 (result.label_metadata or {}).items()},
                                      "initialization": result.initialization}, indent=2) + "\n")


def _synthsr_suffix(path):
    name = Path(path).name
    for suffix in ('.nii.gz', '.nii', '.mgz', '.npz'):
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix
    raise ValueError('SynthSR supports .nii, .nii.gz, .mgz and .npz images')


def _run_synthsr(args):
    from .synthsr import SynthSR

    source, target = Path(args.i), Path(args.o)
    if not source.is_file():
        raise FileNotFoundError(source)
    _synthsr_suffix(source)
    if target.name.endswith(('.nii', '.nii.gz', '.mgz', '.npz')):
        output = target
    else:
        if target.suffix == '.txt':
            raise ValueError('A .txt output list is not supported by the single-image CLI')
        stem, suffix = _synthsr_suffix(source)
        output = target / f'{stem}_synthsr{suffix}'
    model = SynthSR(weights=args.weights, device='cpu' if args.cpu else args.device,
                    lowfield=args.lowfield, v1=args.v1, threads=args.threads)
    model(source, ct=args.ct, disable_flipping=args.disable_flipping,
          disable_sharpening=args.disable_sharpening).image.save(output)
    print(output)


def _run_fast(args):
    from .fast import TorchFAST

    model = TorchFAST(
        device=args.device,
        threads=args.threads,
        init_iterations=args.init_iterations,
        bias_iterations=args.bias_iterations,
        fixed_iterations=args.fixed_iterations,
        bias_fwhm_mm=0.0 if args.no_bias else args.bias_fwhm_mm,
        init_mrf=args.init_mrf,
        mrf=args.mrf,
        mixel_mrf=args.mixel_mrf,
        pve_steps=args.pve_steps,
    )
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "_pve_0.nii.gz": "pve_csf",
        "_pve_1.nii.gz": "pve_gm",
        "_pve_2.nii.gz": "pve_wm",
        "_seg.nii.gz": "hard_segmentation",
        "_pveseg.nii.gz": "pve_segmentation",
        "_mixeltype.nii.gz": "mixel_type",
    }
    if args.save_bias:
        fields["_bias.nii.gz"] = "bias_field"
    if args.save_restored:
        fields["_restore.nii.gz"] = "restored"
    outputs = {suffix: Path(f"{prefix}{suffix}") for suffix in fields}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"output exists: {existing[0]}; use --overwrite")
    result = model(args.image, mask=args.mask)
    for suffix, path in outputs.items():
        _atomic_save(getattr(result, fields[suffix]), path)
        print(path)


def _run_flirt(args):
    import torch

    from .flirt import run_flirt

    torch.set_num_threads(args.threads)
    return run_flirt(
        args.input,
        args.reference,
        output=args.output,
        omat=args.omat,
        init=args.init,
        inweight=args.inweight,
        refweight=args.refweight,
        dof=args.dof,
        cost=args.cost,
        device=args.device,
        overwrite=args.overwrite,
    )


def _run_fnirt(args):
    from .fnirt.cli import _effective_config
    from .fnirt.standalone import run_fnirt

    return run_fnirt(
        args.input,
        args.ref,
        args.aff,
        cout=args.cout,
        iout=args.iout,
        jout=args.jout,
        refmask=args.refmask,
        config=_effective_config(args),
        device=args.device,
        overwrite=args.overwrite,
    )


def _run_applywarp(args):
    from .applywarp import TorchApplyWarp

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite")
    convention = "absolute" if args.absolute else (
        "relative" if args.relative else "auto"
    )
    model = TorchApplyWarp(device=args.device)
    model.run(
        args.input,
        args.reference,
        output,
        warp=args.warp,
        premat=args.premat,
        postmat=args.postmat,
        interpolation=args.interpolation,
        warp_convention=convention,
        output_dtype=args.datatype,
    )
    print(output)


def _run_fast_vbm(args):
    from .fast_vbm import FastVBM, OUTPUT_FILENAMES

    output_dir = Path(args.output_dir)
    report_path = output_dir / "fast_vbm_report.json"
    outputs = [output_dir / filename for filename in OUTPUT_FILENAMES.values()]
    existing = [path for path in (*outputs, report_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"output exists: {existing[0]}; use --overwrite")

    model = FastVBM(
        device=args.device,
        threads=args.threads,
        synthstrip_weights=args.synthstrip_weights,
        synthmorph_weights=args.synthmorph_weights,
        bias_correction=not args.no_bias,
        synthmorph_extent=args.synthmorph_extent,
        synthmorph_hyper=args.synthmorph_hyper,
        synthmorph_steps=args.synthmorph_steps,
        registration_backend=args.registration_backend,
        fnirt_strides=tuple(args.fnirt_strides),
        fnirt_steps=tuple(args.fnirt_steps),
        fnirt_input_fwhm_mm=tuple(args.fnirt_input_fwhm_mm),
        fnirt_reference_fwhm_mm=tuple(args.fnirt_reference_fwhm_mm),
        fnirt_warp_resolution_mm=args.fnirt_warp_resolution_mm,
        fnirt_regularization=tuple(args.fnirt_regularization),
    )
    result = model(
        args.image,
        args.template,
        brain_mask=args.brain_mask,
        reference_mask=args.reference_mask,
    )
    paths = result.save(output_dir, overwrite=args.overwrite)
    for name in OUTPUT_FILENAMES:
        print(paths[name])
    print(report_path)


def _run_connectome(args):
    import nibabel as nib
    import numpy as np

    from .connectome import UKBConnectome_pipeline

    if args.n_seeds < 1:
        raise ValueError("--n-seeds must be positive")
    atlas_names = tuple(args.atlas)
    if len(set(atlas_names)) != len(atlas_names):
        raise ValueError("--atlas names must be distinct")
    if args.download_atlases:
        from .connectome.assets import install_connectome_atlases
        directory = (args.atlas_templates_dir or
                     Path(args.output_dir) / "atlas_templates")
        args.atlas_templates_dir = install_connectome_atlases(
            atlas_names, directory)
    if args.bids_root:
        if not args.subject:
            raise ValueError("--subject is required with --bids-root")
        if any(value is not None for value in
               (args.dwi, args.bvals, args.bvecs, args.t1_segmentation, args.atlas_dwi)):
            raise ValueError("--bids-root cannot be mixed with explicit DWI/T1 segmentation inputs")
        from .connectome.bids import prepare_bids_connectome
        selected = prepare_bids_connectome(
            args.bids_root, args.output_dir, subject=args.subject,
            session=args.session, run=args.run, acquisition=args.acquisition,
            direction=args.direction, t1=args.t1,
            freesurfer_subject_dir=args.freesurfer_subject_dir,
            corrected_dwi=args.corrected_dwi,
            rotated_bvecs=args.rotated_bvecs,
            device=args.device, overwrite=args.overwrite,
        )
        args.dwi, args.bvals, args.bvecs = selected.dwi, selected.bvals, selected.bvecs
        args.t1 = None
        args.freesurfer_subject_dir = selected.freesurfer_subject_dir
        print("preprocessing=" + ",".join(f"{name}:{status}" for name, status in selected.stages.items()))
    else:
        if args.corrected_dwi or args.rotated_bvecs or args.subject:
            raise ValueError("BIDS selection and external correction options require --bids-root")
        if any(value is None for value in (args.dwi, args.bvals, args.bvecs)):
            raise ValueError("provide --bids-root/--subject or corrected --dwi/--bvals/--bvecs")
    if args.freesurfer_subject_dir is not None:
        if any(value is not None for value in (args.t1, args.t1_segmentation, args.atlas_dwi)):
            raise ValueError("--freesurfer-subject-dir cannot be combined with --t1/--t1-segmentation/--atlas-dwi")
        from .connectome import FreeSurferSubject
        subject = FreeSurferSubject(Path(args.freesurfer_subject_dir))
        anatomy_inputs = [subject.brain, subject.aparc_aseg]
    else:
        if any(value is None for value in (args.t1, args.t1_segmentation, args.atlas_dwi)):
            raise ValueError("provide --freesurfer-subject-dir or all of --t1, --t1-segmentation, --atlas-dwi")
        anatomy_inputs = [args.t1, args.t1_segmentation, args.atlas_dwi]
    atlas_inputs = []
    schaefer_tian = {
        "schaefer200+tian-s1": (200, 1),
        "schaefer500+tian-s4": (500, 4),
        "schaefer1000+tian-s4": (1000, 4),
    }
    native_tian = {
        "aparc+tian-s1": "aparc",
        "aparc.a2009s+tian-s1": "aparc.a2009s",
    }
    if "fs-aparc-a2009s" in atlas_names:
        if args.freesurfer_subject_dir is None:
            raise ValueError("fs-aparc-a2009s requires --freesurfer-subject-dir")
        atlas_inputs.extend([
            subject.aparc_a2009s_aseg,
            *(subject.subject_dir / "label" / f"{hemi}.aparc.a2009s.annot"
              for hemi in ("lh", "rh")),
        ])
    glasser_tian = {"glasser+tian-s1": 1, "glasser+tian-s4": 4}
    if not any(name in (*schaefer_tian, *native_tian, *glasser_tian)
               for name in atlas_names) and args.tian_fnirt_coeff:
        raise ValueError("--tian-fnirt-coeff requires a cortical+Tian atlas")
    for atlas_name in atlas_names:
        if atlas_name in (*schaefer_tian, *native_tian, *glasser_tian):
            tian_scale = (schaefer_tian[atlas_name][1] if atlas_name in schaefer_tian
                          else glasser_tian.get(atlas_name, 1))
            if (args.freesurfer_subject_dir is None or args.atlas_templates_dir is None or
                    (atlas_name in (*schaefer_tian, *glasser_tian) and args.fsaverage_dir is None) or
                    (args.mni_template is None) == (args.tian_fnirt_coeff is None)):
                raise ValueError("cortical+Tian needs --freesurfer-subject-dir, --atlas-templates-dir, surface-atlas --fsaverage-dir and exactly one of --mni-template or --tian-fnirt-coeff")
            if args.tian_fnirt_coeff and args.synthmorph_weights:
                raise ValueError("--synthmorph-weights cannot be used with --tian-fnirt-coeff")
            templates = Path(args.atlas_templates_dir)
            atlas_inputs.extend([
                Path(args.mni_template or args.tian_fnirt_coeff),
                templates / f"Tian_Subcortex_S{tian_scale}_3T.nii.gz",
                templates / f"Tian_Subcortex_S{tian_scale}_3T_label.txt",
                *(subject.subject_dir / "surf" / f"{hemi}.{kind}"
                  for hemi in ("lh", "rh") for kind in ("pial", "white")),
                subject.subject_dir / "mri/ribbon.mgz",
            ])
            if atlas_name in schaefer_tian:
                parcels, _ = schaefer_tian[atlas_name]
                atlas_inputs.extend(templates / f"{hemi}.Schaefer2018_{parcels}Parcels_7Networks_order.annot"
                                    for hemi in ("lh", "rh"))
                atlas_inputs.extend(Path(args.fsaverage_dir) / "surf" / f"{hemi}.sphere.reg"
                                    for hemi in ("lh", "rh"))
                atlas_inputs.extend(subject.subject_dir / "surf" / f"{hemi}.sphere.reg"
                                    for hemi in ("lh", "rh"))
            elif atlas_name in glasser_tian:
                atlas_inputs.append(templates / "Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii")
                surfaces = templates.parent / "surfaces"
                atlas_inputs.extend(surfaces / f"{side}.sphere.32k_fs_LR.surf.gii"
                                    for side in ("L", "R"))
                atlas_inputs.extend(surfaces / f"fs_{side}-to-fs_LR_fsaverage.{side}_LR.spherical_std.164k_fs_{side}.surf.gii"
                                    for side in ("L", "R"))
                atlas_inputs.extend(Path(args.fsaverage_dir) / "surf" / f"{hemi}.sphere.reg"
                                    for hemi in ("lh", "rh"))
                atlas_inputs.extend(subject.subject_dir / "surf" / f"{hemi}.sphere.reg"
                                    for hemi in ("lh", "rh"))
            else:
                atlas_inputs.extend(subject.subject_dir / "label" / f"{hemi}.{native_tian[atlas_name]}.annot"
                                    for hemi in ("lh", "rh"))
            if args.synthmorph_weights:
                atlas_inputs.extend(Path(args.synthmorph_weights) / name for name in (
                    "synthmorph.affine.2.h5", "synthmorph.deform.3.h5"))
    inputs = [args.dwi, args.bvals, args.bvecs, *anatomy_inputs,
              args.atlas_dwi, args.t1_segmentation, args.brain_mask,
              args.response_mask, args.fod_mask, args.normalise_mask,
              args.fa_map, args.dwi_to_t1_world, *atlas_inputs]
    inputs = [Path(value) for value in inputs if value is not None]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = Path(args.output_dir)
    multi_output = bool(args.bids_root) or len(atlas_names) > 1

    def atlas_files(name):
        directory = output_dir / "atlases" / name if multi_output else output_dir
        return {
            **{key: directory / f"connectome_{key}.csv" for key in
               ("count", "sift2_fbc", "mean_length", "mean_fa")},
            "atlas": directory / "atlas_dwi.nii.gz",
            "region_labels": directory / "region_labels.csv",
            **({"nodes": directory / "nodes.tsv"} if args.freesurfer_subject_dir else {}),
        }

    files = {
        "five_tissue": output_dir / "five_tissue_dwi_world.nii.gz",
        "gmwmi": output_dir / "gmwmi_dwi_world.nii.gz",
        "fa": output_dir / "fa_dwi.nii.gz",
        "brain_mask": output_dir / "brain_mask_dwi.nii.gz",
        "transform": output_dir / "dwi_to_t1_world.csv",
    }
    output_paths = (*files.values(), *(path for name in atlas_names
                                        for path in atlas_files(name).values()))
    run_state = output_dir / "run_state.json"
    if args.bids_root:
        from .connectome.bids import _fingerprint, _record, _reusable
        run_key = _fingerprint(tuple(inputs), {
            "fnit_version": __version__, "atlas": list(atlas_names),
            "n_seeds": args.n_seeds, "seed": args.seed, "device": args.device,
            "shell_bvals": args.shell_bvals, "compile_arc": args.compile_arc,
        })
        if not args.overwrite and _reusable(run_state, run_key, output_paths):
            print("connectome=skipped (matching inputs and complete outputs)")
            return
    source_paths = {path.resolve() for path in inputs}
    for path in output_paths:
        if path.resolve() in source_paths:
            raise ValueError(f"output would overwrite an input: {path}")
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {path}; use --overwrite")

    transform = None
    if args.dwi_to_t1_world:
        transform = np.loadtxt(args.dwi_to_t1_world, delimiter="," if
                               Path(args.dwi_to_t1_world).suffix == ".csv" else None)
        if transform.shape != (4, 4):
            raise ValueError("--dwi-to-t1-world must contain a 4x4 matrix")
    result = UKBConnectome_pipeline(device=args.device)(
        args.dwi, args.bvals, args.bvecs, args.t1,
        atlas_dwi=args.atlas_dwi,
        t1_segmentation=args.t1_segmentation,
        freesurfer_subject_dir=args.freesurfer_subject_dir,
        atlas=atlas_names[0] if len(atlas_names) == 1 else atlas_names,
        atlas_templates_dir=args.atlas_templates_dir,
        fsaverage_dir=args.fsaverage_dir,
        mni_template=args.mni_template,
        synthmorph_weights=args.synthmorph_weights,
        tian_fnirt_coeff=args.tian_fnirt_coeff,
        brain_mask=Path(args.brain_mask) if args.brain_mask else None,
        shell_bvals=args.shell_bvals,
        response_mask=args.response_mask,
        fod_mask=args.fod_mask,
        normalise_mask=args.normalise_mask,
        fa_map=args.fa_map,
        dwi_to_t1_world=transform,
        n_seeds=args.n_seeds,
        seed=args.seed,
        compile_arc=args.compile_arc,
    )
    print(f"seed_attempts={result.tractogram.seeds_attempted} "
          f"accepted_streamlines={len(result.tractogram.paths)}")
    matrix_names = ("count", "sift2_fbc", "mean_length", "mean_fa")
    if set(result.matrices) != set(matrix_names):
        raise ValueError("connectome result must contain four named matrices")
    output_dir.mkdir(parents=True, exist_ok=True)
    def write_csv(path, array, fmt):
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            np.savetxt(temporary, array, delimiter=",", fmt=fmt)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    from types import SimpleNamespace
    atlas_results = getattr(result, "atlas_results", None) or {
        atlas_names[0]: SimpleNamespace(
            matrices=result.matrices, atlas=result.atlas,
            atlas_affine=result.atlas_affine,
            region_labels=result.region_labels, nodes=getattr(result, "nodes", None),
        ),
    }
    for atlas_name, atlas_result in atlas_results.items():
        selected_files = atlas_files(atlas_name)
        selected_files["atlas"].parent.mkdir(parents=True, exist_ok=True)
        for name in matrix_names:
            write_csv(selected_files[name], atlas_result.matrices[name].detach().cpu().numpy(),
                      "%d" if name == "count" else "%.9g")
            print(selected_files[name])
        path = selected_files["atlas"]
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}.nii.gz")
        try:
            nib.save(nib.Nifti1Image(
                atlas_result.atlas.detach().cpu().numpy().astype(np.int32),
                atlas_result.atlas_affine.detach().cpu().numpy()), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(path)
        write_csv(selected_files["region_labels"],
                  np.asarray(atlas_result.region_labels, dtype=np.int64), "%d")
        if "nodes" in selected_files:
            temporary = selected_files["nodes"].with_name(
                f".nodes.tsv.tmp-{uuid.uuid4().hex}")
            try:
                with temporary.open("w", newline="") as stream:
                    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                    writer.writerow(("index", "original_label", "hemisphere", "name"))
                    writer.writerows((n.index, n.original_label, n.hemisphere, n.name)
                                     for n in atlas_result.nodes)
                os.replace(temporary, selected_files["nodes"])
            finally:
                temporary.unlink(missing_ok=True)
            print(selected_files["nodes"])
        print(selected_files["region_labels"])
    for name, dtype, affine in (
        ("five_tissue", np.float32, result.five_tissue_affine),
        ("gmwmi", np.float32, result.five_tissue_affine),
        ("fa", np.float32, result.dwi_affine),
        ("brain_mask", np.uint8, result.dwi_affine),
    ):
        path = files[name]
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}.nii.gz")
        try:
            data = getattr(result, name).detach().cpu().numpy().astype(dtype)
            nib.save(nib.Nifti1Image(data, affine.detach().cpu().numpy()), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(path)
    write_csv(files["transform"], result.dwi_to_t1_world.detach().cpu().numpy(), "%.9g")
    print(files["transform"])
    if args.bids_root:
        import json
        description = output_dir / "dataset_description.json"
        description.write_text(json.dumps({
            "Name": "FNIT UKBConnectome_pipeline derivatives",
            "BIDSVersion": "1.9.0", "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "Fudan Neuroimaging Toolkit", "Version": __version__,
                             "CodeURL": "https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit"}],
        }, indent=2) + "\n")
        _record(run_state, run_key)
    if args.device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()
        print(f"torch_peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")

def main(argv=None):
    parser = argparse.ArgumentParser(prog='fnit')
    parser.add_argument('--version', action='version', version=f'Fudan Neuroimaging Toolkit (FNIT) {__version__}')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('fslmaths', help='PyTorch implementation of common fslmaths operations')
    strip = commands.add_parser('synthstrip', help='brain extraction')
    strip.add_argument('-i', '--image', required=True)
    strip.add_argument('-o', '--out')
    strip.add_argument('-m', '--mask')
    strip.add_argument('-d', '--sdt')
    strip.add_argument('--weights')
    strip.add_argument('--device', default='cpu')
    strip.add_argument('--no-csf', action='store_true')
    strip.add_argument('-b', '--border', type=float, default=1)
    strip.add_argument('-f', '--fill', type=float)
    strip.add_argument('-j', '--threads', type=int, default=4)
    reg = commands.add_parser('synthmorph', help='rigid/affine/deformable/joint registration')
    reg.add_argument('moving')
    reg.add_argument('fixed')
    reg.add_argument('-m', '--model', choices=('joint', 'deform', 'affine', 'rigid'), default='joint')
    reg.add_argument('--weights', help='directory containing official checkpoint files')
    reg.add_argument('--device', default='cpu')
    reg.add_argument('-o', '--out-moving')
    reg.add_argument('-O', '--out-fixed')
    reg.add_argument('-t', '--trans')
    reg.add_argument('-T', '--inverse')
    reg.add_argument('--fsl-warp', help='moving-to-fixed FSL intent-2006 dense warp')
    reg.add_argument('-i', '--init')
    reg.add_argument('-M', '--mid-space', action='store_true')
    reg.add_argument('-H', '--header-only', action='store_true')
    reg.add_argument('-e', '--extent', choices=(192, 256), type=int, default=256)
    reg.add_argument('-r', '--hyper', type=float, default=0.5)
    reg.add_argument('-n', '--steps', type=int, default=7)
    reg.add_argument('-j', '--threads', type=int, default=4)
    reg.add_argument('-d', '--output-dir')
    apply = commands.add_parser('apply', help='apply an LTA or RAS warp')
    apply.add_argument('transform')
    apply.add_argument('image')
    apply.add_argument('output')
    apply.add_argument('-m', '--method', choices=('linear', 'nearest'), default='linear')
    apply.add_argument('-f', '--fill', type=float, default=0)
    apply.add_argument('-t', '--dtype', choices=('uint8', 'uint16', 'int16', 'int32', 'float32'), default='float32')
    apply.add_argument('-H', '--header-only', action='store_true')
    wmh = commands.add_parser('wmh-synthseg', help='WMH and anatomy segmentation')
    wmh.add_argument('--i', '-i', required=True, help='single 3D input image')
    wmh.add_argument('--o', '-o', required=True, help='segmentation image')
    wmh.add_argument('--csv_vols', '--csv-vols')
    wmh.add_argument('--device', default='cpu')
    wmh.add_argument('--threads', type=int, default=1)
    wmh.add_argument('--crop', action='store_true')
    wmh.add_argument('--save_lesion_probabilities', '--save-lesion-probabilities', action='store_true')
    wmh.add_argument('--weights', help='official checkpoint file or containing directory')
    synthseg = commands.add_parser('synthseg', help='33-class T1 segmentation and soft volumes')
    synthseg.add_argument('--i', '-i', required=True, help='single 3D T1 image')
    synthseg.add_argument('--o', '-o', required=True, help='segmentation image')
    synthseg.add_argument('--csv-vols', '--csv_vols',
                          help='FreeSurfer-style soft volumes CSV')
    synthseg.add_argument('--weights', help='official SynthSeg 2.0 H5 or containing directory')
    synthseg.add_argument('--device', default='cpu')
    synthseg.add_argument('--threads', type=int, default=4)
    synthseg.add_argument('--keep-geometry', action='store_true',
                          help='resample labels back to the input image grid')
    synthseg.add_argument('--color-lut', help='optional FreeSurfer color lookup table')
    synthseg.add_argument('--parc', action='store_true', help='SynthSeg 2.0 cortical parcellation')
    synthseg.add_argument('--parc-weights', help='official synthseg_parc_2.0.h5 or directory')
    synthseg.add_argument('--parc-out', help='optional cortex-only parcel image')
    subregions = commands.add_parser('subregions', help='experimental PyTorch GEMS subregions')
    subregions.add_argument('--i', '-i', required=True, help='native 3-D T1 image')
    subregions.add_argument('--o', '-o', required=True, help='native-grid labels')
    subregions.add_argument('--atlas-root', help='prepared atlas cache; default is FNIT cache')
    subregions.add_argument('--structure', action='append', help='atlas-pack name; repeat for several')
    subregions.add_argument('--coarse-segmentation', help='native-grid coarse labels')
    subregions.add_argument('--cortical-parcellation', help='native-grid DKT cortical labels')
    subregions.add_argument('--wmparc', help='native-grid white matter parcellation')
    subregions.add_argument('--synthseg-weights', help='SynthSeg weights for initialization')
    subregions.add_argument('--synthseg-parc-weights', help='SynthSeg+ cortical weights')
    subregions.add_argument('--output-dir', help='labels, volumes and report output directory')
    subregions.add_argument('--save-highres', action='store_true')
    subregions.add_argument('--report-json', help='machine-readable processing report')
    subregions.add_argument('--no-auto-initialize', action='store_true')
    subregions.add_argument('--em-iterations', type=int, default=8)
    subregions.add_argument('--deform-iterations', type=int, default=0)
    subregions.add_argument('--device', default='cuda:0')
    sr = commands.add_parser('synthsr', help='synthesize a 1 mm T1-weighted image')
    sr.add_argument('--i', '-i', required=True, help='single input image')
    sr.add_argument('--o', '-o', required=True, help='output image or directory for this image')
    sr.add_argument('--device', default='cpu')
    sr.add_argument('--cpu', action='store_true', help='use CPU, matching the original --cpu')
    sr.add_argument('--threads', type=int, default=1)
    sr.add_argument('--ct', action='store_true')
    sr.add_argument('--lowfield', action='store_true')
    sr.add_argument('--v1', action='store_true')
    sr.add_argument('--disable_sharpening', action='store_true')
    sr.add_argument('--disable_flipping', action='store_true')
    sr.add_argument('--weights', '--model', help='official checkpoint file or containing directory')
    fast = commands.add_parser(
        'fast', help='three-tissue T1 segmentation and bias correction')
    fast.add_argument('-i', '--image', required=True,
                      help='brain-extracted, single-channel T1 image')
    fast.add_argument('-o', '--output-prefix', required=True,
                      help='output basename, matching FSL FAST -o')
    fast.add_argument('--mask', help='optional mask on the input grid')
    fast.add_argument('--device', default='cpu')
    fast.add_argument('--threads', type=int, default=1)
    fast.add_argument('-W', '--init-iterations', type=int, default=15)
    fast.add_argument('-I', '--bias-iterations', type=int, default=4)
    fast.add_argument('-O', '--fixed-iterations', type=int, default=4)
    fast.add_argument('-l', '--bias-fwhm-mm', type=float, default=20.0)
    fast.add_argument('-f', '--init-mrf', type=float, default=0.02)
    fast.add_argument('-H', '--mrf', type=float, default=0.1)
    fast.add_argument('-R', '--mixel-mrf', type=float, default=0.3)
    fast.add_argument('--pve-steps', type=int, default=100)
    fast.add_argument('-N', '--no-bias', action='store_true')
    fast.add_argument('-b', '--save-bias', action='store_true')
    fast.add_argument('-B', '--save-restored', action='store_true')
    fast.add_argument('--overwrite', action='store_true')
    flirt = commands.add_parser(
        'flirt',
        help=(
            'Source-derived PyTorch implementation of the supported FLIRT '
            '12-DOF correlation-ratio or 6-DOF normmi path'
        ),
        allow_abbrev=False)
    flirt.add_argument('-in', '--in', dest='input', required=True,
                       help='moving/input image')
    flirt.add_argument('-ref', '--ref', dest='reference', required=True,
                       help='fixed/reference image defining the output grid')
    flirt.add_argument('-out', '--out', dest='output')
    flirt.add_argument('-omat', '--omat')
    flirt.add_argument('-init', '--init')
    flirt.add_argument('-inweight', '--inweight')
    flirt.add_argument('-refweight', '--refweight')
    flirt.add_argument('-dof', type=int, choices=(6, 12), default=12)
    flirt.add_argument('-cost', choices=('corratio', 'normmi'), default='corratio')
    flirt.add_argument('--device')
    flirt.add_argument('--threads', type=int, default=1)
    flirt.add_argument('--overwrite', action='store_true')
    from .fnirt.cli import add_arguments as add_fnirt_arguments
    fnirt = commands.add_parser(
        'fnirt', help='PyTorch FNIRT default, GM, T1 or TBSS registration',
        allow_abbrev=False)
    add_fnirt_arguments(fnirt)
    applywarp = commands.add_parser(
        'applywarp', help='apply an FSL warp field with PyTorch')
    applywarp.add_argument('-i', '--in', dest='input', required=True,
                           help='input image to resample')
    applywarp.add_argument('-r', '--ref', dest='reference', required=True,
                           help='reference image defining the output grid')
    applywarp.add_argument(
        '-w', '--warp',
        help='FSL dense displacement field or FNIRT cubic coefficient file')
    applywarp.add_argument('-o', '--out', dest='output', required=True)
    applywarp.add_argument('--premat', help='input-to-warp-source FLIRT matrix')
    applywarp.add_argument('--postmat', help='warp-reference-to-output FLIRT matrix')
    convention = applywarp.add_mutually_exclusive_group()
    convention.add_argument('--abs', dest='absolute', action='store_true',
                            help='treat an untyped dense field as absolute coordinates')
    convention.add_argument('--rel', dest='relative', action='store_true',
                            help='treat an untyped dense field as relative displacements')
    applywarp.add_argument('--interp', dest='interpolation',
                           choices=('trilinear', 'nearest', 'nn'), default='trilinear')
    applywarp.add_argument('--datatype',
                           choices=('char', 'short', 'int', 'float', 'double'))
    applywarp.add_argument('--device', default='cpu')
    applywarp.add_argument('--overwrite', action='store_true')
    fast_vbm = commands.add_parser(
        'fast-vbm', help='raw T1 to bias-corrected FAST VBM maps')
    fast_vbm.add_argument('-i', '--image', required=True,
                          help='single-frame raw T1 image')
    fast_vbm.add_argument('--template', required=True,
                          help='GM template defining the output grid')
    fast_vbm.add_argument('-o', '--output-dir', required=True)
    fast_vbm.add_argument('--brain-mask',
                          help='optional input-grid mask; skips SynthStrip')
    fast_vbm.add_argument(
        '--reference-mask',
        help=(
            'optional template-grid mask recorded by both backends and used '
            'by FNIRT; pass the FSL dilated MNI mask for UKB/FSL parity'
        ),
    )
    fast_vbm.add_argument('--synthstrip-weights',
                          help='official SynthStrip checkpoint or containing directory')
    fast_vbm.add_argument('--synthmorph-weights',
                          help='official SynthMorph deform checkpoint; used by the synthmorph backend')
    fast_vbm.add_argument('--registration-backend', choices=('synthmorph', 'fnirt'),
                          default='synthmorph',
                          help='nonlinear registration backend')
    fast_vbm.add_argument('--device', default='cpu')
    fast_vbm.add_argument('--threads', type=int)
    fast_vbm.add_argument('--synthmorph-extent', type=int, choices=(192, 256),
                          default=256)
    fast_vbm.add_argument('--synthmorph-hyper', type=float, default=0.5)
    fast_vbm.add_argument('--synthmorph-steps', type=int, default=7)
    fast_vbm.add_argument('--fnirt-strides', type=int, nargs=4,
                          default=(4, 2, 1, 1),
                          metavar=('LEVEL1', 'LEVEL2', 'LEVEL3', 'LEVEL4'))
    fast_vbm.add_argument('--fnirt-steps', type=int, nargs=4,
                          default=(5, 5, 10, 5),
                          metavar=('LEVEL1', 'LEVEL2', 'LEVEL3', 'LEVEL4'))
    fast_vbm.add_argument('--fnirt-input-fwhm-mm', type=float, nargs=4,
                          default=(6.0, 4.0, 2.0, 2.0),
                          metavar=('LEVEL1', 'LEVEL2', 'LEVEL3', 'LEVEL4'))
    fast_vbm.add_argument('--fnirt-reference-fwhm-mm', type=float, nargs=4,
                          default=(4.0, 2.0, 0.0, 0.0),
                          metavar=('LEVEL1', 'LEVEL2', 'LEVEL3', 'LEVEL4'))
    fast_vbm.add_argument('--fnirt-warp-resolution-mm', type=float, default=10.0)
    fast_vbm.add_argument('--fnirt-regularization', type=float, nargs=4,
                          default=(150.0, 75.0, 50.0, 30.0),
                          metavar=('LEVEL1', 'LEVEL2', 'LEVEL3', 'LEVEL4'))
    fast_vbm.add_argument('--no-bias', action='store_true',
                          help='disable TorchFAST bias-field correction')
    fast_vbm.add_argument('--overwrite', action='store_true')
    connectome = commands.add_parser(
        'UKBConnectome_pipeline', aliases=['connectome'],
        help='raw BIDS DWI/T1 or corrected DWI to one or more connectomes',
        allow_abbrev=False)
    connectome.add_argument('--bids-root', help='raw BIDS dataset root')
    connectome.add_argument('--subject', help='BIDS subject label, with or without sub-')
    connectome.add_argument('--session', help='BIDS session label')
    connectome.add_argument('--run', help='BIDS DWI run label')
    connectome.add_argument('--acquisition', help='BIDS DWI acquisition label')
    connectome.add_argument('--direction', help='BIDS DWI direction label')
    connectome.add_argument('--corrected-dwi',
                            help='existing corrected DWI for BIDS mode; requires --rotated-bvecs')
    connectome.add_argument('--rotated-bvecs',
                            help='eddy-rotated FSL bvecs for --corrected-dwi')
    connectome.add_argument('--dwi', help='corrected 4D DWI NIfTI in explicit mode')
    connectome.add_argument('--bvals')
    connectome.add_argument('--bvecs', help='eddy-rotated FSL bvecs in explicit mode')
    connectome.add_argument('--t1', help='paired skull-stripped T1 registration image')
    connectome.add_argument('--t1-segmentation',
                            help='official FreeSurfer recon-all aparc+aseg.mgz')
    connectome.add_argument('--atlas-dwi',
                            help='integer atlas in DWI RAS world coordinates')
    connectome.add_argument('--freesurfer-subject-dir', help='completed recon-all subject directory')
    connectome.add_argument('--atlas', nargs='+', default=['fs-aparc'],
                            choices=('fs-aparc', 'fs-aparc-a2009s',
                                     'aparc+tian-s1', 'aparc.a2009s+tian-s1',
                                     'glasser+tian-s1', 'glasser+tian-s4',
                                     'schaefer200+tian-s1',
                                     'schaefer500+tian-s4', 'schaefer1000+tian-s4'),
                            help='one or more atlas names; tracking and SIFT2 are shared')
    connectome.add_argument('--atlas-templates-dir',
                            help='original UKB atlas directory; Glasser also needs sibling surfaces directory')
    connectome.add_argument('--download-atlases', action='store_true',
                            help='download selected licensed Tian/Schaefer atlas files with size and SHA-256 checks')
    connectome.add_argument('--fsaverage-dir',
                            help='fsaverage subject directory with lh/rh sphere.reg')
    connectome.add_argument('--mni-template',
                            help='MNI152 T1 2 mm image on the Tian atlas voxel grid')
    connectome.add_argument('--synthmorph-weights',
                            help='directory containing affine and deform SynthMorph weights')
    connectome.add_argument('--tian-fnirt-coeff',
                            help='existing FNIRT T1-to-MNI coefficient; original-atlas alternative to SynthMorph')
    connectome.add_argument('--brain-mask',
                            help='optional binary DWI BET mask; default native BET on LAS mean b0')
    connectome.add_argument('--shell-bvals', type=float, nargs='+',
                            help='optional MRtrix response-header shell centers, e.g. 5 999 1997')
    connectome.add_argument('--response-mask', help='fixed response-selection mask on DWI grid')
    connectome.add_argument('--fod-mask', help='fixed CSD mask; default two-pass dilation')
    connectome.add_argument('--normalise-mask', help='fixed mtnormalise mask; default two-pass erosion')
    connectome.add_argument('--fa-map', help='precomputed UKB dti_FA map; default DWI tensor FA')
    connectome.add_argument('--dwi-to-t1-world', help='optional 4x4 RAS-mm transform')
    connectome.add_argument('--output-dir', required=True)
    connectome.add_argument('--device', default='cuda:0')
    connectome.add_argument('--n-seeds', type=int, required=True)
    connectome.add_argument('--seed', type=int, default=0)
    connectome.add_argument('--compile-arc', action='store_true',
                            help='compile the CUDA iFOD2 arc kernel for large seed counts')
    connectome.add_argument('--overwrite', action='store_true')
    # Standalone SynthSeg must not import unrelated pipelines or their dependencies.
    selected = sys.argv[1:] if argv is None else argv
    if selected and selected[0] == 'fslmaths':
        from .fslmaths.cli import main as fslmaths_main
        fslmaths_main(selected[1:])
        return
    if selected and selected[0] == "synthseg":
        _run_synthseg(parser.parse_args(selected))
        return
    if selected and selected[0] == "subregions":
        _run_subregions(parser.parse_args(selected))
        return
    if selected and selected[0] == "flirt":
        _run_flirt(parser.parse_args(selected))
        return
    if selected and selected[0] in ("UKBConnectome_pipeline", "connectome"):
        _run_connectome(parser.parse_args(selected))
        return
    if selected and selected[0] == "synthsr":
        _run_synthsr(parser.parse_args(selected))
        return
    if selected and selected[0] == "wmh-synthseg":
        _run_wmh(parser.parse_args(selected))
        return
    if selected and selected[0] == "fast":
        _run_fast(parser.parse_args(selected))
        return
    if selected and selected[0] == "synthstrip":
        args = parser.parse_args(selected)
        if not any((args.out, args.mask, args.sdt)):
            parser.error('provide at least one -o, -m or -d output')
        from .synthstrip import SynthStrip
        result = SynthStrip(args.weights, args.device, args.no_csf, args.threads)(
            args.image, args.border, args.fill)
        for volume, path in ((result.image, args.out), (result.mask, args.mask),
                             (result.distance, args.sdt)):
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                volume.save(path)
                print(path)
        return
    from .topup.cli import add_parser as add_topup_parser
    from .eddy.cli import add_parser as add_eddy_parser
    from .dtifit.cli import add_parser as add_dtifit_parser
    from .amico_noddi.cli import add_parser as add_amico_noddi_parser
    add_topup_parser(commands)
    add_eddy_parser(commands)
    add_dtifit_parser(commands)
    add_amico_noddi_parser(commands)
    from .mmorf.cli import add_parser as add_mmorf_parser
    from .dmri_pipeline.cli import add_parser as add_dmri_pipeline_parser
    add_mmorf_parser(commands)
    add_dmri_pipeline_parser(commands)
    from .bedpostx.cli import add_parser as add_bedpostx_parser
    add_bedpostx_parser(commands)
    from .probtrackx.cli import add_parser as add_probtrackx_parser
    add_probtrackx_parser(commands)
    from .convertwarp.cli import add_parser as add_convertwarp_parser
    from .invwarp.cli import add_parser as add_invwarp_parser
    add_convertwarp_parser(commands)
    add_invwarp_parser(commands)
    args = parser.parse_args(argv)
    if hasattr(args, '_fnit_handler'):
        args._fnit_handler(args)
        return
    if args.command == 'synthseg':
        _run_synthseg(args)
        return
    if args.command == 'subregions':
        _run_subregions(args)
        return
    if args.command == 'fnirt':
        _run_fnirt(args)
        return
    if args.command == 'applywarp':
        _run_applywarp(args)
        return
    if args.command == 'fast-vbm':
        _run_fast_vbm(args)
        return
    if args.command == 'synthstrip':
        from .synthstrip import SynthStrip
        if not any((args.out, args.mask, args.sdt)):
            parser.error('provide at least one -o, -m or -d output')
        result = SynthStrip(args.weights, args.device, args.no_csf, args.threads)(args.image, args.border, args.fill)
        outputs = ((result.image, args.out), (result.mask, args.mask), (result.distance, args.sdt))
    elif args.command == 'synthmorph':
        import torch
        from .synthmorph import SynthMorph, convert_warp_to_fsl
        if not any((args.out_moving, args.out_fixed, args.trans, args.inverse,
                    args.fsl_warp, args.output_dir)):
            parser.error('provide at least one registration output')
        if args.fsl_warp and args.model in ('affine', 'rigid'):
            parser.error('--fsl-warp requires joint or deform model')
        torch.set_num_threads(args.threads)
        model = SynthMorph(args.weights, args.device, args.model, args.extent, args.hyper, args.steps)
        result = model(args.moving, args.fixed, args.init, args.mid_space, args.header_only, args.output_dir)
        outputs = ((result.moved, args.out_moving), (result.fixed_moved, args.out_fixed),
                   (result.transform, args.trans), (result.inverse, args.inverse),
                   (convert_warp_to_fsl(result.transform, moving=args.moving, fixed=args.fixed)
                    if args.fsl_warp else None, args.fsl_warp))
    else:
        from .synthmorph import apply_transform
        result = apply_transform(
            args.image,
            args.transform,
            args.method,
            args.fill,
            args.dtype,
            args.header_only,
        )
        outputs = ((result, args.output),)
    for volume, path in outputs:
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            volume.save(path)
            print(path)


if __name__ == '__main__':
    main()
