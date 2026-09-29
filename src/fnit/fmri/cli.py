"""Single-run volumetric fMRI command line interface."""

import argparse
import json
from pathlib import Path

from .aroma_pipeline import run_aroma_pipeline
from .end_to_end import run_fmri_pipeline
from .pipeline import run_feat_core
from .surface import SurfaceHemisphere
from .surface_pipeline import SurfacePipelineInputs, run_surface_from_volume


def _bids_options(parser):
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session")
    parser.add_argument("--task", default="rest")
    parser.add_argument("--run")
    parser.add_argument("--acquisition")
    parser.add_argument("--direction")
    parser.add_argument("--reconstruction")
    parser.add_argument("--echo")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--motion-iterations", nargs=3, type=int, default=(35, 25, 15))
    parser.add_argument("--highpass-cutoff-seconds", type=float, default=100)
    parser.add_argument("--overwrite", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-fmri")
    commands = parser.add_subparsers(dest="command", required=True)
    feat = commands.add_parser("feat", help="BIDS BOLD to pre-ICA FEAT core outputs")
    _bids_options(feat)
    feat.add_argument("--brain-mask")
    feat.add_argument("--brain-extraction", choices=("synthstrip", "otsu"), default="synthstrip")
    feat.add_argument("--synthstrip-weights")
    feat.add_argument("--spatial-warp")
    feat.add_argument("--postmat")
    aroma = commands.add_parser("aroma", help="PICA/ICA-AROMA and optional confound regression")
    for name in ("filtered-func-data", "brain-mask", "motion-parameters", "csf-mask",
                 "edge-mask", "outside-mask", "output-dir"):
        aroma.add_argument("--" + name, required=True)
    aroma.add_argument("--n-components", type=int)
    aroma.add_argument("--tr", type=float)
    aroma.add_argument("--mode", choices=("nonaggr", "aggr"), default="nonaggr")
    aroma.add_argument("--device")
    aroma.add_argument("--wm-mask")
    aroma.add_argument("--regress-csf", action="store_true")
    aroma.add_argument("--regress-motion", action="store_true")
    aroma.add_argument("--motion-model", type=int, choices=(6, 12, 24), default=24)
    aroma.add_argument("--bandpass", nargs=2, type=float)
    aroma.add_argument("--global-signal", action="store_true")
    pipeline = commands.add_parser("run", help="raw BIDS to clean MNI152 2-mm BOLD")
    _bids_options(pipeline)
    pipeline.add_argument("--mni-template", required=True)
    pipeline.add_argument("--surface-config", help="JSON paths for registered MNI surfaces and fsLR32k atlas")
    pipeline.add_argument("--surface-subject-dir", help="Precomputed structural surfaces and wmparc directory")
    pipeline.add_argument("--surface-assets-dir", help="Verified HCP fsLR templates directory")
    pipeline.add_argument("--wb-command", default="wb_command", help="Connectome Workbench executable")
    pipeline.add_argument("--mni-brain-mask")
    pipeline.add_argument("--t1w-image")
    pipeline.add_argument("--registration-backend", choices=("synthmorph", "fnirt"), default="synthmorph")
    pipeline.add_argument("--synthstrip-weights")
    pipeline.add_argument("--synthmorph-weights")
    pipeline.add_argument("--ica-n-components", type=int)
    pipeline.add_argument("--ica-max-iter", type=int, default=500)
    pipeline.add_argument("--aroma-mode", choices=("nonaggr", "aggr"), default="nonaggr")
    pipeline.add_argument("--regress-wm", action="store_true")
    pipeline.add_argument("--regress-csf", action="store_true")
    pipeline.add_argument("--regress-motion", action="store_true")
    pipeline.add_argument("--motion-model", type=int, choices=(6, 12, 24), default=24)
    pipeline.add_argument("--bandpass", nargs=2, type=float)
    pipeline.add_argument("--global-signal", action="store_true")
    pipeline.add_argument("--n-splits", type=int, default=1000)
    pipeline.add_argument("--random-state", type=int, default=0)
    surface = commands.add_parser("surface", help="clean native/T1w cortex and MNI subcortex to fsLR32k CIFTI")
    surface.add_argument("--volume-dir", required=True)
    surface.add_argument("--recon-all", required=True)
    surface.add_argument("--surface-assets-dir", required=True)
    surface.add_argument("--output-dir", required=True)
    surface.add_argument("--wb-command", default="wb_command")
    surface.add_argument("--device", default="cpu")
    surface.add_argument("--registration", choices=("msmsulc", "newmsm_experimental", "fs"), default="msmsulc")
    surface.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.command in ("feat", "run"):
        common = dict(
            bids_root=args.bids_root, output_dir=args.output_dir, subject=args.subject,
            session=args.session, task=args.task, run=args.run,
            acquisition=args.acquisition, direction=args.direction,
            reconstruction=args.reconstruction, echo=args.echo,
            highpass_cutoff_seconds=args.highpass_cutoff_seconds,
            device=args.device, batch_size=args.batch_size,
            motion_iterations=tuple(args.motion_iterations), overwrite=args.overwrite,
        )
    if args.command == "feat":
        result = run_feat_core(
            **common, brain_mask=args.brain_mask,
            brain_extraction=args.brain_extraction,
            synthstrip_weights=args.synthstrip_weights,
            spatial_warp=args.spatial_warp, postmat=args.postmat,
        )
        print(result.filtered_func_data)
    elif args.command == "aroma":
        result = run_aroma_pipeline(
            filtered_func_data=args.filtered_func_data, brain_mask=args.brain_mask,
            motion_parameters=args.motion_parameters, csf_mask=args.csf_mask,
            edge_mask=args.edge_mask, outside_mask=args.outside_mask,
            output_dir=args.output_dir, n_components=args.n_components,
            tr=args.tr, mode=args.mode, device=args.device, wm_mask=args.wm_mask,
            regress_csf=args.regress_csf, regress_motion=args.regress_motion,
            motion_model=args.motion_model,
            bandpass=tuple(args.bandpass) if args.bandpass else None,
            global_signal=args.global_signal,
        )
        print(result.denoised_bold)
        if result.confounds_cleaned_bold is not None:
            print(result.confounds_cleaned_bold)
    elif args.command == "surface":
        result = run_surface_from_volume(
            volume_dir=args.volume_dir,
            recon_all=args.recon_all,
            hcp_assets_dir=args.surface_assets_dir,
            output_dir=args.output_dir,
            wb_command=args.wb_command,
            device=args.device,
            registration=args.registration,
            overwrite=args.overwrite,
        )
        print(result.projection.dtseries)
    else:
        surface_inputs = None
        if args.surface_config and args.surface_subject_dir:
            parser.error("--surface-config and --surface-subject-dir are mutually exclusive")
        if args.surface_config:
            config = json.loads(Path(args.surface_config).read_text(encoding="utf-8"))
            surface_inputs = SurfacePipelineInputs(
                left=SurfaceHemisphere(**config["left"]),
                right=SurfaceHemisphere(**config["right"]),
                subject_rois=config["subject_rois"],
                atlas_rois=config["atlas_rois"],
                wb_command=config.get("wb_command", "wb_command"),
                goodvoxels=config.get("goodvoxels"),
            )
        result = run_fmri_pipeline(
            **common,
            mni_template=args.mni_template,
            mni_brain_mask=args.mni_brain_mask,
            t1w_image=args.t1w_image,
            registration_backend=args.registration_backend,
            surface_inputs=surface_inputs,
            surface_subject_dir=args.surface_subject_dir,
            surface_assets_dir=args.surface_assets_dir,
            wb_command=args.wb_command,
            synthstrip_weights=args.synthstrip_weights,
            synthmorph_weights=args.synthmorph_weights,
            ica_n_components=args.ica_n_components,
            ica_max_iter=args.ica_max_iter,
            aroma_mode=args.aroma_mode,
            regress_wm=args.regress_wm,
            regress_csf=args.regress_csf,
            regress_motion=args.regress_motion,
            motion_model=args.motion_model,
            bandpass=tuple(args.bandpass) if args.bandpass else None,
            global_signal=args.global_signal,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        print(result.clean_mni)
        if result.surface is not None:
            print(result.surface.projection.dtseries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
