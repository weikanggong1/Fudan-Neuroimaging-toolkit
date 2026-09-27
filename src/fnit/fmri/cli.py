"""Single-run volumetric fMRI command line interface."""

import argparse

from .aroma_pipeline import run_aroma_pipeline
from .pipeline import run_feat_core


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-fmri")
    commands = parser.add_subparsers(dest="command", required=True)
    feat = commands.add_parser("feat", help="BIDS BOLD to pre-ICA FEAT core outputs")
    feat.add_argument("--bids-root", required=True)
    feat.add_argument("--subject", required=True)
    feat.add_argument("--output-dir", required=True)
    feat.add_argument("--session")
    feat.add_argument("--task", default="rest")
    feat.add_argument("--run")
    feat.add_argument("--acquisition")
    feat.add_argument("--direction")
    feat.add_argument("--reconstruction")
    feat.add_argument("--echo")
    feat.add_argument("--brain-mask")
    feat.add_argument("--spatial-warp")
    feat.add_argument("--postmat")
    feat.add_argument("--highpass-cutoff-seconds", type=float, default=100)
    feat.add_argument("--device")
    feat.add_argument("--batch-size", type=int, default=32)
    feat.add_argument("--motion-iterations", nargs=3, type=int, default=(35, 25, 15))
    feat.add_argument("--overwrite", action="store_true")
    aroma = commands.add_parser("aroma", help="ICA-AROMA and optional confound regression")
    for name in ("filtered-func-data", "brain-mask", "motion-parameters", "csf-mask",
                 "edge-mask", "outside-mask", "output-dir"):
        aroma.add_argument("--" + name, required=True)
    aroma.add_argument("--n-components", type=int, required=True)
    aroma.add_argument("--tr", type=float)
    aroma.add_argument("--mode", choices=("nonaggr", "aggr"), default="nonaggr")
    aroma.add_argument("--device")
    aroma.add_argument("--wm-mask")
    aroma.add_argument("--regress-csf", action="store_true")
    aroma.add_argument("--regress-motion", action="store_true")
    aroma.add_argument("--motion-model", type=int, choices=(6, 12, 24), default=24)
    aroma.add_argument("--bandpass", nargs=2, type=float)
    aroma.add_argument("--global-signal", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "feat":
        result = run_feat_core(
            bids_root=args.bids_root, output_dir=args.output_dir, subject=args.subject,
            session=args.session, task=args.task, run=args.run,
            acquisition=args.acquisition, direction=args.direction,
            reconstruction=args.reconstruction, echo=args.echo,
            brain_mask=args.brain_mask,
            spatial_warp=args.spatial_warp, postmat=args.postmat,
            highpass_cutoff_seconds=args.highpass_cutoff_seconds,
            device=args.device, batch_size=args.batch_size,
            motion_iterations=tuple(args.motion_iterations), overwrite=args.overwrite,
        )
        print(result.filtered_func_data)
    else:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
