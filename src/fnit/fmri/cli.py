"""BIDS fMRI volume and surface commands."""

import argparse

from .end_to_end import fMRIVolume_pipeline
from .surface_pipeline import fMRISurface_pipeline


def _bids_options(parser):
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--derivatives-root", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session")
    parser.add_argument("--task", default="rest")
    parser.add_argument("--run")
    parser.add_argument("--acquisition")
    parser.add_argument("--direction")
    parser.add_argument("--reconstruction")
    parser.add_argument("--echo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-fmri")
    commands = parser.add_subparsers(dest="command", required=True)
    volume = commands.add_parser("volume", help="原始 BIDS 同时生成 preproc 与 clean 体积 BOLD")
    _bids_options(volume)
    volume.add_argument("--mni-template", required=True)
    volume.add_argument("--mni-brain-mask")
    volume.add_argument("--t1w-image")
    volume.add_argument("--no-anatomical-cache", action="store_true",
                        help="本次重新计算解剖结果；默认复用匹配 T1w、模板和配置的缓存")
    volume.add_argument("--bbr-execution", choices=("batched", "reference"), default="batched")
    volume.add_argument("--fnirt-execution", choices=("optimized", "reference"), default="optimized")
    volume.add_argument("--registration-backend", choices=("synthmorph", "fnirt"), default="synthmorph")
    volume.add_argument("--fnirt-preset", choices=("default", "gm", "t1", "tbss"))
    volume.add_argument("--synthstrip-weights")
    volume.add_argument("--synthmorph-weights")
    volume.add_argument("--ica-n-components", type=int)
    volume.add_argument("--ica-max-iter", type=int, default=500)
    volume.add_argument("--aroma-mode", choices=("nonaggr", "aggr"), default="nonaggr")
    volume.add_argument("--regress-wm", action="store_true")
    volume.add_argument("--regress-csf", action="store_true")
    volume.add_argument("--regress-motion", action="store_true")
    volume.add_argument("--motion-model", type=int, choices=(6, 12, 24), default=24)
    volume.add_argument("--bandpass", nargs=2, type=float)
    volume.add_argument("--global-signal", action="store_true")
    volume.add_argument("--batch-size", type=int, default=8)
    volume.add_argument("--motion-iterations", nargs=3, type=int, default=(1, 1, 1))
    volume.add_argument("--highpass-cutoff-seconds", type=float, default=100)
    timing = volume.add_mutually_exclusive_group()
    timing.add_argument("--slice-timing", dest="slice_timing", action="store_true",
                        help="显式开启 preproc 切片时间校正；默认关闭")
    timing.add_argument("--ignore-slice-timing", dest="slice_timing", action="store_false",
                        help="关闭切片时间校正（默认）")
    volume.set_defaults(slice_timing=False)
    volume.add_argument("--slice-time-reference", type=float, default=0.5)
    volume.add_argument("--n-splits", type=int, default=1000)
    volume.add_argument("--random-state", type=int, default=0)
    surface = commands.add_parser("surface", help="已有 volume 到 fsLR32k GIFTI、91k CIFTI、球面及 QC")
    _bids_options(surface)
    surface.add_argument("--recon-all", required=True)
    surface.add_argument("--surface-assets-dir", required=True)
    surface.add_argument("--wb-command", default="wb_command")
    surface.add_argument("--signal", choices=("preproc", "clean"), default="preproc",
                         help="读取对应 volume 分支；默认 preproc，clean 须显式选择")
    surface.add_argument("--fsnative-to-t1w", help="4x4 forward scanner-RAS affine text file")
    surface.add_argument("--registered-spheres", nargs=2, help="L/R registered native spheres; skips estimation")
    surface.add_argument("--goodvoxels", help="optional 3D ROI on the T1w BOLD grid")
    surface.add_argument("--msm-config", help="official MSMSulc configuration file; default HCP schedule")
    surface.add_argument("--msm-execution", choices=("optimized", "reference"), default="optimized")
    surface.add_argument("--msmall-inputs-json", help="Prepared L/R multimodal feature manifest for optional MSMAll refinement")
    surface.add_argument("--msmall-config", help="Official MSMAll configuration; defaults to the HCP three-level refinement")
    args = parser.parse_args(argv)
    common = dict(
        bids_root=args.bids_root, derivatives_root=args.derivatives_root,
        subject=args.subject, session=args.session, task=args.task, run=args.run,
        acquisition=args.acquisition, direction=args.direction,
        reconstruction=args.reconstruction, echo=args.echo,
        device=args.device, overwrite=args.overwrite,
    )
    if args.command == "volume":
        result = fMRIVolume_pipeline(
            **common, mni_template=args.mni_template,
            reuse_anatomical=not args.no_anatomical_cache,
            bbr_execution=args.bbr_execution, fnirt_execution=args.fnirt_execution,
            mni_brain_mask=args.mni_brain_mask, t1w_image=args.t1w_image,
            registration_backend=args.registration_backend,
            fnirt_config=args.fnirt_preset,
            synthstrip_weights=args.synthstrip_weights,
            synthmorph_weights=args.synthmorph_weights,
            ica_n_components=args.ica_n_components,
            ica_max_iter=args.ica_max_iter, aroma_mode=args.aroma_mode,
            regress_wm=args.regress_wm, regress_csf=args.regress_csf,
            regress_motion=args.regress_motion, motion_model=args.motion_model,
            bandpass=tuple(args.bandpass) if args.bandpass else None,
            global_signal=args.global_signal, batch_size=args.batch_size,
            motion_iterations=tuple(args.motion_iterations),
            highpass_cutoff_seconds=args.highpass_cutoff_seconds,
            slice_timing=args.slice_timing,
            slice_time_reference=args.slice_time_reference,
            n_splits=args.n_splits, random_state=args.random_state,
        )
        print(result.clean_mni)
    else:
        result = fMRISurface_pipeline(
            **common, recon_all=args.recon_all,
            hcp_assets_dir=args.surface_assets_dir,
            wb_command=args.wb_command,
            signal=args.signal, fsnative_to_t1w=args.fsnative_to_t1w,
            registered_spheres=args.registered_spheres, goodvoxels=args.goodvoxels,
            msm_config=args.msm_config,
            msm_execution=args.msm_execution,
            msmall_inputs=args.msmall_inputs_json,
            msmall_config=args.msmall_config,
        )
        print(result.dtseries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
