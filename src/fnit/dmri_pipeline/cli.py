"""Single-subject command line interface for the diffusion pipeline."""

import argparse


def _arguments(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-dir", help="directory containing AP.* and optional PA.*")
    source.add_argument("--bids-root", help="root of a raw BIDS dataset")
    parser.add_argument("--subject", help="BIDS subject label; required with --bids-root")
    parser.add_argument("--session", help="BIDS session label")
    parser.add_argument("--run", help="BIDS DWI run label")
    parser.add_argument("--acquisition", help="BIDS DWI acquisition label")
    parser.add_argument("--direction", help="BIDS DWI dir label, such as AP")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument(
        "--registration-backend", choices=("tbss", "mmorf"), default="tbss"
    )
    parser.add_argument("--fnirt-preset", choices=("default", "gm", "t1", "tbss"), help="FNIRT preset; default tbss for TBSS registration")
    parser.add_argument("--fa-template", required=True, help="FMRIB58_FA_1mm or grid-equivalent FA")
    parser.add_argument("--fa-skeleton", help="FMRIB58_FA-skeleton_1mm; required by TBSS")
    parser.add_argument("--t1", help="subject T1w; required by MMORF")
    parser.add_argument("--t1-template", help="MNI152 T1 1 mm brain; required by MMORF")
    parser.add_argument("--tensor-template", help="FSL_HCP1065_tensor_1mm; required by MMORF")
    parser.add_argument("--synthstrip-weights", help="official synthstrip.1.pt; required by MMORF")
    parser.add_argument("--dti-shell", type=float, default=1000)
    parser.add_argument("--dti-tolerance", type=float, default=100)
    parser.add_argument("--bvec-source", choices=("rotated", "raw"), default="rotated")
    parser.add_argument("--noddi-fit-method", choices=("amico", "classic"), default="amico")
    parser.add_argument("--eddy-gp-seed", type=int, help="fixed EDDY GP sampling seed, 1 to 2**32-1; default time-based seed")
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")


def run(args):
    from .pipeline import DMRIPipeline

    pipeline = DMRIPipeline(
        device=args.device,
        registration_backend=args.registration_backend,
        fnirt_config=args.fnirt_preset,
        synthstrip_weights=args.synthstrip_weights,
        dti_shell=args.dti_shell,
        dti_tolerance=args.dti_tolerance,
        bvec_source=args.bvec_source,
        noddi_fit_method=args.noddi_fit_method,
        eddy_gp_seed=args.eddy_gp_seed,
    )
    options = {
        "fa_template": args.fa_template,
        "fa_skeleton": args.fa_skeleton,
        "t1": args.t1,
        "t1_template": args.t1_template,
        "tensor_template": args.tensor_template,
        "overwrite": args.overwrite,
    }
    if args.bids_root:
        if not args.subject:
            raise ValueError("--subject is required with --bids-root")
        result = pipeline.run_bids(
            args.bids_root, args.output_dir, subject=args.subject,
            session=args.session, run=args.run, acquisition=args.acquisition,
            direction=args.direction, **options,
        )
    else:
        result = pipeline.run(args.raw_dir, args.output_dir, **options)
    print(result.output_dir)
    print(result.qc)


def add_parser(commands):
    parser = commands.add_parser(
        "dmri-pipeline", help="TOPUP/EDDY/DTIFIT/NODDI plus TBSS or MMORF"
    )
    _arguments(parser)
    parser.set_defaults(_fnit_handler=run)
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnit-dmri-pipeline")
    _arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
