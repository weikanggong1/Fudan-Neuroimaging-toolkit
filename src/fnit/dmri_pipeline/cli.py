"""Single-subject command line interface for the diffusion pipeline."""

import argparse


def _arguments(parser):
    parser.add_argument("--raw-dir", required=True, help="directory containing AP.* and optional PA.*")
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
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")


def run(args):
    from .pipeline import DMRIPipeline

    result = DMRIPipeline(
        device=args.device,
        registration_backend=args.registration_backend,
        fnirt_config=args.fnirt_preset,
        synthstrip_weights=args.synthstrip_weights,
        dti_shell=args.dti_shell,
        dti_tolerance=args.dti_tolerance,
        bvec_source=args.bvec_source,
    ).run(
        args.raw_dir,
        args.output_dir,
        fa_template=args.fa_template,
        fa_skeleton=args.fa_skeleton,
        t1=args.t1,
        t1_template=args.t1_template,
        tensor_template=args.tensor_template,
        overwrite=args.overwrite,
    )
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
