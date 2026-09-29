"""Command-line BWAS entry point."""

import argparse

from .core import run_bwas


def main(argv=None):
    parser = argparse.ArgumentParser(description="Group voxel-pair BWAS on clean 2 mm BIDS BOLD")
    parser.add_argument("--bids-root", required=True, help="BIDS Derivatives root")
    parser.add_argument("--participants", required=True, help="TSV with participant_id and model columns")
    parser.add_argument("--mask", required=True, help="2 mm standard-space NIfTI analysis mask")
    parser.add_argument("--output-root", required=True, help="empty BIDS Derivatives output directory")
    parser.add_argument("--phenotype", required=True, help="binary or continuous target column")
    parser.add_argument("--covariate", action="append", default=[], help="numeric covariate column; repeat")
    parser.add_argument("--cdt", type=float, default=5.0, help="absolute z cluster threshold")
    parser.add_argument("--block-size", type=int, default=128, help="voxel tile edge length")
    parser.add_argument("--subject-block-size", type=int, default=16, help="subjects per GPU batch")
    parser.add_argument("--num-workers", type=int, default=1, help="parallel BOLD preparation workers")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device")
    parser.add_argument("--fwhm", type=float, help="optional known spatial smoothness in voxels")
    parser.add_argument("--validate-direct-ols", action="store_true",
                        help="compare every voxel pair with unbatched OLS; high host RAM use")
    args = parser.parse_args(argv)
    result = run_bwas(args.bids_root, args.participants, args.mask, args.output_root,
                      phenotype=args.phenotype, covariates=tuple(args.covariate),
                      cdt=args.cdt, block_size=args.block_size,
                      subject_block_size=args.subject_block_size,
                      num_workers=args.num_workers,
                      device=args.device, fwhm=args.fwhm,
                      validate_direct_ols=args.validate_direct_ols)
    print(result)


if __name__ == "__main__":
    main()
