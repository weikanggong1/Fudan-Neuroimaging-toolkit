"""Command line entry for masked NIfTI BigFLICA and new-subject projection."""

import argparse
import json
from pathlib import Path

from .pipeline import apply_model, run_bigflica


def main() -> None:
    parser = argparse.ArgumentParser(prog="fnit-bigflica")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit multimodal components")
    fit.add_argument("--subjects-root", required=True)
    fit.add_argument("--config", required=True, help="JSON modality-to-image-and-mask mapping")
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--subjects-file", help="one subject directory name per line")
    fit.add_argument("--n-components", type=int, required=True)
    fit.add_argument("--migp-dim", type=int)
    fit.add_argument("--dicl-dim", type=int)
    fit.add_argument("--no-mmigp-dicl", action="store_true",
                     help="fit FLICA directly to disk-backed standardized voxels")
    fit.add_argument("--dicl-max-iter", type=int, default=1000)
    fit.add_argument("--flica-max-iter", type=int, default=1000)
    fit.add_argument("--flica-lambda-dims", choices=("o", "R"), default="o")
    fit.add_argument("--top-voxels", type=int, default=1000)
    fit.add_argument("--random-state", type=int, default=0)
    fit.add_argument("--device", default="auto")
    fit.add_argument("--max-gpu-gb", type=float, default=19.0)
    fit.add_argument("--feature-block", type=int, default=2048)
    fit.add_argument("--dicl-batch-size", type=int, default=32)
    fit.add_argument("--dicl-sparse-iterations", type=int, default=120,
                     help="maximum LARS path events per voxel")
    apply = commands.add_parser("apply", help="project one unseen subject")
    apply.add_argument("--model-dir", required=True)
    apply.add_argument("--subject-dir", required=True)
    apply.add_argument("--output-file", required=True)
    apply.add_argument("--ridge", type=float, default=1e-6)
    apply.add_argument("--device", default="auto")
    apply.add_argument("--feature-block", type=int, default=32768)
    args = parser.parse_args()
    if args.command == "fit":
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        subjects = None
        if args.subjects_file:
            subjects = [line.strip() for line in Path(args.subjects_file).read_text(
                encoding="utf-8").splitlines() if line.strip()]
        result = run_bigflica(
            args.subjects_root, config["modalities"], args.output_dir,
            args.n_components, args.migp_dim, args.dicl_dim,
            subjects=subjects, device=args.device,
            dicl_max_iter=args.dicl_max_iter, flica_max_iter=args.flica_max_iter,
            flica_lambda_dims=args.flica_lambda_dims,
            top_voxels=args.top_voxels, random_state=args.random_state,
            max_gpu_gb=args.max_gpu_gb, feature_block=args.feature_block,
            dicl_batch_size=args.dicl_batch_size,
            dicl_sparse_iterations=args.dicl_sparse_iterations,
            use_mmigp_dicl=not args.no_mmigp_dicl,
        )
        print(result)
    else:
        scores = apply_model(args.model_dir, args.subject_dir,
                             ridge=args.ridge, output_file=args.output_file,
                             device=args.device, feature_block=args.feature_block)
        print("\t".join(f"{value:.8g}" for value in scores))


if __name__ == "__main__":
    main()
