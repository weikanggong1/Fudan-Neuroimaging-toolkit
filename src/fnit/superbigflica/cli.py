"""BigFLICA-format NIfTI inputs and a separate subject-matched phenotype CSV."""

import argparse
import json
from pathlib import Path

from .pipeline import apply_model, run_superbigflica


def main() -> None:
    parser = argparse.ArgumentParser(prog='fnit-superbigflica')
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='fit supervised multimodal components')
    fit.add_argument('--subjects-root', required=True)
    fit.add_argument('--config', required=True, help='JSON containing modalities and targets')
    fit.add_argument('--phenotypes-csv', required=True)
    fit.add_argument('--id-column', default='subject_id')
    fit.add_argument('--split-column')
    fit.add_argument('--subjects-file', help='one exact subject directory ID per line')
    fit.add_argument('--output-dir', required=True)
    fit.add_argument('--n-components', type=int, required=True)
    fit.add_argument('--validation-fraction', type=float, default=0.2)
    fit.add_argument('--test-fraction', type=float, default=0.2)
    fit.add_argument('--max-epochs', type=int, default=50)
    fit.add_argument('--batch-size', type=int, default=64)
    fit.add_argument('--learning-rate', type=float, default=0.001)
    fit.add_argument('--dropout', type=float, default=0.2)
    fit.add_argument('--relative-weight', type=float, default=0.5)
    fit.add_argument('--class-weight', choices=('balanced', 'none'), default='balanced',
                     help='classification loss weights computed from training labels only')
    fit.add_argument('--random-state', type=int, default=0)
    fit.add_argument('--device', default='auto')
    fit.add_argument('--max-gpu-gb', type=float, default=19.0)
    fit.add_argument('--feature-block', type=int, default=2048)
    fit.add_argument('--top-voxels', type=int, default=1000)
    fit.add_argument('--no-plots', action='store_true', help='skip summary figures; use plot later')
    plot = commands.add_parser('plot', help='redraw model weights, top components and test predictions')
    plot.add_argument('--model-dir', required=True)
    plot.add_argument('--output-dir')
    plot.add_argument('--labels', help='JSON mapping target columns to display names with units')
    apply = commands.add_parser('apply', help='predict one subject using a frozen model')
    apply.add_argument('--model-dir', required=True)
    apply.add_argument('--subject-dir', required=True)
    apply.add_argument('--device', default='auto')
    apply.add_argument('--output-file', required=True, help='JSON predictions and components')
    args = parser.parse_args()
    if args.command == 'plot':
        from .plotting import plot_superbigflica
        labels = json.loads(Path(args.labels).read_text(encoding='utf-8')) if args.labels else None
        print(plot_superbigflica(args.model_dir, args.output_dir, labels=labels))
        return
    if args.command == 'apply':
        result = apply_model(args.model_dir, args.subject_dir,
                             device=args.device, output_file=args.output_file)
        print(json.dumps(result, ensure_ascii=False))
        return
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    subjects = None
    if args.subjects_file:
        subjects = [line.strip() for line in Path(args.subjects_file).read_text(
            encoding='utf-8').splitlines() if line.strip()]
    result = run_superbigflica(
        args.subjects_root, config['modalities'], args.phenotypes_csv,
        config['targets'], args.output_dir, args.n_components,
        id_column=args.id_column, split_column=args.split_column, subjects=subjects,
        validation_fraction=args.validation_fraction, test_fraction=args.test_fraction,
        max_epochs=args.max_epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, dropout=args.dropout,
        relative_weight=args.relative_weight, class_weight=args.class_weight,
        random_state=args.random_state,
        device=args.device, max_gpu_gb=args.max_gpu_gb,
        feature_block=args.feature_block, top_voxels=args.top_voxels,
        make_plots=not args.no_plots)
    print(result)


if __name__ == '__main__':
    main()
