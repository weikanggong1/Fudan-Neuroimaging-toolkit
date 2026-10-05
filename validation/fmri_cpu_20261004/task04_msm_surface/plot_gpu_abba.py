"""Plot the public complete-API observations and resource peaks, without MRI data."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report_bytes = args.report.read_bytes()
    data = json.loads(report_bytes)
    if data['status'] != 'complete' or data['complete_API_calls'] != 12:
        raise ValueError('Actual complete three-case, twelve-API report required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .7, 'xtick.direction': 'out',
                         'ytick.direction': 'out', 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(2, 3, figsize=(8.4, 4.5))
    titles = ['MSMSulc, 4 levels', 'MSMAll, coarse', 'MSMAll, refine']
    positions = ['A1', 'B1', 'B2', 'A2']
    colors = ['#555555', '#2874A6', '#2874A6', '#555555']
    for index, case in enumerate(data['case_comparisons']):
        rows = [next(row for row in data['observations']
                     if row['case_id'] == case['case_id'] and row['abba_position'] == position)
                for position in positions]
        axis = axes[0, index]
        values = [row['API_seconds'] for row in rows]
        axis.scatter(np.arange(4), values, c=colors, s=30, zorder=3)
        axis.set_xticks(np.arange(4), positions)
        axis.set_xlim(-.4, 3.4)
        axis.set_ylim(0, max(values) * 1.18)
        axis.set_title(titles[index], fontweight='bold')
        axis.set_ylabel('Complete API (s)' if index == 0 else '')
        for x, value in enumerate(values):
            axis.text(x, value + max(values) * .035, f'{value:.1f}', ha='center', fontsize=7)
        axis = axes[1, index]
        metrics = ['cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes',
                   'sampled_owned_process_tree_peak_bytes']
        peaks = [max(row[key] for row in rows) / 1e9 for key in metrics]
        axis.bar(np.arange(3), peaks, color=['#78A6C8', '#A2BED2', '#555555'], width=.6)
        axis.set_xticks(np.arange(3), ['Allocated', 'Reserved', 'Owned tree'], rotation=25, ha='right')
        axis.set_ylim(0, 4.8)
        axis.set_ylabel('GPU peak (GB)' if index == 0 else '')
        for x, value in enumerate(peaks):
            axis.text(x, value + .09, f'{value:.2f}', ha='center', fontsize=7)
    figure.text(.5, .015,
                'A: frozen baseline; B: CPU-optimized source. Shared H100 load; '
                'one full API per process. Owned-tree cap: 20 GB.', ha='center', fontsize=7)
    figure.tight_layout(rect=[0, .05, 1, 1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / 'cpu_optimization_gpu_abba_20261005'
    figure.savefig(prefix.with_suffix('.svg'), bbox_inches='tight')
    figure.savefig(prefix.with_suffix('.png'), dpi=170, bbox_inches='tight')
    plt.close(figure)
    receipt = {'source_report_sha256': hashlib.sha256(report_bytes).hexdigest(),
               'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'figure_sha256': hashlib.sha256(prefix.with_suffix('.svg').read_bytes()).hexdigest(),
               'scope': 'CompleteAPI observations and aggregate resource peaks only',
               'original_API_calls': 12, 'private_individual_data_included': False}
    prefix.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
