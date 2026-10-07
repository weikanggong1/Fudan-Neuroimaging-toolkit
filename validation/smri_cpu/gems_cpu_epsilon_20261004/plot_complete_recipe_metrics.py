"""Plot the preserved real regional Dice; no MRI or atlas read."""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        'encoding', Path(__file__).parent.parent / 'gems_fixes_20261004/report_encoding.py')
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)
    fig, axes = plt.subplots(4, 2, figsize=(9, 13), sharex=True, sharey=True)
    rowmap = {'brainstem': 0, 'thalamus': 1, 'hippo-amygdala-left': 2, 'hippo-amygdala-right': 3}
    for family in ('brainstem', 'thalamus', 'hippo-amygdala'):
        report = encoding.load_report(args.directory / f'score-{family}.public.json')
        for group in report['regional_changes']:
            name, space = group['id'].rsplit('_', 1)
            ax = axes[rowmap[name], 0 if space == 'native' else 1]
            rows = [row for row in group['regions'] if row['dice_delta'] is not None]
            colors = ['#bd2727' if row['passed_before'] and not row['passed_after'] else
                      '#178541' if row['passed_after'] and not row['passed_before'] else '#557691'
                      for row in rows]
            ax.scatter([row['dice_before'] for row in rows], [row['dice_after'] for row in rows],
                       s=28, c=colors, alpha=.8)
            ax.plot([0, 1], [0, 1], color='#999999', lw=.7)
            ax.axhline(.95, color='#999999', lw=.5, ls=':')
            ax.axvline(.95, color='#999999', lw=.5, ls=':')
            ax.set_title(f"{name} / {space}: {group['old_gate']['pass']} → "
                         f"{group['new_gate']['pass']} / {group['new_gate']['nonempty']}", fontsize=10)
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=.15)
    for ax in axes[-1]:
        ax.set_xlabel('Baseline Dice vs official')
    for ax in axes[:, 0]:
        ax.set_ylabel('CPU mixed candidate Dice vs official')
    fig.suptitle('Public sub-02, saved same-input complete recipes\n'
                 'Red: lost old gate; green: newly passed gate; blue: remaining regions', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
