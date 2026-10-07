"""Bind each current CPU timing arm to the corresponding full precision run."""
import argparse
import json
from pathlib import Path

try:
    from compare_cpu_branch import digest, image_comparison
except ModuleNotFoundError:
    from compare_cpu_branch_v1 import digest, image_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base = Path(args.root) / 'runs/smri_cpu_20261004/remaining_20261004/morph'
    reference_folders = {'A1_v7': 'nodecw7_final_v7/joint',
                         'A2_v7': 'nodecw7_final_v7/joint',
                         'C1_v29': 'nodecw7_final_v29/joint',
                         'C2_v29': 'nodecw7_final_v29/joint',
                         'R_reference': 'nodecw7_paired_v2/joint_reference'}
    rows = {}
    for arm, preceding in reference_folders.items():
        folder = base / 'nodecw7_accept_v29' / arm
        rows[arm] = {name: image_comparison(base / preceding / (name + '.nii.gz'), folder / (name + '.nii.gz'))
                     for name in ('forward', 'inverse', 'moved', 'fixed_moved')}
    report = {'scope': __doc__, 'rows': rows, 'worker_sha256': digest(__file__)}
    report['all_gates_passed'] = all(all(value for key, value in row.items() if key.endswith('_equal'))
                                   for arm in rows.values() for row in arm.values())
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_gates_passed']:
        raise RuntimeError('timing arm has different output than its precision run')


if __name__ == '__main__':
    main()
