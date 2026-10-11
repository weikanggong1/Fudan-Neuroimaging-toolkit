"""回看最终输入 white 的完整交叉面对在自产已接受轨迹中的首次出现。"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import nibabel as nib
import numpy as np
from measure_recon_pial_residual_geometry import relations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--hemisphere', choices=['lh', 'rh'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = np.load(args.measurement / 'input_white-self.pairs.private.npy')
    final_white, faces = nib.freesurfer.read_geometry(str(args.case / f'candidate/surf/{args.hemisphere}.white'))
    first_seen = np.full(len(pairs), '', dtype=object)
    records = []
    def observe(name, coordinates):
        first, second = coordinates.astype(np.float64)[faces[pairs[:, 0]]], coordinates.astype(np.float64)[faces[pairs[:, 1]]]
        relation, _, _, _ = relations(first, second)
        proper = relation == 1
        first_seen[(first_seen == '') & proper] = name
        records.append(dict(state=name, final_white_pairs_currently_proper=int(proper.sum())))
    for stage, suffix in [('preaparc', 'orig'), ('white', 'white.preaparc')]:
        observation = args.case / f'{stage}-observation'
        events = json.loads((observation / 'events.json').read_text())
        coordinates, source_faces = nib.freesurfer.read_geometry(str(args.case / f'candidate/surf/{args.hemisphere}.{suffix}'))
        if not np.array_equal(source_faces, faces):
            raise ValueError('upstream ordered faces changed')
        observe(stage + '_input', coordinates)
        for event in events:
            if event['kind'] in ['step', 'cleanup']:
                with np.load(observation / event['file']) as arrays:
                    if event['kind'] == 'step':
                        observe(f'{stage}_accepted_step_{event["step"]}_pass_{event["pass_index"]}', arrays['accepted'])
                    else:
                        observe(stage + '_after_cleanup', arrays['after'])
    observe('final_white_written', final_white)
    if np.any(first_seen == ''):
        raise AssertionError('enumerated final-white pair absent from written final white')
    report = dict(scope='first occurrence only for the enumerated final-white self pair set in observed accepted preaparc/white states',
                  final_white_self_pairs=len(pairs), first_observed_states=dict(Counter(map(str, first_seen))),
                  trajectory=records, transient_nonterminal_pairs='not_enumerated',
                  rejected_trials='not_assessed', earlier_reconstruction_stages='not_assessed',
                  causal_attribution='not_assessed', quality_gates_changed=False,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), status='complete_read_only_pair_trace')
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
