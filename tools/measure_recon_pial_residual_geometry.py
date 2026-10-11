"""只读真实末态：完整 proper 交线尺度、固定面参与及输入 white 来源。"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import nibabel as nib
import numpy as np
from fnit.recon_all.place_surface_quality import proper_pairs
from fnit.recon_all.place_surface_final_cleanup import pial_cleanup_fixed_vertices
from fnit.recon_all.white_pial_crossing_python import _strict_relations, _exact_relation, _rational_triangle


def relations(first, second):
    relation, length, da, db, ambiguous, _ = _strict_relations(first, second)
    for slot in np.flatnonzero(ambiguous):
        relation[slot], length[slot], da[slot], db[slot] = _exact_relation(
            _rational_triangle(first[slot]), _rational_triangle(second[slot]))
    return relation, length, da, db


def statistics(values):
    return dict(count=len(values), minimum=float(values.min()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                p99=float(np.percentile(values, 99)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def measure(vertices, white, faces, masks, cross, output):
    pairs, enumeration = proper_pairs(first_vertices=vertices, faces=faces,
                                      second_vertices=white if cross else None)
    pair_array = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    np.save(output, pair_array)  # Private face IDs remain on the server.
    left, right = pair_array.T
    triangles = vertices.astype(np.float64)[faces]
    target = white.astype(np.float64)[faces] if cross else triangles
    relation, length, da, db = relations(triangles[left], target[right])
    if not np.all(relation == 1):
        raise AssertionError('measured pairs disagree with complete proper enumeration')
    upstream, _, _, _ = relations(white.astype(np.float64)[faces[left]],
                                  white.astype(np.float64)[faces[right]])
    fixed_counts = {}
    for name, mask in masks.items():
        face_count = mask[faces].sum(axis=1)
        labels = np.asarray(['fully_free', 'mixed', 'fully_fixed'])[np.where(face_count == 0, 0, np.where(face_count == 3, 2, 1))]
        keys = labels[left] if cross else np.char.add(np.char.add(labels[left], '__'), labels[right])
        involved = np.unique(faces[np.unique(left if cross else pair_array)])
        fixed_counts[name] = dict(face_pair_categories=dict(Counter(map(str, keys))),
                                 involved_fixed_vertices=int(mask[involved].sum()),
                                 involved_free_vertices=int((~mask[involved]).sum()))
    return dict(enumeration=enumeration, intersection_length_mm=statistics(length),
                first_plane_straddle_depth_mm=statistics(da), second_plane_straddle_depth_mm=statistics(db),
                already_proper_at_same_face_pair_in_input_white=int((upstream == 1).sum()),
                newly_proper_relative_to_same_input_white_pair=int((upstream != 1).sum()),
                fixed_participation=fixed_counts,
                proper_pairs_reclassified_as_legal_contacts=0, legal_contact_count='not_assessed',
                contact_semantics='proper is positive-length open-plane crossing; contact is a separate relation, never a proper exemption')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--hemisphere', choices=['lh', 'rh'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    observation = args.case / 'pial-observation'
    events = json.loads((observation / 'events.json').read_text())
    step = next(event for event in reversed(events) if event['kind'] == 'step')
    cleanup = next(event for event in reversed(events) if event['kind'] == 'cleanup')
    with np.load(observation / step['file']) as arrays:
        ripped = arrays['ripped'].astype(bool)
    with np.load(observation / cleanup['file']) as arrays:
        baseline = arrays['after'].copy()
    white, faces = nib.freesurfer.read_geometry(str(args.case / f'candidate/surf/{args.hemisphere}.white'))
    white = white.astype(np.float32)
    cortex = nib.freesurfer.read_label(str(args.case / f'candidate/label/{args.hemisphere}.cortex.label'))
    fixed = pial_cleanup_fixed_vertices(ripped=ripped, cortex_vertices=cortex)
    candidate = np.load(args.replay / 'candidate_cleanup.private.npy')
    report = dict(schema_version=1, scope='read-only frozen terminal and own cleanup replay; no optimization repeated',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  depth_semantics='minimum of positive and negative corner distances to the other triangle plane; not minimum repair displacement',
                  strict_gate_changed=False, measurement_cutoff_mm=None, status='running')
    def save():
        pending = args.output / 'report.pending'
        pending.write_text(json.dumps(report, indent=2) + '\n')
        pending.replace(args.output / 'report.json')
    save()
    masks = dict(original_rip=ripped, cleanup_fixed_union=fixed)
    for name, vertices in [('input_white', white), ('baseline', baseline), ('fixed_cleanup', candidate)]:
        report[name] = {'coordinates_sha256': hashlib.sha256(np.ascontiguousarray(vertices).tobytes()).hexdigest(), 'coordinates_dtype': str(vertices.dtype), 'coordinates_shape': list(vertices.shape)}
        for cross in [False, True] if name != 'input_white' else [False]:
            kind = 'white_pial' if cross else 'self'
            report[name][kind] = measure(vertices, white, faces, masks, cross,
                                        args.output / f'{name}-{kind}.pairs.private.npy')
            save()
    report['status'] = 'complete_measurement_not_quality_approval'
    save()


if __name__ == '__main__':
    main()
