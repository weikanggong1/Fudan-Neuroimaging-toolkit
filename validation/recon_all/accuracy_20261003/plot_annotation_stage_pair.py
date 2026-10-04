#!/usr/bin/env python3
"""CPU-only real smoothwm/aparc baseline, candidate and difference brain figure."""
import argparse
import datetime
import json
from pathlib import Path
import sys

from compare_annotation_stage_pair import load_arm, receipt


def plot_pair(baseline_path, candidate_path, output, *, dpi=150, formats=('png', 'svg')):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    baseline, bg, bi, bm, ba, bf = load_arm(baseline_path)
    candidate, cg, ci, cm, ca, cf = load_arm(candidate_path)
    if bi != ci:
        raise ValueError('15 relative input SHA bindings differ')
    if baseline['threads'] != candidate['threads'] or baseline['actual_parent_precision'] != candidate['actual_parent_precision']:
        raise ValueError('thread/precision receipts differ')
    # Check all bound smoothwm and sphere.reg geometries before any rendering.
    for key in bm:
        if not np.array_equal(bm[key][0], cm[key][0]) or not np.array_equal(bm[key][1], cm[key][1]):
            raise ValueError('mesh coordinates or ordered faces differ: ' + key)
    report = {'status': 'rendering', 'cpu_only': True, 'data_type': 'real FNIT stage smoothwm and aparc',
              'official_equivalence': 'not_assessed', 'space': 'surface RAS, millimetres',
              'face_color_rule': 'face first vertex label_table_indices maps to that arm color_table RGB/255; -1 is grey; no face or vertex sampling',
              'difference_rule': 'original packed ID differs at same ordered vertex; face colored by first vertex difference, red changed / grey unchanged',
              'renderer': 'Matplotlib Poly3DCollection, zsort=average painter depth ordering, full ordered faces; SVG brain polygons rasterized, text vector',
              'view': {'lh': {'elevation_degrees': 0, 'azimuth_degrees': 180},
                       'rh': {'elevation_degrees': 0, 'azimuth_degrees': 0}},
              'baseline': {'stage': bf['stage'], 'runtime': baseline['runtime']},
              'candidate': {'stage': cf['stage'], 'runtime': candidate['runtime']},
              'input_bindings': [{'kind': key[0], 'relative': key[1], 'size_bytes': bi[key][0], 'sha256': bi[key][1]} for key in sorted(bi)],
              'hemispheres': {}, 'outputs': []}
    figure = plt.figure(figsize=(15, 9), facecolor='white')
    for row, hemi in enumerate(('lh', 'rh')):
        coords, faces = bm[f'{hemi}.smoothwm']
        key = f'{hemi}.aparc'
        left, right = ba[key], ca[key]
        if not np.array_equal(left['color_table'], right['color_table']) or not np.array_equal(left['names'], right['names']):
            raise ValueError('aparc named color-table semantics differ: ' + hemi)
        difference = left['original_annotation_ids'] != right['original_annotation_ids']
        changed = int(np.count_nonzero(difference))
        center = (coords.min(axis=0) + coords.max(axis=0)) / 2
        radius = float(np.max(np.ptp(coords, axis=0)) / 2 * 1.04)
        triangles = coords[faces]  # All original ordered faces; no geometry changes.
        first_vertices = faces[:, 0]
        report['hemispheres'][hemi] = {'vertices': len(coords), 'faces': len(faces), 'different_vertices': changed,
            'different_table_index_vertices': int(np.count_nonzero(left['label_table_indices'] != right['label_table_indices'])),
            'mesh_baseline': baseline['copied_meshes_after'][f'{hemi}.smoothwm'],
            'mesh_candidate': candidate['copied_meshes_after'][f'{hemi}.smoothwm'],
            'baseline_annotation': bf['annotations'][key], 'candidate_annotation': cf['annotations'][key],
            'annotation_file_bytes_same': bf['annotations'][key]['annotation']['sha256'] == cf['annotations'][key]['annotation']['sha256']}
        for column, (name, values) in enumerate((('Baseline', left), ('Candidate', right), ('Difference', None))):
            axes = figure.add_subplot(2, 3, row * 3 + column + 1, projection='3d')
            if values is not None:
                labels = values['label_table_indices'][first_vertices]
                if np.any(labels < -1) or np.any(labels >= len(values['color_table'])):
                    raise ValueError('table row index out of range')
                colors = np.full((len(faces), 3), 0.72, dtype=float)
                valid = labels >= 0
                colors[valid] = values['color_table'][labels[valid], :3] / 255.0
            else:
                colors = np.full((len(faces), 3), 0.80, dtype=float)
                colors[difference[first_vertices]] = (0.85, 0.08, 0.06)
            collection = Poly3DCollection(triangles, facecolors=colors, edgecolors='none', linewidths=0, zsort='average')
            collection.set_rasterized(True)
            axes.add_collection3d(collection)
            axes.set_xlim(center[0] - radius, center[0] + radius)
            axes.set_ylim(center[1] - radius, center[1] + radius)
            axes.set_zlim(center[2] - radius, center[2] + radius)
            axes.set_box_aspect((1, 1, 1))
            axes.set_proj_type('ortho')
            axes.view_init(elev=0, azim=180 if hemi == 'lh' else 0)
            axes.set_axis_off()
            axes.set_title(f'{hemi.upper()} lateral | {name}', fontsize=13)
            if values is None:
                caption = f'{changed:,} / {len(coords):,} vertices differ'
                if changed == 0:
                    caption += '\n0 label differences (no contrast amplification)'
                axes.text2D(0.5, -0.015, caption, ha='center', va='top', transform=axes.transAxes, fontsize=10)
            else:
                axes.text2D(0.5, -0.015, f'{len(coords):,} vertices; {len(faces):,} original faces', ha='center', va='top', transform=axes.transAxes, fontsize=10)
    figure.suptitle('Real smoothwm aparc | surface RAS (mm) | all original faces', fontsize=15)
    figure.text(0.5, 0.015, 'Difference: red = changed first-vertex packed ID; grey = unchanged. File-byte differences are not label differences.', ha='center', fontsize=9)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.08, top=0.92, wspace=0.01, hspace=0.19)
    for fmt in formats:
        path = output / ('aparc.stage_pair.' + fmt)
        figure.savefig(path, dpi=dpi, facecolor='white')
        report['outputs'].append(receipt(path))
    plt.close(figure)
    report['status'] = 'plot_complete'
    report['program'] = receipt(__file__)
    report['comparison_program'] = receipt(Path(__file__).with_name('compare_annotation_stage_pair.py'))
    report['matplotlib_version'] = matplotlib.__version__
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-stage', type=Path, required=True)
    parser.add_argument('--candidate-stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--formats', choices=('png', 'svg'), nargs='+', default=['png', 'svg'])
    args = parser.parse_args()
    output = args.output.resolve()
    if args.dpi <= 0:
        parser.error('dpi must be positive')
    for original in (args.baseline_stage.resolve().parent, args.candidate_stage.resolve().parent):
        if output == original or output in original.parents or original in output.parents:
            parser.error('output must be a new directory separate from both stage directories')
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        report = plot_pair(args.baseline_stage, args.candidate_stage, output, dpi=args.dpi, formats=tuple(dict.fromkeys(args.formats)))
        code = 0
    except Exception as error:
        report = {'status': 'plot_invalid', 'error_type': type(error).__name__, 'error': str(error),
                  'official_equivalence': 'not_assessed', 'cpu_only': True}
        code = 2
    report.update(started_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (output / 'aparc.plot.receipt.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': report['status'], 'receipt': str(output / 'aparc.plot.receipt.json')}))
    return code


if __name__ == '__main__':
    sys.exit(main())
