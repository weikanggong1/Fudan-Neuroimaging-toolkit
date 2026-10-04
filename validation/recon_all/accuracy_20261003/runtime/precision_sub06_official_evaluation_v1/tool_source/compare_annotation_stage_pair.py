#!/usr/bin/env python3
"""Read-only CPU comparison of two bound annotation stage receipts."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import struct

HEMIS = ('lh', 'rh')
ATLASES = ('aparc', 'aparc.a2009s', 'aparc.DKTatlas')
KEYS = {f'{h}.{a}' for h in HEMIS for a in ATLASES}
INPUTS = {('subject', 'mri/aseg.presurf.mgz')} | {
    ('subject', f'{folder}/{h}.{name}') for h in HEMIS
    for folder, name in [('surf', 'smoothwm'), ('surf', 'sphere.reg'), ('label', 'cortex.label')]
} | {('asset', f'average/{h}.{a}.atlas.acfb40.noaparc.i12.2016-08-02.gcs')
     for h in HEMIS for a in ('DKaparc', 'CDaparc', 'DKTaparc')} | {
    ('asset', 'lib/bem/ic4.tri'), ('asset', 'lib/bem/ic7.tri')}
FIELDS = ('original_annotation_ids', 'label_table_indices', 'vertex_indices', 'color_table', 'names')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def receipt(path):
    p = Path(path).resolve()
    return {'path': str(p), 'size_bytes': p.stat().st_size, 'sha256': sha(p)}


def verified(row):
    actual = receipt(row['path'])
    if actual['size_bytes'] != row['size_bytes'] or actual['sha256'] != row['sha256']:
        raise ValueError('file receipt mismatch: ' + row['path'])
    return actual


def bindings(rows):
    result = {}
    for row in rows:
        key = (row['kind'], row['relative'])
        if key in result:
            raise ValueError('duplicate input binding: ' + str(key))
        if len(row['sha256']) != 64:
            raise ValueError('invalid input SHA')
        result[key] = (row['size_bytes'], row['sha256'])
    if set(result) != INPUTS:
        raise ValueError('stage must bind exact 7 subject + 8 asset inputs')
    return result


def array_sha(values, dtype):
    import numpy as np
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order='C')).hexdigest()


def load_arm(path):
    import numpy as np
    import nibabel.freesurfer.io as fsio
    stage = json.loads(Path(path).read_text())
    if stage.get('status') != 'stage_complete' or not stage.get('output_complete'):
        raise ValueError('stage is not complete: ' + str(path))
    if set(stage['annotations']) != KEYS or not stage.get('input_meshes_stable'):
        raise ValueError('six annotations and stable input meshes required')
    initial = bindings(stage['original_inputs_before'])
    for name in ('original_inputs_after', 'copied_inputs_after'):
        if bindings(stage[name]) != initial:
            raise ValueError('input SHA changed during stage: ' + name)
    precision = stage['actual_parent_precision']
    if precision != {'matmul_tf32': True, 'cudnn_tf32': True, 'autocast_enabled': False}:
        raise ValueError('actual parent TF32/FP32 policy not confirmed')
    group = json.loads(Path(verified(stage['group_report'])['path']).read_text())
    if group.get('operation') != 'annotation':
        raise ValueError('group operation must be annotation')
    if set(group.get('workers', {})) != set(HEMIS):
        raise ValueError('both hemisphere worker receipts required')
    if stage['threads'] != {'total': group['total_thread_budget'], 'workers': group['workers_requested'], 'per_worker': group['worker_threads']}:
        raise ValueError('stage/group thread budget differs')
    for worker in group['workers'].values():
        if worker.get('status') != 'complete' or any(worker.get('precision', {}).get(k) is not True for k in ('matmul_tf32', 'cudnn_tf32')):
            raise ValueError('worker completion/TF32 receipt missing')
        budget = worker['thread_budget']
        for library in ('torch', 'numba'):
            if budget[library]['effective'] != group['worker_threads']:
                raise ValueError('actual worker thread budget mismatch')
    if group.get('status') != 'complete':
        raise ValueError('hemisphere group not complete')
    meshes = {}
    files = {'stage': receipt(path), 'group': stage['group_report'], 'annotations': {}}
    for key, row in stage['copied_meshes_after'].items():
        if key not in {f'{h}.{n}' for h in HEMIS for n in ('smoothwm', 'sphere.reg')}:
            raise ValueError('unexpected mesh key')
        verified(row['file'])
        coords, faces = fsio.read_geometry(row['file']['path'])
        if len(coords) != row['vertices'] or len(faces) != row['faces']:
            raise ValueError('mesh shape receipt mismatch')
        if array_sha(coords, '<f8') != row['coordinate_array_sha256_le_f64'] or array_sha(faces, '<i8') != row['ordered_faces_sha256_le_i64']:
            raise ValueError('mesh array SHA mismatch')
        for name in ('copied_meshes_before', 'original_meshes_before'):
            before = stage[name][key]
            for field in ('coordinate_array_sha256_le_f64', 'ordered_faces_sha256_le_i64'):
                if before[field] != row[field]:
                    raise ValueError('mesh changed during stage')
        meshes[key] = (coords, faces)
    if len(meshes) != 4:
        raise ValueError('four mesh receipts required')
    arrays = {}
    for key, row in stage['annotations'].items():
        verified(row['file']); verified(row['semantic_vectors'])
        with np.load(row['semantic_vectors']['path'], allow_pickle=False) as loaded:
            if set(loaded.files) != set(FIELDS):
                raise ValueError('unexpected semantic NPZ fields')
            values = {name: loaded[name].copy() for name in FIELDS}
        ids, indices, vertices, ctab, names = (values[x] for x in FIELDS)
        n = len(meshes[key.split('.')[0] + '.smoothwm'][0])
        if any(v.shape != (n,) or v.dtype.kind not in 'iu' for v in (ids, indices, vertices)):
            raise ValueError('label vectors must be integer ordered vertex vectors')
        if not np.array_equal(vertices, np.arange(n)) or n != row['vertex_count']:
            raise ValueError('ordered vertex index mismatch')
        if ctab.ndim != 2 or ctab.shape[1] != 5 or ctab.dtype.kind not in 'iu' or names.ndim != 1 or len(names) != len(ctab) or names.dtype.kind != 'S':
            raise ValueError('invalid color table/names')
        for field, value in [('label_original_ids_sha256_le_i64', ids), ('label_table_indices_sha256_le_i64', indices), ('color_table_sha256_le_i64', ctab)]:
            if array_sha(value, '<i8') != row[field]:
                raise ValueError('semantic array SHA mismatch: ' + field)
        if hashlib.sha256(json.dumps([bytes(x).hex() for x in names]).encode()).hexdigest() != row['color_table_names_sha256']:
            raise ValueError('names SHA mismatch')
        # Verify raw stored vertex ordering, independently of the semantic reader.
        with Path(row['file']['path']).open('rb') as stream:
            raw_count = struct.unpack('>i', stream.read(4))[0]
            if raw_count != n:
                raise ValueError('raw annotation vertex count mismatch')
            raw_pairs = np.frombuffer(stream.read(n * 8), dtype='>i4').reshape(n, 2)
        if not np.array_equal(raw_pairs[:, 0], vertices) or not np.array_equal(raw_pairs[:, 1], ids):
            raise ValueError('raw annotation vertex order/IDs differ from NPZ')
        # Independently read bound .annot to ensure NPZ still represents that file.
        actual_ids, actual_ctab, actual_names = fsio.read_annot(row['file']['path'], orig_ids=True)
        actual_indices, _, _ = fsio.read_annot(row['file']['path'], orig_ids=False)
        if not all(np.array_equal(a, b) for a, b in [(ids, actual_ids), (indices, actual_indices), (ctab, actual_ctab), (names, np.asarray(actual_names, dtype='S'))]):
            raise ValueError('NPZ differs from bound annotation file')
        arrays[key] = values
        files['annotations'][key] = {'annotation': row['file'], 'npz': row['semantic_vectors']}
    return stage, group, initial, meshes, arrays, files


def dice_rows(left, right, ctab, names):
    import numpy as np
    labels = np.union1d(left, right)
    rows = []
    for label in labels:
        a, b = left == label, right == label
        ca, cb = int(a.sum()), int(b.sum())
        intersection = int(np.count_nonzero(a & b))
        matching = np.flatnonzero(ctab[:, 4] == label)
        rows.append({'packed_original_id': int(label), 'baseline_vertices': ca, 'candidate_vertices': cb,
                     'intersection_vertices': intersection, 'dice': 2 * intersection / (ca + cb),
                     'table_rows': [int(x) for x in matching],
                     'names_hex': [bytes(names[x]).hex() for x in matching]})
    scores = [r['dice'] for r in rows]
    return {'basis': 'packed original annotation ID equality at the same ordered vertex; not anatomical numeric class IDs',
            'summary_scope': 'union of labels present in either arm; absent in both excluded',
            'min': float(np.min(scores)), 'p05': float(np.percentile(scores, 5)),
            'median': float(np.median(scores)), 'labels': rows}


def compare(baseline, candidate):
    import numpy as np
    b, bg, bi, bm, ba, bf = load_arm(baseline)
    c, cg, ci, cm, ca, cf = load_arm(candidate)
    gates = {'input15_relative_sha_same': bi == ci,
             'thread_budget_same': b['threads'] == c['threads'] and
             bg['total_thread_budget'] == cg['total_thread_budget'] and
             bg['worker_threads'] == cg['worker_threads'] and
             bg['workers_requested'] == cg['workers_requested'],
             'actual_precision_same': b['actual_parent_precision'] == c['actual_parent_precision'],
             'worker_precision_same': all(bg['workers'][h]['precision'] == cg['workers'][h]['precision'] for h in HEMIS),
             'precision_policy_same': b['precision_policy'] == c['precision_policy'],
             'device_same': b['device'] == c['device'] and b['gpu_uuid_declared'] == c['gpu_uuid_declared'],
             'runtime_thread_counts_same': all(b['runtime'][x] == c['runtime'][x] for x in ('torch_threads', 'torch_interop_threads'))}
    mesh = {}
    for key in bm:
        mesh[key] = {'coordinates_exact': bool(np.array_equal(bm[key][0], cm[key][0])),
                     'ordered_faces_exact': bool(np.array_equal(bm[key][1], cm[key][1])),
                     'file_bytes_same': b['copied_meshes_after'][key]['file']['sha256'] == c['copied_meshes_after'][key]['file']['sha256']}
    geometry_same = all(v['coordinates_exact'] and v['ordered_faces_exact'] for v in mesh.values())
    annotations = {}
    for key in sorted(KEYS):
        x, y = ba[key], ca[key]
        row = {'annotation_file_bytes_same': bf['annotations'][key]['annotation']['sha256'] == cf['annotations'][key]['annotation']['sha256'],
               'npz_file_bytes_same': bf['annotations'][key]['npz']['sha256'] == cf['annotations'][key]['npz']['sha256'],
               'arrays_exact': {name: (bool(np.array_equal(x[name], y[name])) if geometry_same or name in ('color_table', 'names') else None) for name in FIELDS},
               'ordered_vertex_comparison_performed': geometry_same}
        row['array_differences'] = {}
        for name in FIELDS:
            if row['arrays_exact'][name] is None or row['arrays_exact'][name]:
                continue
            left, right = x[name], y[name]
            if left.shape != right.shape:
                row['array_differences'][name] = {'baseline_shape': list(left.shape), 'candidate_shape': list(right.shape)}
                continue
            positions = np.argwhere(left != right)
            encode = (lambda v: bytes(v).hex()) if name == 'names' else (lambda v: int(v))
            row['array_differences'][name] = {'count': len(positions), 'values_encoding': 'bytes_hex' if name == 'names' else 'integer',
                'entries': [{'index': [int(i) for i in pos], 'baseline': encode(left[tuple(pos)]), 'candidate': encode(right[tuple(pos)])} for pos in positions]}
        if geometry_same:
            row['different_original_id_vertices'] = int(np.count_nonzero(x['original_annotation_ids'] != y['original_annotation_ids']))
            row['different_table_index_vertices'] = int(np.count_nonzero(x['label_table_indices'] != y['label_table_indices']))
            row['dice_original_ids'] = dice_rows(x['original_annotation_ids'], y['original_annotation_ids'], x['color_table'], x['names'])
            row['dice_named_semantics_comparable'] = row['arrays_exact']['color_table'] and row['arrays_exact']['names']
        else:
            row['comparison_skipped_reason'] = 'coordinate or ordered topology differs; vertex indices are not interchangeable'
        annotations[key] = row
    stages = {}
    for name, stage, group, files in [('baseline', b, bg, bf), ('candidate', c, cg, cf)]:
        stages[name] = {'files': files, 'runtime': stage['runtime'], 'timings': stage['timings'], 'threads': stage['threads'],
                        'actual_parent_precision': stage['actual_parent_precision'],
                        'source_provenance': {k: v for k, v in stage.items() if 'source' in k or k in ('script', 'source', 'group_call_arguments')},
                        'group_threads': {k: group[k] for k in ('total_thread_budget', 'worker_threads', 'workers_requested')},
                        'worker_reports': group['workers']}
    exact = all(gates.values()) and geometry_same and all(all(a['arrays_exact'].values()) for a in annotations.values())
    return {'status': 'comparison_complete', 'gates': gates, 'meshes': mesh, 'annotations': annotations,
            'exact_regression_pass': exact, 'official_equivalence': 'not_assessed', 'arms': stages,
            'input_bindings': [{'kind': k[0], 'relative': k[1], 'baseline': bi[k], 'candidate': ci[k]} for k in sorted(bi)],
            'scope': 'same-input annotation stage regression; no whole-case or official equivalence claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-stage', required=True, type=Path)
    parser.add_argument('--candidate-stage', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    for original in (args.baseline_stage.resolve().parent, args.candidate_stage.resolve().parent):
        if output == original or output in original.parents or original in output.parents:
            parser.error('output must be a new directory separate from both stage directories')
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        report = compare(args.baseline_stage, args.candidate_stage)
        code = 0 if report['exact_regression_pass'] else 1
    except Exception as error:
        report = {'status': 'comparison_invalid', 'error_type': type(error).__name__, 'error': str(error),
                  'exact_regression_pass': False, 'official_equivalence': 'not_assessed'}
        code = 2
    report.update(started_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  comparison_script=receipt(__file__), baseline_stage=str(args.baseline_stage.resolve()),
                  candidate_stage=str(args.candidate_stage.resolve()), cpu_only=True)
    (output / 'annotation.pair.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (output / 'README.md').write_text('# Annotation stage 配对比较\n\n'
        + f"状态：{report['status']}；exact_regression_pass={report['exact_regression_pass']}。\n\n"
        + '仅比较绑定输入的 annotation stage；官方整体等效 not_assessed。详见 annotation.pair.json。\n')
    print(json.dumps({'status': report['status'], 'exact_regression_pass': report['exact_regression_pass'], 'report': str(output / 'annotation.pair.json')}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
