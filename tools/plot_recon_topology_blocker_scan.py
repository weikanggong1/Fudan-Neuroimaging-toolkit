"""完成的有序阻挡扫描阶段计时和自产候选在自身MRI上的叠加。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*, input_run, frozen_source, metrics, brain, output):
    """五具名Path→严格完成/源码/产物绑定后的新PNG，不运行GA。

    input_run是完成的第四候选pair目录，frozen_source是其真实源码，
    metrics为安全白名单，brain为同链自身MRI，output为不存在的新PNG。
    验证全部source payload、实际MRI变换模块路径/SHA及两个warm产物
    后绘图。坐标是自产运动终态，不是原程序误差或最终全脑QC。
    """
    run, source, metrics, brain, output = map(Path, (input_run, frozen_source, metrics, brain, output))
    value = json.loads(metrics.read_text())
    query_half = value['status'] == 'complete_fixed_fourth_candidate_query_half_pair'
    variants = ('baseline', 'query_half') if query_half else ('python', 'numba')
    labels = ('Numba baseline', 'Query reuse') if query_half else ('Python loop', 'Numba loop')
    source_prefix = 'query_half/src/' if query_half else 'src/'
    module_root = source/'query_half/src' if query_half else source/'src'
    receipt = json.loads(metrics.with_name('EXPORT_RECEIPT.json').read_text())
    manifest_path = source/'SOURCE_MANIFEST.private.json'
    manifest = json.loads(manifest_path.read_text())
    if (output.exists() or output.suffix.lower() != '.png'
            or value['status'] not in ('complete_fixed_fourth_candidate_numba_blocker_pair',
                                       'complete_fixed_fourth_candidate_query_half_pair')
            or not value['all_controls_and_fields_exact']
            or receipt['source_manifest_sha256'] != sha(manifest_path)
            or value['source_manifest_sha256'] != sha(manifest_path)
            or receipt['actual_private_report_sha256'] != sha(run/'report.private.json')
            or receipt['actual_complete_sha256'] != sha(run/'COMPLETE.json')
            or receipt['actual_exit_sha256'] != sha(run/'EXIT.json')
            or receipt['payload']['metrics']['sha256'] != sha(metrics)
            or receipt['payload']['metrics']['size'] != metrics.stat().st_size
            or not receipt['numeric_values_and_types_preserved']):
        raise ValueError('same completed safe pair and frozen source required')
    for name, item in manifest['payload'].items():
        path = source/name
        if sha(path) != item['sha256'] or path.stat().st_size != item['size']:
            raise ValueError('actual frozen source payload changed')
    if sha(brain) != value['input_sha256']['brain']:
        raise ValueError('same own MRI required')
    sys.path.insert(0, str(module_root))
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(module_root/'fnit/recon_all'))
    import numpy as np
    import nibabel as nib
    from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
    actual = Path(sys.modules[source_topology_ras_to_voxel.__module__].__file__).resolve()
    expected = (module_root/'fnit/recon_all/topology_edge_mri_score.py').resolve()
    if actual != expected or sha(actual) != manifest['payload'][
            source_prefix+'fnit/recon_all/topology_edge_mri_score.py']['sha256']:
        raise ValueError('actual loaded MRI transformation source differs')
    children = value['children']
    points = []
    for backend, child in zip(variants, children):
        warm = next(row for row in child['records'] if row['run'] == 'warm')
        path = run/backend/'candidate/warm.private.npz'
        if sha(path) != warm['candidate_sha256']:
            raise ValueError('same completed candidate artifact required')
        with np.load(path, allow_pickle=False) as state:
            xyz = state['original_vertices'][state['selected_vertices']].copy()
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
            raise ValueError('finite real selected candidate coordinates required')
        points.append(xyz)
    if points[0].dtype != points[1].dtype or points[0].tobytes() != points[1].tobytes():
        raise ValueError('paired selected coordinates differ')
    image = nib.load(brain)
    volume = np.asarray(image.dataobj)
    transform = source_topology_ras_to_voxel(volume_header=image.header)
    voxels = [(np.c_[xyz, np.ones(len(xyz))]@transform.T)[:, :3] for xyz in points]
    plane = int(np.clip(np.rint(np.median(voxels[0][:, 0])), 0, volume.shape[0]-1))
    low = np.maximum(np.floor(np.min(voxels[0][:, 1:], axis=0))-12, 0).astype(int)
    high = np.minimum(np.ceil(np.max(voxels[0][:, 1:], axis=0))+13, np.array(volume.shape[1:])).astype(int)
    if np.any(high <= low):
        raise ValueError('selected candidate MRI slice is outside volume')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    warm = [next(row for row in child['records'] if row['run'] == 'warm') for child in children]
    names = ('cycle', 'retess', 'detect')
    keys = ('complete_cycle_seconds', 'retessellation_seconds', 'face_detection_seconds')
    x = np.arange(3); width = .36
    for offset, row, label in zip((-.18, .18), warm, labels):
        axes[0].bar(x+offset, [row[key] for key in keys], width, label=label)
    axes[0].set(xticks=x, xticklabels=names, ylabel='seconds', title='Actual warm stage')
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, [row['complete_fresh_cli_seconds'] for row in value['process']])
    axes[1].set(ylabel='seconds', title='Fresh CLI: loading + preparation + 3 calls')
    for axis, voxel, label in zip(axes[2:], voxels, labels):
        axis.imshow(volume[plane, low[0]:high[0], low[1]:high[1]].T, cmap='gray', origin='lower',
            extent=[low[0], high[0], low[1], high[1]])
        near = np.abs(voxel[:, 0]-plane) <= 2
        axis.scatter(voxel[near, 1], voxel[near, 2], s=5, c='#ff6633', alpha=.8)
        axis.set(xlabel='voxel Y', ylabel='voxel Z', title=label+': own candidate on own MRI')
    figure.suptitle('Same fixed fourth candidate; 72 fields and full ordered predicate sequence equal', fontsize=11)
    figure.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160); plt.close(figure)
    return dict(status='complete_verified_blocker_stage_plot',
        source_manifest_sha256=sha(manifest_path), metrics_sha256=sha(metrics),
        export_receipt_sha256=sha(metrics.with_name('EXPORT_RECEIPT.json')),
        actual_mri_transform_sha256=sha(actual), brain_sha256=sha(brain), output_sha256=sha(output),
        plotter_sha256=sha(__file__), selected_vertices=len(points[0]), mri_slice_voxel_x=plane,
        own_coordinates_not_reference_error=True, final_mesh_quality='not_assessed', whole_GA=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('fourth', 'first'), default='fourth')
    for name in ('input-run', 'frozen-source', 'metrics', 'brain', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    options = vars(parser.parse_args()); stage = options.pop('stage')
    function = run if stage == 'fourth' else run_first
    print(json.dumps(function(**options), indent=2))


def run_first(*, input_run, frozen_source, metrics, brain, output):
    """完成的默认首缺陷→真实完整墙钟/496分数/自身最终状态脑图。

    五个具名路径与第四图相同。须完整first完成白名单，验证全部manifest/
    产物及实际MRI变换模块。脑图展示原first-selected域在自产最终状态
    的坐标，非全部九缺陷、原软件误差或最终网格QC。
    """
    run, source, metrics, brain, output = map(Path, (input_run, frozen_source, metrics, brain, output))
    value = json.loads(metrics.read_text())
    receipt = json.loads(metrics.with_name('EXPORT_RECEIPT.json').read_text())
    manifest_path = source/'SOURCE_MANIFEST.private.json'
    manifest = json.loads(manifest_path.read_text())
    if (output.exists() or output.suffix.lower() != '.png'
            or value['status'] != 'complete_default_first_GA_blocker_scan' or not value['strict_pair_passed']
            or value['retessellation_calls'] != 497 or value['search_evaluations'] != 495
            or value['final_scores'] != 1 or value['generation_trace_count'] != 29
            or value['execution']['exit_code'] != 0
            or value['source_manifest_sha256'] != sha(manifest_path)
            or receipt['source_manifest_sha256'] != sha(manifest_path)
            or receipt['actual_private_report_sha256'] != sha(run/'candidate/blocker_scan.json')
            or receipt['actual_complete_sha256'] != sha(run/'candidate/BLOCKER_SCAN_COMPLETE.json')
            or receipt['actual_exit_sha256'] != sha(run/'EXIT.json')
            or receipt['payload']['metrics']['sha256'] != sha(metrics)
            or receipt['payload']['metrics']['size'] != metrics.stat().st_size
            or not receipt['numeric_values_and_types_preserved']):
        raise ValueError('same completed full first GA whitelist and source required')
    for name, item in manifest['payload'].items():
        path = source/name
        if sha(path) != item['sha256'] or path.stat().st_size != item['size']:
            raise ValueError('actual frozen source payload changed')
    if sha(brain) != value['optimization']['input_sha256']['brain']:
        raise ValueError('same own MRI required')
    sys.path.insert(0, str(source/'src'))
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(source/'src/fnit/recon_all'))
    import numpy as np
    import nibabel as nib
    from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
    actual = Path(sys.modules[source_topology_ras_to_voxel.__module__].__file__).resolve()
    expected = (source/'src/fnit/recon_all/topology_edge_mri_score.py').resolve()
    if actual != expected or sha(actual) != manifest['payload'][
            'src/fnit/recon_all/topology_edge_mri_score.py']['sha256']:
        raise ValueError('actual loaded MRI transformation source differs')
    for name, digest in value['outputs_sha256'].items():
        filename = name+'.private.npz' if name != 'search_control.json' else name
        if sha(run/'candidate'/filename) != digest:
            raise ValueError('completed actual output changed')
    scores = [np.asarray([row['fitness'] for row in values], np.float64)
        for values in (value['baseline_candidate_timings'], value['candidate_timings'])]
    if scores[0].shape != (496,) or scores[0].tobytes() != scores[1].tobytes():
        raise ValueError('all 495 search and final double scores required')
    with np.load(run/'candidate/first_candidate.private.npz', allow_pickle=False) as state:
        selected = state['selected_vertices'].copy()
    with np.load(run/'candidate/final_state.private.npz', allow_pickle=False) as state:
        xyz = state['original_vertices'][selected].copy()
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('finite real final-state coordinates required')
    image = nib.load(brain); volume = np.asarray(image.dataobj)
    transform = source_topology_ras_to_voxel(volume_header=image.header)
    points = (np.c_[xyz, np.ones(len(xyz))]@transform.T)[:, :3]
    plane = int(np.clip(np.rint(np.median(points[:, 0])), 0, volume.shape[0]-1))
    low = np.maximum(np.floor(np.min(points[:, 1:], axis=0))-12, 0).astype(int)
    high = np.minimum(np.ceil(np.max(points[:, 1:], axis=0))+13, np.array(volume.shape[1:])).astype(int)
    if np.any(high <= low): raise ValueError('selected final-state MRI slice is outside volume')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    old = value['optimization']['baseline_stage_seconds']['connected_search_and_finalization_seconds']
    new = value['optimization']['stage_seconds']['connected_search_and_finalization_seconds']
    old_cli = (value['baseline_execution']['full_cli_seconds'] if
        'full_cli_seconds' in value['baseline_execution'] else
        value['baseline_execution']['complete_fresh_cli_seconds'])
    labels = (('Completed Numba baseline', 'Query reuse') if value.get('query_half_per_scan') else
        ('Completed face-cache baseline', 'Numba blocker scan'))
    x = np.arange(2); width = .36
    axes[0].bar(x-.18, np.array([old, old_cli])/60,
        width, label=labels[0])
    axes[0].bar(x+.18, np.array([new, value['execution']['full_cli_seconds']])/60,
        width, label=labels[1])
    axes[0].set(xticks=x, xticklabels=['search + final API', 'fresh CLI'], ylabel='minutes', title='Observed complete first GA')
    axes[0].legend(fontsize=7)
    for timings, label in zip((value['baseline_candidate_timings'], value['candidate_timings']), labels):
        axes[1].plot([row['candidate'] for row in timings], [row['candidate_seconds'] for row in timings],
            linewidth=.7, label=label)
    axes[1].set(xlabel='score index; last is final', ylabel='candidate seconds', title='All 496 score bytes equal')
    axes[1].legend(fontsize=7)
    axes[2].plot(scores[0], linewidth=.7, label=labels[0]); axes[2].plot(scores[1], '--', linewidth=.7, label=labels[1])
    axes[2].set(xlabel='score index', ylabel='fitness', title='Same search and final result')
    axes[2].legend(fontsize=7)
    axes[3].imshow(volume[plane, low[0]:high[0], low[1]:high[1]].T, cmap='gray', origin='lower',
        extent=[low[0], high[0], low[1], high[1]])
    near = np.abs(points[:, 0]-plane) <= 2
    axes[3].scatter(points[near, 1], points[near, 2], s=8, c='#ff6633', alpha=.8)
    axes[3].set(xlabel='voxel Y', ylabel='voxel Z', title='Own final state: first-selected domain')
    figure.suptitle('Same four-thread budget, separate shared-load windows; first defect only, no whole-pipeline speed claim', fontsize=10)
    figure.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160); plt.close(figure)
    return dict(status='complete_verified_full_first_blocker_plot',
        source_manifest_sha256=sha(manifest_path), metrics_sha256=sha(metrics),
        export_receipt_sha256=sha(metrics.with_name('EXPORT_RECEIPT.json')),
        actual_mri_transform_sha256=sha(actual), brain_sha256=sha(brain), output_sha256=sha(output),
        plotter_sha256=sha(__file__), selected_vertices=len(xyz), mri_slice_voxel_x=plane,
        own_final_state_not_reference_error=True, scores=496, final_mesh_quality='not_assessed', whole_GA=False)


if __name__ == '__main__':
    main()
