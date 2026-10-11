"""完整首缺陷距离配对：真实末态T1叠加与实际分段墙钟，仅事后绘图。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def _plane_labels(voxel_to_surface_ras):
    """根据实际Nib tkreg方向命名切片，不能把体素轴编号当解剖方向。"""
    import nibabel as nib
    codes = nib.orientations.aff2axcodes(voxel_to_surface_ras)
    names = {'L': 'sagittal', 'R': 'sagittal', 'A': 'coronal',
             'P': 'coronal', 'I': 'axial', 'S': 'axial'}
    if any(code not in names for code in codes):
        raise ValueError('valid MRI tkreg axes required for slice labels')
    return list(codes), [names[code] for code in codes]


def _segments(vertices, faces, axis, plane):
    import numpy as np
    tri = vertices[faces]
    signed = tri[:, :, axis] - plane
    rest = [a for a in range(3) if a != axis]
    grouped = {}
    for a, b in ((0, 1), (1, 2), (2, 0)):
        crossing = ((signed[:, a] <= 0) & (signed[:, b] > 0)) | ((signed[:, b] <= 0) & (signed[:, a] > 0))
        ids = np.flatnonzero(crossing)
        alpha = -signed[ids, a] / (signed[ids, b] - signed[ids, a])
        point = tri[ids, a] + alpha[:, None] * (tri[ids, b] - tri[ids, a])
        for index, p in zip(ids, point):
            grouped.setdefault(int(index), []).append(p[rest])
    return np.asarray([points for points in grouped.values() if len(points) == 2])


def plot_completed(*, input_run, publication_directory, frozen_source,
        expected_source_manifest_sha256, expected_metrics_sha256, brain, output_directory):
    """具名原运行/安全目录/冻结源/原MRI→两PNG和仅SHA/数值绘图收据。

    核真实父与两child、81域及产物SHA；MRI仅服务器读取，Nibabel tkreg
    surface RAS到原体素的F64逆变换仅用于显示。不配准/拟合/重排，
    不把首缺陷末态称全九缺陷或最终QC。固定source/metrics两个SHA必填。
    output_directory必须不存在。完整绘图时间单列，不并入GA运行墙钟。
    """
    import numpy as np
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    tick = time.perf_counter()
    D, P, S, out = map(Path, (input_run, publication_directory, frozen_source, output_directory))
    if out.exists():
        raise FileExistsError('fresh figure directory required')
    metrics, receipt = load(P / 'metrics.json'), load(P / 'EXPORT_RECEIPT.json')
    if (sha(P / 'metrics.json') != expected_metrics_sha256
            or receipt['payload']['metrics']['sha256'] != expected_metrics_sha256
            or receipt['actual_report_sha256'] != sha(D / 'candidate/pair.private.json')
            or receipt['actual_complete_sha256'] != sha(D / 'candidate/COMPLETE.json')
            or receipt['actual_exit_sha256'] != sha(D / 'EXIT.json')
            or load(D / 'EXIT.json')['exit_code'] != 0
            or metrics['compared_numeric_fields'] != 81):
        raise ValueError('same completed pair and safe metrics required')
    if (sha(S / 'SOURCE_MANIFEST.private.json') != expected_source_manifest_sha256
            or metrics['source_manifest_sha256'] != expected_source_manifest_sha256):
        raise ValueError('actual scientific source differs')
    for name, item in load(S / 'SOURCE_MANIFEST.private.json')['payload'].items():
        if sha(S / name) != item['sha256'] or (S / name).stat().st_size != item['size']:
            raise ValueError('actual immutable source payload differs')
    if sha(brain) != metrics['records'][0]['report']['input_sha256']['brain']:
        raise ValueError('actual same-chain brain required')
    fields = []
    for row in metrics['records']:
        C = D / 'candidate' / row['label']
        if (sha(C / 'report.private.json') != row['actual_report_sha256']
                or sha(C / 'COMPLETE.json') != row['actual_complete_sha256']
                or sha(D / 'candidate' / (row['label'] + '.EXIT.json')) != row['actual_child_exit_sha256']
                or sha(C / 'final_candidate.private.npz') != row['output_sha256']['final_candidate.npz']):
            raise ValueError('actual final candidate and child identity differ')
        with np.load(C / 'final_candidate.private.npz', allow_pickle=False) as a:
            fields.append((a['original_vertices'].copy(), a['faces'].copy(), a['selected_vertices'].copy()))
    if any(a.tobytes() != b.tobytes() or a.dtype != b.dtype or a.shape != b.shape
           for a, b in zip(fields[0], fields[1])):
        raise ValueError('actual same ordered faces and coordinates required')
    vertices, faces, selected = fields[0]
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or faces.dtype.kind not in 'iu' or not np.isfinite(vertices).all()
            or faces.min() < 0 or faces.max() >= len(vertices) or len(selected) == 0
            or selected.min() < 0 or selected.max() >= len(vertices)):
        raise ValueError('finite real candidate geometry and valid indices required')
    image = nib.load(brain)
    data = np.asarray(image.dataobj)
    voxel_to_surface_ras = np.asarray(image.header.get_vox2ras_tkr(), np.float64)
    axis_codes, plane_labels = _plane_labels(voxel_to_surface_ras)
    matrix = np.linalg.inv(voxel_to_surface_ras)
    voxel = vertices.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
    slices = [int(np.clip(np.rint(np.median(voxel[selected, a])), 0, data.shape[a] - 1)) for a in range(3)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=160)
    counts = []
    for axis, ax in enumerate(axes):
        remaining = [a for a in range(3) if a != axis]
        plane = slices[axis]
        ax.imshow(np.take(data, plane, axis=axis).T, cmap='gray', origin='lower', vmin=0, vmax=180)
        segments = _segments(voxel, faces, axis, plane)
        counts.append(len(segments))
        if len(segments):
            ax.add_collection(LineCollection(segments, colors='#54c3eb', linewidths=.8, label='serial final candidate'))
            ax.add_collection(LineCollection(segments, colors='#ffa861', linewidths=.6, linestyles='dashed', label='parallel final candidate (same)'))
        ax.set_xlim(0, data.shape[remaining[0]]); ax.set_ylim(0, data.shape[remaining[1]])
        ax.set_axis_off(); ax.set_title(plane_labels[axis] + ' / voxel ' + str(plane), fontsize=9)
    axes[0].legend(loc='lower left', fontsize=7, facecolor='black', labelcolor='white')
    fig.suptitle('Same original MRI and final candidate coordinates/faces; first defect only, not whole-mesh QC', fontsize=10)
    fig.tight_layout()
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    fig.savefig(out / 'final_candidate_t1.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    colors = ['#2985ab', '#dc8b47']
    values = [row['fresh_child_cli_seconds'] for row in metrics['records']]
    bars = axes[0].bar(['serial', 'parallel'], values, color=colors)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, value, format(value, '.2f'), ha='center', va='bottom', fontsize=8)
    axes[0].set_ylim(0, max(values) * 1.14); axes[0].set_ylabel('fresh child CLI / seconds')
    keys = ['raw_base', 'ordered_edges', 'search_and_final_API', 'write']
    for i, row in enumerate(metrics['records']):
        axes[1].bar(np.arange(len(keys)) + (i-.5)*.36, [row['report']['stages'][k] for k in keys], width=.36, color=colors[i], label=row['label'])
    axes[1].set_xticks(np.arange(len(keys)), ['raw base', 'edges', 'search + final', 'write'], rotation=20)
    axes[1].set_ylabel('measured substage / seconds'); axes[1].legend(fontsize=8)
    fig.suptitle('Sequential fresh pair, two CPU threads each; shared node load, no whole-pipeline speed claim', fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'observed_timings.png'); plt.close(fig)
    record = dict(status='complete_bound_fullfirst_distance_figures', tool_sha256=sha(__file__),
        source_manifest_sha256=expected_source_manifest_sha256, metrics_sha256=expected_metrics_sha256,
        actual_exit_sha256=sha(D / 'EXIT.json'), brain_sha256=sha(brain),
        surface_ras_to_brain_voxel_display_matrix=matrix.tolist(), slice_voxel_indices=slices,
        voxel_axis_codes_in_surface_ras=axis_codes, slice_plane_labels=plane_labels,
        contour_segments=counts, selected_vertex_count=len(selected),
        coordinates_and_ordered_faces_equal=True, coordinate_fitting_performed=False, mesh_reordered=False,
        interpretation='actual first-defect final candidate, not native-reference error or final whole-mesh QC',
        plotting_seconds=time.perf_counter()-tick,
        payload={p.name:dict(sha256=sha(p), size=p.stat().st_size) for p in out.glob('*.png')})
    (out / 'PLOT_RECEIPT.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-run', 'publication-directory', 'frozen-source', 'brain', 'output-directory'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-source-manifest-sha256', 'expected-metrics-sha256'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(plot_completed(**vars(parser.parse_args()))))


if __name__ == '__main__':
    main()
