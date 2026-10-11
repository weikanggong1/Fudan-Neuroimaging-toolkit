"""真实首候选的MRI顶点叠加、形变和严格坐标误差；只读诊断。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from benchmark_recon_topology_retess_fields import load_retessellation_capture
from benchmark_recon_topology_match_fields import load_match_capture
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel


def run(*, brain, candidate, before, reference_after, output):
    """完整MRI/候选NPZ/两个诊断状态→PNG，surface RAS/mm与体素明确区分。

    所有参数必填具名路径。candidate只读matched/selected/faces；before只
    用于显示实际位移，reference_after只显示事后误差；不生成或改写候选。
    brain是同uint8 XYZ MGZ，output为不存在的PNG。没有算法/验收阈值。
    返回图SHA及selected数、切片和误差范围；图不替代完整网格质量检查。
    """
    if output.exists() or output.suffix.lower() != '.png':
        raise ValueError('fresh PNG output required')
    with np.load(candidate, allow_pickle=False) as data:
        matched, selected, faces = [data[key].copy() for key in ['matched', 'selected_vertices', 'faces']]
    initial = load_retessellation_capture(capture_path=before)
    reference = load_match_capture(capture_path=reference_after)
    if not np.array_equal(faces, reference['faces']) or matched.shape != initial['vertex_fields'][:, :3].shape:
        raise ValueError('ordered vertex/face correspondence required for this plot')
    image = nib.load(brain); volume = np.asarray(image.dataobj)
    matrix = source_topology_ras_to_voxel(volume_header=image.header)
    xyz = matched[selected]
    vox = xyz.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
    kz = int(np.clip(np.rint(np.median(vox[:, 2])), 0, volume.shape[2]-1))
    near = np.abs(vox[:, 2]-kz) <= 2
    delta = np.linalg.norm((matched-initial['vertex_fields'][:, :3]).astype(np.float64), axis=1)[selected]
    error = np.linalg.norm((matched-reference['vertex_fields'][:, :3]).astype(np.float64), axis=1)[selected]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(volume[:, :, kz].T, cmap='gray', origin='lower', vmin=0, vmax=140)
    axes[0].scatter(vox[near, 0], vox[near, 1], s=11, c='orangered', label='FNIT vertices within 2 voxels')
    axes[0].set_title(f'Brain MRI + candidate vertices (voxel Z={kz})')
    axes[0].set_xlabel('voxel X'); axes[0].set_ylabel('voxel Y'); axes[0].legend(fontsize=7)
    for ax, values, title in [(axes[1], delta, 'Retess input to matched displacement (mm)'),
            (axes[2], error, 'Matched minus fixed-source reference (mm)')]:
        points = ax.scatter(xyz[:, 1], xyz[:, 2], c=values, s=18, cmap='viridis',
            vmin=0, vmax=max(float(values.max(initial=0)), .001))
        ax.set_aspect('equal'); ax.set_xlabel('surface RAS Y (mm)'); ax.set_ylabel('surface RAS Z (mm)')
        ax.set_title(title); fig.colorbar(points, ax=ax, shrink=.8)
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=160); plt.close(fig)
    return dict(figure_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), selected_vertices=len(selected),
        voxel_z=kz, overlay_vertices=int(near.sum()), max_displacement_mm=float(delta.max(initial=0)),
        max_reference_error_mm=float(error.max(initial=0)), scope='first candidate diagnostic figure, not final quality acceptance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['brain', 'candidate', 'before', 'reference-after', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args()))))


if __name__ == '__main__':
    main()
