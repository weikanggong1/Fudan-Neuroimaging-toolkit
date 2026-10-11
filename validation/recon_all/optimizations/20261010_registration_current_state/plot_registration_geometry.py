"""同输入注册误差脑图；图形使用smoothwm，误差来自完整sphere.reg。

baseline/candidate/smoothwm是同序表面路径，output-dir必须不存在。
所有顶点与面完整核对并渲染，几何误差独立于光照计算。
输出PNG与输入SHA/完整误差/渲染面数JSON。surface RAS/mm；显示上限
默认0.001mm，仅控制颜色，不是验收阈值。面不对应或参数非法时报错。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'smoothwm', 'output-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--display-max-mm', type=float, default=0.001)
    args = parser.parse_args(argv)
    if args.display_max_mm <= 0:
        raise ValueError('positive display limit required')
    import numpy as np
    from nibabel.freesurfer.io import read_geometry
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    first, faces = read_geometry(str(args.baseline))
    current, current_faces = read_geometry(str(args.candidate))
    original, original_faces = read_geometry(str(args.smoothwm))
    if current.shape != first.shape or original.shape != first.shape or not (
            np.array_equal(faces, current_faces) and np.array_equal(faces, original_faces)):
        raise ValueError('ordered mesh correspondence required for indexed errors')
    difference = np.linalg.norm(current.astype(np.float64)-first.astype(np.float64), axis=1)
    if not np.isfinite(difference).all():
        raise ValueError('finite geometry required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    displayed = faces
    norm = Normalize(0, args.display_max_mm)
    colors = plt.get_cmap('viridis')(norm(difference[displayed].mean(axis=1)))
    triangles = original[displayed]
    surface_normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normal_lengths = np.linalg.norm(surface_normals, axis=1)
    unit_normals = surface_normals/np.maximum(normal_lengths[:, None], np.finfo(float).tiny)
    # 光照只用于展示实际脑形状；误差量和彩条范围保持不变。
    colors[:, :3] *= (0.45+0.55*np.abs(unit_normals[:, 0]))[:, None]
    figure = plt.figure(figsize=(10, 4.5))
    for index, angle in enumerate((180, 0), 1):
        axis = figure.add_subplot(1, 2, index, projection='3d')
        collection = Poly3DCollection(original[displayed], facecolors=colors,
                                      edgecolor='none', rasterized=True)
        axis.add_collection3d(collection)
        for setter, column in ((axis.set_xlim, 0), (axis.set_ylim, 1), (axis.set_zlim, 2)):
            setter(original[:, column].min(), original[:, column].max())
        axis.set_box_aspect(np.ptp(original, axis=0))
        axis.view_init(elev=0, azim=angle)
        axis.set_axis_off()
    figure.suptitle('Same-input registration: candidate minus baseline (mm)\n'
                    f'full mesh max = {difference.max():.6g}; mean = {difference.mean():.6g}')
    mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    figure.colorbar(mappable, ax=figure.axes, shrink=0.65, label='registered sphere distance (mm)')
    figure.savefig(args.output_dir/'registration_error.png', dpi=160)
    plt.close(figure)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'scope': 'full indexed sphere.reg distance displayed on corresponding smoothwm',
              'input_sha256': {key: sha(getattr(args, key)) for key in ('baseline','candidate','smoothwm')},
              'vertices': len(first), 'faces': len(faces), 'displayed_faces': len(displayed),
              'mean_mm': float(difference.mean()), 'p99_mm': float(np.percentile(difference,99)),
              'max_mm': float(difference.max()), 'different_coordinate_elements': int((first != current).sum()),
              'display_max_mm': args.display_max_mm, 'display_range_is_acceptance_threshold': False,
              'shading_changes_geometry_or_error': False,
              'plot_script_sha256': sha(Path(__file__)),
              'figure_sha256': sha(args.output_dir/'registration_error.png')}
    (args.output_dir/'figure.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
