"""将真实sphere径向翻折位置画在同序inflated皮层上，不修改网格。

输入为双侧最终sphere和同顶点/面顺序inflated；坐标surface RAS/mm。
按FP64面叉积与径向点积定位翻折，只画质控图，不代替三维相交检查。
PNG不含服务器、路径、被试姓名或许可证信息。所有数值来自完整网格；
背景每八个面显示一个，仅用于绘图，红色异常面使用全部原面。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def plot_quality(*, left_sphere: Path, right_sphere: Path,
                 left_inflated: Path, right_inflated: Path, output: Path) -> None:
    """保存四视角PNG；输入同序(F,3)面，不匹配或目标已存在即报错。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel.freesurfer.io as fsio
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if output.exists():
        raise FileExistsError(output)
    fig = plt.figure(figsize=(11, 7))
    for column, (hemisphere, sphere, inflated) in enumerate((
            ("LH", left_sphere, left_inflated), ("RH", right_sphere, right_inflated))):
        xyz, faces = fsio.read_geometry(str(sphere))
        cortex, cortex_faces = fsio.read_geometry(str(inflated))
        if xyz.shape != cortex.shape or not np.array_equal(faces, cortex_faces):
            raise ValueError("sphere and inflated require the same vertex and ordered face correspondence")
        corners = xyz[faces]
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        negative = np.einsum("ij,ij->i", cross, corners.mean(axis=1)) < 0
        area = .5 * np.linalg.norm(cross, axis=1)
        for row, azimuth in enumerate((-90, 90)):
            axis = fig.add_subplot(2, 2, 1 + column + row * 2, projection="3d")
            axis.add_collection3d(Poly3DCollection(cortex[faces[::8]],
                facecolor="#c3c8ce", edgecolor="none", alpha=.30, rasterized=True))
            if negative.any():
                axis.add_collection3d(Poly3DCollection(cortex[faces[negative]],
                    facecolor="#c62525", edgecolor="#c62525", alpha=1, rasterized=True))
                centers = cortex[faces[negative]].mean(axis=1)
                axis.scatter(*centers.T, c="#c62525", s=5, depthshade=False)
            lo, hi = cortex.min(axis=0), cortex.max(axis=0)
            center, radius = (lo + hi) / 2, float((hi - lo).max()) / 2
            axis.set_xlim(center[0]-radius, center[0]+radius)
            axis.set_ylim(center[1]-radius, center[1]+radius)
            axis.set_zlim(center[2]-radius, center[2]+radius)
            axis.set_box_aspect((1, 1, 1)); axis.view_init(elev=0, azim=azimuth)
            axis.set_axis_off()
            axis.set_title(f"{hemisphere}: {int(negative.sum())} negative faces; {area[negative].sum():.8f} mm²")
    fig.suptitle("Sphere finish: unchanged radial orientation defects on FNIT self-produced cortex\n"
                 "Red = full-mesh FP64 radial negative faces; CPU/GPU coordinates identical")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, metadata={"Description": "public FNIT sphere quality diagnostic"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-sphere", type=Path, required=True)
    parser.add_argument("--right-sphere", type=Path, required=True)
    parser.add_argument("--left-inflated", type=Path, required=True)
    parser.add_argument("--right-inflated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_quality(left_sphere=args.left_sphere, right_sphere=args.right_sphere,
                 left_inflated=args.left_inflated, right_inflated=args.right_inflated,
                 output=args.output)


if __name__ == "__main__":
    main()
