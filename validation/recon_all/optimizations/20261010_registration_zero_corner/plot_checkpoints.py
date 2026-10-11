"""从私有真实检查点画球面异常分布；只输出 PNG 和哈希，不发布 NPZ/坐标。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corner-checkpoint", type=Path, required=True, help="真实旧角点法向首差 NPZ")
    parser.add_argument("--face-checkpoint", type=Path, required=True, help="真实旧面法向首差 NPZ")
    parser.add_argument("--output", type=Path, required=True, help="只保存脑图与绘图来源收据的目录")
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    corner = np.load(args.corner_checkpoint)
    face = np.load(args.face_checkpoint)
    positions, faces = face["arg0"], face["arg1"]
    bad_vertices = ~np.isfinite(corner["output0"]).all(axis=1)
    bad_faces = ~np.isfinite(face["output1"]).all(axis=1)
    first = positions[faces[:, 1]] - positions[faces[:, 0]]
    second = positions[faces[:, 2]] - positions[faces[:, 0]]
    cross = np.stack((first[:, 1]*second[:, 2]-first[:, 2]*second[:, 1],
                      first[:, 2]*second[:, 0]-first[:, 0]*second[:, 2],
                      first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0]), axis=1)
    squared = cross[:, 0]*cross[:, 0] + cross[:, 1]*cross[:, 1] + cross[:, 2]*cross[:, 2]
    length = np.sqrt(squared.astype(np.float64)).astype(np.float32)
    tiny = length < np.finfo(np.float32).eps
    assert int(bad_vertices.sum()) == 4 and int(bad_faces.sum()) == 2 and int(tiny.sum()) == 280
    figure = plt.figure(figsize=(12, 5), layout="constrained")
    left = figure.add_subplot(1, 2, 1, projection="3d")
    # 仅绘图稀疏背景，异常点保留全部；此抽样不参与数值比较。
    background = positions[::max(1, len(positions)//10000)]
    left.scatter(*background.T, s=.4, color="0.7", alpha=.1)
    centers = positions[faces[tiny]].mean(axis=1)
    left.scatter(*centers.T, s=6, color="#ed9e21", label="280 FZERO faces", alpha=.7)
    points = corner["arg0"][bad_vertices]
    left.scatter(*points.T, s=34, color="#c62828", label="4 old NaN vertices")
    left.set_title("Real sub-06 LH: first accepted sphere update")
    left.set_box_aspect((1, 1, 1));left.legend(loc="upper left", fontsize=8)
    right = figure.add_subplot(1, 2, 2, projection="3d")
    selected = int(np.flatnonzero(bad_faces)[0]);triangle = positions[faces[selected]].astype(np.float64)
    center = triangle.mean(axis=0);scale = max(float(np.ptp(triangle, axis=0).max()), 0.0001)
    distance = np.linalg.norm(positions[faces].mean(axis=1)-center, axis=1)
    local = positions[faces[distance < scale*2]].astype(np.float64)
    if len(local) > 300:
        local = local[np.argsort(distance[distance < scale*2])[:300]]
    # 局部坐标以被标记面的中心为零、单位 µm；省去 3D 坐标轴的大数偏置。
    local = (local-center)*1000
    local = local[(np.abs(local) <= scale*1000).all(axis=(1, 2))]
    right.add_collection3d(Poly3DCollection(local, facecolor="0.75", edgecolor="0.6", alpha=.2, linewidth=.25))
    triangle = (triangle-center)*1000
    closed = np.concatenate((triangle, triangle[:1]))
    right.plot(*closed.T, color="#c62828", linewidth=2)
    right.scatter(*triangle.T, color="#c62828", s=28)
    for axis in (right.set_xlim, right.set_ylim, right.set_zlim):
        axis(-scale*1000, scale*1000)
    right.set_title("Local zero cross face (float32 length = 0)")
    right.set_box_aspect((1, 1, 1))
    left.set_xlabel("surface RAS x / mm");left.set_ylabel("y / mm");left.set_zlabel("z / mm")
    right.set_xlabel("local x / µm");right.set_ylabel("local y / µm");right.set_zlabel("local z / µm")
    args.output.mkdir(parents=True, exist_ok=True)
    image = args.output / "sphere_zero_normals.png"
    figure.savefig(image, dpi=160);plt.close(figure)
    receipt = {"scope": "real checkpoint visualization; no performance measurement or whole validation",
               "plot_script_sha256": sha(Path(__file__)),
               "corner_checkpoint_sha256": sha(args.corner_checkpoint),
               "face_checkpoint_sha256": sha(args.face_checkpoint),
               "old_NaN_vertices": int(bad_vertices.sum()), "old_NaN_faces": int(bad_faces.sum()),
               "FZERO_faces": int(tiny.sum()), "background_vertex_stride": max(1, len(positions)//10000),
               "image_sha256": sha(image), "NPZ_published": False}
    (args.output / "visualization_source.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
