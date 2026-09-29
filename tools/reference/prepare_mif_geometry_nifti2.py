"""独立 MRtrix 基准准备：保留 MIF 精确世界仿射，写入 NIfTI-2。

仅用在原软件对照的输入准备；FNIT 运行时不调用 mrinfo。
"""

import argparse
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


def mrinfo_matrix(mrinfo: Path, mif: Path) -> np.ndarray:
    """输入 mrinfo 程序与 MIF 文件路径，返回体素中心到 RAS 毫米的 4×4 仿射。"""
    transform = subprocess.run(
        [str(mrinfo), str(mif), "-transform"], check=True, capture_output=True, text=True,
    ).stdout
    spacing = subprocess.run(
        [str(mrinfo), str(mif), "-spacing"], check=True, capture_output=True, text=True,
    ).stdout
    affine = np.array([[float(item) for item in line.split()]
                       for line in transform.strip().splitlines()], dtype=np.float64)
    affine[:3, :3] *= np.array([float(item) for item in spacing.split()[:3]])
    return affine


def main() -> None:
    """读取已由参考软件转换的 NIfTI，输出同体素的精确仿射 NIfTI-2。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrinfo", type=Path, required=True, help="独立参考 MRtrix 的 mrinfo 可执行文件")
    parser.add_argument("--mif", type=Path, required=True, help="原始 MIF，提供精确世界仿射")
    parser.add_argument("--nifti", type=Path, required=True, help="同一 MIF 转换后的 NIfTI-1，提供体素数组")
    parser.add_argument("--output", type=Path, required=True, help="输出 NIfTI-2 文件路径")
    args = parser.parse_args()
    mif_affine = mrinfo_matrix(args.mrinfo, args.mif)
    image = nib.load(str(args.nifti))
    mapping = np.linalg.inv(mif_affine) @ image.affine
    rounded = np.rint(mapping)
    residual = float(np.max(np.abs(mapping - rounded)))
    if residual > 0.001 or not np.allclose(np.abs(rounded[:3, :3]).sum(axis=0), 1):
        raise ValueError(f"MIF/NIfTI orientation is not an integer voxel mapping: {residual}")
    exact = mif_affine @ rounded
    corners = np.array(np.meshgrid(*[(0, size - 1) for size in image.shape[:3]],
                                   indexing="ij")).reshape(3, -1).T
    corner_diff = np.linalg.norm(nib.affines.apply_affine(image.affine, corners) -
                                 nib.affines.apply_affine(exact, corners), axis=1)
    corrected = nib.Nifti2Image(np.asanyarray(image.dataobj), exact)
    corrected.header.set_zooms(image.header.get_zooms())
    nib.save(corrected, str(args.output))
    check = nib.load(str(args.output))
    if not np.allclose(check.affine, exact, atol=1e-12):
        raise ValueError("NIfTI-2 failed to preserve the MIF affine")
    print(json.dumps({"mapping": rounded.tolist(), "mapping_max_residual": residual,
                      "exact_affine": exact.tolist(),
                      "nifti1_to_exact_max_corner_displacement_mm": float(corner_diff.max()),
                      "nifti2_affine_max_abs_error": float(np.max(np.abs(check.affine - exact)))},
                     indent=2))


if __name__ == "__main__":
    main()
