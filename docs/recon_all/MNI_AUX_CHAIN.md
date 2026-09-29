# MNI152 仿射与辅助分割

`fnit.recon_all.mni_aux_chain` 生成 `brain.finalsurfs.mgz` 所需的被试 MNI152 变换及两张辅助标签图。它调用已有的 PyTorch SynthMorph、MCA/dura 和静脉窦模型，不调用 FreeSurfer 可执行程序。

## 函数输入与输出

| Python 函数 | 输入 | 输出及返回值 |
| --- | --- | --- |
| `write_mni_voxel_lta(output_file, matrix, source_file, target_file)` | `matrix` 是完整 MNI152 体素到被试体素的 4×4 NumPy 矩阵；`source_file` 是完整 MNI152 模板；`target_file` 是被试 `orig.mgz`；`output_file` 是写入位置 | 写 type-0 `reg.targ_to_invol.lta`，包含矩阵及两端的体积形状、体素大小、方向和中心；返回 `None`。只负责序列化，不执行配准。 |
| `register_mni152_affine(subject_dir, weights_dir, assets_dir, device="cpu", threads=4)` | `subject_dir/mri/orig.mgz`；外置 `synthmorph.affine.2.h5`；`assets_dir/average/mni_icbm152_nlin_asym_09c/reg-targets/` 中 cropped/full 1 mm MNI152 NIfTI 模板 | 写 `mri/transforms/synthmorph.1.0mm.1.0mm/invol.crop.nii.gz`、`aff.lta`、`reg.targ_to_invol.lta`，返回最后一项路径。最终 type-0 4×4 LTA 将完整 MNI152 体素映射到被试体素，并记录两端体积几何信息。 |
| `run_mni_aux_chain(subject_dir, weights_dir, assets_dir, device="cpu", threads=4)` | 上述配准输入；`subject_dir/mri/nu.mgz`、`synthseg.rca.mgz`；MCA/dura 与静脉窦 H5 权重；`assets_dir/average/` 下的三个先验 | 写上述 LTA、`mri/mca-dura.mgz`（标签 0、6101、6102）、`mri/vsinus.mgz`（标签 0、6111、6112、6115、6116、6117）和 `stats/vsinus.stats`；返回 `lta`、`mca_dura`、`vsinus` 三个 `Path` 值的字典。两张标签图保留 `nu.mgz` 的 conform 网格与 float32 MGH 元数据。 |

裁切区域取 conform 后 `orig.mgz` 的非零包围盒，外扩三个体素。SynthMorph 估计裁切后的被试图像至裁切 MNI152 的 world 变换；矩阵组合将完整 MNI152 目标体素转成被试体素，以供先验重采样。辅助模型读取 LTA，在注册先验确定的范围内裁切 `nu.mgz`、执行 PyTorch 推理，再将硬标签贴回被试网格。静脉窦函数会清除 `synthseg.rca.mgz` 标为皮层（3、42）的体素。

Python 调用：

```python
from fnit.recon_all.mni_aux_chain import register_mni152_affine, run_mni_aux_chain

lta_path = register_mni152_affine(
    subject_dir="/path/to/subjects/sub01",  # 被试目录；其 mri/orig.mgz 为配准输入
    weights_dir="/path/to/weights",  # 含 SynthMorph affine 权重
    assets_dir="/path/to/assets",  # 含 cropped/full MNI152 模板
    device="cuda:0",  # PyTorch 推理设备
    threads=4,  # CPU 计算线程数
)
# lta_path 是 mri/transforms/.../reg.targ_to_invol.lta 的路径。

paths = run_mni_aux_chain(
    subject_dir="/path/to/subjects/sub01",  # 还需 mri/nu.mgz、synthseg.rca.mgz
    weights_dir="/path/to/weights",  # 另含 MCA/dura 与静脉窦模型
    assets_dir="/path/to/assets",  # 另含三个分割先验
    device="cuda:0",  # 模型推理设备
    threads=4,  # CPU 线程数
)
# paths["lta"]、paths["mca_dura"]、paths["vsinus"] 分别是三个输出路径。
```

单独写出已有配准矩阵时，可直接调用 LTA 写入函数：

```python
import numpy as np
from fnit.recon_all.mni_aux_chain import write_mni_voxel_lta

mni_to_native_voxel = np.load("mni_to_native_voxel.npy")  # 输入：已计算的 4×4 体素坐标矩阵
write_mni_voxel_lta(
    output_file="reg.targ_to_invol.lta",  # 输出：type-0 LTA 文件路径
    matrix=mni_to_native_voxel,  # 输入：将完整 MNI152 体素映射至被试体素
    source_file="mni152.1.0mm.nii.gz",  # 输入：完整 MNI152 模板，提供源几何
    target_file="orig.mgz",  # 输入：被试原始图像，提供目标几何
)
# 返回 None；矩阵与几何写入 output_file。
```

该写入函数用 NiBabel/NumPy 读取固定模板和 MGH 几何；可选 MNI152 affine 注册也已用 `affine_transform` 移除 Surfa 运行时依赖。通用 SynthMorph 的 `joint`、`deform`、`rigid`、`__call__` 与 `apply_transform` 仍使用 Surfa。辅助分割模块可在禁用 Surfa 导入时加载，但本次没有重跑两张标签图。

命令行：

```bash
python -m fnit.recon_all.mni_aux_chain /path/to/subjects/sub01 \
  --weights /path/to/weights --assets /path/to/assets \
  --device cuda:0 --threads 4
```

CLI 第一个位置参数是含 `mri/orig.mgz`、`nu.mgz`、`synthseg.rca.mgz` 的被试目录；`--weights` 指模型目录，`--assets` 指 MNI152 模板和先验目录，`--device` 选 PyTorch 设备，`--threads` 指 CPU 线程数。输出文件和返回值见上表。

外置资产目录对 MNI152 模板和先验逐文件校验哈希。两张 MNI152 图像按需下载；当前下载器从约 515 MB 的上游归档提取。权重和模板不随 Python 包分发。

## 对应的 FreeSurfer 8.2 命令

```bash
mri_mask -bb 3 orig.mgz orig.mgz invol.crop.nii.gz
mri_synthmorph -m affine -t aff.lta invol.crop.nii.gz \
  mni152.1.0mm.cropped.nii.gz -j 4
mri_concatenate_lta -invert1 -invertout aff.lta \
  reg.crop-to-invol.lta reg.invol_to_croptarg.lta
mri_concatenate_lta -invert2 reg.1.0mm.to.1.0mm.cropped.lta \
  reg.invol_to_croptarg.lta reg.targ_to_invol.lta
mri_mcadura_seg --i nu.mgz --o mca-dura.mgz --threads 4 \
  --synthmorphdir transforms/synthmorph.1.0mm.1.0mm
mri_vsinus_seg --s sub01 --rca-synthseg --threads 4 \
  --synthmorphdir transforms/synthmorph.1.0mm.1.0mm
```

[MNI152 affine 注册去 Surfa 的同输入验收](../../validation/recon_all/python_gpu_port/mni152_affine_no_surfa_20260927/README.md)给出两份 LTA 逐字节相同及稳态时间。[LTA 写入替换的真实 T1 验证](../../validation/recon_all/python_gpu_port/mni_lta_nibabel_20260927/README.md)给出相同矩阵下与原 Surfa 写入的逐字节比较、完整函数的同输入比较和五次写入耗时。[真实 T1 对照](../../validation/recon_all/python_gpu_port/mni_aux_connected_20260927/README.md)逐项比较候选 LTA、两张标签图及后续 `brain.finalsurfs.mgz` 与保存的官方重建。保存试验的 `stats/vsinus.stats` 五个静脉窦区域数值行匹配，但其 eTIV 来自当时的 Talairach LTA；Talairach 精度修改后尚未重测这份统计文件。标准单 T1 流程在 CPU 上调用本模块。它还需要新的整例验收。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
