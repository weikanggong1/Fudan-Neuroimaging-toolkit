# MNI152 仿射与辅助分割

`fnit.recon_all.mni_aux_chain` 生成 `brain.finalsurfs.mgz` 所需的被试 MNI152 变换及两张辅助标签图。它调用已有的 PyTorch SynthMorph、MCA/dura 和静脉窦模型，不调用 FreeSurfer 可执行程序。

标准recon-all GPU profile已显式传递目标设备，所有上述网络运行FNIT GPU实现。`3faa938`在整个MNI辅助调用作用域使用经两例冻结输入验证的cuDNN FP32，包含MNI152 affine及MCA/dura、vsinus；matmul TF32保留，作用域结束恢复原策略。独立函数仍继承调用方精度并保留显式CPU兼容入口，实际前向记录见返回值`runtime`。[精度覆盖修复与对照](SYNTH_AUX_PRECISION.md)和[五阶段串行验证](SERIAL_OPTIMIZATION.md)分别说明阶段与整例范围。

后续诊断发现MNI152初始仿射的GPU matmul TF32会量化几何矩阵，继而放大逆变形差异。CUDA仿射调用现在局部关闭matmul TF32，返回或失败均恢复原设置；其他辅助网络的matmul和其他阶段默认保持TF32。它继续使用GPU、float32输入与权重，未改用CPU或半精度。固定同输入三策略中，完整FP32相对CPU的矩阵最大元素差约5.95×10⁻⁵/1.06×10⁻⁴，优于仅卷积FP32的0.01797/0.06913；矩阵线性系数无量纲，平移项mm，不能将所有元素差统称mm。下游MNI完整回归另报，不能由矩阵误差直接宣布warp通过。

## 函数输入与输出

| Python 函数 | 输入 | 输出及返回值 |
| --- | --- | --- |
| `write_mni_voxel_lta(output_file, matrix, source_file, target_file)` | `matrix` 是完整 MNI152 体素到被试体素的 4×4 NumPy 矩阵；`source_file` 是完整 MNI152 模板；`target_file` 是被试 `orig.mgz`；`output_file` 是写入位置 | 写 type-0 `reg.targ_to_invol.lta`，包含矩阵及两端的体积形状、体素大小、方向和中心；返回 `None`。只负责序列化，不执行配准。 |
| `register_mni152_affine(subject_dir, weights_dir, assets_dir, device="cpu", threads=4)` | `subject_dir/mri/orig.mgz`；外置 `synthmorph.affine.2.h5`；`assets_dir/average/mni_icbm152_nlin_asym_09c/reg-targets/` 中 cropped/full 1 mm MNI152 NIfTI 模板 | 写 `mri/transforms/synthmorph.1.0mm.1.0mm/invol.crop.nii.gz`、`aff.lta`、`reg.targ_to_invol.lta`，返回最后一项路径。最终 type-0 4×4 LTA 将完整 MNI152 体素映射到被试体素，并记录两端体积几何信息。 |
| `run_mni_aux_chain(subject_dir, weights_dir, assets_dir, device="cpu", threads=4)` | 上述配准输入；`subject_dir/mri/nu.mgz`、`synthseg.rca.mgz`；MCA/dura 与静脉窦 H5 权重；`assets_dir/average/` 下的三个先验 | 写上述 LTA、`mri/mca-dura.mgz`（标签 0、6101、6102）、`mri/vsinus.mgz`（标签 0、6111、6112、6115、6116、6117）和 `stats/vsinus.stats`；返回 `lta`、`mca_dura`、`vsinus` 三个路径字段及runtime实际前向记录的字典。两张标签图保留 `nu.mgz` 的 conform 网格与 float32 MGH 元数据。 |

裁切区域取 conform 后 `orig.mgz` 的非零包围盒，外扩三个体素。SynthMorph 估计裁切后的被试图像至裁切 MNI152 的 world 变换；矩阵组合将完整 MNI152 目标体素转成被试体素，以供先验重采样。辅助模型读取 LTA，在注册先验确定的范围内裁切 `nu.mgz`、执行 PyTorch 推理，再将硬标签贴回被试网格。静脉窦函数会清除 `synthseg.rca.mgz` 标为皮层（3、42）的体素。

Python 调用：

```python
import torch
from fnit.recon_all.mni_aux_chain import run_mni_aux_chain

with torch.backends.cudnn.flags(allow_tf32=False):  # 与recon-all相同的局部卷积策略，不改matmul TF32
    paths = run_mni_aux_chain(
        subject_dir="/path/to/subjects/sub01",  # 自产orig、nu、synthseg.rca；无需预先重复注册
        weights_dir="/path/to/weights",  # affine、MCA/dura与静脉窦权重
        assets_dir="/path/to/assets",  # cropped/full MNI152模板和三个先验
        device="cuda:0",  # 所有网络显式使用同一GPU
        threads=4,  # CPU线程预算
    )
# paths["lta"]、paths["mca_dura"]、paths["vsinus"] 分别是三个输出路径。
# paths["runtime"]记录affine、MCA左/右及vsinus实际前向设置。
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

该写入函数用NiBabel/NumPy读取固定模板与MGH几何。早前去Surfa的导入检查只证明模块可导入，不能代替整例；本轮实际辅助链GPU运行、模型缓存和内存裁剪的验证另列。允许的非命令Surfa功能继续用于通用SynthMorph，当前任务没有开展全面移除Surfa。

命令行：

```bash
python -m fnit.recon_all.mni_aux_chain /path/to/subjects/sub01 \
  --weights /path/to/weights --assets /path/to/assets \
  --device cuda:0 --threads 4
```

CLI 第一个位置参数是含 `mri/orig.mgz`、`nu.mgz`、`synthseg.rca.mgz` 的被试目录；`--weights` 指模型目录，`--assets` 指 MNI152 模板和先验目录，`--device` 选 PyTorch 设备，`--threads` 指 CPU 线程数。输出文件和返回值见上表。

外置资产目录对MNI152模板和先验逐文件校验哈希。下载遵循安装器的固定清单与许可证声明；未明确获准再分发的资源从上游获取。本轮复用已授权缓存，没有重复下载。权重和模板不随Python包分发。

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

早期版本证据包括[MNI152 affine同输入验收](../../validation/recon_all/python_gpu_port/mni152_affine_no_surfa_20260927/README.md)、[LTA写入真实T1验证](../../validation/recon_all/python_gpu_port/mni_lta_nibabel_20260927/README.md)及[2026-09-27辅助链对照](../../validation/recon_all/python_gpu_port/mni_aux_connected_20260927/README.md)。它们各自绑定当时的Talairach和代码版本，不作为当前统计量结论。

当前两例冻结自产输入的GPU对照见[三精度策略原始报告](../../validation/recon_all/optimizations/20261001_serial/whole/precision_policy/three_settings_c757_summary.json)：cuDNN FP32/matmul TF32下标签与CPU全部一致，阶段分别5.242/5.783秒；包含加载、传输和写出，仅各一次观察。`61926c7`原始T1整例与局部差异见[完整诊断](../../validation/recon_all/optimizations/20261001_serial/WHOLE_RESULTS.md)；精度修复后的`3faa938`整例另行验证。不能由冻结阶段标签一致推断整例等效。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。

串行候选：recon-all GPU profile的MNI affine、MCA/dura、vsinus随主设备执行；MCA双侧复用模型，裁剪推理改为内存nibabel影像。新增precision_report及结构见[串行说明](SERIAL_OPTIMIZATION.md)。两例同GPU三张标签图零差异，CPU→GPU局部差异单列。
