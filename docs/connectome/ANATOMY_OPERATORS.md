# Connectome 解剖算子：输入、调用与软件对照

本页对应 [UKB-connectomics 追踪脚本](https://github.com/sina-mansour/UKB-connectomics/blob/main/scripts/bash/probabilistic_tractography_native_space.sh)与 [atlas 合并脚本](https://github.com/sina-mansour/UKB-connectomics/blob/main/scripts/python/combine_volumetric_atlases.py)。分割采用官方 FreeSurfer `recon-all` 输出；下列算子运行在 PyTorch CUDA float32/float64 张量上，默认允许 TF32，未使用 float16 或 bfloat16。影像读取和写出示例采用 nibabel。5TT、GMWMI 与固定矩阵采样数据来自公开 ds004666 `sub-01/ses-2mm`；当前 6DOF 求解指标来自匹配 UKB 真实输入。两组数据的完整命令、范围和脑图分别见[公开阶段报告](../../validation/connectome/ds004666/ANATOMY_STAGE_20260927.md)和[脱敏配准报告](../../validation/connectome/ORIGINAL_UKB_FLIRT_STAGE_20260927.md)。

## `freesurfer_five_tissue(segmentation)`

**功能。** 把 FreeSurfer 官方 `aparc+aseg` 的整数标签按 MRtrix `FreeSurfer2ACT_sgm_amyg_hipp` 表转成形状 `[X,Y,Z,5]` 的 one-hot float32，通道依次是皮层灰质、皮层下灰质、白质、脑脊液、病灶。背景五通道全零。输入需是 GPU 上的 3D 有限非负整数标签；不会执行 `recon-all`、FIRST 或脑图分割。打包的 863 项 ID→类别表由 MRtrix 3.0.3 表按 FreeSurfer 8.2 色表名称匹配生成。较旧 FreeSurfer 或加入 FIRST 的标签应重新验证。

**调用。**

```python
import nibabel as nib
import numpy as np
import torch
from fnit.connectome import freesurfer_five_tissue

seg = nib.load("aparc+aseg.mgz")
labels = torch.as_tensor(np.asarray(seg.dataobj).astype(np.int32), device="cuda:0")
five = freesurfer_five_tissue(
    segmentation=labels,  # 输入；官方 FreeSurfer 三维整数标签图
)  # 输出；[X,Y,Z,5] 的 ACT 组织图
nib.save(nib.Nifti1Image(five.cpu().numpy(), seg.affine), "5tt_t1.nii.gz")
```

**软件命令。** `5ttgen freesurfer aparc+aseg.mgz 5tt_t1.mif -nocrop -sgm_amyg_hipp`。原 UKB 脚本额外给 `-first T1_first`；这份 MRtrix 3.0.3 适配参考没有 FIRST。**对照。** 256³×5 中逐值不一致为 0；GPU 核心计算 0.286 s、PyTorch 峰值已分配 1.00 GiB；MRtrix 参考进程 9.56 s（含 I/O）。两侧时间范围不同。

## `gmwmi_from_five_tissue(five_tissue)`

**功能。** 按 MRtrix `5tt2gmwmi` 的三个空间轴灰质与白质占比中心差分计算灰白质界面权重，输出 `[X,Y,Z]` float32。输入应是前述 five_tissue 或相同通道约定的 5TT；该函数不执行追踪。

**调用。**

```python
from fnit.connectome import gmwmi_from_five_tissue

gmwmi = gmwmi_from_five_tissue(
    five_tissue=five,  # 输入；皮层 GM、皮层下 GM、WM、CSF、病灶五通道图
)  # 输出；同网格 float32 灰白质界面图
nib.save(nib.Nifti1Image(gmwmi.cpu().numpy(), seg.affine), "gmwmi_t1.nii.gz")
```

**软件命令。** `5tt2gmwmi 5tt_t1.mif gmwmi_t1.mif`。**对照。** 256³ 中逐值不一致为 0；正值支持为 336,136 个体素，Dice 1.0；GPU 核心计算 0.101 s，与 5TT 连续运行时两者合计峰值已分配 1.00 GiB；MRtrix 参考进程 0.84 s（含 I/O）。

## `TorchFLIRT(device="cuda:0", dof=6, cost="normmi")`

**功能。** 用包内已有 FSL 坐标转换、金字塔和优化器实现 6 自由度刚性 b0→T1 配准；新增 `normmi` smoothed-histogram 代价。返回 FSL scaled-mm 输入到参考 `.mat`、RAS 世界输入到参考矩阵和参考网格重采样 b0。输入宜为**已去脑的** 3D b0 和 T1，且必须使用同一对输入验证；输出矩阵不能直接当成 RAS/world affine。也可从 `fnit.flirt.run_flirt` 或 `fnit-flirt` CLI 调用。默认的 12-DOF/corratio VBM 路径保持原行为。

**调用。**

```python
from fnit.flirt import TorchFLIRT

flirt = TorchFLIRT(
    device="cuda:0",       # GPU 计算设备；默认允许 TF32
    dof=6,                 # 六自由度刚性变换
    cost="normmi",         # 4/2/1 mm 精化的归一化互信息代价
    angular_search=True,   # 8 mm 阶段使用 FSL 默认的 CorrRatio 搜索
)
result = flirt(
    moving="b0_brain.nii.gz",  # 输入；去脑的平均 b0
    fixed="t1_brain.nii.gz",   # 输入；定义输出网格的去脑 T1
    init=None,                 # 输入；无外部初始矩阵
)
world_dwi_to_t1 = result.moving_to_fixed_world
fsl_matrix = result.matrix
```

**软件命令。** `flirt -in b0_brain.nii.gz -ref t1_brain.nii.gz -cost normmi -dof 6 -omat diff2struct_fsl.txt`。**同输入对照。** 当前实现的粗搜索使用 FSL 默认 CorrRatio，精化使用 normmi。匹配 UKB 的矩阵相对 FSL 在世界空间网格的位移均值／95 分位／最大值为 0.186655／0.285483／0.367350 mm；重采样图的非零掩膜 Dice 0.997655、非零强度 Pearson 0.999512、双方非零体素 MAE 75.967 原始强度单位。PyTorch GPU 求解与重采样用时 19.155 s、峰值分配显存 0.746 GiB，FSL CPU 用时 10.02 s。固定 **FSL 矩阵** 后，PyTorch 与 `flirt -applyxfm -init` 重采样的 Pearson 0.999999995、MAE 0.149；两侧分别用时 0.512／0.73 s。详见[匹配 UKB 脱敏报告](../../validation/connectome/ORIGINAL_UKB_FLIRT_STAGE_20260927.md)。

## `resample_labels_nearest(labels, source_affine, target_shape, target_affine, target_to_source_world)`

**功能。** 将整数 atlas 从源网格最近邻采样到目标网格。`target_to_source_world` 是目标/DWI RAS-mm → 源/T1 RAS-mm 的 4×4 齐次矩阵，不能传未经转换的 FSL `.mat`。输入 `labels` 是 3D GPU 张量，输出保留整数标签类型；网格和变换用 float64 计算索引，越界置零。目标必须与参考 b0 图的 shape、affine 一致。

**调用。**

```python
from fnit.connectome import resample_labels_nearest
from fnit.flirt import flirt_to_world_affine

source, target = nib.load("cortical_t1.nii.gz"), nib.load("b0_brain.nii.gz")
t1 = nib.load("t1_brain.nii.gz")
world = flirt_to_world_affine(
    flirt_matrix=np.loadtxt("diff2struct_fsl.txt"),       # 输入；b0→T1 的 FSL scaled-mm 矩阵
    moving_vox2world=target.affine,                       # 输入；b0 网格仿射
    fixed_vox2world=t1.affine,                            # 输入；T1 网格仿射
    moving_shape=target.shape[:3],                        # 输入；b0 三维形状
    fixed_shape=t1.shape[:3],                             # 输入；T1 三维形状
    moving_voxel_sizes=target.header.get_zooms()[:3],    # 输入；b0 体素尺寸
    fixed_voxel_sizes=t1.header.get_zooms()[:3],         # 输入；T1 体素尺寸
)  # 输出；b0→T1 的 RAS 世界坐标矩阵
out = resample_labels_nearest(
    labels=torch.as_tensor(np.asarray(source.dataobj).astype(np.int32), device="cuda:0"),  # 输入；T1 标签
    source_affine=source.affine,     # 输入；标签图体素到 RAS 的仿射
    target_shape=target.shape[:3],  # 输入；b0 输出网格形状
    target_affine=target.affine,    # 输入；b0 输出网格仿射
    target_to_source_world=world,   # 输入；b0→T1 世界坐标矩阵
)  # 输出；b0 网格三维整数标签图
nib.save(nib.Nifti1Image(out.cpu().numpy(), target.affine), "cortical_dwi.nii.gz")
```

**软件命令。** 先执行 `transformconvert diff2struct_fsl.txt b0_brain.nii.gz t1_brain.nii.gz flirt_import diff2struct_mrtrix.txt`，再运行 `mrtransform cortical_t1.nii.gz cortical_dwi.nii.gz -linear diff2struct_mrtrix.txt -inverse -interp nearest -datatype uint32 -template b0_brain.nii.gz`。**对照。** 单张实际 `aparc+aseg` 标签映射 104×104×72 中 0 个体素不一致，GPU 0.836 s、峰值已分配 0.218 GiB；由同一 FreeSurfer 分割构造的皮层/皮层下两张图各为 0 个体素不一致，GPU 0.538/0.023 s、MRtrix 0.723/0.515 s。计时含各自脚本中的张量/进程开销，单张与双图批次不直接合并。

## `combine_cortical_subcortical(cortical, subcortical, cortical_max_label)`

**功能。** 两张同形状、同 DWI 网格的 3D 整数图合并：皮层非零处保留皮层；仅皮层为零且皮层下大于零时，皮层下标签加 `cortical_max_label`；背景为零。`cortical_max_label` 应取皮层 **ColorLUT 最大索引**，不是单个受试者恰好出现的最大标签。

**调用。**

```python
from fnit.connectome import combine_cortical_subcortical

cortical_dwi = out  # 上一步皮层 atlas 的结果
sub_image = nib.load("subcortical_t1.nii.gz")
subcortical_dwi = resample_labels_nearest(
    labels=torch.as_tensor(np.asarray(sub_image.dataobj).astype(np.int32), device="cuda:0"),  # 输入；皮层下标签
    source_affine=sub_image.affine,  # 输入；皮层下图仿射
    target_shape=target.shape[:3],   # 输入；b0 输出网格形状
    target_affine=target.affine,     # 输入；b0 输出网格仿射
    target_to_source_world=world,    # 输入；b0→T1 世界坐标矩阵
)
combined = combine_cortical_subcortical(
    cortical=cortical_dwi,          # 输入；皮层优先的整数标签图
    subcortical=subcortical_dwi,    # 输入；皮层下整数标签图
    cortical_max_label=2,          # 输入；本例皮层 ColorLUT 的最大索引
)  # 输出；同网格的合并整数标签图
nib.save(nib.Nifti1Image(combined.cpu().numpy(), target.affine),
         "combined_dwi.nii.gz")
```

**原脚本命令。** `python scripts/python/combine_volumetric_atlases.py "$MAIN_DIR" "$SUBJECTS_DIR" "$SUBJECT_ID" "$INSTANCE" "$CORTICAL_NAME" "$SUBCORTICAL_NAME"`，要求原仓库的 atlas 路径与 ColorLUT 布局。**对照。** 本例用官方 FreeSurfer 标签构造 2 区皮层、2 区皮层下操作测试图；与原脚本的皮层优先及 ColorLUT 索引偏移公式比较，104×104×72 合并图 0 个体素不一致。已驻留 GPU 上的 PyTorch 合并核心 0.0158 s，原脚本等价的 NumPy 合并公式 0.0058 s；这两项不包含 NIfTI 读取和保存。前述双图 benchmark 计时仅包括各自的映射。该测试图不是原 UKB 的 Tian atlas。

## 安装和资源边界

主页 [`environment.yml`](../../environment.yml) 提供 PyTorch 2.5.1/CUDA 11.8、nibabel、surfa 和仓库包；5TT 数值表、MPL-2.0 与 FSL 6.0 许可证及 BET 参考源码包含在本次 wheel。合并远端更新前的 `0.12.1` 阶段快照 wheel 已在现有 FNIT Conda Python 3.11 环境中以 `pip --no-deps --target` 隔离安装；BET、MSMT-CSD、FIRST 5TT 和 Tian 反向形变的真实输入 CPU 最小调用成功。该阶段 wheel 的 GPU 初始化遇到显存不足，未重新运行 CUDA 5TT/FLIRT 或 10,000 次播种整链；合并后源码尚未再次打包核验。先前源码级 GPU 数值比较见各阶段报告，不能视为本次 wheel 的 GPU 安装核验。[安装核验记录](../../validation/connectome/ds004666/anatomy_conda_install.public.json)保存本次 wheel 哈希、环境和调用范围；没有重建全新 Conda 环境。FreeSurfer、FSL、MRtrix 可执行文件未被打包，仅用于独立软件对照；官方 `recon-all` 仍由用户在包外运行。

本例 5TT/GMWMI 与 atlas 的 PyTorch 峰值已分配显存为 1.00 和 0.22 GiB。峰值是 `torch.cuda.max_memory_allocated()`，不包含其他进程占用；具体每阶段数据在 JSON 报告中。当前 10,000 次播种整链的 Torch 峰值分配显存为 2.720 GiB；其他尺寸、atlas 与原脚本千万次追踪仍需单独测量。
