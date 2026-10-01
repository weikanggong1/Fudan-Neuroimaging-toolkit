# TorchFAST：T1 组织分割与偏置校正

[返回首页](../../README.md) · [源码目录](../../src/fnit/fast/) · [当前验证](../../validation/fast/README.md)

输入脑提取后的单帧 T1w，输出 CSF、GM、WM 三张部分体积图、三种分类图、乘性偏置场和校正图。计算使用 PyTorch；CUDA 的顺序扫描使用 Triton。运行时不调用 FSL，不需要权重。

`execution="fsl"` 对应 FAST4 的三分类 T1、无先验路径，保留其原位更新顺序。独立接口默认仍为 `execution="tensor"`，使用同步更新；需要复现 FSL 时请显式选择 `fsl`。当前 fMRI volume 流程已显式选择 `fsl`。

## 单被试 Python 调用

```python
from fnit.fast import TorchFAST

fast = TorchFAST(
    device="cuda:0",  # 第一张可见 CUDA GPU；也可用 "cpu"
    threads=1,  # PyTorch CPU 线程数
    execution="fsl",  # 按 FAST4 的顺序更新组织后验与混合组织标签
)
result = fast(
    image="T1_brain.nii.gz",  # 脑提取后的单帧 T1w：路径或 nibabel 影像
    mask=None,  # 可选同网格脑掩膜；省略时使用 image > 0
)
result.pve_csf.save(path="T1_brain_pve_0.nii.gz")  # CSF 部分体积，0–1
result.pve_gm.save(path="T1_brain_pve_1.nii.gz")  # GM 部分体积，0–1
result.pve_wm.save(path="T1_brain_pve_2.nii.gz")  # WM 部分体积，0–1
result.hard_segmentation.save(path="T1_brain_seg.nii.gz")  # HMRF 最大后验分类
result.pve_segmentation.save(path="T1_brain_pveseg.nii.gz")  # 最大 PVE 分类
result.mixel_type.save(path="T1_brain_mixeltype.nii.gz")  # 纯组织/两组织混合类型
result.bias_field.save(path="T1_brain_bias.nii.gz")  # acquired image 的乘性偏置场
result.restored.save(path="T1_brain_restore.nii.gz")  # 脑内 image / bias_field
```

显式 `mask="brain_mask.nii.gz"` 会与正输入取交集。输入可为 3D 或只有一帧的 4D NIfTI。mask 的 shape 和 affine 必须与输入一致。负值处理沿用 FAST 的约定：若第 2 百分位小于零，整体减去最小值；否则把负值设为零。

## 单被试命令行

```bash
fnit fast -i T1_brain.nii.gz -o results/T1_brain \
  --execution fsl --device cuda:0 --threads 1 -b -B
```

这条命令在 GPU 上对脑图做三组织分割和偏置校正。`-i` 指定输入，`-o` 指定输出文件前缀，`--execution fsl` 选择顺序实现，`-b` 保存 bias，`-B` 保存校正图。默认同时保存三张 PVE、seg、pveseg、mixeltype；已有文件需加 `--overwrite` 才能覆盖。

对应原软件命令：

```bash
fast -t 1 -n 3 -I 4 -W 15 -O 4 -f 0.02 -l 20 -H 0.1 -R 0.3 \
  -b -B -o results/T1_brain T1_brain.nii.gz
```

这条原命令指定 T1、三分类、四次 bias 更新、十五次初始更新、四次固定 bias 更新，以及组织和 mixel 的空间权重。FNIT 固定实现 `-t 1 -n 3`，所以不另设类别数和影像类型参数。

## 输入输出合同

`FASTResult` 的八张影像均为 `FNITNifti1Image`，继承 `nibabel.Nifti1Image`，保留输入网格和世界坐标。

| Python 字段 | 原 FAST 文件后缀 | 数值与类型 |
|---|---|---|
| `pve_csf` / `pve_gm` / `pve_wm` | `_pve_0` / `_pve_1` / `_pve_2` | float32；CSF / GM / WM，脑内和为 1、脑外为 0 |
| `hard_segmentation` | `_seg` | int32；脑外 0，脑内 1=CSF、2=GM、3=WM |
| `pve_segmentation` | `_pveseg` | int32；PVE 最大分量，标签同上 |
| `mixel_type` | `_mixeltype` | int32；0/1/2=纯 CSF/GM/WM，3/4/5=CSF-GM/CSF-WM/GM-WM；脑外 0 |
| `bias_field` | `_bias` | float32；乘性偏置场，脑外 1 |
| `restored` | `_restore` | float32；脑内 input / bias，脑外 0 |
| `tissue_means` / `tissue_variances` | Python 返回值 | 校正后线性强度的三组织均值 / 方差 |

## 参数

| Python 参数 | 默认值 | CLI / 原 FAST | 作用 |
|---|---:|---|---|
| `device` | `cpu` | `--device` / 无 | CPU 或 CUDA 设备 |
| `threads` | 不调整 | `--threads`（CLI 默认 1）/ 无 | PyTorch CPU 线程数 |
| `execution` | `tensor` | `--execution` / 无 | `fsl` 保留原顺序；`tensor` 同步更新 |
| `init_iterations` | 15 | `-W` | 初始更新次数；实际初始 GMM 为本值加 fixed_iterations，即 19 次 |
| `bias_iterations` | 4 | `-I` | 分割与 bias 联合更新次数 |
| `fixed_iterations` | 4 | `-O` | bias 固定后的分割更新次数 |
| `bias_fwhm_mm` | 20 | `-l`；`-N` 关闭 | 平滑尺度，单位 mm；设为 0 关闭 bias 更新 |
| `init_mrf` | 0.02 | `-f` | bias 阶段及首轮固定阶段的组织空间权重 |
| `mrf` | 0.1 | `-H` | 后续固定阶段的组织空间权重 |
| `mixel_mrf` | 0.3 | `-R` | 纯组织/混合类型的空间权重 |
| `pve_steps` | 100 | `--pve-steps` / 原内部 iterationspve | 最终 PVE 网格步数；mixel evidence 固定按 0.01 积分 |
| `mean_field_iterations` | 5 | Python / 原固定为 5 | 每个 HMRF 外循环的扫描遍数；对照原默认时保持 5 |
| `pve_chunk_size` | 8 | Python / 无 | 并行计算 PVE 候选比例的块大小 |

`FASTConfig.variance_floor_fraction=1e-6` 仅用于 `tensor` 路径。`fsl` 使用原矩估计，不替换非正方差；遇到退化类别会明确报错。原软件的 T2/PD、多通道、其他类别数、先验、手工均值和 `--nopve` 没有对应实现。

## 本轮修复

原 FAST 在每次 HMRF 外循环使用连续的 glibc `rand()` 流初始化 posterior，随后按 z→y→x 原地扫描五遍；混合类型 ICM 原地扫描一遍。旧同步卷积改变了这些依赖。`fsl` 路径用 `level=x+2y+3z` 波前并行：18 个有效邻居中的前序邻居必在较低 level，同 level 没有依赖，因此保留原扫描的更新结果。

此外，FSL NEWIMAGE 对 affine 行列式为正的影像在内部翻转 X。此步骤影响扫描和随机数赋值方向；输出写回时再翻转回输入网格。`TorchFAST` 已显式处理这一步，并使用 NIfTI header pixdim 计算空间权重。

顺序路径同时保留 float32 乘积和逐项 bias 卷积累积、double 矩归约、连续随机流、float32 反复加步长的 PVE 候选网格。初始化指数的精度以本轮原 FAST 二进制同脑控制为准；额外 float32 指数试验未改善匹配，见[精度审计](../../validation/fast/initclass_expf_control.public.json)。

## 当前真实数据 benchmark

一例真实 T1，固定原程序脑提取输出，脑区 1,397,628 个正体素。原参考为 FSL 6.0.7.22 中 FAST4 2111.3；FNIT 没有启动原程序。下面在同输入的正脑区统计。

| 输出 | `tensor` Pearson | `fsl` Pearson | `fsl` RMSE | `fsl` 不同体素 |
|---|---:|---:|---:|---:|
| CSF PVE | 0.990154817 | 0.999999993 | 4.48×10⁻⁵ | 28 |
| GM PVE | 0.982982153 | 0.999999992 | 5.35×10⁻⁵ | 40 |
| WM PVE | 0.993217329 | 0.999999998 | 2.93×10⁻⁵ | 12 |

三张分类图 seg、pveseg、mixeltype 逐体素相同；三张 PVE 的 0.5 Dice 均为 1。PVE 最大差为 0.01：少量近等值最小能量候选选到了相邻离散比例。bias 最大差 1.19×10⁻⁷，restore 最大差 2.44×10⁻⁴。八图的 shape、affine、dtype、qform/sform code 一致；这些结果没有覆盖原软件其他分割模式。

| 实现 | 本轮墙钟时间 | 计时范围 |
|---|---:|---|
| 原 FAST CPU，8 线程环境 | 150.00 s | 独立进程，含启动、读写 |
| FNIT `fsl`，H100 GPU | 10.95 s | 已加载影像到八幅返回影像，含构造、H2D、计算、D2H |
| FNIT `tensor`，同 GPU | 2.01 s | 同上 |

GPU 数字排除压缩写盘、输入读取和比对；Triton 编译缓存已建立。共享服务器的较早同算法运行观测为 20.35 s，本表没有据此计算稳定加速倍数。`fsl` 的 Torch allocated 峰值 872.5 MiB。原 FAST 参考进程退出码为 255，原记录已核验八个输出完整且有限；这条异常和输出哈希保留在报告中。

[当前匿名报告](../../validation/fast/report.public.json)绑定三个运行源码文件和输入/输出 SHA-256。[模板空间示意图](../../validation/fast/figures/metrics.json)把两张 GM 用同一原 MNI warp 映射，仅用于展示分割差异，配准没有重新估计。

![同一真实 T1 的原 FAST 与 FNIT 顺序实现：MNI GM 对照](../../validation/fast/figures/fast_comparison.png)

## 张量接口与代码结构

```python
import nibabel as nib
import numpy as np
import torch
from fnit.fast import FASTConfig, segment_t1

brain_t1_image = nib.load("T1_brain.nii.gz")  # 本例为3D脑提取T1
brain_t1_array = np.asarray(brain_t1_image.dataobj, dtype=np.float32)
flip_internal_x = np.linalg.det(brain_t1_image.affine[:3, :3]) > 0
if flip_internal_x:
    brain_t1_array = brain_t1_array[::-1].copy()  # FSL内部radiological网格
t1_tensor = torch.as_tensor(brain_t1_array, device="cuda:0", dtype=torch.float32)
mask_tensor = t1_tensor > 0  # 与内部扫描网格一致
result = segment_t1(
    image=t1_tensor,
    mask=mask_tensor,
    voxel_size=brain_t1_image.header.get_zooms()[:3],  # header pixdim；mm
    config=FASTConfig(execution="fsl"),  # 原顺序实现
)
gm_tensor = result.pve[1]  # GM；张量接口的三组织轴在最前
if flip_internal_x:
    gm_tensor = torch.flip(gm_tensor, dims=(0,))  # 恢复到原NIfTI体素网格
```

`segment_t1` 不接收 affine，调用者需先把正行列式影像翻 X 为内部扫描网格，返回后再翻回。常规影像调用用 `TorchFAST` 完成这一处理。代码分工为 `pipeline.py` 检查输入和回写几何，`algorithm.py` 组织分割/PVE/bias 步骤，`_fsl_scan.py` 实现连续随机流、顺序波前与卷积。

## 原实现与文献

- [FAST 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/fast.html)；[FAST4 官方代码库](https://git.fmrib.ox.ac.uk/fsl/fast4)。
- Zhang, Brady & Smith, *Segmentation of brain MR images through a hidden Markov random field model and the expectation-maximization algorithm*, IEEE TMI (2001), [doi:10.1109/42.906424](https://doi.org/10.1109/42.906424)。
- 改写及已有上游参考受 [FSL 6.0 非商业许可](../../licenses/FSL-6.0.txt)约束；源码不会在运行时编译或调用原 FAST。
