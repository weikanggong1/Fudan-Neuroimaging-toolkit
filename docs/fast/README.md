# TorchFAST：单通道 T1 组织分割

[返回首页](../../README.md) · [代码目录](../../src/fnit/fast/)

该功能在 PyTorch 中实现三组织 HMRF-EM 分割、平滑乘性 bias field 估计和
partial-volume estimation。它不调用 FSL，也不需要模型权重。输入应为已经完成脑
提取的单帧 T1w；未提供 mask 时，以输入中严格大于零的体素作为脑区。

## Python 调用

```python
from fnit.fast import TorchFAST

fast = TorchFAST(
    device="cuda:0",  # 设备：第一张可见 CUDA GPU
    threads=1,  # CPU 线程：读写和部分归约使用 1 线程
)
result = fast(
    image="T1_brain.nii.gz",  # 输入：已经脑提取的单帧 T1w
    mask=None,  # 脑掩膜：省略时使用 input > 0
)

result.pve_csf.save(path="T1_brain_pve_0.nii.gz")  # 输出路径：CSF 部分体积图
result.pve_gm.save(path="T1_brain_pve_1.nii.gz")  # 输出路径：GM 部分体积图
result.pve_wm.save(path="T1_brain_pve_2.nii.gz")  # 输出路径：WM 部分体积图
result.pve_segmentation.save(path="T1_brain_pveseg.nii.gz")  # 输出路径：三组织硬分割
result.bias_field.save(path="T1_brain_bias.nii.gz")  # 输出路径：乘性偏置场
result.restored.save(path="T1_brain_restore.nii.gz")  # 输出路径：偏置校正影像
```

输入支持路径或 `nibabel.spatialimages.SpatialImage`。也可传入同网格 mask：

```python
result = fast(
    image="T1w.nii.gz",  # 输入：单帧 T1w
    mask="brain_mask.nii.gz",  # 输入：与 T1w 同网格的显式脑掩膜
)
```

显式 mask 会与 `T1w > 0` 取交集。shape 或 voxel-to-world affine 不一致时会
报错，不做隐式重采样。

## 返回值

`TorchFAST(...)` 返回 `FASTResult`：

| 字段 | 数据含义 |
|---|---|
| `pve_csf` | CSF partial-volume fraction，范围 0–1 |
| `pve_gm` | GM partial-volume fraction，范围 0–1 |
| `pve_wm` | WM partial-volume fraction，范围 0–1 |
| `hard_segmentation` | PVE 前的 HMRF 分类；脑内 1=CSF、2=GM、3=WM |
| `pve_segmentation` | PVE 最大分量；脑内 1=CSF、2=GM、3=WM |
| `mixel_type` | 0–2 为纯 CSF/GM/WM，3–5 为 CSF-GM、CSF-WM、GM-WM |
| `bias_field` | 原始图像中的乘性 bias；脑外为 1 |
| `restored` | 脑内 `input / bias_field`；脑外为 0 |
| `tissue_means` | 校正后线性强度中的 CSF、GM、WM 均值 |
| `tissue_variances` | 校正后线性强度中的 CSF、GM、WM 方差 |

所有影像输出均为 `FNITNifti1Image`（`nibabel.Nifti1Image` 子类），保留输入的 shape、voxel size、方向和世界坐标。脑内三个 PVE 之和为
1，脑外为 0。

## 命令行与 FSL 对应

```bash
fnit fast -i T1_brain.nii.gz -o results/T1_brain \
  --device cuda:0 --threads 1 -b -B
```

这条命令在 `cuda:0` 上分割已经提取的脑图，`-o` 是输出 basename。默认写出：

```text
T1_brain_pve_0.nii.gz       CSF PVE
T1_brain_pve_1.nii.gz       GM PVE
T1_brain_pve_2.nii.gz       WM PVE
T1_brain_seg.nii.gz         PVE 前硬分类
T1_brain_pveseg.nii.gz      最大 PVE 分类
T1_brain_mixeltype.nii.gz   pure/mixed tissue 类型
```

`-b` 另写乘性偏置场 `T1_brain_bias.nii.gz`；`-B` 另写校正图
`T1_brain_restore.nii.gz`。已有文件默认不覆盖，重跑时显式加 `--overwrite`。每个文件
先在目标目录写入临时文件，成功后再替换最终路径。

对应的原生 FSL 命令是：

```bash
fast -n 3 -t 1 -b -B -o results/T1_brain T1_brain.nii.gz
```

本包固定实现 FSL 的 `-n 3 -t 1`，即三组织、T1、单通道路径，因此 CLI 不另设
`-n/-t`。两条命令的 `-W/-I/-O/-l/-f/-H/-R/-N/-b/-B/-o` 含义和默认值对应；
本包用 `-i` 明确输入，用 `--device` 选择 Torch 设备。`-N` 只关闭 bias 更新，仍
执行默认 4 次 HMRF 外循环。FSL 的其他类别数、T2/PD、多通道、priors、手工均值
和 `--nopve` 路径未实现。

## 默认计算过程

默认参数对应 FAST4 2111.3 的单通道 T1 设置：

| `TorchFAST` 参数 | 默认值 | 作用 |
|---|---:|---|
| `init_iterations` | 15 | 初始 Gaussian-mixture 更新次数 |
| `bias_iterations` | 4 | 联合分割与 bias 更新次数 |
| `fixed_iterations` | 4 | bias 固定后的分割更新次数 |
| `bias_fwhm_mm` | 20 | bias field 平滑尺度，单位 mm |
| `init_mrf` | 0.02 | bias 估计阶段的组织空间权重 |
| `mrf` | 0.1 | 最终组织空间权重 |
| `mixel_mrf` | 0.3 | pure/mixed tissue 类型空间权重 |
| `pve_steps` | 100 | 最终 PVE 比例的离散间隔数；mixel evidence 固定按 FAST 的 0.01 网格积分 |
| `mean_field_iterations` | 5 | 每轮 HMRF 的同步更新次数 |

计算首先按 FAST 的输入规则处理负值：若第 2 百分位仍小于零，则整体减去最小值；
否则把零以下体素置零。随后在 `log(T1 + 1)` 域以排序数组的 25%、50%、75%
位置初始化 CSF、GM 和 WM。默认初始软 GMM 总更新数为
`init_iterations + fixed_iterations = 19`。EM 更新组织均值、
方差和后验概率。bias 更新对精度加权残差及精度分别做物理尺度 Gaussian 平滑，
随后相除并把脑内 log-bias 均值归零。最终在校正后的线性强度中比较三种纯组织
和三种双组织 mixel，并在选中的 mixel 内估计 PVE。

## 与 FSL FAST 的边界

实现参照 [FAST 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/fast.html)
和 FAST4 2111.3 源码结构。它覆盖本项目 VBM 所需的单通道、T1、三分类、无先验
路径。以下功能尚未实现：

- T2、PD 和多通道分割；
- 标准空间 tissue priors 或手工初始均值；
- 任意组织类别数；
- FSL 命令行文件命名与所有诊断输出。

FSL 每次 HMRF 外循环先用固定种子随机数初始化 posterior，再按体素顺序原地更新；
本实现从 Gaussian likelihood 初始化并使用同步卷积更新，才能在 GPU 上并行。
Gaussian 卷积边界、分位数计算和浮点归约顺序也不同。因此方法阶段和默认参数相
对应，但不能据此把输出称为逐体素或逐位等价。与原生 FAST 的实际一致性需要在
完全相同的 brain-only 输入上分别比较三张 PVE、bias field 和校正图。

FAST4 2111.3 标签的完整、未改动源码随包保存在
[`upstream_fast4/`](../../src/fnit/fast/upstream_fast4/)，只用于许可和溯源，
不会被编译或导入。该部分及本改写受 [FSL 6.0 许可](../../licenses/FSL-6.0.txt)
约束，只允许非商业使用；来源提交和逐文件 SHA-256 见
[`provenance.json`](../provenance.json)。

2026-09-27 用一幅真实 brain-only T1w 对当前 Nibabel I/O 完整回归，并与同一输入的固定 FSL FAST 6.0.7.4 输出比较。候选运行没有调用 FSL。当前 FAST 源码树 SHA-256 为 `433723da856c798faca9a66ea95784bdbb5ad5b13f66f451ce32e7030559943b`。

| 输出 | Pearson | MAE | Dice 0.5 |
|---|---:|---:|---:|
| CSF PVE | 0.988556 | 0.008464 | 0.987640 |
| GM PVE | 0.984627 | 0.015411 | 0.992328 |
| WM PVE | 0.993731 | 0.007022 | 0.997031 |

log-bias Pearson 为 0.99999999936。对应文件的 shape、affine、dtype 及 qform/sform code 一致。当前 GPU 测量进程为 12.55 s，CPU CLI 为 36.24 s；同病例 FSL CPU 参考运行记录为 305.46 s。它们不在同一时段运行，因此不计算稳定加速倍数。GPU Torch 峰值 allocated 2,352 MiB、reserved 3,444 MiB。

本轮单例覆盖当前完整调用链。数值、时间、显存、输出文件哈希和实际命令见[验证记录](../../validation/fast/README.md)与[机器报告](../../validation/fast/report.public.json)。

### FSL FAST 与当前 TorchFAST 示意图

下图使用公开 `sub-02` 的相同 brain-only T1，本轮重新运行当前 TorchFAST。该例 GM Pearson 为 0.978172，Dice 0.5 为 0.987420。

![相同 T1 输入的 FSL FAST 与当前 TorchFAST](../../validation/fast/figures/fast_comparison.png)

## 直接张量接口

低层接口不读取影像文件：

```python
from fnit.fast import FASTConfig, segment_t1

result = segment_t1(
    image=t1_tensor,  # 输入：形状为 [X,Y,Z] 的 T1 张量
    mask=mask_tensor,  # 输入：同形状的布尔脑掩膜
    voxel_size=(1.0, 1.0, 1.0),  # 几何：三个空间轴的体素尺寸，单位 mm
    config=FASTConfig(),  # 配置：使用 TorchFAST 默认迭代与正则参数
)
gm = result.pve[1]
```

`t1_tensor` 和返回张量均为 `(X, Y, Z)` 顺序。该接口适合算法测试；常规影像调用
应使用 `TorchFAST`，由它检查坐标并构造 `FNITNifti1Image` 输出。

## Reference

- 参考文献：Zhang, Brady & Smith, *Segmentation of brain MR images through a hidden Markov random field model and the expectation-maximization algorithm*, IEEE Transactions on Medical Imaging (2001), [doi:10.1109/42.906424](https://doi.org/10.1109/42.906424)。
- 原实现代码库：[FSL `fast4`](https://git.fmrib.ox.ac.uk/fsl/fast4)。
