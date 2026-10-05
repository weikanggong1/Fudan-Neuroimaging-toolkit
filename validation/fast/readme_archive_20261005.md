# README历史归档（2026-10-05）

本页保留文档整理前完整正文，原始正文SHA-256：`59fb77f8e54ba253ec90b7deee23218005781d1c8ef368c1918e387dcd4f83a0`。仅修正移位后的相对链接；最新用户手册见[功能README](../../docs/fast/README.md)。历史测量仍绑定原源码与输入，不改标当前main。

# TorchFAST：T1 组织分割与偏置校正

[返回首页](../../README.md) · [源码目录](../../src/fnit/fast) · [当前验证](README.md)

输入脑提取后的单帧 T1w，输出 CSF、GM、WM 三张部分体积图、三种分类图、乘性偏置场和校正图。计算使用 PyTorch；CUDA 的顺序扫描使用 Triton，CPU 的顺序扫描、随机流、卷积和 PVE 使用 Numba 编译内核。运行时不调用 FSL，不需要权重。

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
| `threads` | 不调整 | `--threads`（CLI 默认 1）/ 无 | PyTorch CPU 线程数；CPU `fsl` 的 Numba 并行线程不超过此预算和安装时的 Numba 上限，调用后恢复原 Numba 线程设置 |
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
| `pve_chunk_size` | 8 | Python / 无 | tensor/CUDA 路径计算 PVE 候选比例的块大小；CPU `fsl` 逐体素按原序检查候选，此参数不改变该内核的分块 |

`FASTConfig.variance_floor_fraction=1e-6` 仅用于 `tensor` 路径。`fsl` 使用原矩估计，不替换非正方差；遇到退化类别会明确报错。原软件的 T2/PD、多通道、其他类别数、先验、手工均值和 `--nopve` 没有对应实现。

## 本轮修复

原 FAST 在每次 HMRF 外循环使用连续的 glibc `rand()` 流初始化 posterior，随后按 z→y→x 原地扫描五遍；混合类型 ICM 原地扫描一遍。旧同步卷积改变了这些依赖。`fsl` 路径用 `level=x+2y+3z` 波前并行：18 个有效邻居中的前序邻居必在较低 level，同 level 没有依赖，因此保留原扫描的更新结果。

此外，FSL NEWIMAGE 对 affine 行列式为正的影像在内部翻转 X。此步骤影响扫描和随机数赋值方向；输出写回时再翻转回输入网格。`TorchFAST` 已显式处理这一步，并使用 NIfTI header pixdim 计算空间权重。

顺序路径同时保留 float32 乘积和逐项 bias 卷积累积、double 矩归约、连续随机流、float32 反复加步长的 PVE 候选网格。初始化指数的精度以本轮原 FAST 二进制同脑控制为准；额外 float32 指数试验未改善匹配，见[精度审计](initclass_expf_control.public.json)。

CPU `fsl` 使用连续 X 行的 z→y→x 原序扫描，避免为每个波前和邻居派发大量小型 PyTorch 运算。它保留更新依赖、随机数赋值方向和相等 PVE 能量的首次候选选择；所有 Numba 内核关闭 `fastmath`。首轮包含即时编译，之后使用 Numba 磁盘缓存。CUDA 和 `tensor` 分支没有启用这些 CPU 内核。Numba 已包含在项目 Conda 环境中，无需安装 FSL 或 C++ 编译器。

## 2026-10-04 最新 CPU 精度修复与 GPU 回归

`execution="fsl"` 的 CPU 指数/对数改为系统 C `expf/logf/exp/log`，用 Numba 并行处理独立元素。真实全脑跟踪的首次分歧来自校正后线性强度的末位舍入，进而影响组织矩和离散 PVE 候选选择。原扫描顺序、随机流、FP32/FP64 使用位置和相等候选的选择规则保留；CUDA 仍使用原 Torch 算子，默认 `tensor` 路径不变。CPU 数学编译器延迟到实际 CPU 调用才导入，不增加默认 GPU 导入依赖。

本例为公开 ds003138 的完整真实 SynthStrip brain，224×288×288、2,843,038 个正脑体素。CPU评测节点 上两侧同一八物理核和线程预算为 8，同锁串行，默认顺序官方→FNIT→FNIT→官方。计时包含新进程、读取、计算和八图保存，不含排队和事后评分。FSL 6.0.7.4 实际主要使用单线程；节点共享负载与每次时钟、RSS 另记录。

| 配置 | 官方完整 CLI | FNIT 完整 CLI | 对官方精度 |
|---|---:|---:|---|
| 默认，各两次 | 411.291 / 368.198 s | 110.173 / 100.174 s | 八图、affine 全部逐值相同 |
| `-N -W 5 -I 2 -O 2 -f 0 -H 0 -R 0` | 119.226 s | 53.592 s | 八图、affine 全部逐值相同 |

默认两侧中位数比为 **3.706 倍**，非默认单组为 **2.225 倍**。八图包括 CSF/GM/WM PVE、seg、pveseg、mixeltype、bias 和 restore；全部不同体素、最大绝对差及 RMSE 为 0。默认 FNIT RSS 约 3.104–3.290 GB、官方 2.240–2.282 GB。结果仅覆盖已声明的 T1 三组织模式、此输入及上述配置。[完整协议、源码哈希、逐图指标和脑图](../smri_cpu/fast_fixes_20261004/README.md)可复核。

![真实同输入 GM PVE 与零差图](../smri_cpu/fast_fixes_20261004/results/gm_pve_match.png)

完整相同脑图的 H100 GPU 旧/新对照在 20 GB allocator 预算内完成：`tensor` 与 `fsl` 各八图数值 SHA 相同，allocated/reserved 同为 `tensor` 3.836/5.505 GB、`fsl` 2.603/5.505 GB。CUDA 内核和数学规则未改。首次完整配对的 API 为 `tensor` 3.850/3.840 s、`fsl` 21.205/23.298 s；最终代码按新→旧→旧→新追加完整回归，`tensor` 为 3.838/3.446/3.751/3.849 s，`fsl` 为 21.713/22.989/23.275/22.333 s。共享 GPU 上快慢方向有波动，按原值记录，不宣称稳定 GPU 速度变化。64³ 区域的检查只作额外回归，不代替上述完整脑验收。

默认 `tensor` 与官方算法仍有差异，CPU 精度修复没有改变默认接口选择。此前同脑 `tensor` 的 CSF/GM/WM PVE RMSE 为 0.05697/0.07594/0.05017；需要本次原序对照结果时显式选择 `execution="fsl"`。

### 最近版本记录

| 版本 | 真实测量与变化 |
|---|---|
| 本版 CPU 标量数学修复 | 默认和一组非默认八图与官方逐值相同；当前 CPU评测节点 计时见上表 |
| `f1cbdab1` CPU 原序编译版 | CPU评测节点 默认官方 388.216/394.790 s，FNIT 121.683/132.939 s；默认 PVE 有 5/27/22 点差异，现已在本例消除 |
| 较早原序 Torch 版 | 小张量派发为主要热点，原扫描和随机流已对齐；见[上一轮诊断](../smri_cpu_20261004/task04/README.md) |

旧记录绑定当时输入、机器和源码，保留作更新记录，不与本轮 CPU评测节点 计算同组速度比。

## 既有官方精度 benchmark：不同输入与环境

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

[当前匿名报告](report.public.json)绑定三个运行源码文件和输入/输出 SHA-256。[模板空间示意图](figures/metrics.json)把两张 GM 用同一原 MNI warp 映射，仅用于展示分割差异，配准没有重新估计。

![同一真实 T1 的原 FAST 与 FNIT 顺序实现：MNI GM 对照](figures/fast_comparison.png)

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
