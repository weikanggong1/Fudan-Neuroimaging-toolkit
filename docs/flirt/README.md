# TorchFLIRT：线性配准与已知矩阵重采样

| 项目 | 内容 |
|---|---|
| 输入 | 个体与参考3D影像，可选初值和连续权重 |
| 输出 | 参考网格影像、正向scaled-mm矩阵和QC |
| 对应原软件 | FSL flirt |
| Python / CLI | run_flirt、TorchFLIRT / fnit flirt、fnit-flirt |
| CPU / GPU | PyTorch CPU/CUDA；CPU融合采样用Numba |

## 1. 功能简介

FNIT提供12自由度CorRatio仿射配准、6自由度NormMI刚体配准和已知矩阵重采样。默认执行完整角度搜索及8/4/2/1mm多层优化，输出固定三线性插值。生产不启动FSL；Numba与CUDA批量NMI所需Triton均通过主页Conda环境安装。

图像和代价使用float32，矩阵及NMI直方图保留float64。CUDA默认允许TF32；坐标与采样中保留已验证的FP32计算。输出的`.mat`为input→reference的FSL scaled-mm矩阵，world-RAS正向/逆向矩阵分别通过结果属性取得。

## 2. Python 调用

```python
from fnit.flirt import run_flirt

moving_image_path = "/data/subject_GM.nii.gz"  # 个体3D灰质图
reference_image_path = "/data/template_GM.nii.gz"  # 模板与目标网格
registered_image_path = "/data/subject_GM_in_template.nii.gz"  # 重采样结果
forward_matrix_path = "/data/subject_GM_to_template.mat"  # input→reference的scaled-mm矩阵
registration_result = run_flirt(
    input=moving_image_path, reference=reference_image_path,
    output=registered_image_path, omat=forward_matrix_path,
    dof=12, cost="corratio", device="cuda:0",  # 仿射配准
)
```

### 输入数据格式

输入为有限3D NIfTI/单帧空间影像，4D只允许最后一维1。reference决定输出网格。权重为各自网格的3D连续权重，仅12/corratio支持。init为4×4 FSL scaled-mm矩阵，不能用world-RAS矩阵直接代替。MGH reference使用scanner RAS写出NIfTI qform/sform。

### 输入参数

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `input` | 是 | 路径 / NIfTI | 无 | 待处理影像，网格与空间由 NIfTI affine 定义。 |
| `reference` | 是 | 路径 / NIfTI / None | 无 | 参考图；决定输出空间、shape、affine。 |
| `output` | 否 | 路径 / None | `None` | 输出文件或前缀；父目录按接口创建。 |
| `omat` | 否 | 路径 / None | `None` | input→reference的4×4 FSL scaled-mm矩阵文件。 |
| `init` | 否 | 路径 / (4,4)数组 / None | `None` | 优化初值；applyxfm时直接应用的矩阵。 |
| `inweight` | 否 | 路径 / NIfTI / None | `None` | input同网格连续权重，仅12/corratio。 |
| `refweight` | 否 | 路径 / NIfTI / None | `None` | reference同网格连续权重，仅12/corratio。 |
| `dof` | 否 | int | `12` | 12/corratio为仿射；6/normmi为刚体。 |
| `cost` | 否 | str | `'corratio'` | corratio或normmi，与dof配对。 |
| `applyxfm` | 否 | bool | `False` | 直接应用init或qform/sform，不估计矩阵。 |
| `usesqform` | 否 | bool | `False` | applyxfm时按世界坐标对齐；不能同时传init。 |
| `device` | 否 | str / torch.device / None | `None` | 计算设备；显式 CUDA 不可用时报错。 |
| `execution` | 否 | str | `'auto'` | 执行策略；本页说明实际支持值。 |
| `candidate_batch_size` | 否 | int | `128` | 每批仿射候选上限，正整数。 |
| `memory_budget_gb` | 否 | float | `20.0` | 批量临时张量预算，单位GiB；20 GiB约21.47GB，不等同20GB。 |
| `overwrite` | 否 | bool | `False` | 是否覆盖已有结果，默认保护原文件。 |

`TorchFLIRT(..., angular_search=True)`另提供角度搜索开关；False对应原软件-nosearch；run_flirt和CLI不暴露该开关。仅支持12/corratio和6/normmi；最终插值固定三线性。

### 输出

`FLIRTResult` 包含：

| 字段 | 类型 | 含义 |
|---|---|---|
| `moved` | `FNITNifti1Image`，是 `nibabel.Nifti1Image` 的子类 | input 在 reference 网格上的图像；按 FSL 规则保留 input 的存盘类型，整数输出强度范围小于 1.5 时转 float32，以保留插值后的 mask 小数。 |
| `matrix` / `fsl_matrix` | NumPy `(4, 4)` 数组 | input→reference 的 FSL scaled-mm 矩阵。 |
| `moving_to_fixed_world` | NumPy `(4, 4)` 数组 | input world-RAS→reference world-RAS 的正向矩阵。 |
| `fixed_to_moving_world` | NumPy `(4, 4)` 数组 | 重采样使用的 reference world-RAS→input world-RAS pull 矩阵。 |
| `qc` | 字典 | 设备、执行路径、TF32 状态、代价函数、搜索层级、评价次数、分阶段墙钟时间和坐标方向。运行时 QC 不把仓库基准解释为当前输入已与 FSL 比较。 |

写盘后的结构为：

```text
subject_GM_to_template.nii.gz  # -out；reference 网格；数据类型按上表的 FSL 规则
subject_GM_to_template.mat     # -omat；4 行×4 列文本矩阵，input→reference
```

配准模式的 qform 和 sform 各自保留 reference 的有效值，缺少一种时从另一种补齐；两种都未设置时使用变换后的 input 空间信息。`applyxfm` 保留 reference 的两种 form 和 code，包括未设置的 code。两种模式都保留 reference header 的 `pixdim`。

FSL scaled-mm 使用 NIfTI header 的 `pixdim` 作为体素大小。视野外填充值采用输出平滑后 input 边缘两层体素的第 10 百分位；存盘类型与整数 mask 的处理规则见上表。

输出先写入同目录临时文件，全部成功后再原子移动到目标位置。没有 `--overwrite` 时，已有文件会使调用停止。

### 应用已知矩阵

`run_flirt(input=..., reference=..., output=..., init=..., applyxfm=True)`直接重采样；用`applyxfm=True, usesqform=True, init=None`按共同world坐标对齐。usesqform要求有效qform或sform。用于MNI分辨率转换时参考图只定义新网格，函数不重新估计配准。

### 设置CPU与显存预算

CPU Python调用前用torch.set_num_threads设置预算；库调用不重置全局PyTorch线程池。独立fnit-flirt可通过OMP_NUM_THREADS/MKL_NUM_THREADS约束线程。

memory_budget_gb的单位是GiB，默认20.0GiB约21.47GB。若作业严格限制十进制20GB，应降低该值并监测整个进程的allocated/reserved；临时张量预算不等于进程总显存。

## 3. 命令行调用

独立入口：

```bash
fnit-flirt \
  -in subject_GM.nii.gz \
  -ref template_GM.nii.gz \
  -out subject_GM_to_template.nii.gz \
  -omat subject_GM_to_template.mat \
  -dof 12 \
  -cost corratio \
  --device cuda:0
```

统一入口将 `fnit-flirt` 换成 `fnit flirt`，配准参数相同；统一入口还可用 `--threads` 设置 PyTorch CPU 线程数，默认 `1`。`--device cuda:0` 选择第一块可见 GPU，CPU 写为 `--device cpu`。

| CLI 参数 | Python 参数 | 含义 |
|---|---|---|
| `-in / -ref / -out / -omat` | `input / reference / output / omat` | 影像与矩阵路径 |
| `-init / -inweight / -refweight` | `init / inweight / refweight` | 初值与权重 |
| `-dof / -cost` | `dof / cost` | 自由度与代价配对 |
| `-applyxfm / -usesqform` | `applyxfm / usesqform` | 应用已知矩阵或world对齐 |
| `--execution` | `execution` | auto/reference/batched |
| `--candidate-batch-size / --memory-budget-gb` | `candidate_batch_size / memory_budget_gb` | 候选分块与GiB预算 |
| `--device / --overwrite` | `device / overwrite` | 设备和覆盖 |
| `fnit flirt --threads` | `torch.set_num_threads` | 统一入口默认1；独立fnit-flirt没有此选项 |

## 4. 原软件调用

```bash
flirt \
  -in subject_GM.nii.gz \
  -ref template_GM.nii.gz \
  -out subject_GM_to_template.nii.gz \
  -omat subject_GM_to_template.mat \
  -dof 12 \
  -cost corratio
```

两套命令的 `-in`、`-ref`、`-out`、`-omat`、`-init`、`-inweight`、`-refweight`、`-dof` 和 `-cost` 含义一致。FNIT 另外提供 `--device`、`--execution`、`--candidate-batch-size`、`--memory-budget-gb` 和 `--overwrite`。当前接口不接受 FSL 的其他 cost、DOF、schedule、搜索范围和插值选项。

去脑 b0 到去脑 T1 的刚性配准改用 6 自由度与归一化互信息：

## 5. 最新精度和运行时间

输入为仓库公开的 CC0 T1w：sub-02 的完整 `156×256×256` 图像配准到 sub-01 的 `256×156×256` 网格。FSL 6.0.7.4 和各冻结 FNIT 源码使用相同输入、有效参数、物理核亲和性和 1/8 线程上限。默认搜索、初始矩阵、无角度搜索、三种连续权重、已知矩阵、qform/sform 以及显式 `batched` 均按完整体积执行。

下表为 v16 默认 CPU `auto/reference` 的四个完整主例；同报告另有 22 项功能观察，均已完成。[机器可读报告](cpu_20261004.public.json)包含该冻结版三份 CPU 源文件、计时 runner、adapter 及源码包的 SHA-256。每个主例完整预热一次后，交替执行三次官方/FNIT 配对；表中完整进程包含启动、导入、读图、完整计算和写盘。预热 API 另行测量全部读图、计算和写盘，与官方完整命令的计时范围不同。

| 模式 | CPU 上限 | FSL 完整进程中位数 | FNIT 完整进程中位数 | FNIT 预热 API 中位数 | 位移 RMS | Pearson |
|---|---:|---:|---:|---:|---:|---:|
| 12/corratio，默认搜索 | 1 | 56.223 s | 94.483 s | 89.100 s | 0.01569 mm | 0.99999596 |
| 6/normmi，默认搜索 | 1 | 123.353 s | 107.927 s | 106.180 s | 0.48752 mm | 0.99713577 |
| 12/corratio，默认搜索 | 8 | 54.257 s | 45.644 s | 40.858 s | 0.01569 mm | 0.99999596 |
| 6/normmi，默认搜索 | 8 | 118.392 s | 83.691 s | 73.869 s | 0.48700 mm | 0.99713927 |

单线程 12/corratio 仍未达到官方速度，其余三项在本例更快。6/normmi 的矩阵与 FSL 存在上述位移差异。四个主例的 shape、affine、float32 dtype、qform/sform 矩阵及 code 均一致；各线程预算内，FNIT 的每次完整进程与预热 API 输出 SHA-256、最终 cost 和评价次数一致，12/corratio 的 1/8 线程输出也一致。NMI 保留原来的 PyTorch float32 熵求和，1/8 线程结果略有差异：最终 cost 分别为 −1.15116012096 和 −1.15116024017，评价次数为 6,427 和 6,400。

八线程上限下，官方 FLIRT 的实际进程 CPU 用量仍约为一个核，FNIT 12/corratio 约为五个核；报告单独保留 CPU 时间与墙钟时间。节点还有其他作业，当前结果对应这份完整输入和本次配对，不推广到其他图像类型。

阶段诊断仅来自 FNIT。下表为三次正式完整进程中各阶段的中位数，完整逐阶段值见报告；各阶段中位数之和不等于完整进程中位数。官方默认命令没有对应的阶段时钟，因此阶段时间不能用于推断官方哪个步骤更快。

| FNIT 阶段 | 12/corratio，1 CPU | 12/corratio，8 CPU | 6/normmi，1 CPU | 6/normmi，8 CPU |
|---|---:|---:|---:|---:|
| 图像准备 | 4.804 s | 2.260 s | 4.495 s | 2.283 s |
| 4 mm 扰动优化 | 20.716 s | 7.419 s | 16.809 s | 25.110 s |
| 2 mm 局部优化 | 5.819 s | 2.377 s | 2.660 s | 2.253 s |
| 1 mm 局部优化 | 40.478 s | 20.103 s | 58.674 s | 29.027 s |

### 2026-10-04 GPU 完整回归（v25，与当前 FLIRT 源码相同）

在 H100 PCIe 上，用同一公开 T1w→T1w 完整体积分别执行 12/corratio 和 6/normmi，CUDA `auto` 使用原有批量路径。比较基线为 FNIT 提交 `1d31e7baaebbb644ab199471f7fe6282721455fd`。每例执行两组 AB/BA 进程配对，每个进程完整预热一次后计时三次，共 16 次完整保存调用。API 时间包含读图、全部配准搜索、重采样和写盘，进程启动与 Torch 导入在计时外；本表保留每组的三个调用中位数。

| 完整模式 | 第 1 组 API：优化前／本版 | 第 2 组 API：优化前／本版 | 峰值 allocation：两源均相同 |
|---|---:|---:|---:|
| 12/corratio | 5.405／5.543 s | 5.291／5.309 s | 2,033,769,472 B（1.894 GiB） |
| 6/normmi | 10.012／10.056 s | 10.021／10.287 s | 5,441,890,816 B（5.068 GiB） |

CPU表绑定v16，GPU表绑定v25并对照FNIT `1d31e7b`；当前文档基线main `140c3739`，最新源码来源核验见审计文件。CPU Intel Xeon Gold6418H、1/8线程预算；GPU H100 PCIe、TF32，图像FP32/NMI直方图FP64；GPU上限20GB。CPU未记录GPU峰值。CPU新版本连续权重/batched的补测详见[后续报告](cpu_followup_20261004.public.json)，不重标v16时间。

图中使用仓库内 OpenNeuro ds000114 v1.0.2 的 CC0 去面部派生数据，sub-02 T1w 为 input，sub-01 T1w 为 reference。FNIT 是本轮 v16 冻结版的完整 12/corratio 输出，参考为 FSL FLIRT 6.0.7.4；两边使用相同强度范围，绝对差单独缩放。该公开数据的许可记录见 [dataset description](https://raw.githubusercontent.com/OpenNeuroDatasets/ds000114/master/dataset_description.json)。

![公开 T1w：本轮 CPU FSL FLIRT、FNIT 和绝对差](figures/cpu_public_12_corratio.png)

官方输出非零区域内 Pearson 为 0.99999596，MAE 为 0.55052，RMSE 为 1.10587，support Dice 为 0.99991224；13³ world-grid 位移 RMS 为 0.01569 mm。shape、affine、float32 dtype、两种 form 的矩阵和 code 均一致。输入、输出、矩阵及 v16 三份 CPU 源文件的 SHA-256 见[图示来源与精度](figures/cpu_public_12_corratio.sources.json)，可下载[矢量 PDF](figures/cpu_public_12_corratio.pdf)。图示由完整体积输出提取，不用于推断速度。

## 6. 最近版本和 benchmark

<!-- 旧文档链接兼容锚点；原始记录在本页第6节的历史链接中。 -->
<a id="修复前后速度与精度"></a>

| 日期 | 更新与验证范围 |
|---|---|
| 2026-10-04，v25/v28 同源 GPU 回归 | H100 的 12/corratio 和 6/normmi 各完成两组 AB/BA、每组完整预热及三次 API 测量；每例 16 次完整保存的全部体素、矩阵、header、扩展和 affine 与优化前一致，allocation 峰值相同。当前四份 FLIRT 生产源文件与该冻结版相同，CPU 各表仍保留 v16/v21/v24 的实际计时来源。 |
| 2026-10-04，v24 | 显式 CPU batched 复用保序箱统计，保留原 pack、compact_sum、float32 cost/熵与搜索。506 项局部回归通过；完整 12/corratio 与 6/normmi 的 15,384 个候选统计量、中间求和、cost 及最终输出逐位一致。补齐 1/8 线程四项最新官方完整对照；单线程 corratio 仍未达到官方速度。 |
| 2026-10-04，v22 | 显式 CPU batched 复用融合采样，保留原批量归约与搜索；435 项局部回归通过，两种完整轨迹共 15,384 个候选逐位一致，补齐 1/8 线程四项官方完整对照。 |
| 2026-10-04，v21 | CPU 单线程连续权重使用八体素采样；修复 4,096/4,097 轴 float32 上边界舍入导致的邻居索引越界。407 项局部回归通过，三种权重的完整搜索统计量和输出逐位一致；补齐 1/8 线程六项官方完整对照。单线程仍慢于官方，八线程本例更快。 |
| 2026-10-04，v16 | CPU 融合采样、保序分箱、独立行并行、单线程八体素向量采样和金字塔准备；309 项局部回归，以及公开完整配准的最终 cost、矩阵、体素位模式与 8,981 次成本评价检查。CPU 官方对照使用相同亲和性和 1/8 线程上限，主配准重复计时与其余功能单次观察分别记录。CUDA 分支保持原实现。 |

## 7. 参考文献、原软件和资源

源码位置：[原软件 `flirt-2111.2/flirt.cc`](../../src/fnit/_vendor_fsl/sources/flirt-2111.2/flirt.cc)；[FNIT `flirt/standalone.py`](../../src/fnit/flirt/standalone.py)。固定 tag/commit、Git tree 及每文件 SHA-256 见[来源清单](../../src/fnit/_vendor_fsl/manifest.json)。

代码改写沿用 [FSL 6.0 非商业许可证](../../licenses/FSL-6.0.txt)，完整来源与再分发要求见[第三方声明](../../THIRD_PARTY_NOTICES.md)。

- 参考文献：Jenkinson et al., *Improved Optimization for the Robust and Accurate Linear Registration and Motion Correction of Brain Images*, NeuroImage (2002), [doi:10.1006/nimg.2002.1132](https://doi.org/10.1006/nimg.2002.1132)。
- 原实现代码库：[FSL `flirt`](https://git.fmrib.ox.ac.uk/fsl/flirt)。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| 本功能无模型权重；用户自备参考影像、掩膜或变换 | 定义目标网格/变换 | 各文件原作者 | 按实际文件 | 按实际文件 | 不随本功能发布用户数据 |

[完整历史说明与调试证据](../../validation/flirt/readme_archive_20261005.md) · [返回主页](../../README.md)

<!-- 旧版文档锚点兼容 -->
<a id="cpu-执行"></a> <a id="输入"></a> <a id="python-单被试调用"></a> <a id="返回值与磁盘输出"></a> <a id="命令行调用"></a> <a id="已知线性变换mni152-分辨率转换"></a> <a id="mat-坐标约定"></a> <a id="gpu-批量执行"></a> <a id="重跑性能与精度对照"></a> <a id="真实数据测量"></a> <a id="2026-10-04-cpu-官方对照"></a> <a id="2026-10-04-gpu-完整回归v25与当前-flirt-源码相同"></a> <a id="本次定位与修复"></a> <a id="此前-gpu-修复的速度与精度"></a> <a id="profile-与保留的参考路径"></a> <a id="其他已验证数据与重采样路径"></a> <a id="公开-openneuro-示例"></a> <a id="本轮-cpu-输出"></a> <a id="此前-gpu-输出"></a> <a id="结果解释"></a> <a id="最近更新"></a> <a id="参考文献与原实现"></a>
