# FNIT MELODIC：单被试空间PICA

| 项目 | 内容 |
|---|---|
| 输入 | 已预处理4D BOLD及同网格mask |
| 输出 | 成分/概率/阈值图与T×K时序 |
| 对应原软件 | FSL MELODIC2601.1单被试symm/pow3 |
| Python / CLI | run_melodic_bids、decompose_spatial_ica；无独立CLI |
| CPU / GPU | CPU/CUDA；PCA/ICA float64，空间图float32 |

## 1. 功能简介

`run_melodic_bids` 在一张已预处理4D BOLD中提取空间独立成分，并将图与混合时序写回BIDS Derivatives。`decompose_spatial_ica` 是同一内核的普通文件入口，fMRI volume流程也复用它。

对应FSL MELODIC2601.1单被试对称pow3 ICA、Laplace PPCA自动定阶及Gaussian/正负Gamma混合模型。读写用nibabel，PCA/ICA与混合模型用PyTorch；运行时不调用FSL。PCA/白化/ICA明确使用float64，避免TF32改变分解；空间图写float32。其他模块的TF32设置不因此改动。

## 2. Python 调用

```python
from fnit.melodic import run_melodic_bids

component_result = run_melodic_bids(
    source_derivatives_root="/data/bids/derivatives/preproc",  # 输入预处理数据集
    derivatives_root="/data/bids/derivatives/fnit-melodic",  # 输出数据集
    input_bold="/data/bids/derivatives/preproc/sub-001/func/sub-001_task-rest_desc-preproc_bold.nii.gz",  # 4D BOLD
    brain_mask="/data/bids/derivatives/preproc/sub-001/func/sub-001_task-rest_desc-brain_mask.nii.gz",  # 同网格脑mask
    n_components=None,  # 自动选择K；整数则固定K
    device="cuda:0",  # 计算设备
    overwrite=False,  # 保留已有输出
)
selected_component_count = component_result.n_components  # 实际K
fastica_converged = component_result.converged  # 检查迭代是否收敛
```

### 输入数据格式

- BOLD：`[X,Y,Z,T]` NIfTI，已完成所需预处理，有限值；时间顺序与源扫描一致。可以处于native或MNI，入口不重配准或重采样。
- mask：`[X,Y,Z]`非零NIfTI，shape、affine、orientation与BOLD相同；所有图沿用源网格。
- 自动定阶要求T≥6。输入空间范围、时间长度和噪声均影响K，自动K不是质量保证。
- BIDS入口要求源根有`dataset_description.json`及`DatasetLinks.raw`，输入位于其`sub-*/[ses-*]/func/`中，实体用于输出命名。
- 普通文件入口没有BIDS结构要求；BOLD信号沿用输入强度单位，经算法噪声归一化后输出空间Z分数。

普通文件内核调用示例：

```python
from fnit.melodic import decompose_spatial_ica

file_result = decompose_spatial_ica(
    input_bold="/data/preproc/bold.nii.gz",  # 已预处理4D BOLD
    brain_mask="/data/preproc/mask.nii.gz",  # 同网格3D mask
    output_dir="/data/results/ica",  # 普通文件结果目录
    n_components=30,  # 固定K，需与数据秩相容
    device="cuda:0",  # 计算设备
)
```

**`run_melodic_bids` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `source_derivatives_root` | 是 | `路径` | `—` | 已有预处理BIDS Derivatives根，含DatasetLinks.raw。 |
| `derivatives_root` | 是 | `路径` | `—` | 输出BIDS Derivatives根目录。 |
| `input_bold` | 是 | `路径` | `—` | 4D已预处理BOLD路径，不在此入口做运动校正或配准。 |
| `brain_mask` | 是 | `路径` | `—` | 与BOLD同shape、affine和orientation的3D非零脑掩膜。 |
| `n_components` | 否 | `int或None` | `None` | 请求的成分数；约束随算法/数据规模，MELODIC的None表示自动定阶。 |
| `device` | 否 | `str/torch.device/None` | `None` | PyTorch 设备；None 自动选择可用 CUDA，否则 CPU。 |
| `voxel_batch_size` | 否 | `int` | `8192` | 每批读入GPU的脑内体素数，正整数。 |
| `max_iter` | 否 | `int` | `500` | 最大迭代/epoch数；正整数，可能按收敛规则提前结束。 |
| `tolerance` | 否 | `float` | `0.001` | FastICA正交化变化停止阈值；默认1e-3。 |
| `random_state` | 否 | `int` | `0` | 初始化和数据划分随机种子。 |
| `mm_threshold` | 否 | `float` | `0.5` | 非背景后验概率阈值，范围0–1。 |
| `overwrite` | 否 | `bool` | `False` | 是否允许覆盖已有结果；默认已有结果时报错。 |

**`decompose_spatial_ica` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `input_bold` | 是 | `str / Path` | `—` | 4D已预处理BOLD路径，不在此入口做运动校正或配准。 |
| `brain_mask` | 是 | `str / Path` | `—` | 与BOLD同shape、affine和orientation的3D非零脑掩膜。 |
| `output_dir` | 是 | `str / Path` | `—` | 本次结果目录；路径按当前工作目录解析。 |
| `n_components` | 否 | `int / None` | `None` | 请求的成分数；约束随算法/数据规模，MELODIC的None表示自动定阶。 |
| `device` | 否 | `str / torch.device / None` | `None` | PyTorch 设备；None 自动选择可用 CUDA，否则 CPU。 |
| `voxel_batch_size` | 否 | `int` | `8192` | 每批读入GPU的脑内体素数，正整数。 |
| `max_iter` | 否 | `int` | `500` | 最大迭代/epoch数；正整数，可能按收敛规则提前结束。 |
| `tolerance` | 否 | `float` | `0.001` | FastICA正交化变化停止阈值；默认1e-3。 |
| `random_state` | 否 | `int` | `0` | 初始化和数据划分随机种子。 |
| `mm_threshold` | 否 | `float` | `0.5` | 非背景后验概率阈值，范围0–1。 |

### 输出

```text
derivatives/fnit-melodic/
├── dataset_description.json
└── sub-001/func/
    ├── *_desc-FNITMELODIC_components.nii.gz
    ├── *_desc-FNITMELODICprob_components.nii.gz
    ├── *_desc-FNITMELODICthresh_components.nii.gz
    ├── *_desc-FNITMELODIC_mixing.tsv
    └── *_desc-FNITMELODIC_decomposition.json
```

| 输出 | 格式与含义 |
|---|---|
| components | `[X,Y,Z,K]` float32；背景分布重新标定的空间Z图。 |
| prob | 同shape float32；非背景后验概率0–1。 |
| thresh | 同shape float32；后验小于mm_threshold的体素置零。 |
| mixing.tsv | `[T,K]`；melodic_0至melodic_(K-1)，样本标准差归一化为1。 |
| decomposition及图旁JSON | 算法、源路径、K、收敛状态和列定义。 |

三张NIfTI保留源BOLD的空间shape、affine、orientation；不是形变或标签图。BIDS实体sub/ses/task/run/space沿用输入、desc被替换。`MelodicBIDSResult`返回五个绝对路径、K与converged。

普通内核的`ICAResult`另有Fourier功率文件、迭代数、最终变化量及PCA解释量；其文件布局见[ica.py](../../src/fnit/melodic/ica.py)。空间成分排序或符号变化时，应先按时间序列配对再比较脑图。

普通文件入口实际产生以下五个文件，不写BIDS dataset_description：

```text
ica/
├── ica_components_z.nii.gz
├── ica_components_pica_thresholded.nii.gz
├── ica_components_signal_probability.nii.gz
├── ica_mixing.tsv
└── ica_frequency_power.tsv
```

普通mixing与frequency TSV没有表头；mixing是float64 `[T,K]` 的文本保存，功率是 `[ceil(T/2),K]`，奇数T先补一个零再做FFT并去掉DC项。功率入口未接收TR，所以行索引是频率bin，不自动标Hz。

运行后先查看 `converged`、`n_iterations` 和 `final_decorrelation_change`；达到max_iter仍未收敛的结果不应按收敛分解解释。增加迭代预算后要保存为另一次结果，并重新进行成分匹配。

BIDS输入的BOLD与mask必须位于同一个func目录，除desc外实体完全匹配。输入根的DatasetType应为derivative，DatasetLinks.raw指向的原始数据集也须有dataset_description.json。

## 3. 命令行调用

本模块没有独立`fnit-melodic`或`fnit melodic`命令。使用以上公开Python入口；完整BOLD流水线的CLI见[fMRI用户手册](../fmri/README.md)。

| 需求 | 公开入口 |
|---|---|
| 对单个普通BOLD文件分解 | `decompose_spatial_ica(...)` |
| 对单个BIDS Derivatives BOLD分解并写BIDS | `run_melodic_bids(...)` |
| 从原始BOLD运行全预处理及ICA | fMRI pipeline入口，处理范围不同 |

Python示例可保存为用户脚本后用主页Conda环境运行。不能把没有提供的CLI开关写成可执行命令。

## 4. 原软件调用

在隔离FSL环境中读取同一BOLD与mask。TR改成实际扫描值。

```bash
melodic -i /data/preproc/bold.nii.gz -m /data/preproc/mask.nii.gz    -o /data/reference/melodic --nobet --bgthreshold=3 --tr=0.735    -d 0 --dimest=lap --nl=pow3 --eps=0.001 --maxit=500 --seed=0    --Ostats --mmthresh=0.5
```

| FNIT | FSL MELODIC |
|---|---|
| n_components=None/整数K | -d 0/-d K |
| max_iter / tolerance / random_state | --maxit / --eps / --seed |
| brain_mask / mm_threshold | --nobet -m / --mmthresh |
| 已支持内核 | 单被试symm、pow3、dimest=lap、Gaussian/Gamma |

未覆盖原软件多被试MIGP/TICA、其他对比函数或Gaussian混合模型回退。`--no_mm`只比较PCA/ICA，处理范围与包含混合模型的FNIT默认调用不同。

## 5. 最新精度和运行时间

最新独立内核对照见[固定真实输入报告](../../validation/fmri/ica_fixed_input.public.json)，绑定ica.py SHA `66b8cfeb…`。同一例490帧、TR0.735 s、99,372脑体素；两边使用相同原FEAT BOLD/mask，参考MELODIC2601.1。本轮没有按140c3739重跑影像或原命令。

| 同输入精度 | 配对结果 |
|---|---:|
| 自动K与ICA迭代 | 双方95成分、40次迭代 |
| 时间序列绝对r中位数/最低 | 0.999999978 / 0.999998346 |
| 空间图绝对r中位数 | 0.999999970 |
| 阈值支持Dice中位数/最低 | 0.999562 / 0.993478 |
| Fourier功率相对L2差 | 0.000414 |

成分先以匈牙利算法匹配并统一符号；61/62排序互换。FNIT计算与写出117.49 s、peak allocated0.504 GB，GPU为共享H100。此控制未重新计时原命令、未独立记录所有阶段和线程，因此无匹配端到端速度比；PCA/ICA为float64，图float32。它不含上游运动、脑提取或配准。

![同真实输入MELODIC与FNIT空间成分及绝对差](figures/ica_fixed_input.png)

图通过同一原配准场进入MNI2mm，只显示原报告绑定的三个匹配成分。完整原流水线的旧651.87 s还包含HTML/统计/绘图，边界不同。fMRI全链新结果不能当成独立ICA重新测量。

## 6. 最近版本和 benchmark

| 日期 | commit/version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-01 | 1d3c0916 | 发布组件控制和490帧验证 | 固定输入与ICA-AROMA分类控制 |
| 2026-10-01 | a720bcf2 | 修正中心化、RNG、PPCA和混合模型尾部 | 95成分同输入对照；高相关与支持差异均保留 |
| 2026-09-30 | 历史源码见原报告 | 单被试PICA接入BIDS/volume | 旧时间序列r中位数0.899301，归档保留 |

更早的debug、profiling和长表保留在[旧README归档](../../validation/melodic/readme_archive_20261005.md)。归档已修复相对链接；旧科学报告与原始产物不修改。

<a id="输入和调用"></a>
<a id="官方对照命令"></a>
<a id="固定真实输入与原-melodic-对照"></a>
<a id="复现固定输入核对"></a>
<a id="实现范围"></a>
<a id="参考文献与原实现"></a>

## 7. 参考文献、原软件和资源

- Beckmann与Smith，2004，[Probabilistic Independent Component Analysis for fMRI](https://doi.org/10.1109/TMI.2003.822821)。
- 原软件：[MELODIC官方文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/melodic.html)、[2601.1代码](https://git.fmrib.ox.ac.uk/fsl/melodic/-/tree/2601.1)，`melpca.cc/melica.cc/melgmix.cc`。
- FNIT：[ica.py](../../src/fnit/melodic/ica.py)、[bids.py](../../src/fnit/melodic/bids.py)。输出结构参照[BIDS时空分解](https://bids-specification.readthedocs.io/en/bep012/derivatives/functional-derivatives.html)。

### 外部资源

本功能不需要预训练权重、图谱或模板，也不自动下载真实输入数据。用户输入与参考软件的许可由各自来源决定。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| 无额外模型资源 | — | — | 不适用 | 不适用 | 不适用 |
