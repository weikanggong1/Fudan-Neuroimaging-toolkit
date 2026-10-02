# FNIT MELODIC：单被试空间 PICA

[返回首页](../../README.md) · [源码目录](../../src/fnit/melodic/) · [固定输入验证](../../validation/fmri/ica_fixed_input.public.json)

`run_melodic_bids` 对一张已预处理的 4D BOLD 做单被试空间 PICA，从 BIDS Derivatives 读取 BOLD 和同网格脑掩膜，将成分图与时间序列写回 BIDS Derivatives。算法内核是 `decompose_spatial_ica`，volume 流程调用同一个内核。数据读写用 nibabel，PCA、ICA 和混合模型用 PyTorch；FNIT 运行时不调用 FSL。

计算顺序对应 FSL MELODIC 2601.1 的单被试 `symm`、`pow3`、`dimest=lap` 分支：

1. 每个体素去时间均值；初始 30 维 PCA 用阈值 2.3 的空间得分估计残差标准差，再除以该标准差进行方差归一化。
2. 空间去均值只用于计算 PCA 协方差，ICA 的输入保留归一化数据的全脑平均时间过程。自动定阶采用原实现的离散 Marchenko–Pastur 谱修正和 Laplace PPCA，而固定整数 `n_components` 直接选择 K 维。
3. 用与 Linux FSL `srand`/`unifrnd` 一致的随机序列初始化，对称 FastICA 迭代至 `tolerance` 或 `max_iter`。PCA 协方差、白化和 ICA 采用 float64，避免 TF32 改变分解结果；其他函数的 TF32 设置不变，写出的空间图仍是 float32。
4. 时间序列按样本标准差归一化为 1，翻转符号使空间图最大绝对值为正，再按空间得分标准差排序。空间成分除以残差噪声及时间解混矩阵的行范数。
5. 拟合一个 Gaussian 背景及正、负 Gamma 分布，按拟合背景重新标定 Z 分数，保留非背景后验概率达到 `mm_threshold` 的体素。原实现拟合时使用的 `1e-4` 似然下限不会用于最终后验比值，极端信号体素的后验概率因此得到保留。

## 输入和调用

`source_derivatives_root` 是已有预处理 BIDS Derivatives 的根目录，必须含 `dataset_description.json` 和指向原始 BIDS 的 `DatasetLinks.raw`。`input_bold` 是其 `sub-*/func/` 下 4D `*_bold.nii.gz`（X×Y×Z×T，自动定阶需要 T≥6）；`brain_mask` 是同目录、同网格的 3D `*_mask.nii.gz`，非零体素参与分解。`derivatives_root` 是本功能输出的 BIDS Derivatives 根目录。三张成分图与混合矩阵用源 BOLD 的 sub、ses、task、run、space 等实体命名，替换原 `desc`。示例中的绝对路径需换成实际数据。

```python
from fnit.melodic import run_melodic_bids

ica = run_melodic_bids(
    source_derivatives_root="/absolute/path/bids/derivatives/preproc",  # 已预处理 BIDS Derivatives 根目录
    derivatives_root="/absolute/path/bids/derivatives/fnit-melodic",    # 输出 BIDS Derivatives 根目录
    input_bold="/absolute/path/bids/derivatives/preproc/sub-0001/func/sub-0001_task-rest_desc-preproc_bold.nii.gz",  # 4D BOLD
    brain_mask="/absolute/path/bids/derivatives/preproc/sub-0001/func/sub-0001_task-rest_desc-brain_mask.nii.gz",  # 同网格 3D 掩膜
    n_components=None,       # None：Laplace PPCA 自动定阶；整数 K：固定 K
    device="cuda:0",        # PyTorch 设备；也可填 "cpu"
    voxel_batch_size=8192,   # 每次处理的脑内体素数
    max_iter=500,             # 对称 FastICA 的最大迭代数
    tolerance=1e-3,           # 正交化变化量停止阈值
    random_state=0,           # Linux FSL 同序随机初始化种子，对应 --seed=0
    mm_threshold=0.5,         # 非背景后验概率阈值
    overwrite=False,          # 是否覆盖同名结果
)
print(ica.n_components)       # 实际成分数 K
print(ica.thresholded)        # 概率阈值后的 IC 图
print(ica.mixing)             # T×K 成分时间序列
```

| 输出 | 内容 |
| --- | --- |
| `*_desc-FNITMELODIC_components.nii.gz` | X×Y×Z×K，float32；按拟合背景分布重新标定的空间成分图。 |
| `*_desc-FNITMELODICprob_components.nii.gz` | X×Y×Z×K，float32；非背景成分后验概率，取值 0–1。 |
| `*_desc-FNITMELODICthresh_components.nii.gz` | X×Y×Z×K，float32；概率小于 `mm_threshold` 的体素置零。 |
| `*_desc-FNITMELODIC_mixing.tsv` | T×K，成分时间序列；表头为 `melodic_0` 至 `melodic_(K-1)`。 |
| `*_desc-FNITMELODIC_decomposition.json` | 算法、源 BIDS 路径、成分数、收敛状态、阈值和时间序列列说明。 |

`MelodicBIDSResult` 返回上述五个绝对路径，以及 `n_components` 和 `converged`。三张图各有相邻 JSON；输出根目录的 `dataset_description.json` 记录源预处理数据集和原始 BIDS 数据集。算法内核的频谱文件只在运行时工作目录存在；运行完应检查收敛状态和空间图、时间序列。自动定阶不是质量保证。

## 官方对照命令

以下命令用于独立验证，不是 FNIT 运行依赖。`-d 0` 对应 `n_components=None`；固定 K 时改为 `-d K`。`--nobet -m` 使用同一掩膜；`--mmthresh=0.5` 对应 `mm_threshold=0.5`。示例 TR 需改成实际数据值。

```bash
melodic -i /absolute/path/filtered_func_data.nii.gz \
  -o /absolute/path/melodic_ref -m /absolute/path/mask.nii.gz \
  --nobet --bgthreshold=3 --tr=0.735 -d 0 \
  --dimest=lap --nl=pow3 --eps=0.001 --maxit=500 --seed=0 \
  --Ostats --mmthresh=0.5
```

当只核对 PCA/ICA 而不运行混合模型时，加 `--no_mm`。只核对一张已生成的 IC 图的混合模型时，可按 [MELODIC 官方用法](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/melodic.html)使用 `--ICs=<单张IC图> --mix=<一值文本>`。

## 固定真实输入与原 MELODIC 对照

新版内核在一例完整 490 帧、TR 0.735 秒的 UKB 数据上核对。两边读取完全相同的原软件 FEAT `filtered_func_data.nii.gz` 和 99,372 体素 EPI 掩膜；因此这项测量隔离了 ICA，未把运动校正、脑提取或配准差异混入结果。原参照为 MELODIC 2601.1，显式设置 `--dimest=lap --nl=pow3 --eps=0.001 --maxit=500 --seed=0 --mmthresh=0.5`。

时间序列先去均值、L2 归一化，再用匈牙利算法最大化配对后的绝对 Pearson r。空间图也按该配对和符号核对；Dice 计算阈值图非零体素的重合率。

| 指标 | FNIT 当前内核 | 原 MELODIC | 对照结果 |
| --- | --- | --- | --- |
| 自动成分数 | 95 | 95 | 相同。 |
| ICA 迭代 | 40 | 40 | 相同；最终变化量分别为 0.00097617、0.00097589。 |
| 成分顺序与符号 | 95 张 | 95 张 | 61/62 号成分排序互换，其他顺序相同；配对后符号全部相同。 |
| 成分时间序列 | T×95，样本标准差为 1 | T×95 | r 中位数 **0.999999978**，5% 分位 0.999999481，最低 0.999998346。 |
| 背景标准化空间图 | X×Y×Z×95 | X×Y×Z×95 | r 中位数 **0.999999970**；每成分 RMSE 中位数 0.000476。 |
| 概率阈值图 | 后验 ≥0.5 | `stats/thresh_zstat*` | 非零支持集 Dice 中位数 **0.999562**，最低 0.993478。 |
| Fourier 功率 | 245×95 | `melodic_FTmix`，245×95 | 配对后相对 L2 差 0.000414。 |
| FNIT 计算与写出 | 117.49 秒，CUDA 分配峰值 0.504 GB | 此控制不重新计时原命令 | 单次共享 GPU 测量；PCA/ICA 为 float64，全局 TF32 保持开启。 |

完整标量、输入与代码哈希见[固定输入控制](../../validation/fmri/ica_fixed_input.public.json)。修复前，同样原输入、同样掩膜的 ICA 时间序列 r 中位数为 0.899301，5% 分位为 0.308246，迭代 74 次；见[同输入修复前控制](../../validation/fmri/ica_fixed_input_before.public.json)。新版将错误的额外空间去均值、不同 RNG、TF32 分解、PPCA 尾谱索引以及输出缩放与排序逐项修正。两次控制的计时分别为 74.39 和 117.49 秒，不代表在相同共享 GPU 负载下测得的速度比。

完整原软件 MELODIC 命令的旧连续流水线测量为 651.87 秒，其中还生成 HTML 报告、单成分统计和图片，计时边界不同。流水线原始输入、步骤及完整耗时见[匹配原步骤的端到端比较](../../validation/fmri/matched_native.md)。

进一步固定原运动参数、原 BBR/FNIRT 变换和 ICA-AROMA 掩膜，只替换为新版 FNIT 的 ICA 输出，得到 95/95 成分和 50/50 噪声成分；配对后的 95 个噪声/信号标签全部一致，频率特征完全相同。四种特征的误差见[分类控制](../../validation/fmri/aroma_corrected_ica_control.public.json)。

下图展示三个匹配成分通过同一原配准场进入 MNI 2 mm 后的阈值图。每列选择原成分非零体素最多的轴位层；上下两幅使用同一切片、同一色标。第三行为绝对差，色标上限 0.01，超出上限的值显示为最亮色；不额外平滑。图像来源和哈希见[图示记录](../../validation/fmri/ica_figure.public.json)。

![同原输入的 MELODIC 与 FNIT 空间成分及绝对差](figures/ica_fixed_input.png)

### 复现固定输入核对

先用上面的原 MELODIC 命令生成参照目录，并让两边使用同一份 BOLD 与掩膜。以下脚本只运行 FNIT，读取已有参照文件进行比较；输出目录中的影像和逐成分配对文件用于本地检查，公开摘要仅含匿名统计和哈希。

```bash
python validation/fmri/compare_ica_fixed_input.py \
  --input-bold /absolute/path/filtered_func_data.nii.gz \
  --brain-mask /absolute/path/mask.nii.gz \
  --reference-dir /absolute/path/melodic_ref \
  --output-dir /absolute/path/fnit_ica_control \
  --device cuda:0
```

[7 项内核与接口测试](../../validation/fmri/ica_tests.public.json)检查相同 Linux RNG、混合矩阵单位样本标准差、奇数帧 FFT 补零、极端 Gamma 后验、NIfTI 网格与头信息，以及 BIDS 文件和来源链接。这些合成单元测试不替代上面的真实影像比较。

### 实现范围

本内核支持单被试对称 `pow3` ICA、Laplace 自动定阶和三类 Gaussian/Gamma 推断。原 MELODIC 的多被试 MIGP/TICA、其他对比函数及 Gaussian 混合模型回退尚未纳入。该真实参照的 95 个成分均使用 Gaussian/Gamma 分支；上述高相关和阈值 Dice 对应该数据与参数，空间得分的小数值差异仍会让少量接近 0.5 的体素改变是否保留。

## 参考文献与原实现

- Beckmann 与 Smith，*Probabilistic Independent Component Analysis for Functional Magnetic Resonance Imaging*，IEEE TMI，2004，[DOI](https://doi.org/10.1109/TMI.2003.822821)。
- 原实现：[FSL MELODIC 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/melodic.html)、[PCA 源码](https://git.fmrib.ox.ac.uk/fsl/melodic/-/blob/2601.1/melpca.cc)、[ICA 源码](https://git.fmrib.ox.ac.uk/fsl/melodic/-/blob/2601.1/melica.cc)、[混合模型源码](https://git.fmrib.ox.ac.uk/fsl/melodic/-/blob/2601.1/melgmix.cc)。
- 输出命名参照 [BIDS 功能导数中的时空分解格式](https://bids-specification.readthedocs.io/en/bep012/derivatives/functional-derivatives.html)。
