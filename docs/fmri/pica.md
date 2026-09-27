# 单被试空间 PICA（PyTorch）

`decompose_spatial_ica` 对一张 4D BOLD 做单被试空间 ICA。输入须先完成所选 FEAT 预处理；函数本身不做运动校正、畸变校正或配准。它先按体素去时间均值，用初始 PCA 的残差估计体素方差，再做时间 PCA 白化和空间对称 FastICA。默认 `n_components=None` 用平滑度修正后的 Laplace PPCA 选阶；空间成分按残差噪声标准化，Gaussian/正负 Gamma 混合模型给每个体素估计“非背景”后验概率，默认保留概率 ≥0.5 的值。该实现仅依赖 PyTorch、NumPy、nibabel，不在实际处理路径调用 FSL。[MELODIC 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/melodic.html)；[当前源码：PCA](https://git.fmrib.ox.ac.uk/fsl/melodic/-/blob/master/melpca.cc)、[ICA](https://git.fmrib.ox.ac.uk/fsl/melodic/-/blob/master/melica.cc)、[混合模型](https://git.fmrib.ox.ac.uk/fsl/melodic/-/blob/master/melgmix.cc)。

## 输入和调用

`input_bold` 是 4D NIfTI（X×Y×Z×T，T≥6）；`brain_mask` 是同网格 3D NIfTI，非零值表示参与分解的体素。两者的前三维尺寸和 affine 必须一致。`output_dir` 是输出目录。示例中的路径需要换成实际绝对路径。

```python
from fnit import decompose_spatial_ica

ica = decompose_spatial_ica(
    input_bold="/absolute/path/filtered_func_data.nii.gz",  # 已预处理的 4D BOLD；X×Y×Z×T
    brain_mask="/absolute/path/mask.nii.gz",                 # 与 BOLD 同网格的 3D 脑掩膜
    output_dir="/absolute/path/pica",                       # 输出目录；不存在则创建
    n_components=None,                                       # None：Laplace PPCA 自动定阶；整数 K：固定 K
    device="cuda:0",                                        # PyTorch 设备；也可填 "cpu"
    voxel_batch_size=8192,                                  # 每次处理的脑内体素数
    max_iter=500,                                            # 对称 FastICA 的最大迭代数
    tolerance=1e-3,                                          # 正交化变化量停止阈值
    random_state=0,                                          # ICA 初始矩阵的随机种子
    mm_threshold=0.5,                                        # 非背景后验概率阈值
)
print(ica.n_components)       # 实际成分数 K
print(ica.thresholded_maps)   # 概率阈值后的 IC 图
print(ica.mixing)             # T×K 成分时间序列
```

| 输出 | 内容 |
| --- | --- |
| `ica_components_z.nii.gz` | X×Y×Z×K，float32；按拟合背景分布重新标定的空间成分图。 |
| `ica_components_signal_probability.nii.gz` | X×Y×Z×K，float32；非背景成分后验概率，取值 0–1。 |
| `ica_components_pica_thresholded.nii.gz` | X×Y×Z×K，float32；概率小于 `mm_threshold` 的体素置零，供 ICA-AROMA 空间特征使用。 |
| `ica_mixing.tsv` | T×K，成分时间序列；每列对应同编号空间图。 |
| `ica_frequency_power.tsv` | ⌊T/2⌋×K，去掉零频的功率谱。 |

返回的 `ICAResult` 提供上述五个绝对路径，以及 `n_components`、参与分解的 `n_voxels`、`n_iterations`、`converged`、`final_decorrelation_change`、`pca_variance_explained`。`model_order_method` 为 `laplace_ppca` 或 `fixed`；`estimated_resels` 仅在自动定阶时有值。运行完应检查 `converged` 和空间图、时间序列；自动定阶不是质量保证。

## 官方对照命令

以下命令用于独立验证，不是 FNIT 运行依赖。`-d 0` 对应 `n_components=None`；固定 K 时改为 `-d K`。`--nobet -m` 使用同一掩膜；`--mmthresh=0.5` 对应 `mm_threshold=0.5`。示例 TR 需改成实际数据值。

```bash
melodic -i /absolute/path/filtered_func_data.nii.gz \
  -o /absolute/path/melodic_ref -m /absolute/path/mask.nii.gz \
  --nobet --bgthreshold=3 --tr=0.735 -d 0 \
  --Ostats --mmthresh=0.5
```

当只核对 PCA/ICA 而不运行混合模型时，加 `--no_mm`。只核对一张已生成的 IC 图的混合模型时，可按 [MELODIC 官方用法](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/melodic.html)使用 `--ICs=<单张IC图> --mix=<一值文本>`。

## 真实影像核对

使用同一例 UKB 原始 rfMRI（88×88×64×490，TR 0.735 秒）和同一张 113,659 体素掩膜。这里故意让两者读同一输入，避免上游 FEAT 差异混入 PICA 核对；原始影像本身不是建议的最终去噪输入。FSL 参考运行使用 `-d 0 --no_mm`。匹配 IC 时先用匈牙利算法配对，再取空间图和时间序列 Pearson 相关的绝对值，因此消除了 ICA 成分排列、符号的不确定性。

| 检查 | FNIT | FSL 参考 | 对照 |
| --- | --- | --- | --- |
| 自动成分数 | 89 | 89 | 一致。 |
| 平滑度 resels | 0.497094 | `smoothest` 0.4971 | 一致。 |
| ICA 时间序列 | 90.74 秒，峰值 PyTorch 已分配显存 0.283 GB | 核心文件在 91.73 秒出现，进程退出 255 | 匹配后 r 中位 0.9570、5% 分位 0.4428。 |
| ICA 空间图 | 89 张 | 89 张 | 匹配后 r 中位 0.9566、5% 分位 0.3791。 |
| PICA 概率图 | 全部有限，范围 0–1 | 核心对照加了 `--no_mm`，无整例官方概率图 | FNIT 概率 ≥0.5 后保留 357,036 / 10,115,740 个脑内 IC 体素。 |

上表的 FNIT 运行采用默认 `tolerance=1e-3`、`max_iter=500`，第 89 次迭代收敛，最终正交化变化量为 0.000956。FSL 返回 255，91.73 秒只是核心输出检查点，不能当作成功整例耗时。峰值显存仅统计 PyTorch 已分配张量，并非进程总显存。

混合模型另用同一张 FSL 第 1 个 IC 图做输入，对照官方 `--ICs` 单图模式：FNIT 与 FSL 后验概率 r=0.9980、MAE=0.0101，0.5 阈值 Dice=0.9533；共同保留体素的 Z 图 r≈1、MAE=0.0758。FNIT 单图 EM 为 0.26 秒，FSL 命令为 1.69 秒；FSL 进程仍退出 255，但概率图和阈值图完整可读。上述只说明该 IC 的混合模型接近，不能外推到所有 89 张图。

## 仍有差异

FSL 的 PPCA、IC 求解及混合模型含重启、Gaussian 混合模型回退和更多推断分支；此处实现单次对称 ICA 与三类 Gaussian/Gamma 推断，尚未逐项复现所有分支。自动定阶一致、IC 中位相关较高不代表每张 IC 一致；5% 分位数反映仍有明显不同的成分。本次标量和源码哈希见[机器可读汇总](../../validation/fmri/pica_summary.json)。具体源码版本、随机初始化、停止阈值、FSL 异常退出和完整阈值图比较都应随基准记录。ICA-AROMA 的分类结果因此不能直接称为与官方逐成分相同。
