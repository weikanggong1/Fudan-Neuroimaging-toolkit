# 真实流线离线 QC 复现

`plot_real_tracking_qc.py` 读取 FNIT 与 MRtrix3 的 TCK 和同一空间的 5TT，在 CPU 上生成三方向投影与保存折线长度统计。两份 TCK 必须已经位于 5TT 的 RAS 世界毫米空间；脚本不做配准。5TT 只在离线显示时重采样为 1.5 mm RAS 网格，生产 tracking 的重采样及轨迹坐标不受影响。

本例 [脑图](real100k_qc.png) 与 [公开统计](real100k_qc.public.json) 来自 **9c compiled baseline** 的真实输出，保留原生产者标签。本轮正式优化的编译组 TCK SHA 与这份基线相同，见[完整 100k 报告](ordered_compiled_100000.public.json)，因此可沿用该图；图不承担性能计时或 SC 矩阵精度验收。FNIT/MRtrix3 分别保存 27,537/27,685 条流线，平均折线长度为 38.967/40.248 mm，KS D 为 0.019626；完整分位数、输入 SHA-256 和运行库版本见公开统计。

## 输入、输出与参数

在含 NumPy、nibabel、SciPy 和 Matplotlib 的 FNIT Conda 环境中调用。脚本不运行 GPU 或原软件命令，也不下载数据。

| 参数 | 输入意义与格式 | 默认值 |
| --- | --- | --- |
| `--fnit-tck` | FNIT TCK；每条流线为有限的 `[N>=2,3]` RAS 世界毫米坐标 | 必填 |
| `--mrtrix-tck` | MRtrix3 TCK；相同坐标与结构要求 | 必填 |
| `--five-tissue` | `[X,Y,Z,5]` 5TT NIfTI，空间单位为 mm；前三通道 cGM+sGM+WM 作为显示背景 | 必填 |
| `--output-dir` | 写入 `tracking_qc.png` 与 `tracking_stats.json` 的目录 | 必填 |
| `--n-seeds` | 调用者明确提供的每组配置种子预算，正整数；不从 TCK header 或流线数推断 | 必填 |
| `--fnit-label` | 调用者提供的 FNIT 版本或变体标签 | 必填 |
| `--dataset-label` | 调用者提供的数据集名称；名称本身不证明来源或许可 | 必填 |
| `--sample-count` | 每组最多显示的流线数，正整数 | 2000 |
| `--sample-seed` | 两组分别重置为相同值的抽样种子，非负整数 | 20261009 |
| `--metadata-manifest` | 可选 JSON，提供本次来源/许可与三份输入 SHA；数据集标签、FNIT 标签及全部 SHA 必须匹配 | 未提供时来源与许可均为 `unknown` |

PNG 显示轴向、冠状和矢状的世界坐标投影，附 RAS 方向标识；背景为 GM+WM 最大强度投影。每份 TCK 独立用相同抽样规则选取流线，RGB 按端点位移的 R/A/S 方向着色。图是离线投影示例，不能作为 voxel exact 对照。

JSON 记录实际文件 SHA、独立读到的流线数，以及保存折线长度和保存点数的均值、中位数、p5/p25/p75/p95、最小值和最大值。长度直接对相邻保存点的欧氏距离求和，采用 float64；未对流线重新采样。保存点间距或 downsampling 会影响该长度，它不等于软件内部积分长度。两组 RNG 不同，KS D 仅描述长度分布距离；未检验流线配对、端点或结构连接矩阵等价。

## 命令行复现

在仓库根目录运行，替换以下输入和输出变量：

```bash
# 使用当前 FNIT Conda 环境中的 Python。
QC_PYTHON=python
# 本地真实 FNIT 基线 TCK，坐标单位为 RAS 世界毫米。
QC_FNIT_TCK=/path/to/fnit_baseline.tck
# 同一真实数据的 MRtrix3 TCK。
QC_MRTRIX_TCK=/path/to/mrtrix_reference.tck
# 两组流线共同使用的 5TT 解剖图像。
QC_FIVE_TISSUE=/path/to/five_tissue.nii.gz
# 离线 QC 输出目录。
QC_OUTPUT_DIR=/path/to/qc_output

"$QC_PYTHON" validation/connectome/tracking_exact_20261009/plot_real_tracking_qc.py \
  --fnit-tck "$QC_FNIT_TCK" \
  --mrtrix-tck "$QC_MRTRIX_TCK" \
  --five-tissue "$QC_FIVE_TISSUE" \
  --output-dir "$QC_OUTPUT_DIR" \
  --n-seeds 100000 \
  --fnit-label '9c compiled baseline' \
  --dataset-label 'OpenNeuro ds004666' \
  --sample-count 2000 \
  --sample-seed 20261009 \
  --metadata-manifest validation/connectome/tracking_exact_20261009/real100k_qc.case_manifest.json
```

查看帮助：

```bash
python validation/connectome/tracking_exact_20261009/plot_real_tracking_qc.py --help
```

使用其他输入时，填写对应种子预算和标签，并提供针对那些文件的 manifest，或省略 `--metadata-manifest`。本例 manifest 只接受三份实际输入的 SHA-256；不能将其许可元数据套用到其他文件。

## 来源、许可与发布范围

本例来源为 OpenNeuro ds004666（EDDEN）。其 [上游 dataset_description](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004666/master/dataset_description.json) 在本轮核对为 CC0；引用工作为 Manzano-Patron 等，Imaging Neuroscience（2024），[doi:10.1162/imag_a_00060](https://doi.org/10.1162/imag_a_00060)。[本例 manifest](real100k_qc.case_manifest.json) 将来源、许可和版本标签绑定到三份实际输入 SHA，并保留原统计报告 SHA。

本目录只包含绘图代码、派生脑图、标量统计及必要的几何和哈希。TCK、5TT 原文件及实际服务器路径未纳入发布。通用脚本生成的本地 JSON 会记录调用者的本地路径；对外发布时使用当前 `real100k_qc.public.json` 的白名单字段方式。Matplotlib 的临时目录仅存缓存，退出时清理，不含输入副本。

参数化 CLI 已用本例真实输入重现；TCK SHA、计数、全部长度统计、KS、抽样索引哈希和 5TT 几何均与原报告完全一致，并完成一次脑图目视检查。已检查默认来源/许可为 unknown，以及不匹配的输入无法继承本例 manifest 元数据。
