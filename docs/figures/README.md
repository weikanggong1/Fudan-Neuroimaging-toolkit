# 当前实现与原软件的影像对照

本页汇总当前源码对应的单被试示意图。数值结论以各功能页和机器可读报告为准。公开样例的来源和 SHA-256 见 [T1w 清单](../../examples/data/SOURCES.json)与 [FLAIR 清单](../../examples/wmh_data/SOURCES.json)；临床回归图只发布去标识化衍生切面，不发布原始影像或病例标识。

## SynthStrip

![当前 SynthStrip 与 FreeSurfer 8.2 的真实 T1w 脑提取对照](../synthstrip/figures/synthstrip_current_real_case01.png)

当前 12 例回归的最低 mask Dice 为 `0.993925`，最低 brain-image Pearson r 为 `0.994261`。完整方法、计时和限制见 [SynthStrip 功能页](../synthstrip/README.md)。

## SynthMorph

![当前 SynthMorph 与 FreeSurfer 8.2 的公开 T1w 配准对照](../synthmorph/figures/synthmorph_public_current.png)

当前 12 例回归的最低 moved-image Pearson r 为 `0.994337`，平均位移向量误差均值为 `0.079139 mm`。公开 OpenNeuro 示例和完整真实数据统计见 [SynthMorph 功能页](../synthmorph/README.md)。

## WMH-SynthSeg

![当前 WMH-SynthSeg 与 FreeSurfer 8.2 的公开 FLAIR 对照](wmh_synthseg_comparison.png)

三幅公开 FLAIR 的最低全标签一致率为 `0.99997810`，最低 WMH Dice 为 `0.99981002`。完整方法、计时和显存见 [WMH-SynthSeg 功能页](../wmh_synthseg/README.md)。

## TorchFLIRT

![当前 TorchFLIRT 与 FSL FLIRT 6.0.7.4 的公开 T1w 配准对照](../flirt/figures/flirt_public_current.png)

当前发布源码的 10 例真实 GM 对照中，CPU 与默认 TF32 GPU 分别有 9/10 例满足矩阵 `rmsdiff <= 0.05 mm`；moved Pearson 中位数分别为 `0.999985` 和 `0.999823`。公开 OpenNeuro 示例、完整时间和误差边界见 [TorchFLIRT 功能页](../flirt/README.md)。
