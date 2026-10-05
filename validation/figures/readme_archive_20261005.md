<!-- 历史资料：迁移前说明，当前接口与结论以 docs 中用户手册为准。 -->

# 当前实现与原软件的影像对照

本页汇总当前源码对应的单被试示意图。数值结论以各功能页和机器可读报告为准。公开样例的来源和 SHA-256 见 [T1w 清单](../../examples/data/SOURCES.json)与 [FLAIR 清单](../../examples/wmh_data/SOURCES.json)；临床回归图只发布去标识化衍生切面，不发布原始影像或病例标识。

## SynthStrip

![当前 SynthStrip 与 FreeSurfer 8.2 的真实 T1w 脑提取对照](../../docs/synthstrip/figures/synthstrip_current_real_case01.png)

当前 12 例回归的最低 mask Dice 为 `0.993925`，最低 brain-image Pearson r 为 `0.994261`。完整方法、计时和限制见 [SynthStrip 功能页](../../docs/synthstrip/README.md)。

## SynthMorph

![当前 SynthMorph 与 FreeSurfer 8.2 的公开 T1w 配准对照](../../docs/synthmorph/figures/synthmorph_public_current.png)

当前 12 例回归的最低 moved-image Pearson r 为 `0.994337`，平均位移向量误差均值为 `0.079139 mm`。公开 OpenNeuro 示例和完整真实数据统计见 [SynthMorph 功能页](../../docs/synthmorph/README.md)。

## WMH-SynthSeg

![当前 WMH-SynthSeg 与 FreeSurfer 8.2 的公开 FLAIR 对照](../../docs/figures/wmh_synthseg_comparison.png)

三幅公开 FLAIR 的最低全标签一致率为 `0.99997810`，最低 WMH Dice 为 `0.99981002`。完整方法、计时和显存见 [WMH-SynthSeg 功能页](../../docs/wmh_synthseg/README.md)。

## TorchFLIRT

![当前 TorchFLIRT GPU 批量实现与 FSL FLIRT 6.0.7.4 的公开 T1w 配准对照](../../docs/flirt/figures/flirt_public_current.png)

图中使用同一组 CC0 公开 OpenNeuro ds000114 v1.0.2 去面部 T1w，比较 FSL 与当前 FNIT GPU 批量结果。修复 header 采样距离与搜索层级后，官方输出非零区域的 moved Pearson 为 `0.9999965`，moving 视野 13³ 个世界坐标点的位移 RMS 为 `0.01355 mm`。源码绑定、CPU/FSL 对照、性能与精度范围见 [TorchFLIRT 功能页](../../docs/flirt/README.md)。
