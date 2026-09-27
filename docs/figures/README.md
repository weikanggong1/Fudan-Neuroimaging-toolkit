# 官方实现与 PyTorch 实现的影像对照

本页只展示单被试输出。T1w 样例来自 OpenNeuro ds000114，FLAIR 样例来自 OpenNeuro ds003592；仓库发布的是按各数据许可制作的去标识衍生文件，来源与 SHA-256 分别见 [T1w 清单](../../examples/data/SOURCES.json)和 [FLAIR 清单](../../examples/wmh_data/SOURCES.json)。这些影像没有人工标注，图和指标用于比较原版与本包输出。

![原始 T1w 与两种脑提取结果](synthstrip_comparison.png)

SynthStrip 图使用 `sub-02`。从左到右是输入、FreeSurfer 输出和本包输出；上、下是同一原网格的轴位和冠状位切面。两种输出的掩膜 Dice 为 1，不一致体素为 0，脑图最大绝对差为 0，输出几何相同。

![移动与固定影像及两种 joint 配准结果](synthmorph_comparison.png)

SynthMorph 图以 `sub-02` 为 moving、`sub-01` 为 fixed，使用默认 `joint` 模型。从左到右是 moving、fixed、FreeSurfer 配准结果和本包配准结果；两列配准结果都位于 fixed 网格。完整保存影像的 NRMSE 为 `1.7880e-5`，RAS 位移向量最大误差为 `0.000622 mm`，输出几何相同。

![同一公开 FLAIR 的 FreeSurfer 与本包 WMH-SynthSeg 输出](wmh_synthseg_comparison.png)

WMH-SynthSeg 图使用公开 `sub-04` FLAIR。中、右列分别叠加 FreeSurfer 与本包输出的标签 77；两次单被试推理使用同一官方 checkpoint、CUDA 和 `--crop`。完整三维输出的标签不一致体素为 0，WMH Dice 为 1，概率图最大绝对差为 0，数值仿射相同。机器可读指标见 [WMH 图示数据](wmh_metrics.json)。

当前 SynthStrip 数值核对见[无 Surfa 迁移验证](../../validation/synthstrip_no_surfa_20260927/README.md)；本页 SynthStrip 图为此前公开样例的同权重输出。SynthMorph 既有 12 例汇总见[历史汇总](../../benchmark/public_report/summary.md)，WMH 的 12 例报告见[WMH 报告](../../validation/wmh/README.md)。FastVBM 的当前数值验证见 [0.9 正式报告](../../validation/fast_vbm/README.md)。
