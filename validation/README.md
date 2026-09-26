# 验证记录

每个功能只保留当前使用的说明与最新公开 benchmark：

- SynthStrip / SynthMorph：[`benchmark/public_report/summary.md`](../benchmark/public_report/summary.md) 与机器可读 CSV。
- WMH-SynthSeg：[`wmh/README.md`](wmh/README.md) 与 [`wmh/report.public.json`](wmh/report.public.json)。
- SynthSeg：[`synthseg/report.public.json`](synthseg/report.public.json)。
- SynthSR：[`synthsr/README.md`](synthsr/README.md)。
- TorchFAST：[`fast/README.md`](fast/README.md) 与 [`fast/report.public.json`](fast/report.public.json)。
- TorchApplyWarp：[`applywarp/report.json`](applywarp/report.json)。
- FLIRT：[`flirt/report.public.json`](flirt/report.public.json)。
- FNIRT 与 FastVBM：[`fast_vbm/README.md`](fast_vbm/README.md) 及 0.9 正式文件。
- TOPUP：[`topup/report.public.json`](topup/report.public.json) 与[功能页](../docs/topup/README.md)。

## 功能子页面审计

本表不包含 recon-all 和 fsLR32k MS-HBM；这两项按项目范围单独维护。每个“有”都表示该材料直接出现在对应功能子页面，而不只是
保存在机器可读报告中。

| 功能子页面 | 原软件对照 | 输出一致性 | 运行时间 | example image |
|---|---|---|---|---|
| [SynthStrip](../docs/synthstrip/README.md) | FreeSurfer 8.2，12 例 | 有：脑图、mask、distance | 有：原版/本包 CPU 与 GPU | 有：公开 T1w 脑提取 |
| [SynthMorph](../docs/synthmorph/README.md) | FreeSurfer 8.2，四种模式 | 有：变换、正反向图像与边界差异 | 有：原版/本包 CPU 与 GPU | 有：公开 T1w joint 配准 |
| [WMH-SynthSeg](../docs/wmh_synthseg/README.md) | FreeSurfer 官方源码，12 例 | 有：标签、WMH 概率、软体积 | 有：原版/本包 CPU 与 GPU | 有：公开 FLAIR WMH overlay |
| [SynthSeg](../docs/synthseg/README.md) | FreeSurfer 8.2，3 例公开 T1w | 有：标签、几何、dtype、软体积 | 有：原版 CPU / 本包 H100 | 有：公开 T1w 标签与 mismatch |
| [SynthSR](../docs/synthsr/README.md) | FreeSurfer TensorFlow，12 例 | 有：shape、affine、dtype、体素差 | 有：原版/本包 CPU 与 GPU | 有：公开 FLAIR 合成 T1w |
| [TorchFAST](../docs/fast/README.md) | FSL FAST，10 例 | 有：GM PVE、Dice、体积、bias | 有：FSL CPU / 本包 GPU | 有：GM overlay、差值和 bias correction |
| [FastVBM](../docs/fast_vbm/README.md) | UKB v1 / FSL，10 例 | 有：warped GM、Jacobian、modulated GM | 有：两种后端与历史 FSL 记录，边界已标注 | 有：三类输出的十例平均 |
| [FLIRT](../docs/flirt/README.md) | FSL 6.0.7.4，10 例 | 有：`.mat` 与 reference-grid 图像 | 有：FSL CPU / 本包 H100 | 有：十例平均配准 GM 与差值 |
| [FNIRT](../docs/fnirt/README.md) | FSL 6.0.7.4，10 例 matched input | 有：coefficient、field、iout、jout、modulated GM | 有：FSL CPU / 本包 H100 | 有：十例平均 warped GM 和 Jacobian |
| [applywarp](../docs/applywarp/README.md) | FSL 6.0.7.4，11 项 | 有：dense/coefficient、linear/nearest、header/dtype | 有：FSL CPU / 本包 CPU 与 H100 | 有：相同 warp 的输出与差值 |
| [TOPUP](../docs/topup/README.md) | FSL 6.0.7.4，1 例真实 UKB 格式 dMRI | 有：field、corrected images、Jacobian、coefficient/movement/header | 有：FSL CPU / 本包 H100，各 3 次 | 有：原始 AP/PA、校正均值、field 与差值 |

公开记录不含账号、私有绝对路径、源病例 ID、权重或临床原图。公开样例及其来源校验见 [T1w 示例](../examples/README.md)和 [FLAIR 示例](../examples/WMH.md)。没有人工真值的报告只衡量与参考实现的一致性。
