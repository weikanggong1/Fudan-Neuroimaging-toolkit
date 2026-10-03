# dMRI 参数图流程验证

2026-10-02 的最新 FNIRT 完整 FastVBM、volume、dMRI 验收与完整 4D 重采样统一见[本轮报告](../registration_lossless_20261002/README.md)。以下记录按各自日期和源码保留。

[功能和调用方式](../../docs/dmri_pipeline/README.md) · [既有 UKB TBSS 对照报告](tbss_e2e.real.current.json) · [经典 NODDI 接入报告](pipeline_classic_real.public.json) · [九图比较脚本](compare_current_eddy_pipeline.py) · [官方 TBSS 参考脚本](run_official_tbss.sh)

## 最新：2026-10-03，公开十人双分支固定对照

[十人协议、调用和结果入口](public10_20261002/README.md)覆盖完整 117 帧 AP、PA b0、配对 T1，以及 TBSS/FNIRT 与 T1/tensor MMORF 两个分支。冻结 `bf339a0` 的 FNIT 20/20 完成；原软件 19/20 完成，已核对 432/450 张图。TBSS 为 10 对，MMORF 为 9 对；case10 MMORF 的原 GPU EDDY 三次完整尝试均失败，保留其时钟及缺失分母。逐图误差、步骤时间、脑图和失败历史已公布；输出仍非数值等价。该数据集与下面的既有单例不同，源码、协议和时间不能合并。

## 历史：2026-10-02，SynthStrip＋源码对应 TOPUP 单例整链

当前 b0 掩膜使用 FNIT PyTorch SynthStrip；TOPUP 补齐默认 regrid 和源码对应的场/运动优化、周期平滑、样条采样。真实 raw AP/PA 的 FNIT 与独立官方 SynthStrip 参考链生成全部 27 张参数图。最新计时、精度、显存、源码、两种参考协议及脑图见[完整报告](end_to_end_synthstrip_topup_20261002.md)、[运行 JSON](report.synthstrip_topup_20261002.public.json)、[官方 SynthStrip 对照](comparison.synthstrip_topup_20261002.public.json)、[历史 BET 对照](comparison.historical_bet_20261002.public.json)与[上游检查](upstream.synthstrip_topup_20261002.public.json)。TOPUP 独立同输入验收见[分功能报告](../topup/README.md)。输出仍非逐值相等。

## 历史：2026-10-02，阈值 mask 与旧 TOPUP 的同 raw 完整 TBSS＋AMICO

[完整报告与脑图](end_to_end_20261002.md)、[逐次时间与源码](report.end_to_end_20261002.public.json)、[27 图比较](comparison.end_to_end_20261002.public.json)、[上游比较](upstream.end_to_end_20261002.public.json)记录真实原始 AP/PA 的两次 FNIT 和两次独立 FSL＋AMICO 整链；各阶段从零生成，共 27 张最终图。

处理时间中位数为 FNIT **481.14 s / 8.02 分钟**、参考 **2080.64 s / 34.68 分钟**。标准九图固定模板 ROI r=0.419–0.810、骨架 r=0.197–0.710，脑 mask Dice=0.888，尚未数值等价；4.32 是本例不同输出流程的观察时间比。shape/affine/有限值检查通过，各实现内部两次 27 张解码图及 header binary block 分别精确相同。原软件参考使用独立 BET 和官方 Python AMICO 等本例适配，细节在报告中；不称为未修改的 UKB v1.5。

## 2026-10-02：保持已有 FNIT 输出的组件优化

[详细报告](lossless_20261002.md)与[机器报告](report.lossless_20261002.public.json)记录相对冻结 FNIT `954ad19` 的真实同输入验收。完整八轮 EDDY、两种全脑 NODDI、固定 warp 的 TBSS/MMORF 九图及参考索引复用全部保持逐值、几何、header 与模型 QC 一致。EDDY 一次配对计时 484.75→404.31 s；固定 warp 的热调用九图传播约 2.02/1.91 倍。NODDI 尚无稳定速度收益，经典模式整体中位数变慢。

该组件报告未运行整链；随后完成的同 raw 整链复测按其实际源码分别记录，见上方当前与历史两节。下表和后文保留历史整链及原软件对照的具体输入边界。

## 既有整链与原软件对照

| 验证边界 | 结果 | 证据 |
|---|---|---|
| 原始 BIDS、无 T1w、TBSS 完整命令 | 九张 native、standard、skeleton 图均通过输出检查 | [当前整链](bids_tbss.real.current.json) |
| 原始 BIDS、带 T1w、MMORF 2 mm 完整命令 | 九张 native、standard 图和 MMORF 形变场均通过输出检查 | [当前整链](bids_mmorf_2mm.real.current.json) |
| 原始 BIDS、带 T1w、MMORF 1 mm 完整命令 | native 图、T1 脑图和两份仿射完成；非线性阶段因共享 GPU 显存不足退出 | [失败阶段](bids_mmorf.shared_gpu_oom.json) |
| 既有 UKB TBSS 对 FSL 参考 | 九图数值比较；参考输入边界与 raw-to-standard 候选不同 | [对照报告](tbss_e2e.real.current.json) |
| 经典 NODDI 接入 | 24 个真实体素的拟合与九图接口检查，非完整整链 | [阶段报告](pipeline_classic_real.public.json) |

## 原始 BIDS 入口：真实 AP/PA 采集

把一例真实 UKB AP/PA DWI 按原始 BIDS 的 `sub-*/dwi/` 结构组织，并分别测试不含 T1w 和加入配对 T1w 的目录。源图像仍留在计算节点，不上传仓库。[最终源码的输入预检](bids_preflight.real.final.json)确认两种目录都选中同一条 105 volume DWI；暂存的 AP/PA 图像、AP bval/bvec 与 PA bval 同原文件逐字节相同，两方向的相位编码向量及读出时间也相同。没有反向图时跳过 TOPUP、多场图 `IntendedFor` 选择、3D `fmap/*_epi` 转换、BIDS 元数据继承和覆盖时清理旧 PA，另由 21 项自动测试覆盖。

[无 T1w 的整链报告](bids_tbss.real.current.json)记录 `fnit-dmri-pipeline --bids-root ... --registration-backend tbss` 的真实运行。TOPUP 确实启用，acqparams 两行分别为 `(0,-1,0,0.069)` 和 `(0,1,0,0.069)`；九张 native 图同为 104×104×72，九张 standard 图与九张 skeleton 图同为 182×218×182，全部 float32、有限值且各组 affine 一致。完整进程 wall 为 1649.82 s；TOPUP 与 EDDY 准备 187.55 s、EDDY 1009.60 s、DTIFIT 15.63 s、NODDI 65.97 s、TBSS 配准和九图传播 366.39 s。这是共享 H100 节点的一次观察值，不能用作相对 FSL 的独占 GPU 加速比。

无 T1w 整链启动后，BIDS 选择器又修正了 subject 层场图、多 T1w 的 TBSS 忽略规则以及覆盖时清理旧 PA。该病例无 session、无 T1、已有唯一反向 DWI，也未覆盖输出，因此这些分支不参与所测计算；整链报告中的 inspected-source hash 不是启动时的精确 BIDS 文件 hash。最终源码另以同一真实输入完成上述预检，并通过自动测试。该验收证明 BIDS 输入正确进入既有计算链、输出结构正确；与原版 FSL 的数值差异应按下文的匹配输入对照判读，不能从输入格式验收推出逐体素等价。

[两次运行的 native 九图比较](bids_native_branch_compare.real.json)在非零体素并集上的 r 为 0.982398–0.999929。两次 TorchEDDY 的 `gp_seed_override` 都为 null，日志中的首轮 GP seed 分别为 1790694888 和 1790696344。因此独立运行的上游图并非固定种子的配对试验，这些数值差异不能归因于 T1w；T1w 在本流程只进入后面的 MMORF 配准。

带 T1w 的 MMORF 分支在同一真实 AP/PA+T1w 采集上，以由官方 1 mm FA、T1、tensor 模板重采样的 2 mm 网格完成 `run_bids()` 整链。九张 native 图为 104×104×72，九张 standard 图为 91×109×91；图像均为 float32、有限值且组内 affine 一致。MMORF 形变场为 91×109×91×3，九图与 T1 模板网格一致。暂存的 AP/PA 图像和梯度与原始文件逐字节相同，SynthStrip 权重大小和 SHA-256 与 FNIT 清单一致。完整命令 wall 为 1829.25 s；TOPUP/EDDY 准备、EDDY、DTIFIT、NODDI 和配准传播分别为 168.91、958.84、13.34、50.95 和 633.69 s；组件内 CUDA allocator 峰值最高为 NODDI 的 11.59 GiB。[机器报告](bids_mmorf_2mm.real.current.json)记录了源码、模板及权重哈希。2 mm 模板改变了输出网格，因此不能把这组图直接与 1 mm 原版 FSL 图逐体素比较。

另一次使用官方 1 mm 模板的 MMORF 整链已完成 BIDS 选择、TOPUP、EDDY、九张 native 图、SynthStrip 脑图及两份 FLIRT 仿射，随后在 tensor 有限应变旋转时因共享 GPU 显存不足退出，未生成标准空间九图。失败时该 GPU 剩余 228.69 MiB，进程占用约 10.59 GiB；命令运行 36:20.19，退出码 1。详见[失败阶段记录](bids_mmorf.shared_gpu_oom.json)。

## 既有 UKB TBSS 与 FSL 对照

2026 年 9 月 29 日在 gpucw1 用一例真实 UKB 格式 AP/PA 数据，从原始图像运行 TBSS 分支。结果图像仍留在计算节点；仓库只保存汇总数值。

随后 TorchEDDY 的样条权重改为无布尔索引计算；固定种子的完整八轮校正图与改动前文件 SHA-256 相同，其他数值输出也逐值相同。本节与 FSL 的九图相似性指标来自改动前的整链实测，未按上方 BIDS 运行的输出重新计算；下述 14:06.89 也只代表该次 UKB 运行。新版 EDDY 的单独计时见 [EDDY 验证页](../eddy/README.md)。

TBSS 的第一个参考固定 TOPUP 系数、掩膜和其余输入，改用 FSL `eddy_cuda10.2` 校正图，再由相同 FNIT DTIFIT、NODDI、TBSS 代码生成九图；该配对比较隔离 EDDY 输入的影响。九张 native 图 r 为 0.978640–0.999882，standard 图 r 为 0.940917–0.988332，skeleton 图 r 为 0.949470–0.991235。shape 和 affine 全部匹配。

第二个参考由先前准备的官方 UKB native 参数图开始，经 FSL 6.0.7.4 weighted FLIRT、三阶段 FNIRT 和 applywarp 生成 standard/skeleton 图。其输入边界与本次 raw-to-standard 流程不同：standard 九图 r 为 0.358268–0.699381，skeleton 九图 r 为 0.098878–0.651267。这些差异不能单独定位到 EDDY 或 FNIRT，也不能将 FSL 的配准计时与完整 FNIT 计时相除。各图 MAE、RMSE 和有效体素数见[机器报告](tbss_e2e.real.current.json)。

2026-10-02 对照脚本按官方 `bb_tbss_3_postreg` 补上 FA 的 `applywarp --rel`，覆盖三阶段 FNIRT 的 `--iout`；本节历史数值来自补齐前的命令和既有官方 native 图，最新从相同 raw AP/PA 起步的官方整链已使用补齐后的命令，见[完整复测](end_to_end_20261002.md)。

| 参数图 | 匹配 FSL EDDY：native r | 匹配 FSL EDDY：standard r | 原版 FSL TBSS：standard r | 匹配 FSL EDDY：skeleton r | 原版 FSL TBSS：skeleton r |
|---|---:|---:|---:|---:|---:|
| FA | 0.999224 | 0.988332 | 0.667522 | 0.991235 | 0.517921 |
| MD | 0.999882 | 0.981015 | 0.649976 | 0.954968 | 0.489054 |
| L1 | 0.999797 | 0.979714 | 0.591519 | 0.952986 | 0.426730 |
| L2 | 0.999823 | 0.981538 | 0.661351 | 0.961240 | 0.553910 |
| L3 | 0.999819 | 0.981895 | 0.699381 | 0.966816 | 0.596406 |
| MO | 0.978640 | 0.940917 | 0.511447 | 0.968428 | 0.640657 |
| ICVF | 0.986409 | 0.977894 | 0.358268 | 0.949470 | 0.098878 |
| OD | 0.993978 | 0.976456 | 0.632424 | 0.974554 | 0.651267 |
| ISOVF | 0.999170 | 0.978553 | 0.690850 | 0.962090 | 0.599029 |

本次 TBSS 完整进程 wall 为 14:06.89；EDDY 阶段 673.42 s，配准、九图传播和 skeleton 阶段 58.02 s。EDDY 进程内 CUDA allocation 峰值为 4.54 GiB；全流程最大组件峰值为 NODDI 的 12.87 GiB。共享 GPU 未隔离。

比较脚本对两份相同网格的结果计算非零并集上的 Pearson r、MAE、RMSE，并检查 shape 与 affine。运行前需分别准备 FNIT 候选目录、匹配 FSL EDDY 的下游目录、官方 TBSS `stats` 目录：

```bash
python validation/dmri_pipeline/compare_current_eddy_pipeline.py \
  --candidate-root subject_tbss \
  --reference-root fsl_eddy_with_fnit_downstream \
  --reference-label matched_FSL_EDDY_plus_FNIT_downstream \
  --official-standard official_tbss/stats \
  --official-skeleton official_tbss/stats \
  --branch tbss \
  --output tbss_comparison.json
```

`--candidate-root` 和 `--reference-root` 均为含 `native/`、`registration/standard/` 的单被试结果根；TBSS 还读取 `registration/skeleton/`。`--official-standard` 和 `--official-skeleton` 接收 FSL `stats/` 目录。脚本输出 JSON，不改写图像。

## MMORF 分支

当前 MMORF 模块的 T1w、FA、tensor 真实数据配对验证见 [MMORF 报告](../mmorf/report.public.json)。本页 2 mm BIDS 整链已生成九张标准图；与原版 FSL MMORF 的配对精度仍按独立 MMORF 验证页判读，不能将不同模板网格的图像直接逐体素比较。

## 经典 NODDI 接入

在同一例真实 EDDY 校正 DWI 的固定 24 个脑内体素上，`DMRIPipeline(noddi_fit_method="classic")` 实际调用 `select_shell`、`TorchDTIFIT` 和经典 `TorchAMICONODDI`。九张 native 和 standard 图键完整；ICVF、OD、ISOVF 与相同输入的独立经典 NODDI 输出最大绝对差均为零。EDDY 直接接入预先校正的真实数据，配准使用 identity stub；报告只验证 NODDI 的 pipeline 接入，不代表完整 raw-to-MNI 结果。运行记录见[机器报告](pipeline_classic_real.public.json)。
