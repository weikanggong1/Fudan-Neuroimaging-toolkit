# dMRI 参数图流程验证

[功能和调用方式](../../docs/dmri_pipeline/README.md) · [TBSS 当前机器报告](tbss_e2e.real.current.json) · [经典 NODDI 接入报告](pipeline_classic_real.public.json) · [九图比较脚本](compare_current_eddy_pipeline.py) · [官方 TBSS 参考脚本](run_official_tbss.sh)

## 原始 BIDS 入口：真实 AP/PA 采集

把一例真实 UKB AP/PA DWI 按原始 BIDS 的 `sub-*/dwi/` 结构组织，并分别测试不含 T1w 和加入配对 T1w 的目录。源图像仍留在计算节点，不上传仓库。[最终源码的输入预检](bids_preflight.real.final.json)确认两种目录都选中同一条 105 volume DWI；暂存的 AP/PA 图像、AP bval/bvec 与 PA bval 同原文件逐字节相同，两方向的相位编码向量及读出时间也相同。没有反向图时跳过 TOPUP、多场图 `IntendedFor` 选择、3D `fmap/*_epi` 转换、BIDS 元数据继承和覆盖时清理旧 PA，另由 21 项自动测试覆盖。

[无 T1w 的整链报告](bids_tbss.real.current.json)记录 `fnit-dmri-pipeline --bids-root ... --registration-backend tbss` 的真实运行。TOPUP 确实启用，acqparams 两行分别为 `(0,-1,0,0.069)` 和 `(0,1,0,0.069)`；九张 native 图同为 104×104×72，九张 standard 图与九张 skeleton 图同为 182×218×182，全部 float32、有限值且各组 affine 一致。完整进程 wall 为 1649.82 s；TOPUP 与 EDDY 准备 187.55 s、EDDY 1009.60 s、DTIFIT 15.63 s、NODDI 65.97 s、TBSS 配准和九图传播 366.39 s。这是共享 H100 节点的一次观察值，不能用作相对 FSL 的独占 GPU 加速比。

无 T1w 整链启动后，BIDS 选择器又修正了 subject 层场图、多 T1w 的 TBSS 忽略规则以及覆盖时清理旧 PA。该病例无 session、无 T1、已有唯一反向 DWI，也未覆盖输出，因此这些分支不参与所测计算；整链报告中的 inspected-source hash 不是启动时的精确 BIDS 文件 hash。最终源码另以同一真实输入完成上述预检，并通过自动测试。该验收证明 BIDS 输入正确进入既有计算链、输出结构正确；与原版 FSL 的数值差异应按下文的匹配输入对照判读，不能从输入格式验收推出逐体素等价。

[两次运行的 native 九图比较](bids_native_branch_compare.real.json)在非零体素并集上的 r 为 0.982398–0.999929。两次 TorchEDDY 的 `gp_seed_override` 都为 null，日志中的首轮 GP seed 分别为 1790694888 和 1790696344。因此独立运行的上游图并非固定种子的配对试验，这些数值差异不能归因于 T1w；T1w 在本流程只进入后面的 MMORF 配准。

带 T1w 的 `--registration-backend mmorf` 也实际启动了完整命令。BIDS 选择、TOPUP、EDDY、九张 native 图、SynthStrip 脑图及两份 FLIRT 仿射均已完成；随后 MMORF 的 tensor 有限应变旋转在共享 GPU 上因显存不足退出，未生成标准空间九图。失败时该 GPU 剩余 228.69 MiB，进程占用约 10.59 GiB；命令运行 36:20.19，退出码 1。详见[失败阶段记录](bids_mmorf.shared_gpu_oom.json)。这次不能作为带 T1w 整链通过的证据；最终源码的真实 BIDS 输入预检和该分支的单元调用测试仍已通过。

2026 年 9 月 29 日在 gpucw1 用一例真实 UKB 格式 AP/PA 数据，从原始图像运行 TBSS 分支。结果图像仍留在计算节点；仓库只保存汇总数值。

随后 TorchEDDY 的样条权重改为无布尔索引计算；固定种子的完整八轮校正图与改动前文件 SHA-256 相同，其他数值输出也逐值相同。本页九图相似性指标来自改动前的整链实测，TBSS 分支尚未用改动后源码重跑；下述整链运行时间也不代表现版耗时。现版 EDDY 单独计时见 [EDDY 验证页](../eddy/README.md)。

TBSS 的第一个参考固定 TOPUP 系数、掩膜和其余输入，改用 FSL `eddy_cuda10.2` 校正图，再由相同 FNIT DTIFIT、NODDI、TBSS 代码生成九图；该配对比较隔离 EDDY 输入的影响。九张 native 图 r 为 0.978640–0.999882，standard 图 r 为 0.940917–0.988332，skeleton 图 r 为 0.949470–0.991235。shape 和 affine 全部匹配。

第二个参考由先前准备的官方 UKB native 参数图开始，经 FSL 6.0.7.4 weighted FLIRT、三阶段 FNIRT 和 applywarp 生成 standard/skeleton 图。其输入边界与本次 raw-to-standard 流程不同：standard 九图 r 为 0.358268–0.699381，skeleton 九图 r 为 0.098878–0.651267。这些差异不能单独定位到 EDDY 或 FNIRT，也不能将 FSL 的配准计时与完整 FNIT 计时相除。各图 MAE、RMSE 和有效体素数见[机器报告](tbss_e2e.real.current.json)。

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

当前 MMORF 模块的 T1w、FA、tensor 真实数据配对验证见 [MMORF 报告](../mmorf/report.public.json)。旧求解器的 raw-to-standard 整链指标已移除；这次 BIDS 整链尝试在非线性阶段因共享 GPU 显存耗尽，完整标准空间输出仍待验证。

## 经典 NODDI 接入

在同一例真实 EDDY 校正 DWI 的固定 2,048 个脑内体素上，`DMRIPipeline(noddi_fit_method="classic")` 实际调用 `select_shell`、`TorchDTIFIT` 和经典 `TorchAMICONODDI`。九张 native 和 standard 图键完整；ICVF、OD、ISOVF 与相同输入的独立经典 NODDI 输出最大绝对差均为零。EDDY 直接接入预先校正的真实数据，配准使用 identity stub；报告只验证 NODDI 的 pipeline 接入，不代表完整 raw-to-MNI 结果。运行记录、源码和输入 SHA-256 见[机器报告](pipeline_classic_real.public.json)。
