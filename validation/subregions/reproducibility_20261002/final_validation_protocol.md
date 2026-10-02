# 最终完整流程验证协议

状态：最终六次完整流程、24 组/440 行 CPU 指标、源码/输入/几何/数值审计及 V2 分步骤计时提取均已完成。精简结果见 [最终验收](final_all_analysis/README.md)、[440 行指标](final_all_analysis/final_roi_metrics.tsv)、[24 组结果](final_all_analysis/final_reproducibility_compact.json)、[完整步骤表](final_all_analysis/final_steps.tsv)。

## 输入和来源

每种输入分别运行三个独立进程，每次均为 `structures=all`：脑干、丘脑、左海马/杏仁核、右海马/杏仁核。原始 T1 使用公开 ds000114 数据；同阶段输入使用本次三份官方重复共同的 `norm/aseg/wmparc`。参考图只在拟合结束后评分。

最终快照的运行文件数量从 `source_manifest.json` 动态计算；冻结时仓库有 431 个 Python 文件，其中 7 个无生产调用的原软件脚本不进入验证快照，实际导出 424 个原生 Python 模块，加上 3 个 LUT 和验证 driver 共 428 个文件；清单 SHA `04f952cfc43d7c75458b8e71cd87ae18d8ce049bb51c52580d9fd62caf6db254`。排除清单由快照清单记录。旧 `4178a48` 的 450 个文件/415 个运行文件、三次完整重复独立核对和保留，不与最终实现合并。

CPU 分析核对实际输入、源码、图谱、权重、包装器、输出文件的大小和 SHA-256；检查三个重复的调用 scope、模型选择、配置和物理 GPU UUID 一致。实际预处理数组的 dtype、shape、affine 和字节 SHA 用于记录重复差异，不以数值哈希相同作为排除随机预处理波动的条件。

## 指标

正式六次均为 `structures=all`，每种输入固定一个 GPU UUID（同阶段 GPU 0、原始 T1 GPU 1），4 线程、默认 TF32、阶段内相同输入和源码配置。所有 421 个非空 label-space 内部最低 Dice 为 1，全部 440 行最多不同 voxel 为 0；19 个 FNIT 空标签重复 Dice 为 NA。软体积三次逐值完全相同，最大 CV 为 0。全部预处理 data/coarse/parc/wmparc/brainmask 的 dtype/shape/字节 SHA 也一致。官方本例三次非空重复 Dice 为 1、不同 voxel 为 0；跨实现偏差独立列出。

6 个家族 × 4 个空间共 24 组，110 个标签 × 4 个空间共 440 行。原网格固定为实际输入网格；高分辨率网格沿用全新官方第 1 次输出的轴、间距和整数相位，对优化前、最终和官方联合扩展 FOV。每个家族按共同前景边界裁剪指标计算区域，保留全网格几何和裁剪坐标。

逐 ROI 记录官方内部重复、FNIT 内部重复及跨实现 Dice、Jaccard、不同 voxel 数、硬体积、软体积及软体积 CV。双方均空的硬标签 Dice 为 NA。三次重复提供观察范围，不提供人群随机范围或统计置信区间。

FNIT 自身重复波动与官方自身重复波动比较；跨实现差异单独记录。稳定但存在跨实现 voxel 差异的算法不因官方重复 Dice=1 而自动判定精度不合格。官方存在而 FNIT 硬标签为空的区域单独列出。

所有原始输出标签检查有限值和整数值，原网格 shape/affine 与输入匹配；各结构 Jacobian 有限且大于 0，所有硬/软体积有限且非负。

## 时间口径

最终 runner 用 Popen 前到独立 `process.wait()` 线程确认退出的 monotonic 时间计量 `process_wall_seconds`，单独记录源码/输入检查和 GPU 可用预算等待。API 计算和保存、导入、评分与包装器收尾均包含在进程时间中。

API 计时保留验证 observer 的实际开销。`observer_seconds` 是 context 回调中数组哈希和当次 JSON 写入的子计时；包装器开始/结束的源码哈希和最终 JSON 保存另在进程时间内，不用直接相减推导无 observer 的生产时间。GPU 自身显存以真实 PID 采样，限制 19073 MiB；采样峰值不等于瞬时峰值。共享 GPU 的其他进程占用同时记录。

### 官方已保存的显式阶段计时（秒）

| 结构 | 初始化/预处理 | 初始图谱对齐 | 合成阶段准备与拟合 | 强度阶段准备与拟合 | 后处理 |
|---|---:|---:|---:|---:|---|
| 脑干 | 10–11 | 2 | 17 | 201–210 | 缺测 |
| 丘脑 | 7 | 2 | 85–89 | 298–311 | 缺测 |
| 左海马/杏仁核 | 13–14 | 1 | 75–77 | 251–257 | 缺测 |
| 右海马/杏仁核 | 14–15 | 1 | 69–72 | 219–229 | 缺测 |

以上来自 [9 份日志的显式整数计时、行号及 SHA](official_explicit_step_timers.json)。官方 `process.py` 的合成/强度计时包括对应准备步骤，与 FNIT 分列的准备和拟合字段范围有所不同。官方 `extract/postprocess/cleanup` 没有独立 timer，不填 0。中间阶段没有共同保存的标签输出时，不填造出的 Dice；最终原网格/高分辨率标签精度由完整比较表报告。

### 历史原始 T1 预处理

[历史 recon-all 来源核对](reconall_historical_lineage_audit.json)记录 2026-09-23 的完整 `recon-all -all -openmp 4` 耗时 4727 秒、退出码 0。保留的 `orig/001.mgz` 与当前公开 T1 shape 和 float32 体素逐值相同，affine 最大差 7.63e-6。历史启动没有记录压缩文件 SHA，此处核实的是保留输入的数值和几何。

本轮官方三个细核子流程的分段时间合计为 1474.19–1496.97 秒。加上历史 recon-all 得到分段耗时合计 6201.19–6223.97 秒（约 103.35–103.73 分钟）；该合计没有本轮单次完整原始 T1 重跑的计时含义。

## CPU 工具

- [最终完整流程分析](analyze_final_full_pipeline.py)：严格排除 partial/full scope 混合，实际运行源码、输入、配置、GPU、数值和几何审计，优化前与最终分别统计。
- [阶段时间提取 V2](extract_final_step_timing_v2.py)：读取完整 API/solver timers 与官方显式 timer，TH/HIP 多层准备、拟合和 solver 后续整理 timer 全部提取；未独立计量的 affine/后处理保持空值，剩余开销合并列出。旧 V1 的加载 SHA 日志保留为运行来源记录，最终发布步骤证据来自 V2。
- [分析协议检查](release_tests/final_analysis_contract.json)：六项 CPU contract fixture，仅验证元数据与分类规则，不是影像 benchmark。
- [observer 协议检查](release_tests/context_observer_contract.json)：四项 CPU contract fixture，仅核实调用、同对象返回、哈希及 hook 恢复，不是影像 benchmark。
