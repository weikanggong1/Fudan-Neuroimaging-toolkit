# HIP 低 Dice 的真实数据定位

## 固定方法

本轮 5 个部分 HIP 诊断使用冻结 source450，按声明配置执行：右侧 stage fast / joint fast / balanced，左侧 stage / raw balanced。独立 observer 只读取拟合输出，缓存 rawfit / native support 到服务器；原求解器、初始化和预测没有被观察代码修改。只下载数值报告和日志。每个 case 的原 LCC6 由 CPU helper 重建，native 与 highres 均为 0 voxel 差异。

后处理对照参数在官方评分之前固定：六邻域球、半径 1 highres voxel（0.33333 mm），closing 的选择掩膜与原 foreground 并集，再选其主连通块，最后与原 raw foreground 相交。它只保留原模型已经预测的细核标签，不新增标签，不改 label ID。另有 dilation 对照，没有按官方结果调节半径。

## 右侧 AAA 的原因

joint fast 原始 argmax / valid / raw 均有 627 个 AAA highres voxel；原 LCC6 仅保留 1 个，native 为 0。最大被丢连通块为 629 voxel、23.2956 mm³，包含 AAA 613 和 Corticoamygdaloid-transition 16。其至主 foreground 的最短六邻路线只有 1 个空 highres voxel，最近 voxel 中心欧氏距离 0.4714 mm。balanced 同样因窄断点丢弃 AAA：原始 631，LCC 后 0。实际 posterior argmax 的 invalid 区 AAA 为 0，native support 过滤也没有再删 AAA，所以原因在 hard foreground 的连通块选择。

官方与 FNIT 原规则均为 scipy 默认六邻接，并且在 HIP/Amyg foreground 并集上取一次 LCC；这不是官方连通定义不同。算法路径造成的亚毫米拓扑差异，被 hard LCC 放大成整个核消失。

## 固定 closing 的效果

|Case|HIP weighted native Dice（前→后）|Amyg weighted native Dice（前→后）|AAA native Dice（前→后）|
|---|---:|---:|---:|
|右 stage fast|0.830807→0.830807|0.929114→0.929114|0.956522→0.956522|
|右 stage joint fast|0.853946→0.853813|0.920830→0.937903|0→0.933333|
|右 stage balanced|0.902432→0.902432|0.928916→0.945989|0→0.933333|
|左 stage balanced|0.781051→0.781051|0.868861→0.868861|0.761905→0.761905|
|左 raw balanced|0.787963→0.787963|0.879666→0.879666|0.736842→0.736842|

每个 label 的 native 前后计数 / Dice / Dice 差值、highres 计数和 candidate-grid Dice 都保存在 `hippo_topology_region_comparison.tsv`，完整组件组成见 `hippo_topology_summary.json`。highres 评分在 candidate fit grid，仅用于本轮局部方法对照；正式结果另用固定官方轴与 union-FOV evaluator。

## 左侧 balanced 回退与采用范围

左 stage balanced 的低 Dice 不是 AAA 被整块删掉：原 LCC 只删 1 个 AAA highres voxel，closing 不改变任何 native family 指标。既有 stage fast 的固定 native-grid HIP/Amyg weighted Dice 为 0.894839 / 0.953639，joint 为 0.898531 / 0.956967。本轮 balanced 为 0.781051 / 0.868861，因此不能把右侧 balanced 的更高 Dice 当成双侧统一预算应升级的证据。生产采用由 root 决定为 closing-only，保留现有 fast/balanced 求解器参数，不设左右不同默认。

## raw 输入范围核实与更正

执行的 `api_report.json` 明确记录 raw-left-balanced 调用一次 SynthSegPlus、零次 SynthSeg。冻结 pipeline.py:158 对任何 HIP 设置 need_parc=True；context.py:103–122 用 SynthSegPlus 同时提供 coarse 与 cortical parcellation。此前从部分结构名称推断 HIP-only raw 使用 SynthSeg 的说法错误，已更正。BS-only raw 则在实际 API 记录一次 SynthSeg、零次 Plus。具体 bytes / SHA 和 source call sites 在 `hippo_raw_preprocessing_scope_audit.json`。

本轮 raw-left-balanced 是单个 HIP、balanced 配置，其输出位于原始 public T1 网格；不能当 default fast structures=all 的重复或全功能 benchmark。正式生产验收另执行 all 模式 3 次 stage + 3 次 raw。

## 时间和显存

5 个 case API compute 为 98.32 / 110.38 / 111.18 / 113.75 / 150.13 秒，观察与缓存额外 1.50–1.76 秒已经包含其中；CPU topology 对照 / 评分另用 11.29–14.50 秒。所有 case 物理 GPU1 串行，4 CPU threads，自身采样峰值 9226–18926 MiB，全部低于 19073 MiB 监控上限。它们是带观察的诊断耗时，不能替代正式 all benchmark。
