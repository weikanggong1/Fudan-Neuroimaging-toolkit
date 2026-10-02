# segment_4_subregions：真实数据重复性与精度证据

## 最终版本验收

最终不可变源码清单 SHA 为 `04f952cfc43d7c75458b8e71cd87ae18d8ce049bb51c52580d9fd62caf6db254`，核实 428 个文件、424 个原生运行模块。使用同一张公开、去面部 T1（OpenNeuro ds000114，CC0），原始 T1 与同阶段输入各运行三个独立完整进程，每次均分割脑干、丘脑及双侧海马/杏仁核。FreeSurfer 8.2.0-1 在相同 `norm/aseg/wmparc`、4 线程和同一服务器执行三次全新分割。

24 组、440 行全部完成。FNIT 的 421 个非空 label-space 内部最低 Dice 为 1；全部 440 行最多不同 voxel 为 0；19 个 FNIT 空标签的内部 Dice 为 NA。软体积三次逐值相同，最大 CV 为 0，与本例三次官方重复观察一致。实际预处理 data/coarse/parc/wmparc/brainmask 的 dtype、shape 和字节 SHA 也完全相同。

用户接受稳定算法的 voxel 差异。本次自身重复和跨实现偏差分别报告，较低跨实现 Dice 保留为实际算法差异，不设定跨实现 Dice 必须为 1。三次运行提供本例观察范围，不提供人群随机范围或统计置信区间。

## 指标、脑区与耗时

- [最新家族指标、最低 Dice 脑区及分步骤主表](final_all_analysis/README.md)
- [完整 440 行逐 ROI 指标](final_all_analysis/final_roi_metrics.tsv)
- [24 组精简结果与软体积](final_all_analysis/final_reproducibility_compact.json)
- [来源、GPU、几何、Jacobian、有限值核对](final_all_analysis/final_source_numerical_audit.json)
- [完整分步骤表](final_all_analysis/final_steps.tsv)及[计时出处](final_all_analysis/final_steps.json)
- [已完成验证设计](final_validation_protocol.md)

同阶段 API 含保存时间为 353.44–362.21 秒；原始 T1 为 260.65–348.87 秒。自身 GPU 采样峰值分别为 9726 / 16468 MiB，均在 19073 MiB 预算内。阶段内三个重复固定同一物理 GPU UUID：同阶段 GPU 0、原始 T1 GPU 1；两种输入同时运行。进程时间、来源核对、GPU 等待及 observer 子计时分别记录。

原网格固定为实际输入 affine；高分辨率比较使用全新官方第 1 次输出的轴、间距和整数相位，对优化前、最终和官方共同扩展 FOV。没有为评分拟合配准或相位。双方均空的硬标签 Dice 为 NA。

唯一官方存在而 FNIT 硬标签为空的项为 raw 高分辨率 `Left-VM`（8130）：官方 1 个高分辨率 voxel，FNIT 0；软体积为 14.4602 / 14.2063 mm³。跨实现硬 Dice 为 0，FNIT 内部空标签 Dice 为 NA。完整逐 ROI 表保留该项。

## 优化前历史参照：4178a48

优化前行为版本固定为 `4178a48`，独立快照为 450 个文件、415 个运行模块；原始 T1 与同阶段输入各三次完整运行。该历史实现与最终实现分别统计：[历史 24 组精简结果](final_all_analysis/before_reproducibility_compact.json)、[历史 440 行表](final_all_analysis/before_roi_metrics.tsv)。

历史丘脑、海马和杏仁核的重复硬标签完全相同。脑干同阶段重复相差 7–14 个原网格 / 69–102 个高分辨率 voxel，原始输入相差 4–10 / 40–104 个。最终实现的脑干也逐体素一致，软体积 CV 为 0。同阶段丘脑跨实现加权 Dice 从 0.953076 提高到 0.970813，高分辨率从 0.955107 提高到 0.972354；其余家族总体接近历史结果。

## 候选与 scope 审阅记录

候选单例仅评估跨实现精度。增加最终层或合成拟合步数曾提高右海马 aggregate，但损失 AAA 硬标签；最终实现保留原 fitter 参数，为主连通块选择加入固定亚毫米闭运算，最后仍与原预测前景相交，AAA 保留。最终同阶段右 AAA 原网格为 22 个 voxel、Dice 0.956522，软体积 44.3618 mm³；官方为 24 个、45.2047 mm³。

完整四结构流程与单独丘脑调用具有不同预处理/输出合并上下文，混合 scope 的旧范围已撤回为重复性证据。最终六次全部采用 `structures=all`。旧官方右 AAA 参考仅为 0/1 个原网格/高分辨率 voxel；本次三份全新官方均为 24/626，旧参考没有混入重复统计。

AAA 诊断已完成：[拓扑审阅](hippo_topology_review.md)、[精简组件结果](hippo_topology_summary.json)。右侧 joint 原始 argmax/有效前景有 627 个 AAA 高分辨率 voxel，原六邻域 LCC 只保留 1 个；AAA 与主前景之间仅有一个空 highres voxel 的断点。posterior argmax 的 invalid 区和 native support 没有再删 AAA，首次整体丢失发生在主连通块选择。官方也使用同一六邻域规则，因此是亚毫米预测拓扑差异被 LCC 放大。

固定 closing 可以保留已预测 AAA。左侧 balanced 的回退发生在拟合输出，closing 不能修复：同阶段左 HIP/Amyg weighted Dice 为 0.781051/0.868861，低于原 fast 的 0.894839/0.953639。因此最终采用 closing-only，维持原 fast20/balanced30 参数，没有统一提高预算或左右不同默认。

精简历史审阅见 [发布 scope 审阅](publication_scope_review.json)。大型候选、逐配对 JSON、网格清单及早期单例 cross 文件保留在服务器和仓库外本地 archive；精简来源清单记录其路径、大小和 SHA，支持重新分析。

## 官方步骤与原始 T1 预处理

官方已保存的 72 个显式整数 timer 覆盖初始化、图谱对齐、合成准备/拟合和强度准备/拟合：[官方 timer 行号及 SHA](official_explicit_step_timers.json)。后处理没有独立 timer。FNIT 的完整步骤表使用实际 API/solver timers，未独立计量的 affine/后处理保留缺测；未保存双方共同中间标签的阶段不填造出的 Dice。

[历史 recon-all 来源核对](reconall_historical_lineage_audit.json)记录 2026-09-23 完整 `recon-all -all -openmp 4` 耗时 4727 秒、退出码 0；保留原始输入与当前公开 T1 的 float32 体素逐值相同、affine 最大差 7.63e-6。历史启动未记录压缩文件 SHA。加上本轮官方细核子流程 1474.19–1496.97 秒，分段合计为 6201.19–6223.97 秒（103.35–103.73 分钟），没有作为本轮一条原始 T1 全流程重跑计时。

## 发布与验证清单

[精简文件大小/SHA](final_all_analysis/publication_artifacts.json)、[完成后的衍生文件传输清单](final_complete_fetch_manifest.json)、[合并后分割依赖来源核对](release_tests/postmerge_source_audit.json)、[合并后公共接口测试](release_tests/postmerge_public.log)。每份回收文件均在写入前核对远端大小和 SHA；没有回收影像、权重、图谱或 license 内容。

函数输入、输出、参数、完整 Python/CLI 示例、流程图、脑图、原软件命令及参考文献见 [segment_4_subregions 说明](../../../docs/subregions/README.md)。官方方法见 [FreeSurfer subregion segmentation](https://surfer.nmr.mgh.harvard.edu/fswiki/SubregionSegmentation)。
