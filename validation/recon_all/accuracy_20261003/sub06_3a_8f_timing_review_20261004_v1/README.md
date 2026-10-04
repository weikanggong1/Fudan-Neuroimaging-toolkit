# sub-06：3a 与 8f 实际计时审查

读取两次同一原始 T1 的 actual pipeline、launch、completion 与 monitor，保留文件路径/SHA；未执行 GPU、算法、冻结或新比较。旧源码为 `8f3e51f58c37d28e1283d5bb8e248e93530cd398`，新源码为 `3a0c9aba6321b4981fd8174b4b191515459aa38b`。两次整例 complete，138 项产物齐全；这不是官方数值等效判定。

## 可以相加的阶段树

```text
命令入口墙钟
├─ API 总墙钟
│  ├─ 输入/依赖验证
│  ├─ pipeline_seconds
│  │  ├─ stages[] 所有顺序父阶段（60行；包含下述步骤）
│  │  │  ├─ surface_hemisphere_group
│  │  │  │  └─ 左右worker并行步骤/内部子指标（不再加到pipeline）
│  │  │  ├─ defects_lh → defects_rh
│  │  │  ├─ register_hemisphere_group（左右worker并行）
│  │  │  ├─ annotation_hemisphere_group（左右worker并行）
│  │  │  ├─ finish_surface_hemisphere_group（左右放置并行，defer_metrics=True）
│  │  │  ├─ finish_metrics_lh → finish_metrics_rh（父进程串行；独立可加）
│  │  │  │  └─ metric_seconds thickness/area/curv等（嵌套，不重复加）
│  │  │  ├─ stats_lh_* → stats_rh_*（独立顺序；同半球复用cache）
│  │  │  └─ mesh_validation
│  │  └─ 未剖析残差：阶段间报告写出/控制/其他代码，非推定计算阶段
│  └─ public wrapper残差（线程设置/恢复及末尾报告返回等）
└─ 命令入口与API差额（导入/调用/退出等，不能称算法时间）
```

`native_free.py` 3a 的 stage 调用在1000–1002行完成并返回组结果，1024–1030行才调用 `finish_metrics_lh/rh`；684行 worker 显式 `defer_metrics=True`。因此 metrics 不是 finish_group 的虚拟嵌套行。父 `stages[]` 全部可相加，而 `group.values[*].stages`、`surfaces[*].metric_seconds`、MNI 的 postprocess/timings 和 ROI 缓存内部指标都不可重复加。Profiler 的阶段秒数已经包含 CUDA 前/后同步；parent/child CPU 秒不是额外墙钟。

## 实际差额

| 范围 | 8f 秒 | 3a 秒 | 新减旧秒 |
|---|---:|---:|---:|
| 命令入口 | 2755.779450 | 2948.558978 | +192.779528 |
| API | 2752.458063 | 2945.172644 | +192.714581 |
| pipeline | 2746.529304 | 2939.520808 | +192.991504 |
| 顺序父阶段总和 | 2739.162028 | 2930.506142 | +191.344114 |
| pipeline 未剖析残差 | 7.367276 | 9.014666 | +1.647390 |

主要可加差额：annotation +35.467 秒，surface +35.069 秒，finish_metrics 左 +25.818/右 +26.686 秒，MNI +17.547 秒，input +14.995 秒，mesh_validation +14.458 秒，mri_fill +10.761 秒。register 为 −46.153 秒；finish worker 组为 −3.686 秒。完整60行差额、CPU和同步秒数在 `timing_delta.json`，不按绝对值挑选后累加。

新旧 finish 子指标表明：thickness 从左3.546/右2.916变为13.189/12.887秒，curv及curv.pial各从约0.4秒变为8–9秒。它们已包含在两侧 finish_metrics 中。这些具体实现文件并非五个精度改动的目标，不能由此断言五个改动导致52.5秒回退。

## 启动与缓存事实

新3a finish组两侧均首次启动失败、`operation_entered=False`，第二次成功进入并完成；旧8f全组均一次成功。新finish启动等待9.408秒、旧2.965秒；实际worker span新444.175/旧454.445秒，因此组总墙钟仍下降。这里已经有真实整例启动恢复的一个成功批次，不能沿用此前“真实测试未触发重试”的说法，也不能据此估计恢复率。失败重试、组内复制/发布/回收时间都已包含在组父阶段。

新 `parent_idle_cuda_cache` 在finish组耗时0.00209秒。两次分配器环境均 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`；报告的PyTorch缓存counter为0不说明显存使用为0，也不证明empty_cache降低了实际显存。

ROI 新版将“逐面float32 area/3”转为double后按顶点归约，旧版先float32顶点求和再转double；新版多出独立缓存 `roi_vertex_area_computations=2`/半球，保留geometry_reads=2、principal_curvature_computations=2、roi_summary_transfers=6。后续同表面图谱复用确实存在，但第一次white/pial stats各包含基础量计算，不能只看首aparc代表所有ROI成本。首左右aparc墙钟均增加约9.2秒，parent CPU同样约9.8秒；不是只在边界CUDA同步里等待。

3a还合入 `mris_register_average_numba.py` 的source-order CPU归约：`@njit(cache=True, parallel=True, fastmath=False)`/prange、worker线程预算内并行，旧版循环串行且该函数没有磁盘cache=True。本次register缩短与该改变相容，但没有隔离对照，不能给因果结论。曲率邻接基础函数没有Numba装饰器，不能把finish或首ROI的全部增加笼统称为JIT。现有回执未记录每个dispatcher冷/热命中或编译耗时；Python源目录/cache身份变化也可能影响其他Numba函数，此处只能列待测因素。

## 负载与证据边界

两次同GPU UUID、总线程4。启动loadavg旧28.36/44.39/48.98，新8.34/8.94/10.34：新启动时负载较低，不能说新慢是后台负载更高。单点loadavg也不证明整次运行专用CPU、GPU、存储无干扰；monitor只保留owned PID显存采样，没有持续外部GPU占用/CPU调度/IO负载证据。

旧采样峰10,951,327,744B（1318样本，2秒请求，最大gap3.218秒），新13,570,670,592B（1402样本，最大gap4.622秒）；均 `continuous_peak_verified=False`。只能称owned同时采样峰，不能称连续真实峰或精度改动显存增量。

3a与8f之间还包含注册并行、volmask及共享模块等变化，不能将本次192.8秒差额归因于五项精度修改。一次非交替整例也不足以给稳定性能回退结论。

## 最小下一验证方案（仅计划）

先复用**同一份3a已完成white/pial/annotation/thickness**，在同GPU/四线程、无其他受控任务下，使用新进程分别运行现有 `_finish_cortical_metrics` 和左右全套ROI统计，每臂先复制自产 checkpoint 到独立目录，原版与3a使用相同输入，固定seed交替AB/BA各一次。每臂独立输出/cache，明确冷第一次/同进程第二次，记录CPU墙钟、内部metrics、CUDA同步、实际文件/source SHA及数值差异。这仅验证最显著的52.5秒指标与首ROI变化，不重跑整例、不读取官方结果生成候选；不会据结果放宽精度门槛。若仍要归因“五项精度修改的整例代价”，才准备9f7主线+同启动修复、其余策略完全相同的控制，核对实际diff后同输入交替整例；此处未启动或冻结任何控制版本。

实际远程冻结目录的9个关键源文件/版本共18个SHA均与本地审阅commit匹配，见 `actual_source_hashes.json` 和 `source_match.json`；未用准备文档SHA替代源码SHA。
