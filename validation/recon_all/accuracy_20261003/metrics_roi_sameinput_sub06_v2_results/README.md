# sub06 同输入 metrics + ROI：v2 真实阶段观察

两臂隐藏 CUDA 的接口预检均通过，数值算子全部 mock，因此预检不计 benchmark。随后原8f与精度候选3a使用相同10份3a自产输入，GPU1、总4线程、默认TF32、FP32、关闭AMP和PyTorch缓存分配器，在共享锁与GPU1锁内各执行真实cold/warm。两臂及队列均complete，exit0，所有登记进程退出，无算法重试。

|左半球阶段|8f cold|3a cold|8f warm|3a warm|
|---|---:|---:|---:|---:|
|metrics（同步，秒）|33.557|33.154|30.219|30.547|
|第一白面ROI（秒）|10.700|10.715|10.493|10.936|
|第一pial ROI（秒）|10.827|10.799|10.330|10.434|
|整遍入口含输出序列化（秒）|56.066|55.484|52.014|52.908|

每遍新建SurfaceStatsCache，因此warm实际重算全部metrics和ROI；两版每遍geometry_reads=2、principal_curvature_computations=2、roi_summary_transfers=6。8f使用vertex_area_computations=2，3a使用roi_vertex_area_computations=2，反映既定ROI面积累加定义修正。

两臂采样显存峰均2,046,820,352字节；0.5秒请求下分别188/189样本、最大间隔0.710/0.668秒。它们是离散进程采样峰，不是连续真实峰。allocator关闭时PyTorch memory counters不可代表显存下降。

这次同输入阶段未出现原整例的metrics约25.8秒版本差；两版均重现较慢的约30秒级metrics及10秒级首ROI。无法据此指定原整例差异的输入、运行环境、进程历史或负载原因，也不能推广整例速度结论。后续数值比较已完成，见NUMERICAL_RESULTS.md与numerical_summary.json。原等待控制器pid80035在进入command前取消、清理回执保留；冻结比较脚本随后在隐藏CUDA、CPU1的新目录只读分析，耗时不计benchmark。原v1 TypeError保留，属于诊断接口错误、算法前失败，不能当作OOM或算法退化。

机器回执：receipts_20261004T1459.json；计时观察：timing_observation.json。后续数值结果独立追加，不覆盖上述来源。
