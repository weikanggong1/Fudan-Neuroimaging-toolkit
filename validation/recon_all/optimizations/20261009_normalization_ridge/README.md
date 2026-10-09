# 限定medial ridge消费范围：真实两例和完整归一化

固定公开ds000114 sub06/sub07的803aec50自产norm/aseg.presurf/brainmask检查点，
没有读取官方结果作为生产输入。真实aseg.presurf为MGH float32存储的整数标签。
源码祖先为5f6ca781，实际测试是冻结v4依赖＋既有v7初始偏场两文件＋新v10 ridge两文件。
完整报告和source_binding.json记录实际代码SHA，不把祖先commit当作完整候选版本。

- profile_v9：两例各cold+2warm，外背景传播16.01–29.06秒是主要热点；过滤仅0.12–0.16秒。
- stage_v10：4次完整ridge/过滤对照，全部WM与26邻域float32距离、ridge/control/removed逐位一致。
- api_v11：同GPU1、CPU4–7四线程、TF32、无半精度、cacheoff的full→local→local→full。
  8/8完整MGZ逐字节一致，初始和两轮中间float32/控制图也一致。
  完整API中位数60.063→38.958、59.703→33.947秒；全部逐次数据保留。
- memory_v13：额外cache-enabled完整API同样零差异，只用于显存统计。
  allocated754,974,720、reserved773,849,088字节；未知进程树峰值保持null。
- plot_v14：公开自产brain原网格切片和零误差图；没有发布MRI、权重或许可证。
- diagnostic_logs：默认完整场兼容测试1/1（9种mask）及原显存诊断收尾日志。

原背景pass处理16,158,505/16,227,609点，限域处理179,549/156,319点；内侧完整传播不变。
限定范围覆盖非极大值选择的梯度与floor/ceil采样角点，原堆顺序/动态更新前缀保持；
远背景辅助距离不再计算，不能当成通用完整signed_distance。该默认公共接口仍为完整场。

每次完整API含校验、加载、CPU传播、GPU搬运/初始偏场、完整反馈及写出，GPU同步指定目标设备。
冷诊断进程收据另含导入/初始化/SHA比较和监控收尾；不能误标为纯生产CLI时间。
共享负载导致baseline明显波动；另有0.374秒CPU兼容测试在ABBA期间执行，记录其范围。
没有新的官方重复性、原始T1整例或整体指标等效结论，未推算整例提速。

显存诊断API完成，但其最初绘图调用误用normalized.mgz文件名而失败，退出码1和日志保留；
正确brain.mgz在独立plot_v14重新生成，退出码0。没有重新标记失败调用。
NVML采样与整卡共享上界见完整报告；缓存关闭的统计不可用，不填0。

[七节中文功能页](../../../../docs/recon_all/NORMALIZATION_RIDGE_LOCAL.md)列全部输入输出、参数、证明和原命令。
[reproduce.sh](reproduce.sh)使用自行准备的已授权冻结源码/输入，结果目录必须尚不存在。
数据：[OpenNeuro ds000114](https://openneuro.org/datasets/ds000114)，CC0；路径和主机已去敏感。
