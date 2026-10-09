# 本轮原始T1整例：中断收据

冻结 `c7231316-plus-frozen-v2-module-hashes` 从原始公开 sub-01 T1 和空目录运行。
报告记录实际输入、源码、资产、程序SHA、线程、GPU及硬件。
这是控制臂，候选臂尚未启动。已保存42个完成阶段记录，原调度状态仍为running。

headcw已认证连接可用，但gpucw1新SSH先Connection refused、后No route to host；
原验证tmux会话不再存在，没有CLI返回码/最终报告。原因没有确认，不归因OOM。
不能把现有138文件存在性、阶段相加或手动补跑称为整例完成。

- [只读中断收据](interruption_receipt.json)：分离执行、完整性、网格、严格复现、退化及整体等效状态。
- [原外层报告](control/benchmark.json)：源码、资源与输入身份；初版未周期保存进程采样，不能推断整例峰值。
- [原阶段报告](control/subject/fnit-native-free-run.json)：42阶段原始记录，不修改running字段。
- [原CLI日志](control/run.log)：不包含输入影像、凭据或许可证内容。

关键本次观测为N4 122.778秒、GCA 444.146秒、surface group 706.712秒、
register group 214.085秒、finish surface group 344.745秒；并行group不是两个半球耗时相加。
GCA中线性搜索400.532秒，是继续扩大候选评分块并批量余子式求逆的实测依据。
`mri_fill` 53.164秒已是FNIT Python/Numba路径，不因阶段名称将其称为原生C++。

初版控制先运行、候选后运行，Numba磁盘缓存与共享负载可能影响时间；即使两臂完成，
单次有序配对也不能证明稳定整例提速。恢复后使用新空目录和均衡的独立缓存策略。
新的包装脚本增加每30秒原子检查点和外部信号收据，保留部分同期显存采样；
SIGKILL或节点故障仍只能保存最后完整检查点，不能恢复未写出的结果。
