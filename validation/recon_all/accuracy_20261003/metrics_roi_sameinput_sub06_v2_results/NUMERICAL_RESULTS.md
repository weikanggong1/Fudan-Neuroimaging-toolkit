# v2 同输入数值回归（算法后 CPU1 分析）

本轮仅左半球同输入；真实两GPU臂已在GPU1、4线程、双锁下完成。后续比较仅读取固定NPZ和文本，不调用影像算子或GPU；使用冻结脚本SHA `0a471625e604dff306a9c7ae378ba1f44350dba71ed2f49a0df76e9396f30b98`，隐藏CUDA，CPU数值库均1线程。这个报告分析的耗时不纳入cold/warm或整例benchmark。原等待控制器80035未进入command，取消后interrupted/130，SIGTERM和自有进程清理记录保留。

四组比较（8f cold/warm、3a cold/warm、跨版本cold、跨版本warm）均满足输入SHA、相同白面/pial坐标及同序faces依据。每个形态图有130,416个顶点，读出dtype为FreeSurfer大端float32；ROI文本数值读出float64，按原格式舍入。

5个已有算子门槛全部通过、outlier_count=0：area/area.pial使用既有atol0.001和rtol0.001；thickness/curv/curv.pial使用既有atol0.005和rtol0.001。thickness四组逐点完全相等；跨版本curv.pial最大绝对差cold=1.704692840576172e-5，warm=4.744529724121094e-5。area.mid最大差4.76837158203125e-7，volume最大差1.9073486328125e-6；这两图只报告差异，没有新增等效阈值。

六ROI文本表的命名行、NumVert及9个数值列，在四组比较均完全相等：aparc34行、a2009s74行、DKTatlas31行、aparc.pial34行、BA_exvivo14行、BA_exvivo.thresh14行。它说明当前文本输出稳定，不证明舍入前内部浮点累加逐位相等；3a的ROI面积累加定义修正仍按既定实现保留，未要求旧新字节一致。

这次对照未发现该左半球同输入阶段的新数值漂移，也未复现原整例metrics的约25.8秒版本耗时差。它不能确定整例差额因果，不支持整例性能或正式整体等效声明；没有为结果放宽门槛，没有重跑真实算法。

远程结果位于 `FNIT/runs/recon_accuracy_20261003/metrics_roi_sameinput_sub06_v2/comparison_cpu1_v2`。原comparison.json SHA为 `da07e13b4ecf813476b34520f0f9f58cc8bfe7d64f8e8f5474509036f3703ddd`；本地格式化副本以numerical_summary.json记录自身SHA。完整来源与取消记录在numerical_receipts_20261004T1507.json。

现场证据重核：冻结工具清单全部匹配；8f/3a源码和程序库存分别475/501文件重核通过，输入manifest与两配置、实际报告/监测回执一致，28个已比较NPZ重核通过。绑定记录见evidence_binding.json。显存仅为采样峰；仅三个直接导入模块显式记录加载路径，间接模块源文件由完整库存约束，未逐模块跟踪实际加载路径。取消竞争单测覆盖兼容队列，v2未单独执行该竞争动态测试。
