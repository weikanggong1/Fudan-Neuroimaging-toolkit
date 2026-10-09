# 离散曲率真实回归

`report_v2.json`：gpucw1 H100，ds000114 sub-07/sub-08双侧冻结smoothwm，
原生八图重复2次，CPU/GPU/GPU/CPU。完整候选API含拓扑准备、传输和读写。
K逐位一致；原先探索门没有全部通过，未切换默认，未评估整例等效。

`failure_v1.json`保留不同度数网格padding索引越界；v2仅修无效padding索引。
`diagnostics/report.json`对已保存候选做ULP、坐标、区域、面面积与脑图定位。
最大H绝对误差所在点为1ULP，BE最大绝对误差为3ULP；不删除这些点。

`diagnostics/acos_cpu_v2.json`在headcw固定几何，只替换CPU反三角函数作来源隔离。
四半球libm acosf重建H与原参考逐位一致；GPU中间量隔离未完成。
`acos_cpu_endian_bug_v1.json`和初版`acos_cpu.json`的位图差异统计未统一大端dtype，
其different_bits字段无效。原数值比较不受影响，保留首版以避免掩盖诊断错误。

源码/脚本/表面/参考程序SHA在报告中；主机不同的CPU来源诊断不视为GPU配对计时。
PyTorch allocated/reserved单列，未测完整进程占用或整例显存。
