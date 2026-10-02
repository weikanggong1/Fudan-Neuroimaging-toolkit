# recon-all 验证索引

[当前五阶段优化整例结果](optimizations/20261001_serial/FINAL_RESULTS.md)：
两例从原始T1和空目录运行，138项均存在。完整命令4242.884→3453.936秒、
4043.589→3712.568秒；同目标H100与4线程，单次共享服务器观察。
对优化前七类标签Dice为1，68区厚度/面积/体积/曲率不变，有序white/pial不变。
MNI三个文件仍有差异；相对官方严格诊断为6/138、2/138，整体指标等效尚未判定。

完整机器报告、两例脑图、双向表面距离和局部异常均在该结果页。
源码、资源与程序SHA见[版本绑定](optimizations/20261001_serial/whole/integrated_main/source_after_main_merge.json)；
[复现脚本](optimizations/20261001_serial/REPRODUCE.md)区分冻结阶段、自产连续链和原始T1整例。
当前白/pial穿越阳性与优化前相同，不以旧的单表面拓扑检查代替互不穿越验收。

## 仍用于差异定位的历史证据

[2026-09-27连续链](python_gpu_port/NATIVE_FREE_CONNECTED_20260927.md)对应早期实现，
其52项缺失和1805.7秒不能代表当前标准流程。
[比较JSON](python_gpu_port/native_free_sub01_20260927/comparison.json)、
[严格138 JSON](python_gpu_port/native_free_sub01_20260927/strict_138.json)、
[旧运行与哈希](python_gpu_port/native_free_sub01_20260927/run.json)
保留为历史排错输入；[阶段移植](python_gpu_port/README.md)和
[门槛说明](python_gpu_port/RELEASE_GATES.md)不因性能通过而降低严格比较标准。
