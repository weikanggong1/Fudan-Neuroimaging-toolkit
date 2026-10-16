# recon-all 验证索引

[2026-10-16 cbaffa 两例原始 T1 整例与完成的 v2 事后评估](optimizations/20261016_accelerator-a-1_cbaffa_two_case/README.md)：
两例均退出 0，138/138 项输出齐全，生产网格门通过；严格逐文件诊断为 14/138、23/138。分区 Dice、双向表面距离、脑区统计和局部异常已评估；扩展网格检查仍发现 white/pial 互穿。整体指标等效、受控退化和同环境提速均未判定。

[2026-10-14 f016 两例部分运行与 sub-07 事后评估](optimizations/20261014_accelerator-a-1_f016_partial/README.md)：
sub-07 从原始 T1 完成且 138/138 输出，sub-06 在拓扑阶段失败；两例整体验收与整体指标等效均未判定。该记录绑定旧 f016 科学源码，不代表后续修复版结果。

[2026-10-01 五阶段优化历史整例结果](optimizations/20261001_serial/FINAL_RESULTS.md)：
两例从原始T1和空目录运行，138项均存在。完整命令4242.884→3453.936秒、
4043.589→3712.568秒；同目标H100与4线程，单次共享服务器观察。
对优化前七类标签Dice为1，68区厚度/面积/体积/曲率不变，有序white/pial不变。
MNI三个文件仍有差异；相对官方严格诊断为6/138、2/138，整体指标等效尚未判定。

完整机器报告、两例脑图、双向表面距离和局部异常均在该结果页。
源码、资源与程序SHA见[版本绑定](optimizations/20261001_serial/whole/integrated_main/source_after_main_merge.json)；
[复现脚本](optimizations/20261001_serial/REPRODUCE.md)区分冻结阶段、自产连续链和原始T1整例。
该版白/pial穿越阳性与优化前相同，不以旧的单表面拓扑检查代替互不穿越验收。

## 仍用于差异定位的历史证据

[2026-09-27连续链](python_gpu_port/NATIVE_FREE_CONNECTED_20260927.md)对应早期实现，
其52项缺失和1805.7秒不能代表当前标准流程。
[比较JSON](python_gpu_port/native_free_sub01_20260927/comparison.json)、
[严格138 JSON](python_gpu_port/native_free_sub01_20260927/strict_138.json)、
[旧运行与哈希](python_gpu_port/native_free_sub01_20260927/run.json)
保留为历史排错输入；[阶段移植](python_gpu_port/README.md)和
[门槛说明](python_gpu_port/RELEASE_GATES.md)不因性能通过而降低严格比较标准。
