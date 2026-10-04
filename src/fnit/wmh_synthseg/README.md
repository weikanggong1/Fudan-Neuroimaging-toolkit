# WMH-SynthSeg 源码目录

这里实现 WMH 与脑结构联合分割。`model.py` 定义与官方权重匹配的3D U-Net，`spatial.py` 处理方向和1 mm重采样，`pipeline.py` 完成预处理、推理和标签/概率/软体积输出。单例Python、全部参数、命令行和原版指令见[功能说明](../../../docs/wmh_synthseg/README.md)。

2026-10-04生命周期修复：CPU strict-load权重后一次迁移；无梯度eval释放已消费缓冲；GPU裁剪分块nearest拼接，并在大GroupNorm后释放inactive缓存。FP32/TF32、GN/Conv算子、参数及累加不变。训练、有梯度和容器/global hooks走原Module调用；公开调用完成或异常后恢复原模式。

原始公开T1完整GPU `crop=True` 在20,000,000,000 B限额下，allocated/reserved峰值18,373,921,792 /18,438,160,384 B，分割、WMH概率和匿名CSV文件SHA及57个实际卷积kernel均与旧GPU相同。nodecw7同8核CPU标签、概率、33CSV数值和header/affine全同，完整官方172.064/107.942 s、候选77.134/83.876 s。共享负载使这些时间不能证明稳定性能保证。

GPU默认 `crop=False` 保留原forward、decoder及概率调度，以保留旧GPU输出；原full allocated37,997,219,328 B，**仍未达到20 GB**。同实例full→crop→full输出SHA和模式恢复已验证。20 GB环境需显式 `crop=True`；裁剪可能移除视野边缘。完整时间分步、原型失败、权重/输入/源码SHA及复现见[本轮记录](../../../validation/smri_cpu/synth_fixes_20261004/README.md)。

上一轮保存头信息修复与历史真实脑图仍见[功能说明](../../../docs/wmh_synthseg/README.md)及[上一轮CPU记录](../../../validation/smri_cpu_20261004/t2_seg/README.md)；新输出网格使用原版默认float32 NIfTI header，保留该修复。
