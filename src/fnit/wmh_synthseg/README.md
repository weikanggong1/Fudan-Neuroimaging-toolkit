# WMH-SynthSeg 源码目录

这里实现 WMH 与脑结构联合分割。`model.py` 定义与官方权重匹配的 3D U-Net，`spatial.py` 处理图像方向和 1 mm 重采样，`pipeline.py` 完成预处理、推理及标签和病灶概率输出。单例 Python 与命令行用法、对应的原版指令及验证结果见[功能说明](../../../docs/wmh_synthseg/README.md)。

2026-09-27 当前源码使用三幅公开真实 FLAIR 重跑。对 FreeSurfer 8.2 CPU 参考，最低全标签一致率为 0.99997810，最低 WMH Dice 为 0.99981002；GPU 完整命令中位数为 11.34 s。源码树 SHA-256 为 `39cd9b4c380a93370c1244e72eea4c3eaa9f277dd8d70ef44a4ec2293e6a5860`。显存 allocated 峰值 29,730 MiB，当前超过 20 GB 目标。输入输出、命令、逐例数值和当前图见[功能说明](../../../docs/wmh_synthseg/README.md)与[验证记录](../../../validation/wmh/README.md)。
