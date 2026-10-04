# WMH-SynthSeg 源码目录

这里实现 WMH 与脑结构联合分割。`model.py` 定义与官方权重匹配的 3D U-Net，`spatial.py` 处理图像方向和 1 mm 重采样，`pipeline.py` 完成预处理、推理及标签和病灶概率输出。单例 Python 与命令行用法、对应的原版指令及验证结果见[功能说明](../../../docs/wmh_synthseg/README.md)。

2026-10-04 在 nodecw10 的同一组 8 个物理核和 8 个计算线程上，原始公开 FLAIR 的 crop/full 模式均与 FreeSurfer 8.2 CPU 的硬标签、数值 CSV 和 WMH 概率逐值相同。本次修复仅涉及新输出网格的默认 NIfTI header，不继承输入扫描 XML；真实 FLAIR full/T1 crop 的新 header 也与官方相同。GPU 在 20 GB allocator 限额下的原模型 GroupNorm/初始化 OOM 如实保留，未列为性能回归通过。输入输出、完整矩阵状态、各冻结文件 SHA 和脑图见[功能说明](../../../docs/wmh_synthseg/README.md)与[本轮记录](../../../validation/smri_cpu_20261004/t2_seg/README.md)。2026-09-27 的派生样例结果保留在功能说明的历史部分。
