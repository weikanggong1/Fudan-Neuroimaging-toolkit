# WMH-SynthSeg

这里实现 WMH 与脑结构联合分割。`model.py` 定义与官方权重匹配的 3D U-Net，`spatial.py` 处理图像方向和 1 mm 重采样，`pipeline.py` 完成预处理、推理及标签和病灶概率输出。单例 Python 与命令行用法、对应的原版指令及验证结果见[功能说明](../../../docs/wmh_synthseg/README.md)。

当前输出由仓库内 `Volume` 保存，单例 Python/CLI 不需要导入 Surfa。同一真实 FLAIR 的标签、病灶概率、仿射、软体积和 NIfTI/MGZ 字节核对及耗时见[本次迁移验证](../../../validation/wmh_no_surfa_20260928/README.md)。旧版 12 例记录见[历史验证](../../../validation/wmh/README.md)，不作为新输出路径的多例验收。
