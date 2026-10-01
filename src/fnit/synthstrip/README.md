# SynthStrip 源码目录

这里实现脑提取。`model.py` 定义官方 U-Net，`pipeline.py` 实现影像处理及 `SynthStrip`、`StripResult`，`__init__.py` 导出接口。单被试命令使用统一的 `fnit synthstrip` 入口。

```python
from fnit.synthstrip import SynthStrip

model = SynthStrip(
    weights="/path/to/weights",  # 权重输入：官方 PT 文件或权重目录
    device="cuda:0",  # 计算设备；可改为 "cpu"
    no_csf=False,  # 使用保留 CSF 的默认模型
    threads=4,  # 预处理和后处理使用的 CPU 线程数
)
result = model(
    image="subject_T1w.nii.gz",  # 输入：3D T1w 或逐 frame 处理的 4D 图像
    border=1,  # 距离场小于 1 mm 的体素归入脑 mask
    fill=0,  # 脑外输出填充值
)
result.mask.save(path="subject_mask.nii.gz")  # 输出路径：二值脑掩膜
```

调用返回三个 `FNITNifti1Image`（`nibabel.Nifti1Image` 子类）：去颅骨图像 `image`、脑掩膜 `mask` 和符号距离图 `distance`。三者保留输入几何；输入可为 3D 图像或逐帧处理的 4D 图像。

当前固定真实 SBRef/T1 控制中，1 mm LIA 数组和归一化网络输入与官方 Surfa 逐元素一致。同一网络预测回采样后的脑 mask 也逐元素一致；对独立官方 GPU 推理，mask Dice 分别为 `0.999995 / 0.999991`，有 `1 / 25` 个体素不同。候选保留默认 TF32。

网络、1 mm 重采样和 SDT 回采样使用所选 PyTorch 设备；nibabel 读写影像，SciPy 处理 SDT 扩展和连通域。控制计时为 `7.61 / 4.83 s`，不含输入 conform、Python 启动和最终写盘，不能视为完整 CLI 耗时。[当前机器可读报告](../../../validation/fmri/synthstrip_geometry_control.public.json)记录输入、权重、源码哈希及重复结果。

[完整参数、CLI、源码分析与验证](../../../docs/synthstrip/README.md) · [权重](../../../docs/WEIGHTS.md)
