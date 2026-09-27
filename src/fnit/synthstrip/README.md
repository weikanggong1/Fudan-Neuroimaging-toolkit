# SynthStrip

这里实现脑提取。`model.py` 定义官方 U-Net，`pipeline.py` 实现影像处理及 `SynthStrip`、`StripResult`，`__init__.py` 导出接口。单被试命令使用统一的 `fnit synthstrip` 入口。

```python
from fnit.synthstrip import SynthStrip

model = SynthStrip(
    weights="/path/to/weights",  # 输入：模型权重文件或目录
    device="cuda:0",           # 输入：推理设备
    no_csf=False,              # 输入：是否使用排除脑脊液的权重
    threads=4,                 # 输入：PyTorch CPU 线程数
)
result = model(
    image="subject_T1w.nii.gz",  # 输入：T1 影像路径
    border=1,                   # 输入：脑掩膜距离阈值，mm
    fill=0,                     # 输入：掩膜外的强度
)
result.mask.save("subject_mask.nii.gz")  # 输出：二值脑掩膜
```

调用返回去颅骨图像 `image`、脑掩膜 `mask` 和符号距离图 `distance`。路径或 nibabel 输入返回仓库内 `Volume`，包含 `.data`、`.affine`、`.geom.vox2world.matrix`、`.geom.voxsize`、`.save(path)`；已有 Surfa 内存对象仍返回同类对象。输入可为 3D 图像或逐帧处理的 4D 图像。

同输入、官方命令和本包命令的实际体素、几何与耗时记录见[无 Surfa 迁移验证](../../../validation/synthstrip_no_surfa_20260927/README.md)。

[完整参数、CLI、源码分析与验证](../../../docs/synthstrip/README.md) · [权重](../../../docs/WEIGHTS.md)
