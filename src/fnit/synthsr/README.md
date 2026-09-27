# SynthSR

这里实现单幅 3D MRI/CT 到 1 mm 合成 T1w 的推理。`model.py` 定义与官方 HDF5 权重对应的 PyTorch U-Net，`spatial.py` 处理重采样、方向和填充，`pipeline.py` 连接读图、推理、后处理和写盘；`__init__.py` 导出公开接口。推理不调用 FreeSurfer、TensorFlow 或 Surfa。

```python
from fnit.synthsr import SynthSR

sr = SynthSR(
    weights=None,       # 已配置的通用 v2 权重
    device="cuda:0",    # 运行设备
    lowfield=False,     # 通用模型
    v1=False,           # v2 模型
    threads=4,          # CPU 线程数
)
result = sr(
    image="case_FLAIR.nii.gz",  # 输入：单幅 3D FLAIR
    ct=False,                  # MRI 输入
    disable_flipping=False,    # 双次翻转预测
    disable_sharpening=False,  # 保留锐化
)
result.image.save(path="case_synthsr.nii.gz")  # 输出：1 mm T1w
```

`result.image.data` 是写盘前量化为 `uint8` 的 0–255 体素，`result.image.affine` 描述 1 mm 输出网格。MGZ 经 nibabel 保存后重新读入会报告 `float32` 存储类型，数值仍是这组量化值。构造时加载一次权重，可复用于后续单被试调用。接口参数、原版 `mri_synthsr` 对应关系、权重和输出格式见[完整说明](../../../docs/synthsr/README.md)。

当前 SynthSR 的路径和单例 CLI 不导入 Surfa。同一真实 FLAIR 的新旧输出字节、官方体素/仿射及时间比较见[迁移验证](../../../validation/synthsr_no_surfa_20260928/README.md)。此前 12 例 T1w 的[历史基准](../../../validation/synthsr/README.md)不作为本次改写的多例验收。
