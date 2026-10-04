# SynthSR 源码目录

2026-10-04 的[相同 CPU 资源对照](../../../validation/smri_cpu/strip_sr_20261004/README.md)完成 9 个参数和 7 个域/格式场景，NIfTI 量化容差通过：最差 exact99.992227%、max1。默认 T1 浮点 NPZ 在固定 rtol1e−5/atol1e−3 下仍有3,522个超门槛点、max0.0191345。相同官方预测经过 FNIT 后处理可逐值恢复官方输出，误差已定位网络 FP32 计算。CPU channels-last/BN 原型未通过、不接入生产；CPU 构造保持调用方 CUDA 后端设置，GPU 路径不变。默认两例完整 CLI 中位数官方/FNIT62.810/42.800、80.869/45.947 s，但实际低场和 EPI 有慢于官方的场景。详细表格、阶段时间和脑图见[完整说明](../../../docs/synthsr/README.md)。下面的2026-09-27数据为历史对照。

这里实现单幅 3D MRI/CT 到 1 mm 合成 T1w 的推理。`model.py` 定义与官方 HDF5 权重对应的 PyTorch U-Net，`spatial.py` 处理重采样、方向和填充，`pipeline.py` 连接读图、推理、后处理和写盘；`__init__.py` 导出公开接口。推理不调用 FreeSurfer 或 TensorFlow。

```python
from fnit.synthsr import SynthSR

super_resolution_model = SynthSR(
    weights=None,  # 权重输入：None 按 FNIT 配置查找官方 HDF5
    device="cuda:0",  # 计算设备；可改为 "cpu"
    lowfield=False,  # 是否使用低场专用模型
    v1=False,  # False 使用默认 v2，而非 2021 年 v1
    threads=4,  # 预处理和后处理使用的 CPU 线程数
)
super_resolution_result = super_resolution_model(
    image="case_FLAIR.nii.gz",  # 输入：单幅 3D MRI 或 CT
    ct=False,  # False 按 MRI 强度处理
    disable_flipping=False,  # 保留左右翻转测试增强
    disable_sharpening=False,  # 保留输出锐化
)
super_resolution_result.image.save(path="case_synthsr.nii.gz")  # 输出路径：1 mm 合成 T1w
```

`result.image.data` 是写盘前量化为 `uint8` 的 0–255 体素，`result.image.affine` 描述 1 mm 输出网格。MGZ 经 nibabel 保存后重新读入会报告 `float32` 存储类型，数值仍是这组量化值。构造时加载一次权重，可复用于后续单被试调用。接口参数、原版 `mri_synthsr` 对应关系、权重和输出格式见[完整说明](../../../docs/synthsr/README.md)。

2026-09-27 当时默认 TF32 和 Nibabel I/O 已在 12 幅真实临床 T1w 上重跑。对原版 CPU，shape、`uint8` 和 affine 全部一致，平均 MAE 为 0.02314 灰度级；GPU 完整命令中位数为 14.45 s。源码树 SHA-256 为 `7b5bc19e1afa806fe8698ea70b6358bacaab23b19877d20d21d2e7c5f3560543`，Torch 显存 allocated 峰值为 13,780 MiB。该历史命令、逐例指标和公开图见[完整说明](../../../docs/synthsr/README.md)与[验证记录](../../../validation/synthsr/README.md)。
