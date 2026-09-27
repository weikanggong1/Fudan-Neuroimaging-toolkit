# SynthSeg 与皮层分区模块

本目录包含 FreeSurfer 8.2 SynthSeg 2.0 的 PyTorch 推理代码。

- `synthseg.py` 提供公开的 33 类 `SynthSeg`、`SynthSegResult` 和软体积 CSV；
- `segment.py`、`preprocess.py`、`postprocess.py` 实现网络推理、影像预处理和标签后处理；
- `pipeline.py` 与 `model.py` 提供 GPU recon-all 使用的皮层分区路径。

独立入口只覆盖单幅 T1 的非 robust、非 parcellated 33 类路径；输入可为路径或 nibabel 空间影像，分割输出为 `nibabel.Nifti1Image` 子类：

```python
from fnit import SynthSeg

model = SynthSeg(
    weights=None,  # 权重输入：None 按 FNIT 配置查找四个官方文件
    device="cuda:0",  # 计算设备；可改为 "cpu"
    threads=4,  # 预处理和后处理使用的 CPU 线程数
)
result = model(
    image="subject_T1w.nii.gz",  # 输入：单幅 3D T1w
    keep_geometry=False,  # False 输出预处理后的 RAS、约 1 mm 网格
    color_lut=None,  # 可选 FreeSurfer LUT；None 不附加外部色表
)
result.segmentation.save(path="subject_synthseg.nii.gz")  # 输出路径：33 类硬分割图
result.write_volumes_csv(
    source="subject_T1w.nii.gz",  # CSV 中记录的原始输入标识
    path="subject_synthseg.vol.csv",  # 输出：软体积 CSV
)
```

单被试 CLI、原版 `mri_synthseg` 参数对应、权重和验证边界见[功能说明](../../../docs/synthseg/README.md)。

2026-09-27 当前源码已在三幅公开真实 T1w 上重跑。shape、affine、`int32` dtype、qform code 0 和 sform code 2 均与 FreeSurfer 8.2 原版一致；最低标签一致率为 0.99998817，GPU 完整命令中位数为 9.48 s。源码树 SHA-256 为 `39fa204aea7674ad7c6e09652d0f8750dd2872b1b78799812ab0d71b5b6c8972`。默认和 `keep_geometry=True` 的文件头契约均有保存/重载测试。当前命令、逐例数值和图见[功能说明](../../../docs/synthseg/README.md)与[验证记录](../../../validation/synthseg/README.md)。
