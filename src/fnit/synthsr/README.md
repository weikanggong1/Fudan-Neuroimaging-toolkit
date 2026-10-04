# SynthSR 源码目录

`model.py` 定义官方 HDF5 对应的成熟 PyTorch U-Net，`spatial.py` 处理重采样、方向和填充，`pipeline.py` 连接读图、推理、后处理和保存。输入单幅 3D MRI/CT，输出1 mm合成T1w；生产不调用FreeSurfer或TensorFlow。

本轮特别修复 CPU FP32 网络尾差：`_cpu_inference.py` 延迟选择受保护的 CPU 推理，复用成熟卷积/池化/上采样及checkpoint，以临时CL3D权重视图保持公开参数stride；`_cpu_math.py` 为独立NumBa oneDNN ELU与Eigen BN公式。限Linux x86-64/SSE2/FMA、MKLDNN、CPU F32单volume/eval/no-grad。训练、autograd、CPU autocast、hooks和替换叶模块走原Module路径；NumBa线程mask返回后恢复。CUDA沿用原forward正文且不导入helper，主页Conda依赖不变。

2026-10-04的冻结`source_sr_v2`，同一真实原始T1在nodecw7相同8核/8线程CPU ABBA：两CNN共22,020,096值及最终9,072,000浮点/uint8全同官方冻结整图参考，原固定浮点门0失败，affine/header/NIfTI文件SHA同。正常CLI旧28.24/29.08 s、新29.91/26.77 s，基本持平；RSS旧约10.22GB、新7.50GB。H100默认TF32旧新完整CNN/float/uint8/header/输出文件SHA全同，allocated10,006,443,008/reserved13,845,397,504 B峰值相同，仍在20GB cap内。共享GPU97–100%利用率下时间只作观察。v1/低场/关闭翻转/关闭锐化及第二T1、FLAIR、真实CT、64mT和EPI首通道量化图均全同官方；EPI NPZ原浮点门通过但仍有尾差。参数分支、全部时钟、原型失败与融合证据见[本轮验证](../../../validation/smri_cpu/synth_fixes_20261004/sr_cpu_followup/README.md)。

## Python 示例

```python
from fnit.synthsr import SynthSR

super_resolution_model = SynthSR(
    weights=None,  # None 按 FNIT 固定权重配置查找；可给HDF5路径或目录
    device="cuda:0",  # 显式计算设备；CPU用"cpu"
    lowfield=False,  # 是否选择低场单输入模型
    v1=False,  # 是否选择2021年v1，优先于lowfield
    threads=8,  # CPU线程预算
)
super_resolution_result = super_resolution_model(
    image="case_FLAIR.nii.gz",  # 输入路径或nibabel SpatialImage
    ct=False,  # 真实HU CT为True，MRI为False
    disable_flipping=False,  # 保留左右翻转预测平均
    disable_sharpening=False,  # 保留末端锐化
)
super_resolution_result.image.save(path="case_synthsr.nii.gz")  # 1 mm uint8量化图
```

`result.image.data` 为0–255 uint8，`result.image.float_data` 为NPZ使用的量化前浮点，`result.image.affine` 描述1 mm输出RAS网格；MGZ保存后读取报告float32，但数值仍是量化值。构造只加载一次权重。完整七节说明包含每个参数、所有输入输出、CLI、原软件命令、最新版及历史精度/时间、脑图和文献，见[功能说明](../../../docs/synthsr/README.md)。

旧nodecw10的9参数/7域格式量化记录及默认浮点3,522点失败保留为修复前历史，[历史报告](../../../validation/smri_cpu/strip_sr_20261004/README.md)不代替本轮候选验收。较早ELU和Eigen完整原型未通过原浮点门，未接入生产。
