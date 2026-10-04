# SynthStrip 源码目录

2026-10-04 的[相同 CPU 资源验证](../../../validation/smri_cpu/strip_sr_20261004/README.md)发现原始 NIfTI 的 `pixdim` 被共享图像构造重写，改变了 1 mm 网格。共享修复 `dc2fc052` 保留原头文件；本模块 `6b7aeafd` 在 CPU oneDNN 启用时使用 `channels_last_3d`，CPU 构造保持调用方 CUDA 后端设置。nodecw10 同 8 线程与 8 物理核，26 次 CLI、11 个真实场景全部通过：brain/mask 逐值同官方，SDT `rtol=1e-5, atol=1e-4 mm`，最大全场景差 `4.3392e-5 mm`。两例默认完整 CLI 中位数官方/FNIT 为 45.561/14.901、49.199/14.027 s；冷导入与卷积时间另列。相同修复输入的 GPU ABBA 三份输出逐值同、组件显存峰值同。详细参数、真实域输入、逐项结果与脑图见[功能说明](../../../docs/synthstrip/README.md)。下面的 `cfb7beee` 数据是既有 GPU 对照。

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

CUDA 卷积默认使用 `cudnn.benchmark=False`、`cudnn.deterministic=True`，matmul/cuDNN 的 TF32 仍默认开启。真实 T1 的新进程控制发现旧 autotune 策略会产生 19 个掩膜边界差异；仅关闭 benchmark 后，两次掩膜与 SDT 逐值一致。本修复在 SynthStrip 子函数内完成，网络、官方权重、几何及公开接口沿用原实现。

[跨进程验证驱动](../../../validation/synthstrip/check_repeatability.py)使用真实输入和本地官方权重，保存私有掩膜/SDT、实际模型状态与网络输入/预测哈希，公开报告只有标量和 SHA-256；[验证说明](../../../validation/synthstrip/README.md)包含参数和复现命令。

最新受测源码 `cfb7beee` 的同输入完整三维对照中，EPI/T1 mask Dice 为 `0.999994968 / 0.999998569`，分别有 `1 / 4` 个边界体素不同；T1 脑图完整三维 RMSE 为 `0.389388`，空间 r 为 `0.999999655`。当前 T1 脑图和掩膜与 `1eb9c417` 图示输入逐值相同，既有固定原 FNIRT 场的示例继续有效。

FNIT 复用模型子函数阶段为 `1.1654 / 1.4442 s`，含影像处理、推理和阶段输出保存，不含模型构造；原 FreeSurfer 独立进程为 `8.5569 / 9.3769 s`，含启动、模型加载和影像读写。完整网格指标、实际源码与官方权重 SHA-256 见[最新版原程序报告](../../../validation/synthstrip/latest_native_comparison.public.json)。

此前冻结源码的真实 SBRef/T1 几何控制中，1 mm LIA 数组和归一化网络输入与官方 Surfa 逐元素一致。同一网络预测回采样后的脑 mask 也逐元素一致；与独立官方 GPU 推理的 mask Dice 分别为 `0.999995 / 0.999991`，有 `1 / 25` 个体素不同。该报告采用旧卷积选择策略，其源码哈希及计时边界保留在[历史几何报告](../../../validation/fmri/synthstrip_geometry_control.public.json)。当前跨进程重复性和相对原程序的精度是两个独立检查。

网络、1 mm 重采样和 SDT 回采样使用所选 PyTorch 设备；nibabel 读写影像，SciPy 处理 SDT 扩展和连通域。

## 最近版本

| 冻结版本 | 更新与 benchmark |
|---|---|
| `cfb7beee` | 最新原程序对照：EPI/T1 Dice 为 0.999994968/0.999998569；FNIT 子函数 1.1654/1.4442 秒，原独立进程 8.5569/9.3769 秒，计时边界见[当前报告](../../../validation/synthstrip/latest_native_comparison.public.json)。 |
| `44364a8` | 固定卷积算法选择，TF32 和 API 沿用既有默认值；[跨进程报告](../../../validation/synthstrip/cudnn_repeatability.public.json)分别保存当前默认策略验收及 `original_diagnostic` 中的旧策略控制。调用时间含哈希观测，进程时间含启动、加载和写盘；范围限于 SynthStrip。 |
| `1eb9c417` | 包含 `1db5917` 的几何修复；真实 SBRef/T1 控制为 7.61/4.83 秒，含模型构造、预测、双实现回采样和比较，未含启动、输入 conform/归一化及写盘。 |

[完整参数、CLI、源码分析与验证](../../../docs/synthstrip/README.md) · [权重](../../../docs/WEIGHTS.md)
