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

当前 12 例真实 T1w、H100 单进程 CLI 对照中，FreeSurfer 8.2 / FNIT
墙钟中位数为 `16.916 / 16.395 s`；最低 mask Dice 为 `0.993925`，最低 brain
image Pearson r 为 `0.994261`。FNIT 单例 PyTorch peak allocation 为
`8.316 GB`。候选使用默认 TF32，官方参考关闭 TF32，因此不声明逐体素完全相同。
当前报告绑定 `pipeline.py`、`model.py`、`_nib.py` 和 `weights.py` 的源码
SHA-256；逐例指标和图见[功能说明](../../../docs/synthstrip/README.md)。

[完整参数、CLI、源码分析与验证](../../../docs/synthstrip/README.md) · [权重](../../../docs/WEIGHTS.md)
