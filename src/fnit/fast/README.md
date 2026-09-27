# TorchFAST 源码目录

`TorchFAST` 对脑提取后的单通道 T1w 做 CSF、GM、WM 三组织分割，同时估计乘性 bias field 和 partial-volume fraction。算法使用 HMRF-EM 与 PyTorch，可在 CPU 或 CUDA 上运行；不调用 FSL，也不需要模型权重。

## Python 单被试示例

```python
from fnit.fast import TorchFAST

model = TorchFAST(
    device="cuda:0",  # 计算设备；可改为 "cpu"
    threads=1,  # 影像读取、写盘和部分归约使用的 CPU 线程数
)
result = model(
    image="T1_brain.nii.gz",  # 输入：脑提取后的单帧 3D T1w
    mask=None,  # 可选输入：同网格脑 mask；None 使用 image > 0
)
result.pve_gm.save(path="T1_brain_pve_1.nii.gz")  # 输出路径：GM 部分体积图
result.bias_field.save(path="T1_brain_bias.nii.gz")  # 输出路径：乘性偏置场
result.restored.save(path="T1_brain_restore.nii.gz")  # 输出路径：偏置校正影像
```

输入可以是路径或 nibabel 空间影像。PVE 顺序固定为 CSF、GM、WM；所有影像输出是 `nibabel.Nifti1Image` 子类，并保留输入 shape 和 affine。`bias_field` 是 acquired image 中的乘性场；脑内满足 `restored = input / bias_field`。

## 命令行与原软件对应

```bash
fnit fast -i T1_brain.nii.gz -o T1_brain --device cuda:0 --threads 1 -b -B
fast -b -B -o T1_brain T1_brain.nii.gz
```

`-i` 是 brain-only T1w，`-o` 是输出 basename，`-b` 写 bias field，`-B` 写 bias-corrected 图像。FNIT 还写三张 PVE、hard segmentation、PVE segmentation 和 mixel type；完整文件树见[功能说明](../../../docs/fast/README.md)。

2026-09-27 当前 Nibabel I/O 已用一幅真实 brain-only T1w 完整回归。GM PVE 对 FSL FAST 的 Pearson、MAE 和 0.5 Dice 分别为 0.984627、0.015411 和 0.992328；对应文件的 shape、affine、dtype 和 qform/sform code 一致。GPU 测量进程为 12.55 s，Torch 显存 allocated 峰值 2,352 MiB。源码树 SHA-256 为 `433723da856c798faca9a66ea95784bdbb5ad5b13f66f451ce32e7030559943b`。GPU 同步 HMRF 与 FSL 逐体素更新不同，因此不声明逐体素等价。当前命令、完整输出和图见[功能说明](../../../docs/fast/README.md)与[验证记录](../../../validation/fast/README.md)。

该移植和保留的 `upstream_fast4/` 源码受非商业 FSL 6.0 许可证约束；上游源码不由 Python 包编译或导入。
