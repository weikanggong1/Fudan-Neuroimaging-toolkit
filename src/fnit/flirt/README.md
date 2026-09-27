# TorchFLIRT 源码目录

本目录实现 FNIT 的单被试线性配准：

- `core.py`：12-DOF/correlation-ratio 仿射配准和 6-DOF/NMI 刚体配准；
- `coordinates.py`：FSL scaled-mm 与 world-RAS 的矩阵转换；
- `types.py`：nibabel 输入检查和 `FLIRTResult`；
- `standalone.py`：FSL 风格路径接口与原子写盘；
- `cli.py`、`__main__.py`：`fnit-flirt` 命令行入口。

候选运行时只使用 PyTorch、NumPy 和 nibabel，不调用 FSL。CUDA 使用 float32 并默认允许 TF32。

## Python 单被试示例

```python
from fnit.flirt import run_flirt

result = run_flirt(
    input="subject_GM.nii.gz",  # moving 3D 图像
    reference="template_GM.nii.gz",  # fixed 图像；决定输出网格
    output="subject_GM_to_template.nii.gz",  # reference 网格上的 float32 图像
    omat="subject_GM_to_template.mat",  # input→reference 的 FSL scaled-mm 4×4 矩阵
    init=None,  # 可选 FSL scaled-mm 初始矩阵
    inweight=None,  # 可选 input 网格连续权重图
    refweight=None,  # 可选 reference 网格连续权重图
    dof=12,  # 当前仿射配置使用 12
    cost="corratio",  # 当前 12-DOF 配置使用 correlation ratio
    device="cuda:0",  # 计算设备；可改为 "cpu"
    overwrite=False,  # 是否覆盖已有输出
)
```

返回值 `result` 包含 `moved`、`matrix`、`moving_to_fixed_world`、`fixed_to_moving_world` 和 `qc`。`moved` 是 `nibabel.Nifti1Image` 子类；`matrix` 是 input→reference 的 FSL scaled-mm 矩阵。

## 命令行

```bash
fnit-flirt \
  -in subject_GM.nii.gz \
  -ref template_GM.nii.gz \
  -out subject_GM_to_template.nii.gz \
  -omat subject_GM_to_template.mat \
  -dof 12 \
  -cost corratio \
  --device cuda:0
```

对应的原软件命令为：

```bash
flirt \
  -in subject_GM.nii.gz \
  -ref template_GM.nii.gz \
  -out subject_GM_to_template.nii.gz \
  -omat subject_GM_to_template.mat \
  -dof 12 \
  -cost corratio
```

`-in` 是 moving 图像，`-ref` 是 fixed 图像和输出网格，`-out` 保存重采样结果，`-omat` 保存 input→reference 的 FSL scaled-mm 矩阵。当前还支持 `-dof 6 -cost normmi`；其他 FSL 配置会被拒绝。

10 例真实 GM 对照由 `flirt/core.py` `f5315f…` 生成；当前 `ce375d…` 对 **12-DOF/corratio** 的数值路径保持源码等价，没有 fresh current-hash 完整重跑。CPU 和默认 TF32 GPU 均有 9/10 例满足矩阵 `rmsdiff <= 0.05 mm`，moved Pearson 中位数分别为 0.999985 和 0.999823。完整命令中位时间为 FSL CPU 27.705 s、FNIT CPU 82.159 s、FNIT H100 30.165 s。该结果不构成逐元素或完整数值等价，也不覆盖已改变的 6-DOF/normmi 路径。源码链见[配置限定的证明](../../../validation/runtime_dependencies/flirt_profile_source_equivalence.public.json)。

输入、输出结构、参数说明、坐标公式、当前源码报告和公开示意图见[功能说明](../../../docs/flirt/README.md)。
