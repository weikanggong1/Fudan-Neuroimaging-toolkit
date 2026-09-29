# TorchFLIRT 源码目录

本目录实现 FNIT 的单被试线性配准及已知变换的重采样：

- `core.py`：12-DOF/correlation-ratio 仿射配准、6-DOF/NMI 刚体配准及 `applyxfm`；
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

MNI152 同一世界空间的 1 mm→2 mm 重采样使用 `fnit-flirt -in MNI152_T1_1mm.nii.gz -ref MNI152_T1_2mm.nii.gz -applyxfm -usesqform -out MNI152_T1_1mm_on_2mm.nii.gz`。已有 FSL `.mat` 时把 `-usesqform` 换成 `-init input_to_reference.mat`。对应原命令是 `flirt -in MNI152_T1_1mm.nii.gz -ref MNI152_T1_2mm.nii.gz -applyxfm -usesqform -out MNI152_T1_1mm_on_2mm.nii.gz`；[完整参数和 Python 示例](../../../docs/flirt/README.md#已知线性变换mni152-分辨率转换)及[真实模板对照](../../../validation/flirt/applyxfm_mni.cpu.json)见链接。

真实 UKB b0→T1 的 6-DOF/normmi 配准，经本次候选搜索自由度修订后，相对 FSL 矩阵的世界坐标位移 RMS 从旧版 GPU 的 0.195330 mm 降至 CPU 0.009932 mm、H100 TF32 0.009983 mm；一例结果见[刚性配准报告](../../../validation/connectome/original_ukb_flirt.public.json)。12-DOF/corratio 的 10 例 GM CPU 测量有 9/10 例满足矩阵 `rmsdiff <= 0.05 mm`；H100 完成 4 例，4/4 满足门限。完整命令时间及不能直接比较的负载条件见[功能说明](../../../docs/flirt/README.md)。10 例测量源码与最终源码的 12-DOF 分支核对见[源码范围记录](../../../validation/runtime_dependencies/flirt_profile_source_equivalence.public.json)。这些结果不构成逐矩阵或逐体素等价。

输入、输出结构、参数说明、坐标公式、当前源码报告和公开示意图见[功能说明](../../../docs/flirt/README.md)。
