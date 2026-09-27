# PyTorch FLIRT：输入、输出与官方对照

`fnit.flirt.TorchFLIRT` 实现两种已验证的 FSL FLIRT 参数组合：`-dof 12 -cost corratio` 仿射配准，以及原 UKB 结构连接流程使用的 `-dof 6 -cost normmi` 刚性 b0→T1 配准。两种组合都使用 8/4/2/1 mm 金字塔、角度搜索和 MISCMATHS Brent 坐标优化器。只指定 `-cost normmi` 时，FSL 在 8 mm 角度搜索中仍使用默认 `searchcost=corratio`，随后才使用 normmi；FNIT 与该规则一致。运行时只依赖包内 PyTorch 实现，不启动 FSL。CUDA 默认开启 TF32 矩阵计算，张量仍为 float32/float64，不使用 float16/bfloat16。

实现参考本仓库保存的 FSL FLIRT 2111.2 源码，受[非商业 FSL 许可证](../../licenses/FSL-6.0.txt)约束；详见[第三方声明](../../THIRD_PARTY_NOTICES.md)。当前支持的范围是上述两种参数组合；其他 FSL 代价函数、自由度、日程和权重图等选项不在公开接口中。

## 输入、输出和调用

```python
from fnit.flirt import run_flirt

result = run_flirt(
    input="b0_brain.nii.gz",              # 必填：单帧 3D 移动图像，配准从此图开始
    reference="T1_brain.nii.gz",          # 必填：单帧 3D 固定图像，定义输出形状和几何信息
    output="b0_in_T1.nii.gz",             # 输出：T1 网格上的 float32 重采样 b0
    omat="b0_to_T1_fsl.mat",              # 输出：输入到参考的 FSL scaled-mm 4×4 矩阵
    init=None,                             # 输入：可选 FSL scaled-mm 初始 4×4 矩阵或路径
    dof=6,                                 # 6 为刚性；12 为仿射
    cost="normmi",                         # 6DOF 用 normmi；12DOF 用 corratio
    device="cuda:0",                      # PyTorch 设备；省略时自动选择可用 CUDA/CPU
    overwrite=False,                      # False 拒绝覆盖已存在输出
)
```

`input`、`reference` 也可传入单帧 `surfa.Volume`；`init` 可传入有限、可逆的齐次 NumPy 4×4 矩阵。`output` 与 `omat` 至少提供一个。写入前会检查输出路径与输入路径不相同；模型完成后再提交输出文件。

`result` 是 `FLIRTResult`，字段结构如下：

| 字段 | 结构和方向 |
|---|---|
| `moved` | `surfa.Volume`，移动图像重采样到参考图像的 3D 网格 |
| `matrix` / `fsl_matrix` | NumPy 4×4，输入→参考，FSL scaled-mm 坐标 |
| `moving_to_fixed_world` | NumPy 4×4，输入→参考，RAS 世界坐标 |
| `fixed_to_moving_world` | NumPy 4×4，参考→输入，RAS 世界坐标，用于反向取样 |
| `qc` | Python 字典：设备、TF32、搜索和主代价、日程、评估次数、源码版本与验证范围 |

直接模型调用适合已有图像对象的程序，且不写文件：

```python
from fnit.flirt import TorchFLIRT

model = TorchFLIRT(
    device="cuda:0",       # PyTorch 设备
    angular_search=True,   # 启用默认 8 mm 角度搜索
    dof=6,                 # 刚性 6 自由度
    cost="normmi",         # 4/2/1 mm 主代价函数
)
result = model(
    moving="b0_brain.nii.gz",  # 移动 3D 图像或 surfa.Volume
    fixed="T1_brain.nii.gz",   # 固定 3D 图像或 surfa.Volume
    init=None,                  # 可选输入→参考 FSL scaled-mm 初始矩阵
)
```

命令行等价调用：

```bash
fnit-flirt -in b0_brain.nii.gz -ref T1_brain.nii.gz \
  -out b0_in_T1.nii.gz -omat b0_to_T1_fsl.mat \
  -dof 6 -cost normmi --device cuda:0
```

`fnit flirt` 提供相同接口。`-in` 是移动图像；`-ref` 定义参考网格；`-out` 是参考网格 NIfTI；`-omat` 是 FSL scaled-mm 矩阵；`-init` 是同一坐标约定的可选初始矩阵；`-dof`、`-cost` 选择上述配对组合；`--device` 选择 CPU/CUDA；`--overwrite` 允许覆盖。扩展名省略时按 `FSLOUTPUTTYPE` 选择 NIfTI 格式。

对应官方命令：

```bash
flirt -in b0_brain.nii.gz -ref T1_brain.nii.gz \
  -out b0_in_T1_fsl.nii.gz -omat b0_to_T1_fsl.mat \
  -dof 6 -cost normmi
```

## 矩阵坐标

FSL `.mat` 表示**输入 scaled-mm → 参考 scaled-mm**，不能直接当作 NIfTI RAS 世界坐标矩阵。令输入和参考的 voxel→RAS 仿射分别为 `W_in`、`W_ref`，对应的 voxel→FSL scaled-mm 变换为 `S_in`、`S_ref`，FSL 矩阵为 `A`，则：

```text
输入 RAS → 参考 RAS = W_ref @ inverse(S_ref) @ A @ S_in @ inverse(W_in)
```

`S` 由体素大小和 NIfTI 仿射行列式定义。FNIT 在 `moving_to_fixed_world` 中完成转换；与 FreeSurfer LTA 或世界坐标 API 交互时必须使用转换后的矩阵。

## 实数据验证

### 原 UKB b0→T1：6DOF/normmi

使用原流程从同一真实配对 DWI/T1 制作平均 b0 和 BET 脑图，FSL 6.0.7.4 与当前 FNIT 版本分别从**相同的图像**估计矩阵。世界坐标 13³ 网格位移平均/最大为 `0.186655/0.367350 mm`；T1 网格重采样图像前景 Dice `0.997655`、双方非零 Pearson `0.999512`、交集 MAE `75.967` 未归一化强度单位。FNIT H100 GPU 求解＋重采样调用 `19.155 s`、PyTorch CUDA 已分配峰值 `0.746 GiB`；FSL CPU 整条命令 `10.02 s`。固定相同 FSL 矩阵后，单独重采样图像 Pearson `0.999999995`、交集 MAE `0.1490`。矩阵与图像仍未逐值一致，主要剩余差异在求解阶段。详见[同输入脱敏报告](../../validation/connectome/ORIGINAL_UKB_FLIRT_STAGE_20260927.md)。UKB 脑图和个体矩阵留在私有工作区。

### T1 GM→群体模板：12DOF/corratio

旧版 12DOF 路径未改动。FSL 6.0.7.4 与 FNIT 在 10 个真实 T1 衍生的 FAST GM 图上比较，官方 UKB 群体 GM 模板为参考；矩阵 `rmsdiff ≤ 0.05 mm` 的功能门槛通过 10/10，median `0.008544 mm`，最大 `0.028984 mm`。完整命令产出的图像 Pearson median `0.999895`、Dice median `0.997457`、MAE median `0.002883`。FSL CPU 完整命令墙钟 median `27.705 s`，FNIT H100 GPU median `23.021 s`；包括启动、读写、求解和重采样，设备不同，不作等硬件加速比解释。[完整聚合记录](../../validation/flirt/report.public.json)及其[公开示例脑图](figures/flirt_fsl_comparison.png)对应此 12DOF 测试，不对应上面的 UKB b0→T1 测试。

![公开数据的 12DOF FLIRT 对照脑图](figures/flirt_fsl_comparison.png)

`qc["reference_validation_matrix_gate_passed"]` 仅描述 12DOF 十例固定验证集。`qc["current_input_compared_with_fsl"]` 与 `qc["validated_fsl_equivalent"]` 始终为 `False`；模型不会在推理时调用 FSL，也不声明逐值等价。6DOF 的一例 UKB 比较是当前版本的实数据证据，不是对未来新输入的通过判定。
