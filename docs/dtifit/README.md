# TorchDTIFIT：扩散张量拟合

| 项目 | 内容 |
|---|---|
| 输入 | 已校正DWI、同网格mask、bval/bvec |
| 输出 | FA、MD、MO、S0、特征值/方向及可选tensor |
| 对应原软件 | FSL DTIFIT默认OLS |
| Python / CLI | TorchDTIFIT / fnit-dtifit |
| CPU / GPU | CPU/CUDA；float64拟合、float32写出 |

## 1. 功能简介

`TorchDTIFIT` 对一名被试的扩散信号做七参数对数线性最小二乘拟合，生成常用扩散张量参数图。结果保留输入diffusion网格；多shell输入可先用 `select_shell` 抽取b0和目标shell。

数值路径对应 FSL FDT 2202.6 的默认 OLS。读写使用 nibabel，计算使用 PyTorch，运行时不调用FSL。`weighted=True` 尚未实现并明确报错；近简并特征值对应的特征方向可能不同，不能只按有符号向量逐值判断方向一致性。

## 2. Python 调用

```python
from fnit.dtifit import TorchDTIFIT

tensor_model = TorchDTIFIT(
    device="cuda:0",  # 指定计算设备，可改为cpu
    chunk_size=131072,  # 每批拟合的脑内体素数
)
tensor_result = tensor_model.run(
    data="/data/dwi/selected_dwi.nii.gz",  # b0和目标shell的4D DWI
    mask="/data/dwi/brain_mask.nii.gz",  # 相同diffusion网格的3D掩膜
    bvecs="/data/dwi/selected.bvec",  # 与所选volume对应的旋转后方向
    bvals="/data/dwi/selected.bval",  # 与所选volume对应的b-value
    output_prefix="/data/results/dti",  # 不带扩展名的输出basename
    save_tensor=True,  # 另保存六分量tensor
    overwrite=False,  # 已有文件时停止
)
fractional_anisotropy_image = tensor_result.maps["FA"]  # 内存中的nibabel影像
quality_control = tensor_result.qc  # 设备、精度、时间与体素数
```
### 输入数据格式

- DWI：单被试 NIfTI，shape 为 `[X,Y,Z,N]`，至少含 b0 和扩散方向；信号强度沿用输入单位。读入转 float32，不在此入口做运动、涡流或磁敏感校正。
- mask：NIfTI `[X,Y,Z]`，须与 DWI 的 affine、orientation、体素尺寸和原生 diffusion 空间一致。不自动配准或重排梯度；mask 外结果为零。
- bval：纯文本 N 个有限 b-value，单位 s/mm²，顺序与第四维一致。
- bvec：纯文本 `3×N` 或 `N×3`，每个非 b0方向对应同一 volume；已完成 EDDY 时用旋转后的 bvec。
- 路径接受字符串或 PathLike。示例使用用户自己的真实文件；多被试调度由调用方组织。
OLS最好采用与原协议相同的shell与梯度；多shell直接拟合会改变模型输入。核心读入只检查mask形状，调用者仍须先核对affine与方向。mask值>0的体素参与计算。

`select_shell` 是公开的文件准备接口：按原第四维顺序写所选DWI及同basename `.bval/.bvec`，不重新估计或旋转梯度。

```python
from fnit.dtifit import select_shell

selected_dwi = select_shell(
    data="/data/eddy/data.nii.gz",  # 多shell校正后的DWI
    bvals="/data/dwi/bvals",  # 原有全部b-value
    bvecs="/data/eddy/data.eddy_rotated_bvecs",  # 校正后的全部方向
    output="/data/dwi/selected_dwi.nii.gz",  # 所选DWI及配套梯度输出
    shell=1000,  # 目标b-value，s/mm²
    tolerance=100,  # 双侧容差，s/mm²
    include_b0=True,  # 同时保留b0
    overwrite=False,  # 不覆盖已有文件
)
```

**`TorchDTIFIT.__init__` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `device` | 否 | `str/torch.device/None` | `None` | PyTorch 设备；None 自动选择可用 CUDA，否则 CPU。 |
| `weighted` | 否 | `bool` | `False` | 仅默认 False 的 OLS 已实现；True 报 NotImplementedError。 |
| `chunk_size` | 否 | `int` | `131072` | 每批并行拟合的脑内体素数，必须为正整数。 |

**`TorchDTIFIT.run` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `data` | 是 | `路径` | `—` | 已校正的 4D DWI 文件，第四维为采集 volume。 |
| `mask` | 是 | `路径` | `—` | 与 DWI 相同 shape、affine 和方向的 3D 脑掩膜。 |
| `bvecs` | 是 | `路径` | `—` | FSL 3×N 或 N×3 梯度方向文本；优先用 EDDY 旋转后的方向。 |
| `bvals` | 是 | `路径` | `—` | N 个 b-value，单位 s/mm²，顺序与 DWI 一致。 |
| `output_prefix` | 是 | `路径` | `—` | 不带 .nii/.nii.gz 扩展名的输出 basename。 |
| `save_tensor` | 否 | `bool` | `False` | 是否另写六分量tensor文件；内存结果始终包含tensor。 |
| `overwrite` | 否 | `bool` | `False` | 是否允许覆盖已有结果；默认已有结果时报错。 |

**`select_shell` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `data` | 是 | `路径` | `—` | 已校正的 4D DWI 文件，第四维为采集 volume。 |
| `bvals` | 是 | `路径` | `—` | N 个 b-value，单位 s/mm²，顺序与 DWI 一致。 |
| `bvecs` | 是 | `路径` | `—` | FSL 3×N 或 N×3 梯度方向文本；优先用 EDDY 旋转后的方向。 |
| `output` | 是 | `路径` | `—` | 用户指定输出文件路径。 |
| `shell` | 否 | `int` | `1000` | 选取的目标 b-value，单位 s/mm²。 |
| `tolerance` | 否 | `int` | `100` | 目标 shell 的双侧 b-value 容差，单位 s/mm²。 |
| `include_b0` | 否 | `bool` | `True` | 是否同时保留 b≤100 的 b0 volume。 |
| `overwrite` | 否 | `bool` | `False` | 是否允许覆盖已有结果；默认已有结果时报错。 |

**`TorchDTIFIT.__call__` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `data` | 是 | `路径` | `—` | 已校正的 4D DWI 文件，第四维为采集 volume。 |
| `mask` | 是 | `路径` | `—` | 与 DWI 相同 shape、affine 和方向的 3D 脑掩膜。 |
| `bvecs` | 是 | `路径` | `—` | FSL 3×N 或 N×3 梯度方向文本；优先用 EDDY 旋转后的方向。 |
| `bvals` | 是 | `路径` | `—` | N 个 b-value，单位 s/mm²，顺序与 DWI 一致。 |

**`DTIFITResult.save` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `output_prefix` | 是 | `路径` | `—` | 不带 .nii/.nii.gz 扩展名的输出 basename。 |
| `save_tensor` | 否 | `bool` | `False` | 是否另写六分量tensor文件；内存结果始终包含tensor。 |
| `overwrite` | 否 | `bool` | `False` | 是否允许覆盖已有结果；默认已有结果时报错。 |

### 输出

```text
results/
├── dti_{FA,MD,MO,S0,L1,L2,L3}.nii.gz
├── dti_{V1,V2,V3}.nii.gz
└── dti_tensor.nii.gz                 # save_tensor=True时写
```

| 输出 | 格式、shape、单位与空间 |
|---|---|
| FA、MO | 3D float32；无量纲；保留DWI affine和orientation。 |
| S0 | 3D float32；拟合基线信号，沿用输入强度单位。 |
| L1/L2/L3、MD | 3D float32，mm²/s；特征值按降序排列。 |
| V1/V2/V3 | `[X,Y,Z,3]` float32；按输入bvec坐标约定的单位方向轴，正负号等价。 |
| tensor | `[X,Y,Z,6]` float32；顺序 Dxx、Dxy、Dxz、Dyy、Dyz、Dzz，mm²/s。 |
| result.maps | 上述所有nibabel影像；即使save_tensor=False，内存中仍有tensor。 |
| result.qc | Python字典，包含数值路径和计算记录；run不另写QC JSON。 |

`tensor_model(...)` 只返回内存结果；`run()`调用保存。mask外为零。`output_prefix`含NIfTI扩展名会报错。方向比较用符号不变夹角，近简并方向另外检查特征值间隙。

## 3. 命令行调用

```bash
fnit-dtifit -k /data/dwi/selected_dwi.nii.gz    -m /data/dwi/brain_mask.nii.gz -r /data/dwi/selected.bvec    -b /data/dwi/selected.bval -o /data/results/dti    --device cuda:0 --save_tensor
```

统一入口 `fnit dtifit` 使用相同参数。

| CLI | Python参数 | 含义 |
|---|---|---|
| `-k/--data` | `data` | 4D DWI |
| `-m/--mask` | `mask` | 3D脑掩膜 |
| `-r/--bvecs`、`-b/--bvals` | `bvecs`、`bvals` | 梯度文本 |
| `-o/--out` | `output_prefix` | 无扩展名basename |
| `--device` | 构造函数`device` | 计算设备 |
| `--save_tensor` | `save_tensor=True` | 保存tensor |
| `--overwrite` | `overwrite=True` | 覆盖已有文件 |

CLI不开放 `weighted`、`chunk_size` 或 `select_shell`；需要这些设置时用Python。独立 `--help` 不执行影像拟合。

## 4. 原软件调用

下列命令在隔离的原软件benchmark环境运行，读取同一所选shell和mask。

```bash
dtifit -k /data/dwi/selected_dwi.nii.gz    -m /data/dwi/brain_mask.nii.gz -r /data/dwi/selected.bvec    -b /data/dwi/selected.bval -o /data/reference/dti --save_tensor
```

| FNIT | FSL DTIFIT |
|---|---|
| `data/mask/bvecs/bvals` | `-k/-m/-r/-b` |
| `output_prefix`、`save_tensor` | `-o`、`--save_tensor` |
| 默认OLS | 默认OLS |
| `weighted=True` | 原软件`--wls`，FNIT未实现 |

没有模型权重。FNIT不覆盖FSL的全部DTIFIT扩展选项；文件名相同不能代替逐图对照。

## 5. 最新精度和运行时间

最新独立真实对照仍是[0.14.0报告](../../validation/dtifit/report.public.json)：一例完整脑掩膜，242,261体素，5个b0和50个b≈1000 volume；参考FSL6.0.7.4/FDT2202.6。报告绑定core/CLI/共享dMRI I/O的SHA。core/CLI与原报告SHA相同，共享_dmri.py已有后续更新；本轮未重新运行MRI，实测日期和版本保留原记录。

| 端到端指标 | FNIT | FSL |
|---|---:|---:|
| 三次完整进程时间中位数 | 8.52 s | 6.47 s |
| 三次时间 | 8.92/8.52/8.13 s | 6.58/6.47/6.25 s |
| 最大RSS中位数 | 960 MB | 244 MB |
| FA MAE/最大绝对差 | 4.25e-9 / 1.19e-7 | 比较参照 |
| MD MAE/最大绝对差 | 3.44e-14 / 4.66e-10 | 比较参照 |
| MO r/最大绝对差 | 0.99999918 / 0.259 | 比较参照 |

Intel Xeon Gold6430配H100 GPU，CPU参照使用同一真实输入；计时含进程启动、读取、CUDA初始化、拟合、写出。计算为float64，输出float32；CUDA峰值显存和线程预算未在本报告完整记录。无独立分步骤计时。V1–V3平均符号不变夹角为0.0065–0.0069°，最大方向差集中于近简并体素。本例FNIT总耗时较高，未建立速度收益。

![真实FA：FSL、FNIT及绝对差](figures/dtifit_fsl_comparison.png)

图示为原报告绑定的真实输出；更多逐图指标和绘图来源保留在原JSON，不把单例推广成全队列结论。

## 6. 最近版本和 benchmark

| 日期 | commit/version | 变化 | benchmark |
|---|---|---|---|
| 2026-09-28 | 2ad53c5b | 刷新独立真实数据与资源审计 | DTIFIT report.public.json，0.14.0 |
| 2026-09-27 | f63413cc | 加入TorchDTIFIT OLS及shell选择 | 真实全脑逐图对照；WLS未实现 |

源码历史仅有以上两次相关实质更新，未为满足数量补造版本。

更早的debug、profiling和长表保留在[旧README归档](../../validation/dtifit/readme_archive_20261005.md)。归档已修复相对链接；旧科学报告与原始产物不修改。

<a id="输入和原命令对应"></a>
<a id="python-调用"></a>
<a id="输出合同"></a>
<a id="算法对应"></a>
<a id="当前发布验收状态"></a>
<a id="真实数据对照"></a>
<a id="reference"></a>

## 7. 参考文献、原软件和资源

- 原文档：[FSL DTIFIT](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/dtifit.html)。
- 原代码：[FDT 2202.6 dtifit.cc](https://git.fmrib.ox.ac.uk/fsl/fdt/-/blob/2202.6/dtifit.cc)；FNIT位置：[core.py](../../src/fnit/dtifit/core.py)、[cli.py](../../src/fnit/dtifit/cli.py)。
- Basser、Mattiello与LeBihan，1994，[MR diffusion tensor spectroscopy and imaging](https://doi.org/10.1016/S0006-3495(94)80775-1)。

### 外部资源

本功能不需要预训练权重、图谱或模板，也不自动下载真实输入数据。用户输入与参考软件的许可由各自来源决定。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| 无额外模型资源 | — | — | 不适用 | 不适用 | 不适用 |
