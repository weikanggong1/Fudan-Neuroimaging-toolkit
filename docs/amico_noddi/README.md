# TorchAMICONODDI：NODDI 微结构拟合

`fnit.amico_noddi.TorchAMICONODDI` 按 [AMICO 2.0.3](https://github.com/daducci/AMICO/tree/v2.0.3) 的 NODDI fitting 流程实现三个 PyTorch/CUDA 求解阶段。实现固定对应 commit `df540093b60240c38a6ff2ea4ceb1181c4f3e936`，不在运行时导入或调用 AMICO，也不依赖已安装的 DIPY。

代码在包内复现 AMICO 使用的 DIPY OLS 主方向和 legacy Descoteaux-2007 order-12 spherical-harmonic 基，并使用相同的 b-value 取整、500-direction LUT、145-atom dictionary 和三阶段求解：完整 dictionary NNLS、positive elastic-net support selection、support-constrained NNLS debias。response kernel 与影像为 float32；活动集线性代数为 float64，以保持 SPAMS 求解结果。CUDA float32 运算允许 TF32，没有使用 float16 或 bfloat16。500 个 LUT direction 默认按最多 400 个一批求解；各 direction 相互独立，分批只限制显存，不改变模型或精度。可用 `AMICONODDIConfig(lut_batch_size=...)` 调整。

该功能不下载模型权重。包内包含 AMICO 2.0.3 的 500-direction、hash 和 gradient 表；许可见 [`licenses/AMICO-2.0.txt`](../../licenses/AMICO-2.0.txt)。kernel 根据每例 bval/bvec 在运行时生成。

## 输入

| 参数 | 内容 | 要求 |
|---|---|---|
| `data` / `-k` | EDDY 校正后的 4D DWI | 至少一个 b0 和一个 diffusion-weighted volume |
| `mask` / `-m` | 脑 mask | 与 DWI 前三维及 affine 对齐；值等于 1 的体素参与 fitting |
| `bvecs` / `-r` | FSL 3×N 或 N×3 b-vector | 应使用 EDDY 输出的 rotated bvec；列数等于 DWI volume 数 |
| `bvals` / `-b` | FSL b-value | 数量等于 DWI volume 数；默认按 100 s/mm² 取整，`b<=100` 为 b0 |
| `output_dir` / `-o` | 输出目录 | 已存在文件需显式设置 `overwrite=True` 或 `--overwrite` |

## Python 调用

```python
from fnit import TorchAMICONODDI

noddi = TorchAMICONODDI(
    device="cuda:0",  # 运行设备：第一张可见 CUDA GPU；无权重文件
    config=None,  # 配置：使用 AMICO 2.0.3 的默认 NODDI 参数
)
result = noddi.run(
    data="eddy/data.nii.gz",               # 单被试、EDDY 校正后的 4D DWI
    mask="eddy/nodif_brain_mask.nii.gz",  # 同一扩散网格上的 3D mask
    bvecs="eddy/data.eddy_rotated_bvecs", # EDDY 旋转后的梯度方向
    bvals="AP.bval",                       # 与 DWI 顺序一致的 b-value
    output_dir="noddi",                   # 写入五个 NIfTI 文件的目录
    naming="amico",                       # 使用官方 AMICO 的 fit_* 文件名
    overwrite=False,                       # 默认不覆盖已有结果
)

ndi = result.ndi               # nibabel.Nifti1Image，neurite density
odi = result.odi               # orientation dispersion
fwf = result.fwf               # isotropic/free-water fraction
directions = result.directions # 包内数值复现的 AMICO/DIPY OLS tensor 主方向
rmse = result.rmse             # normalized-signal fitting RMSE
print(result.qc)               # 版本、精度、体素数、耗时和峰值显存
```

构造对象只选择设备和参数；调用 `run()` 完成读取、归一化、kernel 生成、主方向估计、三个 fitting 阶段及保存。若只需内存对象，可调用：

```python
result = noddi(
    data="eddy/data.nii.gz",  # 输入：EDDY 校正后的 4D DWI
    mask="eddy/nodif_brain_mask.nii.gz",  # 输入：同网格 3D 脑掩膜
    bvecs="eddy/data.eddy_rotated_bvecs",  # 输入：旋转后的梯度方向
    bvals="AP.bval",  # 输入：与 DWI 顺序一致的 b-value
)
```

## 命令行调用

```bash
fnit amico-noddi \
  -k eddy/data.nii.gz \
  -m eddy/nodif_brain_mask.nii.gz \
  -r eddy/data.eddy_rotated_bvecs \
  -b AP.bval \
  -o noddi \
  --naming amico \
  --device cuda:0
```

这条命令对一个病例完成 NODDI fitting。`--naming amico` 写官方 `fit_*` 名称；`--naming ukb` 写 UK Biobank pipeline 名称。默认遇到已有文件即停止，确认需要覆盖时加入 `--overwrite`。包只提供单被试接口；多个病例由调用方在包外分配进程与 GPU。

## 与官方 AMICO 指令的对应

官方 AMICO 2.0.3 没有一个等价的单行 shell 命令。其 Python 序列为：

```python
import amico

amico.core.setup()
ae = amico.Evaluation(
    study_path=".",  # 输入根目录：包含当前被试和共享 kernel 目录
    subject=".",  # 被试目录：相对于 study_path
    output_path=None,  # 输出：None 使用官方默认 AMICO/NODDI 目录
)
amico.util.fsl2scheme(
    bvalsFilename="bvals",  # 输入：FSL b-value 文件
    bvecsFilename="bvecs",  # 输入：FSL 3×N b-vector 文件
    schemeFilename="bvals.scheme",  # 输出：AMICO acquisition scheme
    bStep=100,  # 将 b-value 归到 100 s/mm² 步长
)
ae.load_data(
    dwi_filename="DWI.nii.gz",  # 输入：4D DWI
    scheme_filename="bvals.scheme",  # 输入：上一步生成的 scheme
    mask_filename="mask.nii.gz",  # 输入：同网格 3D mask
    b0_thr=100,  # b0 判定阈值，单位 s/mm²
)
ae.set_model(
    model_name="NODDI",  # 模型：官方 NODDI compartment model
)
ae.generate_kernels(
    regenerate=True,  # 强制重新生成当前 acquisition 的 kernels
)
ae.load_kernels()
ae.set_solver(
    lambda1=0.5,  # NODDI solver 的第一项正则参数
    lambda2=1e-3,  # NODDI solver 的第二项正则参数
)
ae.CONFIG["doComputeRMSE"] = True
ae.fit()
ae.save_results()
```

FNIT 的一次 `TorchAMICONODDI.run(..., naming="amico")` 对应上述全部步骤。默认参数固定为官方 NODDI 默认值；两边共同输入 DWI、mask、bval 与 bvec，输出文件和参数定义如下。

| Python 字段 | `naming="amico"` | `naming="ukb"` | 含义 |
|---|---|---|---|
| `ndi` | `fit_NDI.nii.gz` | `NODDI_ICVF.nii.gz` | 细胞内/神经突密度指数 |
| `odi` | `fit_ODI.nii.gz` | `NODDI_OD.nii.gz` | 方向离散指数 |
| `fwf` | `fit_FWF.nii.gz` | `NODDI_ISOVF.nii.gz` | 各向同性自由水分数 |
| `directions` | `fit_dir.nii.gz` | `NODDI_dir.nii.gz` | DTI 主方向，最后一维长度为 3 |
| `rmse` | `fit_RMSE.nii.gz` | `NODDI_RMSE.nii.gz` | 归一化信号 RMSE |

所有输出为 float32 NIfTI，mask 外为零。真实数据检查中，shape、dtype、affine、qform、sform、pixdim、空间单位和 intent 均与官方文件一致。

## 当前源码逐体素验证

[`report.public.json`](../../validation/amico_noddi/report.public.json) 由 FNIT 0.14.0 生成，在 gpucw1 上以一例真实 UKB 格式、官方 FSL EDDY 校正后的 `104×104×72×105` dMRI 与 AMICO 2.0.3 配对。数据包含 5 个 b0、50 个 b≈1000、50 个 b≈2000；同一 mask 内 242,261 个体素全部参加比较。两边使用相同 DWI、mask、bval 和 rotated bvec。报告记录的 `__init__.py`、`cli.py`、`core.py`、`kernels.py`、`solver.py` 和共享 DWI I/O 与当前 0.16.0 数值文件的 SHA-256 逐个相同，因此这些精度和计时结果仍覆盖当前数值路径。其中 `kernels.py` SHA-256 为 `46f1044aa5f1c2dbffd12dd33707c0e192ef1de01dc23157ae03a0977b3d0b59`。

| 输出 | MAE | 最大绝对误差 | 误差 > `1e-7` |
|---|---:|---:|---:|
| NDI | `1.394e-9` | `5.960e-8` | `0 / 242,261` |
| ODI | `1.292e-12` | `5.960e-8` | `0 / 242,261` |
| FWF | `7.090e-13` | `5.960e-8` | `0 / 242,261` |
| normalized RMSE | `0` | `0` | `0 / 242,261` |
| direction | component MAE `0` | component max `0` | `0 / 726,783` |

本页把标量图 `max_abs <= 1e-7` 且方向分量逐元素相同定义为数值等价。NDI、ODI 和 FWF 的最大差异为 float32 一个 ULP；RMSE 和方向图逐元素相同。shape、dtype、affine、qform、sform、pixdim、空间单位、intent 与官方文件一致。

| 实现 | 设备 | 完整 wall time | 内部总时间 | solver | 峰值显存 |
|---|---|---:|---:|---:|---:|
| AMICO 2.0.3 | Intel Xeon Gold 6430，32 threads | `29.24 s` | — | `16.90 s` | — |
| FNIT 0.14.0 实测（数值文件与当前 0.16.0 相同） | NVIDIA H100 PCIe 80 GB | `66.35 s` | `28.05 s` | `23.90 s` | `9.95 GB` |

FNIT 的 `66.35 s` 从模型构造开始，包含 NIfTI 读取、kernel、方向估计、fitting 和五张图保存；外部验证进程为 `73.19 s`，还包含 Python 启动和逐图比较。本次 GPU 有其他常驻进程，AMICO 参考与候选也不在同一时间窗，因此不计算稳定加速比。FNIT 默认按 400 个 LUT direction 分块，当前峰值 allocation 低于 20 GB。

NumPy 的 LAPACK/OpenBLAS 构建会影响退化张量的特征向量符号与伪逆末位。上述逐体素结果使用 NumPy 1.26.4 官方 CPython 3.11 manylinux wheel（OpenBLAS64 0.3.23.dev）。仓库 [`environment.yml`](../../environment.yml) 以 PyPI 官方 URL和 SHA-256 `666dbf...31d5` 固定该 wheel；`result.qc` 同时返回 `numpy_version`、`numpy_blas_name`、`numpy_blas_version` 和 `validated_numpy_build`。只有该已验证构建才把 `amico_numerically_equivalent` 置为 `True`，表示运行环境具备已验证的数值路径；普通 API 不运行 AMICO oracle，因此 `current_input_compared_with_amico` 保持 `False`。本页逐体素结论来自独立验证驱动。其他 NumPy 构建仍可运行，但不能继承本页结论。

![AMICO 2.0.3 与当前 FNIT 的 NDI、ODI、FWF 和逐体素绝对差](figures/amico_noddi_comparison.png)

图使用同一真实病例。最终当前输出与图示运行的 NDI/ODI/FWF/RMSE/方向数组分别逐元素相同或仅有上述一个 ULP 差异，因此图示仍对应当前数值结果。原始病例、官方输出和开发期 AMICO/DIPY oracle 不进入仓库。当前源码的 OLS 主方向、Descoteaux-2007 spherical-harmonic basis 与 500-direction rotation basis 对 DIPY 1.12.1 的误差均为 `0`；验证脚本和报告见 [`compare_no_dipy.py`](../../validation/amico_noddi/compare_no_dipy.py) 与 [`no_dipy_equivalence.public.json`](../../validation/amico_noddi/no_dipy_equivalence.public.json)。

## Reference

- 参考文献：Daducci et al., *Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data*, NeuroImage (2015), [doi:10.1016/j.neuroimage.2014.10.026](https://doi.org/10.1016/j.neuroimage.2014.10.026)。
- 参考文献：Zhang et al., *NODDI: Practical in vivo neurite orientation dispersion and density imaging of the human brain*, NeuroImage (2012), [原文](https://www.sciencedirect.com/science/article/pii/S1053811912003539)。
- 原实现代码库：[AMICO 2.0.3](https://github.com/daducci/AMICO/tree/v2.0.3)。
