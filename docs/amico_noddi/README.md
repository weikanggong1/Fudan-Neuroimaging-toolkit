# PyTorch AMICO-NODDI

`fnit.amico_noddi.TorchAMICONODDI` 复现 [AMICO 2.0.3](https://github.com/daducci/AMICO/tree/v2.0.3) 的 NODDI fitting，并把三个求解阶段放到 PyTorch/CUDA。实现固定对应 commit `df540093b60240c38a6ff2ea4ceb1181c4f3e936`，不在运行时导入或调用 AMICO。

代码使用相同的 b-value 取整、DIPY OLS 主方向、order-12 spherical harmonics、500-direction LUT、145-atom dictionary 和三阶段求解：完整 dictionary NNLS、positive elastic-net support selection、support-constrained NNLS debias。response kernel 与影像为 float32；活动集线性代数为 float64，以保持 SPAMS 求解结果。CUDA float32 运算允许 TF32，没有使用 float16 或 bfloat16。

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

noddi = TorchAMICONODDI(device="cuda:0")  # 在 cuda:0 构建 PyTorch fitting；无权重文件
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
directions = result.directions # DIPY OLS tensor 主方向
rmse = result.rmse             # normalized-signal fitting RMSE
print(result.qc)               # 版本、精度、体素数、耗时和峰值显存
```

构造对象只选择设备和参数；调用 `run()` 完成读取、归一化、kernel 生成、主方向估计、三个 fitting 阶段及保存。若只需内存对象，可调用 `result = noddi(data, mask, bvecs, bvals)`。

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
ae = amico.Evaluation(".", ".")
amico.util.fsl2scheme("bvals", "bvecs", bStep=100)
ae.load_data(
    dwi_filename="DWI.nii.gz",
    scheme_filename="bvals.scheme",
    mask_filename="mask.nii.gz",
    b0_thr=100,
)
ae.set_model("NODDI")
ae.generate_kernels(regenerate=True)
ae.load_kernels()
ae.set_solver(lambda1=0.5, lambda2=1e-3)
ae.CONFIG["doComputeRMSE"] = True
ae.fit()
ae.save_results()
```

FNIT 的一次 `TorchAMICONODDI.run(..., naming="amico")` 对应上述全部步骤。默认参数固定为官方 NODDI 默认值；两边共同输入 DWI、mask、bval 与 bvec，输出文件和参数定义如下。

| Python 字段 | `naming="amico"` | `naming="ukb"` | 含义 |
|---|---|---|---|
| `ndi` | `fit_NDI.nii.gz` | `NODDI_ICVF.nii.gz` | intracellular/neurite density index |
| `odi` | `fit_ODI.nii.gz` | `NODDI_OD.nii.gz` | orientation dispersion index |
| `fwf` | `fit_FWF.nii.gz` | `NODDI_ISOVF.nii.gz` | isotropic free-water fraction |
| `directions` | `fit_dir.nii.gz` | `NODDI_dir.nii.gz` | DTI 主方向，最后一维长度为 3 |
| `rmse` | `fit_RMSE.nii.gz` | `NODDI_RMSE.nii.gz` | normalized-signal RMSE |

所有输出为 float32 NIfTI，mask 外为零。真实数据检查中，shape、dtype、affine、qform、sform、pixdim、空间单位和 intent 均与官方文件一致。

## 逐体素验证

在 gpucw1 上用一例真实 UKB 格式、官方 FSL EDDY 校正后的 `104×104×72×105` dMRI 验证。数据包含 5 个 b0、50 个 b≈1000、50 个 b≈2000；同一 mask 内有 242,261 个体素。AMICO 2.0.3 与 FNIT 使用完全相同的 DWI、mask、bval 和 rotated bvec，比较覆盖全部 mask 体素。

| 输出 | Pearson r | MAE | 最大绝对误差 | 误差 > 1e-7 的体素 |
|---|---:|---:|---:|---:|
| NDI | 1.0000000000 | 1.39e-9 | 5.96e-8 | 0 / 242,261 |
| ODI | 1.0000000000 | 1.29e-12 | 5.96e-8 | 0 / 242,261 |
| FWF | 1.0000000000 | 7.09e-13 | 5.96e-8 | 0 / 242,261 |
| normalized RMSE | 1.0000000000 | 0 | 0 | 0 / 242,261 |
| direction | — | component MAE 0 | component max 0 | 0 / 726,783 components |

NDI、ODI 和 FWF 的最大差异为 float32 的一个 ULP 量级；RMSE 与方向图逐元素相同。本页所称“数值等价”采用预先固定的 `max_abs <= 1e-7` 标准。

| 实现 | 设备 | 总 wall time | fitting/solver | 最大 RSS | 峰值显存 |
|---|---|---:|---:|---:|---:|
| AMICO 2.0.3 | Intel Xeon Gold 6430，32 threads | 29.24 s | 16.90 s | 3.30 GB | — |
| FNIT 0.12.1 | NVIDIA H100 PCIe 80 GB | 32.87 s（3 次中位数） | 12.77 s | 1.76 GB | 12.43 GB |

计时来自共享节点上的 fresh-process 运行，包含 Python 启动、kernel、方向估计、fitting 和 NIfTI 读写。官方记录为 1 次；最终代码的 FNIT 三次为 `29.44/32.87/34.19 s`，表中报告中位数。GPU wall time 中位数为官方 CPU 的 1.12 倍。活动集优化将精确版初始的总时间从 108.24 s 降至 32.87 s（3.29×），solver 从 92.42 s 降至 12.77 s（7.24×）；主要改动是只构造 active atom 小矩阵、batched Cholesky、病态子集 CG fallback 和 top-k index extraction。代价是峰值显存从 7.78 GB 增至 12.43 GB。

![AMICO 2.0.3 与 FNIT 0.12.1 的 NDI、ODI、FWF 和逐体素绝对差](figures/amico_noddi_comparison.png)

机器可读结果和验收阈值见 [`validation/amico_noddi/report.public.json`](../../validation/amico_noddi/report.public.json)。原始病例、官方输出和开发期 oracle 不进入仓库。
