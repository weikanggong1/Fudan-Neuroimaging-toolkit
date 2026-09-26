# PyTorch AMICO-NODDI：GPU dictionary fitting

`fnit.amico_noddi.TorchAMICONODDI` 在 PyTorch/CUDA 中实现 AMICO 风格的 NODDI dictionary fit，输出 neurite density、orientation dispersion 和 free-water fraction。运行时不调用 AMICO，也不需要下载权重或预生成 kernel。

实现参照 [AMICO 2.0.3](https://github.com/daducci/AMICO/tree/v2.0.3) commit `df540093b60240c38a6ff2ea4ceb1181c4f3e936` 的 NODDI 参数、三阶段求解顺序和输出定义。response kernel 在运行时由 NODDI 方程数值生成：12 个 IC volume fraction、12 个 IC orientation dispersion 和一个 isotropic atom，共 145 个 atom。官方 AMICO 使用 order-12 spherical-harmonic kernel、500-direction LUT、compiled NNLS/LASSO；FNIT 使用连续方向的 Watson 数值积分和 batched non-negative FISTA。因此两者的 map 定义和文件合同相同，但不是逐 voxel 数值等价。

CUDA 路径使用 float32 并允许 TF32；不自动使用 float16 或 bfloat16。AMICO 2.0 软件许可仅允许研究和教育用途，完整条款见 [`licenses/AMICO-2.0.txt`](../../licenses/AMICO-2.0.txt)。

## 输入

NODDI 应使用 EDDY 校正后的多 shell DWI 和旋转后的 b-vector：

- 4D DWI，包含至少一个 b0 和一个非零 shell；真实验证使用 b0、b≈1000、b≈2000。
- 与 DWI 前三维相同的脑 mask。
- FSL 3×N b-vector；推荐使用 `<eddy_out>.eddy_rotated_bvecs`。
- FSL 1×N b-value。

## Python 调用

```python
from fnit import TorchAMICONODDI

model = TorchAMICONODDI(device="cuda:0")
result = model.run(
    data="eddy/data.nii.gz",                       # EDDY 校正 4D DWI
    mask="eddy/nodif_brain_mask.nii.gz",          # 同一网格脑 mask
    bvecs="eddy/data.eddy_rotated_bvecs",          # 旋转后梯度方向
    bvals="AP.bval",
    output_dir="noddi",
    naming="ukb",                                 # UKB 文件名
)

ndi = result.ndi
odi = result.odi
fwf = result.fwf
print(result.qc)
```

`result` 同时包含方向图和 normalized-signal RMSE。`result.qc` 记录 AMICO 参考版本、dictionary 大小、求解器、迭代次数、设备、TF32、耗时和峰值 CUDA memory。

## 命令行调用

```bash
fnit amico-noddi \
  -k eddy/data.nii.gz \
  -m eddy/nodif_brain_mask.nii.gz \
  -r eddy/data.eddy_rotated_bvecs \
  -b AP.bval \
  -o noddi \
  --naming ukb \
  --device cuda:0
```

- `-k/-m/-r/-b` 分别是单被试 DWI、mask、b-vector 和 b-value。
- `-o` 是输出目录，不是 basename。
- `--naming ukb` 写 UKB pipeline 使用的 `NODDI_ICVF/OD/ISOVF`；`--naming amico` 写 AMICO 2.x 的 `fit_NDI/ODI/FWF`。
- `--device` 选择 CUDA 或 CPU；H100 是正式验证路径。

## 输出

| Python 字段 | `naming="ukb"` | `naming="amico"` | 含义 |
|---|---|---|---|
| `ndi` | `NODDI_ICVF.nii.gz` | `fit_NDI.nii.gz` | intracellular/neurite density index |
| `odi` | `NODDI_OD.nii.gz` | `fit_ODI.nii.gz` | orientation dispersion index |
| `fwf` | `NODDI_ISOVF.nii.gz` | `fit_FWF.nii.gz` | isotropic free-water fraction |
| `directions` | `NODDI_dir.nii.gz` | `fit_dir.nii.gz` | DTI 主方向 |
| `rmse` | `NODDI_RMSE.nii.gz` | `fit_RMSE.nii.gz` | normalized signal fit RMSE |

所有 parameter map 在 mask 外为零，NDI、ODI 和 FWF 位于 `[0,1]`。输入 affine、shape 和 voxel size 保留到输出。

## 求解步骤

1. 用全部 shell 的 log-linear DTI 估计每个 voxel 的主方向。
2. 在该连续方向上计算 144 个 coupled IC+EC response，并加入一个 isotropic atom。
3. 对完整 dictionary 做非负 fit，先估计 isotropic contribution。
4. 从非零 b-value signal 中减去 isotropic contribution，对归一化 WM dictionary 做 non-negative elastic-net support selection；参数与 AMICO 默认值一致：`lambda1=0.5`、`lambda2=1e-3`。
5. 在选定 support 加 isotropic atom上重新做非负 debias fit。
6. 按 AMICO 定义从 coefficient 加权求 NDI、ODI 和 FWF。

默认迭代数为 `100/200/100`。真实数据测试显示早期的 `20/50/25` 配置会过早停止，尤其低估 ODI 和 FWF；当前默认值使用收敛后的结果。

## 与 AMICO 2.0.3 的真实数据对照

输入为官方 FSL EDDY 校正的一例真实 UKB 格式 `104×104×72×105` dMRI，同一 mask、bval 和 rotated bvec，共 242,261 个脑内 voxel。官方 AMICO 使用 32 CPU threads；FNIT 使用一张 H100 PCIe 80 GB。节点处于共享状态。

| map | Pearson r | MAE | RMSE | AMICO mean | FNIT mean |
|---|---:|---:|---:|---:|---:|
| NDI | 0.83073 | 0.06286 | 0.11944 | 0.41628 | 0.38357 |
| ODI | 0.92383 | 0.04862 | 0.10589 | 0.43326 | 0.39903 |
| FWF | 0.94662 | 0.07356 | 0.13216 | 0.24315 | 0.17479 |
| normalized RMSE | 0.36556 | 0.00245 | 0.08529 | 0.04016 | 0.04131 |

NDI 的差异最大。主方向在高 FA voxel 中大多一致，差异主要来自 official SH/LUT response 和 compiled active-set solver 与连续 Watson integration/FISTA 的不同。不能把这些结果描述为 AMICO 的仅浮点误差复现。

| 实现 | 单进程总 wall time | 主要计算时间 | 最大 RSS | 峰值 CUDA memory |
|---|---:|---:|---:|---:|
| AMICO 2.0.3，32 CPU threads | 29.24 s | generate 3.49 + load 0.53 + fit 16.90 + save 0.45 s | 3.30 GB | 不适用 |
| FNIT TorchAMICONODDI，H100 | 22.98 s | 14.81 s | 1.26 GB | 0.93 GB |

包含启动和 I/O 的总耗时加速为 `1.27×`。这是单病例单次实测；NODDI 的 kernel generation 和 NIfTI I/O 占比高，GPU 收益小于 EDDY。

![AMICO 2.0.3 与 FNIT 的 NDI、ODI、FWF 及绝对差](figures/amico_noddi_comparison.png)

机器可读数值见 [`validation/amico_noddi/report.public.json`](../../validation/amico_noddi/report.public.json)。图由 [`tools/plot_dmri_comparisons.py`](../../tools/plot_dmri_comparisons.py) 生成；原始病例数据不进入仓库。
