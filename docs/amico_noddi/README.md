# TorchAMICONODDI：NODDI 微结构拟合

`fnit.amico_noddi.TorchAMICONODDI` 默认按 [AMICO 2.0.3](https://github.com/daducci/AMICO/tree/v2.0.3) 的 NODDI fitting 流程运行；`fit_method="classic"` 增加经典连续 Watson NODDI 模型的逐体素非线性拟合。两条路径都由 PyTorch 执行，不在运行时导入或调用 AMICO、NODDI Toolbox 或 DIPY。

代码在包内复现 AMICO 使用的 DIPY OLS 主方向和 legacy Descoteaux-2007 order-12 spherical-harmonic 基，并使用相同的 b-value 取整、500-direction LUT、145-atom dictionary 和三阶段求解：完整 dictionary NNLS、positive elastic-net support selection、support-constrained NNLS debias。response kernel 与影像为 float32；活动集线性代数为 float64，以保持 SPAMS 求解结果。CUDA float32 运算允许 TF32，没有使用 float16 或 bfloat16。500 个 LUT direction 默认按最多 400 个一批求解；各 direction 相互独立，分批只限制显存，不改变模型或精度。可用 `AMICONODDIConfig(lut_batch_size=...)` 调整。

该功能不下载模型权重。包内包含 AMICO 2.0.3 的 500-direction、hash 和 gradient 表；许可见 [`licenses/AMICO-2.0.txt`](../../licenses/AMICO-2.0.txt)。kernel 根据每例 bval/bvec 在运行时生成。

## 输入

| 参数 | 内容 | 要求 |
|---|---|---|
| `data` / `-k` | EDDY 校正后的 4D DWI | 至少一个 b0 和一个 diffusion-weighted volume |
| `mask` / `-m` | 脑 mask | 与 DWI 前三维及 affine 对齐；值等于 1 的体素参与 fitting |
| `bvecs` / `-r` | FSL 3×N 或 N×3 b-vector | 应使用 EDDY 输出的 rotated bvec；列数等于 DWI volume 数 |
| `bvals` / `-b` | FSL b-value | 数量等于 DWI volume 数；AMICO 按 100 s/mm² 取整，经典拟合使用原始值；`b<=100` 为 b0 |
| `output_dir` / `-o` | 输出目录 | 已存在文件需显式设置 `overwrite=True` 或 `--overwrite` |

## Python 调用

```python
from fnit import TorchAMICONODDI

noddi = TorchAMICONODDI(
    device="cuda:0",  # 运行设备：第一张可见 CUDA GPU；无权重文件
    config=None,  # 配置：使用 AMICO 2.0.3 的默认 NODDI 参数
    fit_method="amico",  # 拟合方式："amico" 离散字典；"classic" 连续 Watson 非线性拟合
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
directions = result.directions # AMICO 模式为 OLS tensor 主方向；经典模式为非线性拟合主方向
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

需要连续 Watson 非线性拟合时，在同一命令中加入 `--fit-method classic`。

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
| `directions` | `fit_dir.nii.gz` | `NODDI_dir.nii.gz` | AMICO 模式为 DTI 主方向；经典模式为非线性拟合主方向；最后一维长度为 3 |
| `rmse` | `fit_RMSE.nii.gz` | `NODDI_RMSE.nii.gz` | 归一化信号 RMSE |

所有输出为 float32 NIfTI，mask 外为零。真实数据检查中，shape、dtype、affine、qform、sform、pixdim、空间单位和 intent 均与官方文件一致。

## AMICO 路径逐体素验证

[`report.public.json`](../../validation/amico_noddi/report.public.json) 由 FNIT 0.14.0 生成，在 gpucw1 上以一例真实 UKB 格式、官方 FSL EDDY 校正后的 `104×104×72×105` dMRI 与 AMICO 2.0.3 配对。数据包含 5 个 b0、50 个 b≈1000、50 个 b≈2000；同一 mask 内 242,261 个体素全部参加比较。两边使用相同 DWI、mask、bval 和 rotated bvec。该报告是旧源码的整脑验证；本次增加模式分流后 `core.py` 和 `cli.py` 的 SHA-256 已改变，下面另列当前源码的真实数据复测。

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
| FNIT 0.14.0 历史实测 | NVIDIA H100 PCIe 80 GB | `66.35 s` | `28.05 s` | `23.90 s` | `9.95 GB` |

FNIT 的 `66.35 s` 从模型构造开始，包含 NIfTI 读取、kernel、方向估计、fitting 和五张图保存；外部验证进程为 `73.19 s`，还包含 Python 启动和逐图比较。本次 GPU 有其他常驻进程，AMICO 参考与候选也不在同一时间窗，因此不计算稳定加速比。FNIT 默认按 400 个 LUT direction 分块，当前峰值 allocation 低于 20 GB。

NumPy 的 LAPACK/OpenBLAS 构建会影响退化张量的特征向量符号与伪逆末位。旧整脑结果使用 NumPy 1.26.4 官方 CPython 3.11 manylinux wheel（OpenBLAS64 0.3.23.dev）。仓库 [`environment.yml`](../../environment.yml) 以 PyPI 官方 URL和 SHA-256 `666dbf...31d5` 固定该 wheel；`result.qc` 同时返回 `numpy_version`、`numpy_blas_name`、`numpy_blas_version` 和 `validated_numpy_build`。只有该已验证构建且使用 `fit_method="amico"` 时，`amico_numerically_equivalent` 才可能为 `True`；普通 API 不运行 AMICO oracle，因此 `current_input_compared_with_amico` 保持 `False`。其他 NumPy 构建不能继承旧整脑逐值结论。

![AMICO 2.0.3 与当前 FNIT 的 NDI、ODI、FWF 和逐体素绝对差](figures/amico_noddi_comparison.png)

图使用旧整脑运行的同一真实病例，展示当时的 NDI/ODI/FWF 与逐体素误差。原始病例、官方输出和开发期 AMICO/DIPY oracle 不进入仓库。OLS 主方向、Descoteaux-2007 spherical-harmonic basis 与 500-direction rotation basis 的原验证见 [`compare_no_dipy.py`](../../validation/amico_noddi/compare_no_dipy.py) 与 [`no_dipy_equivalence.public.json`](../../validation/amico_noddi/no_dipy_equivalence.public.json)。

当前源码的 `fit_method="amico"` 在同一真实病例固定的 2,048 个脑内体素上重新运行，使用相同官方 AMICO 文件对照。当前测试环境的 NumPy 1.26.4 BLAS 构建与旧整脑验证不同；NDI/ODI/FWF 对官方的 MAE 分别为 `5.09e-6`、`1.21e-5`、`1.32e-5`，Pearson r 均大于 `0.999998`，个别体素最大差异达 `0.0234`。本轮 H100 含读写时间 `22.64 s`，进程峰值 CUDA allocation `99.4 MB`，LUT 每批 100 个方向。输入、源码及官方文件哈希见 [`amico_current_real_2048.public.json`](../../validation/amico_noddi/amico_current_real_2048.public.json)。

## 经典连续 Watson 拟合

`fit_method="classic"` 使用与原版 NODDI 对应的细胞内 stick、Watson 方向分布、tortuosity 细胞外室和自由水室，固定 `d_par=1.7e-3`、`d_iso=3e-3` mm²/s。它先用现有 AMICO 求解器为每个体素提供起点，再用 PyTorch float64 对细胞内体积分数、ODI、自由水分数和两个主方向角进行最多 30 次阻尼 Gauss–Newton 更新。经典拟合使用原始 b-value 与归一化的原始 b-vector，b0 幅度固定为该体素 b0 均值。目标函数为 Rician 负对数似然；噪声尺度按原版 `EstimateSigma` 从 b0 的总体标准差和 `0.02 × b0 均值` 取较大者，再除以 100。stick 的 Watson 卷积使用 12 阶偶数 Legendre 展开与 64 点积分。默认 AMICO 初始化 LUT 批量从 400 降至 100，以减少显存。

输入与上表相同；`run()` 仍写五张 NIfTI，`result.qc["fit_method"]` 记录实际模式。`ndi` 是组织内的细胞内分数，`fwf` 是整体自由水分数，`odi=2 atan(1/kappa)/π`。经典模式的 `directions` 为连续拟合的主方向；默认 AMICO 模式保留 DTI 主方向。`rmse` 是对 b0 归一化观测信号的均方根残差。两种算法使用同一文件名、网格与数据类型，但不会逐体素相同。

```python
from fnit import TorchAMICONODDI
from fnit.amico_noddi import AMICONODDIConfig

classic_config = AMICONODDIConfig(
    lut_batch_size=100,  # AMICO 初始化的每批 LUT 方向数；减少共享 GPU 显存
)
classic_noddi = TorchAMICONODDI(
    device="cuda:0",  # 计算设备；也可设为 "cpu"
    config=classic_config,  # 固定扩散系数与 AMICO 初始化配置
    fit_method="classic",  # 连续 Watson 五参数 Rician 非线性拟合
)
classic_result = classic_noddi.run(
    data="eddy/data.nii.gz",  # 输入：EDDY 校正后的四维多壳 DWI
    mask="eddy/nodif_brain_mask.nii.gz",  # 输入：同网格三维二值脑掩膜
    bvecs="eddy/data.eddy_rotated_bvecs",  # 输入：与校正图对应的旋转后 b-vector
    bvals="AP.bval",  # 输入：与第四维顺序一致的 b-value
    output_dir="classic_noddi",  # 输出：五张 AMICO 命名的 NIfTI 文件
    naming="amico",  # 文件名使用 fit_NDI、fit_ODI、fit_FWF 等
    overwrite=False,  # 已存在输出时停止
)
print(classic_result.qc)  # 拟合方式、耗时、显存和样本数
```

`AMICONODDIConfig` 参数如下；经典模式中的字典与正则参数只用于 AMICO 起点。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `d_par` | `1.7e-3` | 细胞内和细胞外平行扩散率，mm²/s |
| `d_iso` | `3.0e-3` | 自由水各向同性扩散率，mm²/s |
| `ic_vfs` | 12 个 `0.1–0.99` 值 | AMICO 起点的细胞内分数字典网格 |
| `ic_ods` | 12 个 `0.03–0.99` 值 | AMICO 起点的 ODI 字典网格 |
| `lambda1` | `0.5` | AMICO 起点的稀疏正则第一参数 |
| `lambda2` | `1e-3` | AMICO 起点的稀疏正则第二参数 |
| `b0_threshold` | `100` | 视作 b0 的最大 b-value，s/mm² |
| `b_step` | `100` | AMICO 输入 b-value 的取整步长，s/mm² |
| `kkt_tolerance` | `1e-11` | AMICO 活动集的 KKT 收敛阈值 |
| `cg_tolerance` | `1e-13` | AMICO 共轭梯度回退的收敛阈值 |
| `maximum_active_steps` | `40` | AMICO 活动集最多更新次数 |
| `lut_batch_size` | AMICO 模式 `400`；经典模式 `100` | 每批处理的 LUT 方向数；仅影响显存和速度 |

原版 NODDI Toolbox 的对应调用是：

```matlab
CreateROI('eddy/data.nii.gz', 'eddy/nodif_brain_mask.nii.gz', 'noddi_roi.mat'); % 输入四维 DWI 和三维 mask，输出 ROI
protocol = FSL2Protocol('AP.bval', 'eddy/data.eddy_rotated_bvecs', 100); % 输入梯度；100 为 b0 阈值
noddi_model = MakeModel('WatsonSHStickTortIsoV_B0'); % 经典 Watson NODDI 模型
batch_fitting_single('noddi_roi.mat', protocol, noddi_model, 'FittedParams.mat'); % 逐体素非线性拟合
SaveParamsAsNIfTI('FittedParams.mat', 'noddi_roi.mat', 'eddy/nodif_brain_mask.nii.gz', 'noddi'); % 输出参数图
```

FNIT 保留 AMICO 起点；原版 Toolbox 使用网格搜索和 MATLAB `fmincon`，因此两者的优化路径不同。DIPY 的 [Bingham `k2odi`](https://docs.dipy.org/stable/reference/dipy.reconst.html#dipy.reconst.bingham.k2odi) 使用相同的 Watson 浓度参数到 ODI 换算，但不是原版 NODDI 的逐体素拟合器。

在一例真实、经 EDDY 校正的 `104×104×72×105` DWI 中，从 242,261 个脑内体素先按种子 `20260929` 取 2,048 个，再按种子 `20260930` 固定取 24 个，向本机 MATLAB R2023b 的原版 NODDI Toolbox 1.05 传入同一 DWI 信号、原始 bval 和旋转后 bvec。24 个原版拟合均返回错误码 0。FNIT H100 进程限制为总显存的 10%；两种机器的耗时不可用来计算加速比。

| 参数 | 与原版的 MAE | 最大绝对差 | Pearson r |
|---|---:|---:|---:|
| ICVF/NDI | `0.002592` | `0.017821` | `0.999707` |
| ODI | `0.002765` | `0.022067` | `0.999544` |
| ISOVF/FWF | `0.000470` | `0.005221` | `0.999994` |

轴向等价的主方向角差中位数为 `0.013°`，90 百分位为 `3.28°`。这是 24 个固定真实体素的数值一致性检查；输入与原版 Toolbox 的 SHA-256、运行环境和实测耗时见 [`classic_original_real_24.public.json`](../../validation/amico_noddi/classic_original_real_24.public.json)。

同一病例的全脑 `242,261` 个 mask 体素也完成了当前经典模式的一次运行：H100 PCIe 上设置进程显存分配上限 10%，包含读写耗时 `340.86 s`，其中 AMICO 初始化 `211.43 s`、连续拟合 `123.39 s`；PyTorch 峰值 allocation `2.30 GB`。测试时 GPU 在运行前后均有其他作业，利用率 100%。五张结果图全部为有限值、mask 外为零；归一化 RMSE 的中位数 `0.0362`、99 百分位 `0.1025`，另有 19 个体素超过 1。全脑输入、输出哈希和逐图检查见 [`classic_whole_brain.public.json`](../../validation/amico_noddi/classic_whole_brain.public.json)。

## Reference

- 参考文献：Daducci et al., *Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data*, NeuroImage (2015), [doi:10.1016/j.neuroimage.2014.10.026](https://doi.org/10.1016/j.neuroimage.2014.10.026)。
- 参考文献：Zhang et al., *NODDI: Practical in vivo neurite orientation dispersion and density imaging of the human brain*, NeuroImage (2012), [原文](https://www.sciencedirect.com/science/article/pii/S1053811912003539)。
- 原实现代码库：[AMICO 2.0.3](https://github.com/daducci/AMICO/tree/v2.0.3)。
- 原版经典实现：[NODDI Matlab Toolbox 1.05](https://www.nitrc.org/projects/noddi_toolbox)；其模型与拟合方法见上列 Zhang et al. 论文。
