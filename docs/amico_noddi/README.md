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

所有输出为 float32 NIfTI，mask 外为零。当前真实数据检查确认五张图的 shape、dtype 和 affine；逐图数值结果见下文。

## 优化前 AMICO 路径逐体素验证

此前实现使用一例真实 UKB 格式、官方 FSL EDDY 校正后的 `104×104×72×105` DWI 和完整的 `242,261` 体素脑掩膜完成默认 AMICO 模式验证，将五张输出图与同输入的官方 AMICO 2.0.3 逐体素比较。数据包含 5 个 b0、50 个 b≈1000 和 50 个 b≈2000；两边使用相同 DWI、mask、bval 与旋转后的 bvec。下面是优化前、由报告中源码 SHA-256 标识的原软件对照；2026-10-02 更新与旧 FNIT 的真实差分和计时见[本次验收报告](../../validation/dmri_pipeline/lossless_20261002.md)。

| 输出 | 全脑 MAE | 最大绝对误差 | Pearson r |
|---|---:|---:|---:|
| NDI | `2.13e-6` | `0.04073` | `0.9999997` |
| ODI | `8.08e-6` | `0.05376` | `0.9999989` |
| FWF | `3.15e-6` | `0.06689` | `0.9999998` |
| normalized RMSE | `7.78e-9` | `0.000354` | `≈1` |

主方向的轴向等价角差中位数为 `0°`，99 百分位为 `1.21e-6°`。五张图均为有限值、float32、与输入同网格，mask 外为零。H100 PCIe 上设置进程显存分配上限 20%，LUT 批量 400；该次含读写耗时 `81.79 s`，其中 solver `71.20 s`，PyTorch 峰值 allocation `9.95 GB`。测试前后 GPU 有其他作业、利用率为 99–100%。此前同例官方 AMICO CPU 运行耗时 `29.24 s`，与该次 GPU 运行不在同一时间窗，不能计算可靠加速比。逐图误差、输入/输出和源码 SHA-256、GPU 负载及计时见[优化前全脑报告](../../validation/amico_noddi/report.public.json)。

![AMICO 2.0.3 与当前 FNIT 的真实全脑 NDI、ODI、FWF 和逐体素绝对误差](figures/amico_noddi_comparison.png)

图为该病例第 36 层轴位切片；误差图的显示上限为 `0.001`，全脑最大误差以表中数值为准。原始病例、官方输出和开发期 AMICO/DIPY oracle 不进入仓库。OLS 主方向、Descoteaux-2007 spherical-harmonic basis 与 500-direction rotation basis 的独立验证见 [`compare_no_dipy.py`](../../validation/amico_noddi/compare_no_dipy.py) 和 [`no_dipy_equivalence.public.json`](../../validation/amico_noddi/no_dipy_equivalence.public.json)。

## 经典连续 Watson 拟合

`fit_method="classic"` 使用与原版 NODDI 对应的细胞内 stick、Watson 方向分布、tortuosity 细胞外室和自由水室，固定 `d_par=1.7e-3`、`d_iso=3e-3` mm²/s。它先用现有 AMICO 求解器为每个体素提供起点，再用 PyTorch float64 对细胞内体积分数、ODI、自由水分数和两个主方向角进行最多 30 次阻尼 Gauss–Newton 更新。经典拟合使用原始 b-value 与归一化的原始 b-vector，b0 幅度固定为该体素 b0 均值。目标函数为 Rician 负对数似然；噪声尺度按原版 `EstimateSigma` 从 b0 的总体标准差和 `0.02 × b0 均值` 取较大者，再除以 100。stick 的 Watson 卷积使用 12 阶偶数 Legendre 展开与 64 点积分。AMICO 初始化默认每批处理 400 个 LUT 方向；显存不足时可显式设为 100。

输入与上表相同；`run()` 仍写五张 NIfTI，`result.qc["fit_method"]` 记录实际模式。`ndi` 是组织内的细胞内分数，`fwf` 是整体自由水分数，`odi=2 atan(1/kappa)/π`。经典模式的 `directions` 为连续拟合的主方向；默认 AMICO 模式保留 DTI 主方向。`rmse` 是对 b0 归一化观测信号的均方根残差。两种算法使用同一文件名、网格与数据类型，但不会逐体素相同。

```python
from fnit import TorchAMICONODDI
from fnit.amico_noddi import AMICONODDIConfig

classic_config = AMICONODDIConfig(
    lut_batch_size=400,  # AMICO 初始化的每批 LUT 方向数；默认 400，显存不足时可设为 100
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
| `lut_batch_size` | `400` | 每批处理的 LUT 方向数；显存不足时可改为 `100` |

原版 NODDI Toolbox 的对应调用是：

```matlab
CreateROI('eddy/data.nii.gz', 'eddy/nodif_brain_mask.nii.gz', 'noddi_roi.mat'); % 输入四维 DWI 和三维 mask，输出 ROI
protocol = FSL2Protocol('AP.bval', 'eddy/data.eddy_rotated_bvecs', 100); % 输入梯度；100 为 b0 阈值
noddi_model = MakeModel('WatsonSHStickTortIsoV_B0'); % 经典 Watson NODDI 模型
batch_fitting_single('noddi_roi.mat', protocol, noddi_model, 'FittedParams.mat'); % 逐体素非线性拟合
SaveParamsAsNIfTI('FittedParams.mat', 'noddi_roi.mat', 'eddy/nodif_brain_mask.nii.gz', 'noddi'); % 输出参数图
```

FNIT 保留 AMICO 起点；原版 Toolbox 使用网格搜索和 MATLAB `fmincon`，因此两者的优化路径不同。DIPY 的 [Bingham `k2odi`](https://docs.dipy.org/stable/reference/dipy.reconst.html#dipy.reconst.bingham.k2odi) 使用相同的 Watson 浓度参数到 ODI 换算，但不是原版 NODDI 的逐体素拟合器。

在一例真实、经 EDDY 校正的 `104×104×72×105` DWI 中，从 242,261 个脑内体素先按种子 `20260929` 取 2,048 个，再按种子 `20260930` 固定取 24 个，向本机 MATLAB R2023b 的原版 NODDI Toolbox 1.05 传入同一 DWI 信号、原始 bval 和旋转后 bvec。24 个原版拟合均返回错误码 0。FNIT 使用当前默认 LUT 批量 400，H100 进程限制为总显存的 10%；两种机器的耗时不可用来计算加速比。

| 参数 | 与原版的 MAE | 最大绝对差 | Pearson r |
|---|---:|---:|---:|
| ICVF/NDI | `0.002592` | `0.017821` | `0.999707` |
| ODI | `0.002765` | `0.022067` | `0.999544` |
| ISOVF/FWF | `0.000470` | `0.005221` | `0.999994` |

轴向等价的主方向角差中位数为 `0.013°`，90 百分位为 `3.28°`。这是 24 个固定真实体素的数值一致性检查；输入与原版 Toolbox 的 SHA-256、运行环境和实测耗时见 [`classic_original_real_24.public.json`](../../validation/amico_noddi/classic_original_real_24.public.json)。

同一病例的全脑 `242,261` 个 mask 体素也完成了优化前经典模式的一次运行：H100 PCIe 上设置进程显存分配上限 20%，包含读写耗时 `209.55 s`，其中 AMICO 初始化 `84.63 s`、连续拟合 `118.09 s`；PyTorch 峰值 allocation `9.95 GB`。此前 LUT 批量 100 的单次运行耗时 `340.86 s`、峰值 allocation `2.30 GB`；改为 400 后，五张全脑图与旧图最大逐值差异为 `5.4e-7`。这两次测试均为共享 GPU，该次测试前后利用率为 99–100%；计时不能解释为隔离条件下的加速比。五张结果图全部为有限值、mask 外为零；归一化 RMSE 的中位数 `0.0362`、99 百分位 `0.1025`，另有 19 个体素超过 1。优化前的全脑输入、输出哈希和逐图检查见 [`classic_whole_brain.public.json`](../../validation/amico_noddi/classic_whole_brain.public.json)。

## 计算复用更新与差分验收

2026-10-02 更新在第一阶段 NNLS 与第三阶段 debias 之间复用同一 LUT 批次的完整 dictionary Gram `AᵀA`；`yA` 和原 `lambda1` 减法在每次求解中按原 float64 形状与顺序重新计算，各阶段重新建立活动集。中间的组织 dictionary 和信号不同，仍独立计算。Cholesky 是否失败的布尔结果也只读取一次，保留原 CG 回退。经典模式将接受更新数保存在设备端 int64 标量，全部体素完成后读取一次；计数只用于 `accepted_updates` QC，不参与候选接受或停止。LUT 批量仍为 400，经典体素批量、迭代数、KKT/CG 阈值、精度和输出文件都使用原设置。

最初同时保留 Gram 和 linear 的候选在同一真实全脑输入上产生完全相同的五张图，但未获得耗时收益，且增加峰值显存；因此当前默认只保留 Gram，移除了跨组织求解存活的大 linear 张量。内部差分测试检查完整 Gram 形状、各求解独立生成 linear、两次 NNLS 与独立组织求解、相同输出/支撑集和精确 QC 计数。

最终版本在同一份 `104×104×72×105` 真实 EDDY 校正 DWI、`242,261` 个 mask 体素上，与旧 FNIT 进行 H100 的 ABBA 串行比较，进程显存上限为 20 GB。计时从 CUDA 初始化完成后的 `run()` 调用开始，到五张图写出并同步完成，包含影像读写。每条路径各运行两次，原始 wall time 如下：

| 拟合方式 | 旧 FNIT 两次耗时（s） | 最终版本两次耗时（s） | 中位数：旧→新（s） |
|---|---|---|---|
| AMICO | `27.740670`、`33.799191` | `22.144909`、`32.757829` | `30.76993`→`27.45137` |
| classic | `74.303144`、`61.999586` | `69.524937`、`90.094424` | `68.15137`→`79.80968` |

两种拟合的五张输出图均与旧 FNIT 完全相同：未舍入的解码数组 SHA-256、保存文件、header、affine 和结果 QC 全部一致；性能字段另行比较。classic 的 `accepted_updates` 均为 `3,710,053`。PyTorch allocated 峰值由 `9.95255 GB` 小幅增加到 `10.02013 GB`。AMICO 中位耗时较低，classic 整体未提速；共享 GPU 上两次运行的波动较大，本轮未建立稳定的耗时收益。保留 Gram 复用和设备端 QC 计数的依据是本例没有数值回归、显存增量较小；精度结论限于该真实输入，不等同于与原软件逐值等价。

另用固定、未舍入的 AMICO 初始化，在同一真实全脑上单独比较 classic 的 QC 读取位置，按主机→设备→设备→主机顺序运行。该计时只含连续拟合，不含影像 I/O 或 AMICO 初始化；主机逐轮读取为 `38.649866`、`58.084653 s`，设备端累计为 `38.488963`、`38.694473 s`，输出与 QC 均完全相同。这验证了消除逐轮主机读取没有数值回归；末次主机计时明显较高，原因未隔离；共享 GPU 上不能据此声称大的加速。源码、输入绑定、逐图检查和计时记录见[本次验收报告](../../validation/dmri_pipeline/lossless_20261002.md)。

开发者可在仓库中使用内部 [A/B hook](../../validation/amico_noddi/lossless_hooks.py)，对同一份真实 EDDY 校正输入切换矩阵复用和 QC 读取；这些开关不属于用户配置或命令行接口：

```python
from fnit import TorchAMICONODDI
from validation.amico_noddi.lossless_hooks import noddi_work_reuse, qc_differences

noddi_inputs = {
    "data": "eddy/data.nii.gz",  # 同一真实 EDDY 校正四维 DWI
    "mask": "eddy/nodif_brain_mask.nii.gz",  # 同一三维二值脑 mask
    "bvecs": "eddy/data.eddy_rotated_bvecs",  # 固定的校正后梯度方向
    "bvals": "AP.bval",  # 固定且与 DWI 第四维对应的 b-value
}
noddi_model = TorchAMICONODDI(device="cuda:0", fit_method="classic")
with noddi_work_reuse(reuse_gram=False, defer_classic_qc=False):
    baseline_result = noddi_model(**noddi_inputs)  # 两次 NNLS 都重算 Gram、每次迭代读取 QC
with noddi_work_reuse(reuse_gram=True, defer_classic_qc=True):
    candidate_result = noddi_model(**noddi_inputs)  # 仅复用完整 Gram、设备端精确计数
print(qc_differences(baseline_result.qc, candidate_result.qc))  # 模型和结果 QC 应相同
```

上述上下文临时替换本进程中的内部函数，应串行使用。五张图要比较全部输出值、dtype、shape、affine、mask 外零值及方向定义；`support_size_min/median/max`、`accepted_updates`、`rician_sigma_median`、LUT 数量和批量大小要逐项一致。计时、峰值显存另做性能比较，记录共享 GPU 负载和冷/热缓存。完整旧提交与新源码的对照仍是最终验收依据。

| 更新日期 | 内容与验收记录 |
|---|---|
| 2026-10-02（临时显存） | 公开十人 `case02` 的 20 GB 失败定位到填充行 Cholesky 分配。改为非空行求解、约 128 MiB 临时分块、按原方向广播 CG Gram；16 项小矩阵内存/数值差分测试通过，结合既有 AMICO/lossless 共 41 项通过。真实组件检查已完成：`case01` 五图解码值与 shape/affine 与旧版逐值相同；原版失败的 `case02` 在 20 GB 上限内完成五张有限值图。组件 allocated 峰值分别为 3,674,249,216 和 5,391,976,448 bytes；两个分支共 20 个 FNIT 完整流程将统一重跑，整链对照尚待完成。 |
| 2026-10-02（计算复用） | 完整 Gram+linear 候选未获耗时收益且增加显存，最终只复用 Gram 并保留 Cholesky 检查复用与 classic 设备整数 QC 计数。同一真实全脑五图、文件、header、affine 和结果 QC 完全相同；AMICO 中位 `30.76993`→`27.45137 s`，classic `68.15137`→`79.80968 s`，共享 GPU 下未建立稳定提速。见[本次报告](../../validation/dmri_pipeline/lossless_20261002.md)。 |
| 2026-09-30 | 经典模式与 MATLAB NODDI 1.05 的固定 24 个真实脑体素比较，见上列 `classic_original_real_24.public.json`。 |
| 2026-09-29 | LUT 批量 100→400 的真实全脑输出比较和共享 GPU 耗时观察，见上列 `classic_whole_brain.public.json`；该批量设置在本次更新中保留。 |

## 2026-10-02：LUT 填充行与临时求解矩阵的显存修复

公开十人双分支测试中，`case02` 的 AMICO 阶段触发了 20,000,000,000 bytes 的进程分配上限。错误位置是 `_masked_solve`：它按所有 LUT 方向的最大 voxel 数补齐批次，再为每一个填充行分配 passive-set Cholesky 矩阵。即使该行的 passive mask 全为零，也占用同样的临时矩阵空间；方向上的体素分布越不均匀，这部分浪费越大。该次矩阵分配请求约 `2.96 GiB`，当时本进程已有 allocated 约 `12.41 GiB`，另有约 `3.58 GiB` reserved 未分配缓存。整张卡仍有超过 `60 GiB` 可用，不应把这次失败解释为服务器显卡总显存耗尽。

本次只调整内部求解的内存调度：

- passive count 为零的行直接保留精确零输出，仅对 count 大于零的行建立 Cholesky 矩阵。
- 每行保留原来的全局 compact width 和 `topk(sorted=False)`；压缩后的行仍按原始 `flat_row // slots` 选择所属方向的 Gram，避免把后一个方向的体素配到前一个方向。
- Cholesky 按约 **128 MiB 的临时工作集估计**分块，下一块开始前释放上一块矩阵及其 diagonal view；这不限制输出、LUT、信号等持续存活张量，也不代替 PyTorch/cuSOLVER 自身的 workspace。
- 奇异系统继续使用原 CG。每个失败方向只广播一份 `(1,C,C)` Gram 到 `(F,1,C)` rhs/mask，并另按 full-column 向量工作集分块，避免为每个失败体素复制完整 Gram。

`float64` solver、字典、LUT 批量 400、正则、KKT/CG 阈值、迭代上限与输出文件契约均保持原设置；不修改 TF32 开关，不降低数值精度，API 与 CLI 无新增参数。约 128 MiB 是内部临时分块目标，不是完整 NODDI 拟合只使用这点显存的承诺。

[差分测试](../../tests/amico_noddi/test_solver_memory.py)以小尺寸旧 `_masked_solve` 为独立数值 oracle，覆盖共享与分方向 Gram、全部/部分填充行、不同 passive width、强制跨方向分块、正则和奇异 CG 回退，以及完整活动集的值、支撑集和停止轮数。CPU 测试中的小矩阵用于检查算法与索引，不能代替真实影像 benchmark。

随后使用真实 pipeline 已保存的 EDDY 校正 DWI、脑掩膜、旋转梯度和原 bval，运行[独立 NODDI 组件检查](../../validation/dmri_pipeline/public10_20261002/check_noddi_memoryfix.py)。它重新拟合并保存五张 NODDI 图，不重跑 TOPUP、EDDY 或配准，显存上限固定为 **20,000,000,000 bytes**。

| 真实输入 | 输出检查 | PyTorch peak allocated（bytes） | PyTorch peak reserved（bytes） |
|---|---|---:|---:|
| `case01` | 五张图的全部解码值、shape 与 affine 均与旧版逐值相同 | 3,674,249,216 | 3,829,399,552 |
| `case02` | 旧版在 20 GB 上限内失败；修复后完成五张图，全部为有限值；没有旧版完整输出作逐值对照 | 5,391,976,448 | 6,490,685,440 |

两例组件运行均在指定上限内完成。`reserved` 是 allocator 向 CUDA 保留的显存，包含 `allocated`；两列不能相加。这些数值是独立 NODDI 组件进程的 allocator 峰值，不是整个 pipeline 的峰值，也不包含 CUDA context 或其他进程的显存。

检查绑定修复提交 `bf339a0368a7711d2c6ca3477c8d7dc1fc17e75a`，其中 `solver.py` 的 SHA-256 为 `1d4887270f267a83967ee4cc9336b306110bf838dde78b6a00ea16e204a1f74e`；solver float64、TF32 设置与参数保持原样。本记录更新时修复提交仍在本地，尚未推送 main。后续十人两个分支共 **20 个 FNIT 完整流程**将统一使用修复源码重跑，旧版首人整链结果保留为历史记录。

本次组件检查支持 `case01` 没有输出数值回归，并确认 `case02` 的分配失败已解除；`case02` 与原软件的精度仍由独立整链比较验收。十人端到端对照尚未完成，不把组件耗时写成 end-to-end 时间，也不以两例组件成功宣称十人通过。此前本页的单被试耗时来自不同修订，保留其历史范围。

## Reference

- 参考文献：Daducci et al., *Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data*, NeuroImage (2015), [doi:10.1016/j.neuroimage.2014.10.026](https://doi.org/10.1016/j.neuroimage.2014.10.026)。
- 参考文献：Zhang et al., *NODDI: Practical in vivo neurite orientation dispersion and density imaging of the human brain*, NeuroImage (2012), [原文](https://www.sciencedirect.com/science/article/pii/S1053811912003539)。
- 原实现代码库：[AMICO 2.0.3](https://github.com/daducci/AMICO/tree/v2.0.3)。
- 原版经典实现：[NODDI Matlab Toolbox 1.05](https://www.nitrc.org/projects/noddi_toolbox)；其模型与拟合方法见上列 Zhang et al. 论文。
