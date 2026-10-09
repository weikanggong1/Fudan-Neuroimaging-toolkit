# TorchAMICONODDI：NODDI微结构拟合

| 项目 | 内容 |
|---|---|
| 输入 | 已校正多shell DWI、二值mask和bval/bvec |
| 输出 | NDI、ODI、FWF、方向及RMSE图 |
| 对应原软件 | AMICO2.0.3；classic对应NODDI Toolbox |
| Python / CLI | TorchAMICONODDI / fnit-amico-noddi |
| CPU / GPU | CPU/CUDA；float32信号、float64求解 |

## 1. 功能简介

`TorchAMICONODDI` 从单被试扩散信号估计神经突密度、方向离散和自由水分数。默认 `fit_method="amico"` 使用离散响应字典；`classic` 使用AMICO起点后进行连续Watson模型的非线性拟合，两者模型输出含义相同但数值不会相同。

FNIT使用PyTorch生成响应核并拟合，用nibabel读写；运行时不调用AMICO、NODDI Toolbox或DIPY。500方向LUT和145原子字典索引随包提供。信号/核为float32，活动集及classic求解使用float64；CUDA允许TF32，不使用FP16/BF16。

## 2. Python 调用

```python
from fnit.amico_noddi import AMICONODDIConfig, TorchAMICONODDI

noddi_configuration = AMICONODDIConfig(
    lut_batch_size=400,  # 同时处理的LUT方向数，显存不足时可调小
)
noddi_model = TorchAMICONODDI(
    device="cuda:0",  # 计算设备
    config=noddi_configuration,  # 使用公开的AMICO配置
    fit_method="amico",  # classic时改用连续Watson拟合
)
noddi_result = noddi_model.run(
    data="/data/eddy/data.nii.gz",  # 完整多shell校正DWI
    mask="/data/eddy/brain_mask.nii.gz",  # 同diffusion网格二值mask
    bvecs="/data/eddy/data.eddy_rotated_bvecs",  # 旋转后梯度
    bvals="/data/dwi/bvals",  # 完整b-value
    output_dir="/data/results/noddi",  # 五张参数图的输出目录
    naming="amico",  # 使用fit_*文件名
    overwrite=False,  # 不覆盖已有结果
)
neurite_density_image = noddi_result.ndi  # nibabel.Nifti1Image
quality_control = noddi_result.qc  # 拟合方式、耗时、精度与显存
```
### 输入数据格式

- DWI：单被试 NIfTI，shape 为 `[X,Y,Z,N]`，至少含 b0 和扩散方向；信号强度沿用输入单位。读入转 float32，不在此入口做运动、涡流或磁敏感校正。
- mask：NIfTI `[X,Y,Z]`，须与 DWI 的 affine、orientation、体素尺寸和原生 diffusion 空间一致。不自动配准或重排梯度；mask 外结果为零。
- bval：纯文本 N 个有限 b-value，单位 s/mm²，顺序与第四维一致。
- bvec：纯文本 `3×N` 或 `N×3`，每个非 b0方向对应同一 volume；已完成 EDDY 时用旋转后的 bvec。
- 路径接受字符串或 PathLike。示例使用用户自己的真实文件；多被试调度由调用方组织。
AMICO入口按uint8转换后值等于1选择mask体素，最好明确提供0/1mask。至少需要一个b0和扩散加权volume；用于NODDI推断的数据采集应与参考协议一致。AMICO把b-value按 `b_step` 取整；classic用原始bval及单位化bvec。

`noddi_model(data, mask, bvecs, bvals)` 返回内存对象，`run()`再保存五张图。自定义配置不会自动变成CLI参数。改变扩散率、字典网格或正则化会改变模型，不能直接沿用默认benchmark。

**`TorchAMICONODDI.__init__` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `device` | 否 | `str/torch.device/None` | `None` | PyTorch 设备；None 自动选择可用 CUDA，否则 CPU。 |
| `config` | 否 | `配置对象或None` | `None` | 对应配置对象；None 使用默认配置。 |
| `fit_method` | 否 | `str` | `'amico'` | amico 为离散字典拟合；classic 为连续 Watson 非线性拟合。 |

**`TorchAMICONODDI.run` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `data` | 是 | `路径` | `—` | 已校正的 4D DWI 文件，第四维为采集 volume。 |
| `mask` | 是 | `路径` | `—` | 与 DWI 相同 shape、affine 和方向的 3D 脑掩膜。 |
| `bvecs` | 是 | `路径` | `—` | FSL 3×N 或 N×3 梯度方向文本；优先用 EDDY 旋转后的方向。 |
| `bvals` | 是 | `路径` | `—` | N 个 b-value，单位 s/mm²，顺序与 DWI 一致。 |
| `output_dir` | 是 | `路径` | `—` | 本次结果目录；路径按当前工作目录解析。 |
| `naming` | 否 | `str` | `'ukb'` | ukb 使用 NODDI_* 名称；amico 使用 fit_* 名称。 |
| `overwrite` | 否 | `bool` | `False` | 是否允许覆盖已有结果；默认已有结果时报错。 |

**`AMICONODDIConfig` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `d_par` | 否 | `float` | `0.0017` | 细胞内及细胞外平行扩散率，单位 mm²/s。 |
| `d_iso` | 否 | `float` | `0.003` | 各向同性自由水扩散率，单位 mm²/s。 |
| `ic_vfs` | 否 | `tuple[float, ...]` | `12点：0.1–0.99` | 细胞内体积分数字典网格；默认12个从0.1到0.99的值。 |
| `ic_ods` | 否 | `tuple[float, ...]` | `0.03、0.06及0.09–0.99的10点` | ODI 字典网格；默认0.03、0.06及10个从0.09到0.99的值。 |
| `lambda1` | 否 | `float` | `0.5` | AMICO 稀疏支持选择的第一正则参数。 |
| `lambda2` | 否 | `float` | `0.001` | AMICO 稀疏支持选择的第二正则参数。 |
| `b0_threshold` | 否 | `float` | `100.0` | AMICO 判定 b0 的最大 b-value，单位 s/mm²。 |
| `b_step` | 否 | `float` | `100.0` | AMICO b-value 取整步长，单位 s/mm²；classic 使用原值。 |
| `kkt_tolerance` | 否 | `float` | `1e-11` | 活动集 KKT 收敛容差。 |
| `cg_tolerance` | 否 | `float` | `1e-13` | 共轭梯度回退求解的收敛容差。 |
| `maximum_active_steps` | 否 | `int` | `40` | 活动集最大更新次数。 |
| `lut_batch_size` | 否 | `int` | `400` | 每批处理的 LUT 方向数；减小可降低显存占用。 |

**`TorchAMICONODDI.__call__` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `data` | 是 | `路径` | `—` | 已校正的 4D DWI 文件，第四维为采集 volume。 |
| `mask` | 是 | `路径` | `—` | 与 DWI 相同 shape、affine 和方向的 3D 脑掩膜。 |
| `bvecs` | 是 | `路径` | `—` | FSL 3×N 或 N×3 梯度方向文本；优先用 EDDY 旋转后的方向。 |
| `bvals` | 是 | `路径` | `—` | N 个 b-value，单位 s/mm²，顺序与 DWI 一致。 |

**`AMICONODDIResult.save` 参数**

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `output_dir` | 是 | `路径` | `—` | 本次结果目录；路径按当前工作目录解析。 |
| `naming` | 否 | `str` | `'ukb'` | ukb 使用 NODDI_* 名称；amico 使用 fit_* 名称。 |
| `overwrite` | 否 | `bool` | `False` | 是否允许覆盖已有结果；默认已有结果时报错。 |

### 输出

```text
noddi/
├── fit_NDI.nii.gz
├── fit_ODI.nii.gz
├── fit_FWF.nii.gz
├── fit_dir.nii.gz
└── fit_RMSE.nii.gz
```

| 返回字段 | amico文件名 | ukb文件名 | shape与含义 |
|---|---|---|---|
| `ndi` | fit_NDI | NODDI_ICVF | `[X,Y,Z]`；组织内细胞内/神经突分数 |
| `odi` | fit_ODI | NODDI_OD | `[X,Y,Z]`；方向离散指数 |
| `fwf` | fit_FWF | NODDI_ISOVF | `[X,Y,Z]`；整体自由水分数 |
| `directions` | fit_dir | NODDI_dir | `[X,Y,Z,3]`；AMICO为OLS主方向，classic为拟合方向轴 |
| `rmse` | fit_RMSE | NODDI_RMSE | `[X,Y,Z]`；b0归一化信号拟合RMSE |

上表basename均加 `.nii.gz`。所有图float32，保留输入DWI affine、orientation和原生空间，mask外为零；分数/ODI/RMSE无量纲。方向正负等价，不是位移场。`result.qc`为内存字典；保存仅写上述五张NIfTI，不编造额外QC文件。

## 3. 命令行调用

```bash
fnit-amico-noddi -k /data/eddy/data.nii.gz -m /data/eddy/brain_mask.nii.gz    -r /data/eddy/data.eddy_rotated_bvecs -b /data/dwi/bvals    -o /data/results/noddi --naming amico --fit-method amico --device cuda:0
```

统一入口 `fnit amico-noddi` 相同；连续拟合改为 `--fit-method classic`。

| CLI | Python参数 | 含义 |
|---|---|---|
| `-k/--data`、`-m/--mask` | `data`、`mask` | DWI与mask |
| `-r/--bvecs`、`-b/--bvals` | `bvecs`、`bvals` | 梯度文本 |
| `-o/--output-dir` | `output_dir` | 结果目录 |
| `--naming` | `naming` | 默认ukb，可用amico |
| `--fit-method` | 构造函数`fit_method` | 默认amico |
| `--device` | 构造函数`device` | 计算设备 |
| `--overwrite` | `overwrite=True` | 覆盖已有图 |

配置对象的12个字段仅能通过Python传入，CLI没有对应开关。多被试由调用方分配进程和GPU。

## 4. 原软件调用

AMICO2.0.3没有对应独立单行CLI；在隔离参考环境按以下Python序列处理相同输入。

```python
import amico

amico.core.setup()  # 原软件初始化rotation cache
amico.util.fsl2scheme(
    bvalsFilename="bvals",  # 同一b-value
    bvecsFilename="bvecs",  # 同一旋转后方向
    schemeFilename="acquisition.scheme",  # 输出scheme
    bStep=100,  # 原协议的取整步长
)
reference_evaluation = amico.Evaluation(study_path=".", subject=".")  # 参考输入目录
reference_evaluation.load_data(
    dwi_filename="DWI.nii.gz",  # 同一DWI
    scheme_filename="acquisition.scheme",  # 上一步scheme
    mask_filename="mask.nii.gz",  # 同一二值mask
    b0_thr=100,  # b0阈值，s/mm²
)
reference_evaluation.set_model("NODDI")  # 选择NODDI
reference_evaluation.generate_kernels(regenerate=True)  # 当前采集kernel
reference_evaluation.load_kernels()
reference_evaluation.set_solver(lambda1=0.5, lambda2=1e-3)  # 同正则参数
reference_evaluation.CONFIG["doComputeRMSE"] = True
reference_evaluation.fit()
reference_evaluation.save_results()
```

| FNIT参数 | AMICO参数 |
|---|---|
| `b_step`、`b0_threshold` | `fsl2scheme(bStep)`、`load_data(b0_thr)` |
| `lambda1/lambda2` | `set_solver(lambda1/lambda2)` |
| `naming="amico"` | 官方fit_*文件名 |

classic原参考使用NODDI Toolbox1.05：

```matlab
CreateROI('DWI.nii.gz','mask.nii.gz','roi.mat'); % 同DWI与mask
protocol = FSL2Protocol('bvals','bvecs',100); % 原始梯度和b0阈值
model = MakeModel('WatsonSHStickTortIsoV_B0'); % Watson模型
batch_fitting_single('roi.mat',protocol,model,'fit.mat'); % 非线性拟合
SaveParamsAsNIfTI('fit.mat','roi.mat','mask.nii.gz','noddi'); % 参数图
```

原Toolbox用网格搜索和MATLAB fmincon；FNIT用AMICO起点与阻尼Gauss–Newton，不能当成同一优化器。第三方依赖仅存在于参考环境。

## 5. 最新精度和运行时间

最新完整真实流程证据为[固定十人报告](../../validation/dmri_pipeline/public10_20261002/README.md)，绑定 `bf339a0`、433文件清单SHA `f13a4098…`。OpenNeuro ds003138 v1.0.1，十名被试、各117帧AP与PA b0；AMICO模式分别使用两侧各自EDDY结果，不是同输入的独立拟合对照。H100、8CPU线程、20,000,000,000字节allocator上限，precision保持源码float32/float64策略；本轮未按140c3739重跑。

| 完整pipeline内NODDI阶段 | FNIT中位数 | 官方AMICO2.0.3中位数 |
|---|---:|---:|
| TBSS分支，n=10 | 17.05 s | 29.00 s |
| MMORF分支，FNIT n=10/参照n=9 | 20.41 s | 28.29 s |

阶段含拟合与保存，完整pipeline时间见[dMRI页](../dmri_pipeline/README.md)。不同EDDY输入的差异也进入NODDI图，以上不是受控AMICO加速比；独立NODDI完整进程与原软件配对峰值显存未在十人摘要单列。

`bf339a0`修复真实整链中的被动集合填充行临时显存问题。两例固定输入回归分别完成五图；其中一例对旧版解码数组、shape和affine逐值相同，另一例旧版OOM没有完整结果可比。完整20次修正版pipeline均完成，不能把组件回归当作所有采集逐值等价。

独立同输入真实原软件精度仍见[AMICO全脑报告](../../validation/amico_noddi/report.public.json)：NDI/ODI/FWF MAE为2.13e-6/8.08e-6/3.15e-6；它绑定显存修复前源码。classic的[24真实体素对照](../../validation/amico_noddi/classic_original_real_24.public.json)对Toolbox1.05的三项MAE为0.002592/0.002765/0.000470，原机器不同不能比较速度。

![显存修复前同输入AMICO与FNIT参数图及差图](figures/amico_noddi_comparison.png)

此图属于所链接旧源码精度记录；最新源码的独立AMICO/classic原软件同输入全脑精度未重测。classic最新完整raw-to-MNI真实队列benchmark未测。


<!-- FNIT-UNIFIED-BENCHMARK-20261008 -->
### 本轮统一 benchmark 摘要（2026-10-08）

最新可公开的单被试 GPU 对照为 H100、104×104×72×105 DWI、242,261 体素全脑 mask、进程显存上限 20%：`fit_method="amico"` 为 **81.79 s / 9.95 GB**，`fit_method="classic"` 为 **209.55 s / 9.95 GB**（均含读写；classic 初始化 84.63 s、Watson 拟合 118.09 s）。两次运行时共享 GPU 利用率 99–100%，因此只报告观察值，不外推稳定加速比。五张输出图均为有限值、同网格且 mask 外为零；另有 24 个真实体素与 MATLAB NODDI 数值核对。CPU 同输入完整 wall clock 本轮未形成可复核公开记录。完整汇总见 [统一 benchmark 索引](../BENCHMARK_INDEX.md)。

## 6. 最近版本和 benchmark

| 日期 | commit/version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-03 | a81ffcef | 发布十人双分支真实benchmark | bf339a0冻结20流程；19配对 |
| 2026-10-02 | bf339a03 | 限制被动集合工作区、跳过填充行 | 真实组件回归及20GB整链验收 |
| 2026-10-02 | 7473452 | NNLS复用Gram、classic设备端QC计数 | 同真实全脑五图不变；classic未稳定提速 |
| 2026-09-29 | be4fbc4c | LUT批量默认400及classic路径 | 24体素Toolbox与共享GPU全脑观测 |

更早的debug、profiling和长表保留在[旧README归档](../../validation/amico_noddi/readme_archive_20261005.md)。归档已修复相对链接；旧科学报告与原始产物不修改。

<a id="输入"></a>
<a id="python-调用"></a>
<a id="命令行调用"></a>
<a id="与官方-amico-指令的对应"></a>
<a id="优化前-amico-路径逐体素验证"></a>
<a id="经典连续-watson-拟合"></a>
<a id="计算复用更新与差分验收"></a>
<a id="2026-10-02lut-填充行与临时求解矩阵的显存修复"></a>
<a id="reference"></a>

## 7. 参考文献、原软件和资源

- 原代码：[AMICO v2.0.3](https://github.com/daducci/AMICO/tree/v2.0.3)，`amico/models.py`与`amico/core.py`；[NODDI Toolbox官方](https://www.nitrc.org/projects/noddi_toolbox/)。
- Zhang等，2012，[NODDI](https://doi.org/10.1016/j.neuroimage.2012.03.072)；Daducci等，2015，[Accelerated Microstructure Imaging via Convex Optimization](https://doi.org/10.1016/j.neuroimage.2014.10.026)。
- FNIT位置：[core.py](../../src/fnit/amico_noddi/core.py)、[solver.py](../../src/fnit/amico_noddi/solver.py)、[classic.py](../../src/fnit/amico_noddi/classic.py)。

### 外部资源

不需要预训练权重。下面三项LUT随包包含，按[AMICO许可](../../licenses/AMICO-2.0.txt)保留署名及科研/教育使用条款；资源是方向表，不是MRI数据。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---|---|---|
| grad_500.npy | 500方向梯度LUT | [AMICO v2.0.3](https://github.com/daducci/AMICO/tree/v2.0.3) | 12,128 bytes | `e07dd95378f12556aba7898a3b3396b3b35e1295e4932e361bb7309ebbf73e3e` | 按随包AMICO许可，科研/教育使用及署名；不扩展商业授权。 |
| ndirs_500.bin | 500方向float64表 | [AMICO v2.0.3](https://github.com/daducci/AMICO/tree/v2.0.3) | 12,000 bytes | `96e6638e756c53a98b8c16aed760e43fcdf636aa2b1b0bd2bfce21aacfc9f328` | 按随包AMICO许可，科研/教育使用及署名；不扩展商业授权。 |
| htable_ndirs_500.bin | 方向索引查找表 | [AMICO v2.0.3](https://github.com/daducci/AMICO/tree/v2.0.3) | 65,522 bytes | `682f11416844f2a09225c16f412bf7a224a5085a244b59d47f51c51217e4a30b` | 按随包AMICO许可，科研/教育使用及署名；不扩展商业授权。 |

[原资产目录](../../src/fnit/amico_noddi/assets/)与本轮实测散列见审核JSON；未重新下载资源。
