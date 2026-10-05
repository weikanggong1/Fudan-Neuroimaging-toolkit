# TorchEDDY：DWI运动和涡流校正

| 项目 | 内容 |
|---|---|
| 输入 | 4D DWI、脑mask、PE采集参数、逐帧索引和bvec/bval |
| 输出 | 校正4D DWI、旋转梯度、运动/涡流参数与离群QC |
| 对应原软件 | FSL EDDY2111.0 volume-to-volume默认组合 |
| Python / CLI | TorchEDDY、run_ukb_eddy / fnit eddy、fnit-eddy |
| CPU / GPU | PyTorch CPU/CUDA；原始输入helper使用FNIT SynthStrip |

## 1. 功能简介

`TorchEDDY` 使用 PyTorch 实现 FSL EDDY 2111.0 的 UK Biobank volume-to-volume 路径：先配准 b0，再按 8 次迭代交替拟合球面 Gaussian process、检查离群切片和更新 DWI 的运动及二次涡流场，最后旋转 b-vector，替换离群切片并做 Jacobian 重采样。运行时只需 FNIT 的 Python 依赖，不调用 FSL。单次调用处理一个 4D DWI；多个被试由调用方安排作业。

此实现对应下面固定的 FSL 命令组合，不覆盖 EDDY 的 slice-to-volume、multi-band 或动态 susceptibility 选项。当前一例真实数据达到预设的图像和参数相关性阈值；完整数值、时间及未达到逐体素相同的部分见[验证记录](../../validation/eddy/README.md)。

## 2. Python 调用

```python
from fnit.eddy import TorchEDDY

input_diffusion_path = "/data/AP.nii.gz"  # 完整4D DWI
brain_mask_path = "/data/nodif_brain_mask.nii.gz"  # 同DWI网格3D脑mask
acquisition_parameters_path = "/data/acqparams.txt"  # PE方向与总读出时间
volume_acquisition_index_path = "/data/eddy_index.txt"  # N个一开始的acqp行号
gradient_directions_path = "/data/AP.bvec"  # FSL 3×N梯度方向
b_values_path = "/data/AP.bval"  # N个b值，s/mm²
corrected_output_prefix = "/data/eddy/data"  # 文件根名
eddy_result = TorchEDDY(device="cuda:0").run(
    imain=input_diffusion_path, mask=brain_mask_path,
    acqp=acquisition_parameters_path, index=volume_acquisition_index_path,
    bvecs=gradient_directions_path, bvals=b_values_path,
    out=corrected_output_prefix, gp_seed=12345,  # 固定GP种子便于复核
)
```

### 输入数据格式

| 参数 | 输入含义 | 格式与要求 |
|---|---|---|
| `imain` | 待校正的单被试 DWI | 4D NIfTI，第四维为 N 个 volume |
| `mask` | 用于 GP 拟合和离群切片判断的脑掩膜 | 与 `imain` 前三维同网格的 3D NIfTI |
| `acqp` | 每种采集的相位编码方向和总读出时间 | 每行 4 列：PE 的 x/y/z 分量和秒数 |
| `index` | 每个 volume 使用 `acqp` 的哪一行 | N 个从 1 开始的整数，顺序与 DWI 相同 |
| `bvecs` | 扩散梯度方向 | FSL 3×N 文本矩阵 |
| `bvals` | 每个 volume 的 b-value | N 个数值，与 DWI 顺序相同 |
| `topup` | TOPUP 的场系数和运动参数前缀 | 例如 `fieldmap_out`，读取同名前缀的 `_fieldcoef.nii.gz` 和 `_movpar.txt`；无反向 PE 数据时可设 `None`，此时不施加 TOPUP 场 |
| `ref_scan_no` | 固定运动参考帧 | 从 0 开始的 volume 编号；默认 0 |
| `gp_seed` | GP 选点的随机种子 | 做逐次比较时与 FSL `--initrand` 取相同整数；默认 `None`，按当前时间选种子 |
| `out` | 输出文件前缀 | 例如 `eddy/data`，不是目录或 NIfTI 文件名 |

### 输入参数

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `device` | 否 | str / torch.device / None | `'cuda:0'` | 计算设备；显式 CUDA 不可用时报错。 |
| `config` | 否 | EDDYConfig / None | `None` | 构造 TorchEDDY 的配置；None 使用下表默认值，Python 不接收配置文件名。 |
| `imain` | 是 | str / Path | 无 | 完整 4D DWI 的 NIfTI 文件；不直接接收内存数组或影像对象。 |
| `mask` | 是 | str / Path | 无 | 与DWI前三维同shape/affine的3D脑掩膜。 |
| `acqp` | 是 | str / Path | 无 | 每行PE向量三分量+总读出时间(秒)。 |
| `index` | 是 | str / Path | 无 | N个从1开始的acqp行号。 |
| `bvecs` | 是 | str / Path | 无 | FSL 3×N梯度方向，无量纲。 |
| `bvals` | 是 | str / Path | 无 | N个b值，s/mm²。 |
| `topup` | 否 | 路径 / None | `None` | TOPUP系数和movpar文件前缀；None不校正susceptibility。 |
| `ref_scan_no` | 否 | int / None | `None` | 零起始运动参考帧；None使用配置默认0。 |
| `gp_seed` | 否 | int / None | `None` | GP选点种子；None按时间初始化。 |
| `out` | 是 | 路径 | 无 | 结果文件根名/前缀。 |
| `overwrite` | 否 | bool | `False` | 是否覆盖已有结果，默认保护原文件。 |

### 配置参数

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `niter` | 否 | int | `8` | EDDY迭代数。 |
| `fwhm_mm` | 否 | tuple[float, ...] | `(10.0, 8.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0)` | 各轮/层平滑FWHM，mm。 |
| `ff` | 否 | float | `10.0` | GP预测噪声放大系数。 |
| `nvoxhp` | 否 | int | `1000` | GP超参数拟合脑体素数。 |
| `ol_nstd` | 否 | float | `4.0` | 离群切片标准差阈值。 |
| `ol_nvox` | 否 | int | `250` | 离群切片最小脑体素数。 |
| `ref_scan_no` | 否 | int | `0` | 默认零起始参考帧。 |
| `gp_nm_maxiter` | 否 | int | `500` | GP超参数Nelder–Mead最大迭代数。 |
| `b0_threshold` | 否 | float | `100.0` | b0判定阈值，s/mm²。 |
| `shell_tolerance` | 否 | float | `100.0` | b-shell容差，s/mm²。 |
| `use_tf32` | 否 | bool | `True` | CUDA默认允许TF32。 |
| `spline_precision` | 否 | float | `1e-08` | 样条预滤收敛阈值。 |
| `enable_post_eddy_shell_alignment` | 否 | bool | `True` | 开启EDDY后shell对齐。 |

### 原始输入和脑掩膜准备

`run_ukb_eddy`封装准备和校正，返回(result,inputs)。校正b0均值交给FNIT SynthStrip，固定border=1mm/no_csf=False；没有TOPUP iout时只平均b<100帧。不会把完整DWI一起平均。

| 参数 | 必需 | 类型 | 默认值 | 含义 |
|---|---|---|---|---|
| `raw_dir` | 是 | 路径 | 无 | 含AP/PA影像和采集侧文件的单被试目录。 |
| `topup_dir` | 是 | 路径 | 无 | 已有TOPUP结果和acqparams目录。 |
| `output_dir` | 是 | 路径 | 无 | 本次调用的多文件输出目录。 |
| `device` | 否 | str / torch.device / None | `None` | 计算设备；显式 CUDA 不可用时报错。 |
| `overwrite` | 否 | bool | `False` | 是否覆盖已有结果，默认保护原文件。 |
| `synthstrip_weights` | 否 | 路径 / None | `None` | 标准SynthStrip权重；None按本地解析器查找。 |
| `ref_scan_no` | 否 | int / None | `None` | 零起始运动参考帧；None使用配置默认0。 |
| `gp_seed` | 否 | int / None | `None` | GP选点种子；None按时间初始化。 |

AP目录含AP.nii.gz/AP.bval/AP.bvec；topup目录含acqparams、fieldcoef/movpar与可选iout。ref_scan_no=None自动选b0，显式值须指向b<100帧；直接TorchEDDY使用调用者mask。

### 输出

```text
eddy/
├── data.nii.gz
├── data.eddy_rotated_bvecs
├── data.eddy_parameters
├── data.eddy_movement_rms
├── data.eddy_restricted_movement_rms
├── data.eddy_outlier_map
├── data.eddy_outlier_n_stdev_map
├── data.eddy_outlier_n_sqr_stdev_map
├── data.eddy_outlier_report
└── data.eddy_qc.json
```

| 文件 | 形状或内容 | 用途 |
|---|---|---|
| `data.nii.gz` | X×Y×Z×N，原 DWI 网格 | 校正后的 4D DWI；`result.corrected` |
| `data.eddy_rotated_bvecs` | 3×N | 按逐帧运动旋转的梯度方向；`result.rotated_bvecs` |
| `data.eddy_parameters` | N×16 | 前 3 列平移 mm、后 3 列旋转 rad、最后 10 列二次涡流场系数；`result.parameters` |
| `data.eddy_movement_rms` | N×2 | 相对参考位姿和前一 volume 的位移 RMS，mm；`result.movement_rms` |
| `data.eddy_restricted_movement_rms` | N×2 | 去掉 PE 平移参数后按全 3D 位移计算的 RMS，mm；`result.restricted_movement_rms` |
| `data.eddy_outlier_map` | N×Z，0/1 | 每帧每切片的离群标记；`result.outlier_map` |
| `data.eddy_outlier_n_stdev_map`、`data.eddy_outlier_n_sqr_stdev_map` | 各 N×Z | 切片残差标准差分数及平方分数；对应 `result` 同名字段 |
| `data.eddy_outlier_report` | 文本 | 离群切片的帧、切片及分数；`result.outlier_report_lines` |
| `data.eddy_qc.json` | JSON | FNIT 额外保存的运行配置及迭代记录；`result.qc` |

校正NIfTI为float32，与imain shape/affine相同，仍为原始DWI空间；参数每行对应输入帧。相位编码空间轴须统一，符号/读出时间可按index区分。梯度旋转依据对应逐帧运动，不改变b值。

## 3. 命令行调用

```bash
fnit eddy \
  --imain AP.nii.gz \
  --mask nodif_brain_mask.nii.gz \
  --topup fieldmap_out \
  --acqp acqparams.txt \
  --index eddy_index.txt \
  --bvecs AP.bvec \
  --bvals AP.bval \
  --out eddy/data \
  --ref-scan-no 0 \
  --gp-seed 12345 \
  --device cuda:0
```

| CLI 参数 | Python 参数 | 含义 |
|---|---|---|
| `--imain / --mask` | `imain / mask` | DWI和同网格mask |
| `--acqp / --index / --bvecs / --bvals` | `acqp / index / bvecs / bvals` | 采集与梯度 |
| `--topup / --out` | `topup / out` | 上游场前缀和输出前缀 |
| `--ref-scan-no / --gp-seed` | `ref_scan_no / gp_seed` | 参考帧和种子 |
| `--raw-dir / --topup-dir / --output-dir` | `raw_dir / topup_dir / output_dir` | 原始输入helper模式 |
| `--synthstrip-weights` | `synthstrip_weights` | helper权重 |
| `--device / --overwrite` | `device / overwrite` | 设备与覆盖 |

## 4. 原软件调用

与本页实测配对的原软件命令为：

```bash
eddy_cuda10.2 \
  --imain=AP.nii.gz --mask=nodif_brain_mask.nii.gz \
  --topup=fieldmap_out --acqp=acqparams.txt --index=eddy_index.txt \
  --bvecs=AP.bvec --bvals=AP.bval --out=eddy/data \
  --ref_scan_no=0 --initrand=12345 \
  --flm=quadratic --resamp=jac --slm=linear --niter=8 \
  --fwhm=10,8,4,2,0,0,0,0 --ff=10 --sep_offs_move \
  --nvoxhp=1000 --repol --rms
```

`--flm=quadratic` 拟合 10 参数二次涡流场；`--resamp=jac` 在重采样时校正局部体积变化；`--slm=linear` 对涡流参数施加线性 shell 模型。`--fwhm` 指定 8 轮平滑尺度，`--ff` 控制 GP 预测中的噪声放大，`--sep_offs_move` 分离 PE 方向的场偏移与运动。`--nvoxhp` 指定 GP 超参数拟合选取的脑体素数，`--repol` 替换离群切片，`--rms` 写出运动位移。`--initrand` 只用于使这次对照可复核。

## 5. 最新精度和运行时间

最新独立官方上游的完整EDDY对照绑定2026-10-02集成提交b3ccafe0a48f4c1c396ae6285d0bdbe07a7664c9，一例104×104×72×105 DWI，5个b0及两组各50个非b0。FSL6.0.7.4/EDDY2111.0，Python3.11.16/Torch2.5.1，H100 GPU1、双方8CPU线程、固定种子12345；FNIT默认TF32/FP32、20GB上限。双方各自生成TOPUP和SynthStrip上游输入，因此误差包含完整上游差异。

| 本轮指标 | 独立官方 SynthStrip 上游参考 |
|---|---:|
| FNIT／参考脑体素数；共同脑体素数 | 271,075／271,080；271,073 |
| 脑 mask Dice | 0.9999834 |
| 完整 105 volume，固定官方脑区 Pearson r | 0.999764184 |
| 同 ROI MAE／RMSE，原信号单位 | 23.2314／43.0074 |
| 同 ROI 最大绝对差，原信号单位 | 2655.9336 |
| 100 个非 b0 旋转梯度平均／最大夹角 | 0.05585°／0.12146° |
| 离群图 FNIT／参考条目数；不同条目数 | 20／21；1 |
| EDDY 读取、计算与保存，b3ccafe0 FNIT／参考 | 331.41／645.88 s |
| FNIT EDDY 阶段 allocator 已分配／保留峰值 | 4.92910／11.32672 GB |

### 分步骤 benchmark

| 范围 | FNIT | 原软件 |
|---|---:|---:|
| EDDY读取、求解、D2H与保存 | 331.41s | 645.88s |
| GP/离群/运动更新各子步 | 最新上游配对未单列；[旧固定输入阶段记录](../../validation/eddy/report.public.json) | 未单列 |

共享节点各一次观察；[实际源码/完整上游报告](../../validation/dmri_pipeline/upstream.synthstrip_topup_20261002.public.json)与[端到端整链报告](../../validation/dmri_pipeline/end_to_end_synthstrip_topup_20261002.md)保留来源。本次文档未重跑当前main，原版和FNIT仍非逐值一致。新版脑图见整链报告；本页不把旧mask固定输入图重标为最新。

## 6. 最近版本和 benchmark

| 日期 | commit/version | 变化 | benchmark |
|---|---|---|---|
| 2026-10-02 | b3ccafe0 | SynthStrip及匹配TOPUP上游重新配对 | [一例完整105帧](../../validation/dmri_pipeline/upstream.synthstrip_topup_20261002.public.json) |
| 2026-10-02 | 参数转发修复 | 原始输入CLI正确转发ref_scan_no/gp_seed，接口兼容 | CLI/wrapper回归，未改写已有整链时钟 |
| 2026-10-02 | 无损执行优化 | 融合采样/缓存并保留八轮算法 | [固定输入记录](../../validation/dmri_pipeline/lossless_20261002.md) |
| 2026-09-29 | FSL2111严格路径 | 完整volume-to-volume默认组合 | [独立EDDY报告](../../validation/eddy/README.md) |

## 7. 参考文献、原软件和资源

源码位置：[原软件 `eddy-2111.0/eddy.cpp`](../../src/fnit/_vendor_fsl/sources/eddy-2111.0/eddy.cpp)；[FNIT `eddy/fsl2111_strict/pipeline.py`](../../src/fnit/eddy/fsl2111_strict/pipeline.py)。固定 tag/commit、Git tree 及每文件 SHA-256 见[来源清单](../../src/fnit/_vendor_fsl/manifest.json)。

代码改写沿用 [FSL 6.0 非商业许可证](../../licenses/FSL-6.0.txt)，完整来源与再分发要求见[第三方声明](../../THIRD_PARTY_NOTICES.md)。

- 参考文献：Andersson & Sotiropoulos, *An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging*, NeuroImage (2016), [doi:10.1016/j.neuroimage.2015.10.019](https://doi.org/10.1016/j.neuroimage.2015.10.019)。
- 原实现代码库：[FSL `eddy`](https://git.fmrib.ox.ac.uk/fsl/eddy)。
- 脑提取参考：Hoopes et al., *SynthStrip: Skull-Stripping for Any Brain Image*, NeuroImage (2022), [doi:10.1016/j.neuroimage.2022.119474](https://doi.org/10.1016/j.neuroimage.2022.119474)；[FreeSurfer `mri_synthstrip`](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthstrip)。

| 资源 | 用途 | 官方来源 | 大小 | SHA-256 | 是否允许 FNIT 再分发 |
|---|---|---|---:|---|---|
| synthstrip.1.pt；仅原始输入helper需要 | b0脑掩膜 | [FreeSurfer SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) | 30,851,709B | 37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33 | 允许按CC BY 4.0条款，见[逐文件资源清单](../RESOURCE_MANIFEST.md)和[权重说明](../WEIGHTS.md)；优先固定assets-v1 |

[完整历史说明与调试证据](../../validation/eddy/readme_archive_20261005.md) · [返回主页](../../README.md)

<!-- 旧版文档锚点兼容 -->
<a id="输入与-fsl-命令"></a> <a id="python-调用"></a> <a id="ukb-输入准备与-synthstrip-脑掩膜"></a> <a id="输出文件"></a> <a id="真实数据对照与版本记录"></a> <a id="2026-10-02当前-main-的-synthstrip匹配-topup-上游复测"></a> <a id="2026-10-02synthstrip-固定输入"></a> <a id="2026-09-29固定-eddy-输入"></a> <a id="最近更新与无损验收"></a> <a id="2026-10-02-真实病例完整八轮结果"></a> <a id="reference"></a>
