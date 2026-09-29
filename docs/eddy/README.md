# TorchEDDY：单被试 DWI 运动与涡流校正

[返回首页](../../README.md) · [源码](../../src/fnit/eddy/) · [真实数据验证](../../validation/eddy/README.md)

`TorchEDDY` 使用 PyTorch 实现 FSL EDDY 2111.0 的 UK Biobank volume-to-volume 路径：先配准 b0，再按 8 次迭代交替拟合球面 Gaussian process、检查离群切片和更新 DWI 的运动及二次涡流场，最后旋转 b-vector，替换离群切片并做 Jacobian 重采样。运行时只需 FNIT 的 Python 依赖，不调用 FSL。单次调用处理一个 4D DWI；多个被试由调用方安排作业。

此实现对应下面固定的 FSL 命令组合，不覆盖 EDDY 的 slice-to-volume、multi-band 或动态 susceptibility 选项。当前一例真实数据达到预设的图像和参数相关性阈值；完整数值、时间及未达到逐体素相同的部分见[验证记录](../../validation/eddy/README.md)。

## 输入与 FSL 命令

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

当前版本要求所有 volume 的 PE 向量共用同一个空间轴；符号和读出时间可以由 `index` 分别指定。b0 和 DWI 都必须存在。`topup=None` 适用于没有可用反向 PE 采集的情况，不等于已经完成 susceptibility 校正。

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

FNIT 命令行调用同一组输入：

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

这些命令行参数与上表的同名 Python 参数一一对应。`--device cuda:0` 指定第一张可见 GPU；缺少反向 PE 数据时省略 `--topup`。FSL 的固定算法参数已经写入 `EDDYConfig` 默认值，因而无需在这条 FNIT 命令中重复。

## Python 调用

```python
from fnit import TorchEDDY

eddy = TorchEDDY(
    device="cuda:0",  # 计算设备；此基准在 H100 上运行
)
result = eddy.run(
    imain="AP.nii.gz",  # 输入：单被试 4D DWI
    mask="nodif_brain_mask.nii.gz",  # 输入：DWI 网格的 3D 脑掩膜
    acqp="acqparams.txt",  # 输入：PE 方向及总读出时间
    index="eddy_index.txt",  # 输入：各 volume 对应的 acqp 行号
    bvecs="AP.bvec",  # 输入：3×N 梯度方向
    bvals="AP.bval",  # 输入：N 个 b-value
    topup="fieldmap_out",  # 输入：TOPUP 结果前缀；无 TOPUP 时设 None
    ref_scan_no=0,  # 输入：从 0 开始的参考 volume 编号
    gp_seed=12345,  # 输入：GP 选点随机种子；可复核比较时固定
    out="eddy/data",  # 输出：FSL 风格文件前缀
    overwrite=False,  # 写盘策略：已有输出时不覆盖
)
```

`eddy(...)` 返回内存中的 `EDDYResult`，`eddy.run(...)` 还会调用 `result.save(out)` 写盘。`result.corrected` 是 NIfTI 影像；其余数值结果是 NumPy 数组。`EDDYConfig` 可用于检查固定参数并传入 `TorchEDDY(device="cuda:0", config=EDDYConfig(...))`；本页对照使用默认参数。

若数据已有 UKB 风格的 `AP.nii.gz`、`AP.bval`、`AP.bvec` 和 TOPUP 目录，可调用 `run_ukb_eddy(raw_dir=..., topup_dir=..., output_dir=..., device="cuda:0", overwrite=False)`。它为这一个被试生成 `eddy_index.txt` 和脑掩膜，返回 `(result, inputs)`；`inputs` 是实际传给 EDDY 的输入路径及参考帧。对应单被试命令为 `fnit eddy --raw-dir subject/raw --topup-dir subject/topup --output-dir subject/eddy --device cuda:0`。目录准备规则见[源码说明](../../src/fnit/eddy/README.md)。

## 输出文件

设 `out="eddy/data"`，默认生成下列文件。所有矩阵的行次序与输入 DWI 的 volume 次序相同。

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

默认 NIfTI 输出扩展名随 `FSLOUTPUTTYPE` 变化；本页基准使用 `.nii.gz`。以上是本命令共享的核心输出；FSL 的命令快照等辅助文件不属于 FNIT 输出合同。

## 实测对照

同一真实 UKB 格式 AP/PA 病例包含 105 个 volume（5 个 b0、50 个 b≈1000、50 个 b≈2000），使用同一 AP、脑掩膜、TOPUP 系数、acqp、index、bval、bvec、参考帧及 GP 种子。基准软件为 gpucw1 的 FSL 6.0.7.4 `eddy_cuda10.2`，FNIT 在同节点 H100 GPU 上运行。指标在同一脑掩膜内计算；过程和机器可读结果见[验证记录](../../validation/eddy/README.md)。

| 指标 | 本版 FNIT 对 FSL GPU |
|---|---:|
| 4D 脑内 Pearson r | 0.999738 |
| 4D 脑内 MAE，原图像强度单位 | 19.47 |
| b≈1000 / b≈2000 每体素跨方向 r 中位数 | 0.996331 / 0.998301 |
| 平移 / 旋转参数 MAE | 0.03590 mm / 0.0003973 rad |
| 旋转 b-vector 平均夹角 | 0.03842° |
| 官方离群切片召回 | 14/15，即 93.3% |
| 实测 wall time | FSL GPU 10:21.19（启动至写盘）；FNIT GPU 10:38.75（CUDA 初始化后调用至写盘） |

数值由完整 4D 影像计算。两侧在共同脑掩膜内的非零输出掩膜相同，但离群切片图不完全相同。FNIT 的 cubic B-spline 权重改用无布尔索引的分段计算后，完整八轮输出与修改前逐值相同，NIfTI 的 SHA-256 也相同。单轮真实病例在同一 GPU、固定种子下的两次计时为修改前 249.68 s、修改后 147.69 s；共享 GPU 负载未隔离。表中的 FSL 与 FNIT 计时边界不同，不能据此计算稳定加速比。本版 CUDA allocation 峰值为 4.51 GiB。对 FSL 的实测误差仍超过浮点舍入，不能描述为逐体素完全相同。新版的真实切片对照图已在计算节点生成，公开图像尚待授权；旧版图不对应当前代码，已移除。

## Reference

- 参考文献：Andersson & Sotiropoulos, *An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging*, NeuroImage (2016), [doi:10.1016/j.neuroimage.2015.10.019](https://doi.org/10.1016/j.neuroimage.2015.10.019)。
- 原实现代码库：[FSL `eddy`](https://git.fmrib.ox.ac.uk/fsl/eddy)。
