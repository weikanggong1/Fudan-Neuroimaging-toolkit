# TorchBEDPOSTX：扩散方向后验估计

[返回首页](../../README.md) · [源码目录](../../src/fnit/bedpostx/) · [验证记录](../../validation/bedpostx/README.md)

`TorchBEDPOSTX` 从单被试扩散 MRI 中估计最多三条交叉纤维的方向、体积分数和不确定性。模型采用 ball-and-stick 信号、Gaussian 残差、次要纤维稀疏先验和 Metropolis 采样；默认 `model=2` 用 Gamma 分布描述多壳层扩散率。CPU 和 CUDA 均使用 float32，CUDA 默认允许 TF32，不使用 float16 或 bfloat16。运行时不调用 FSL。

## 输入

`subject_dir` 必须包含以下文件：

```text
subject/
├── data.nii.gz                 # 4D、已完成运动/涡流/磁敏感校正的 DWI
├── nodif_brain_mask.nii.gz     # 与 DWI 前三维和 affine 一致的 3D 脑掩膜
├── bvals                       # N 个 b-value
└── bvecs                       # 3×N 或 N×3；通常使用 EDDY 旋转后的方向
```

| Python 参数 | 类型与默认值 | 含义 |
|---|---|---|
| `subject_dir` | 路径，必需 | 上述单被试输入目录。 |
| `output_dir` | 路径或 `None` | 输出目录；省略时为 `<subject_dir>.bedpostX`。 |
| `device` | `"cpu"`、`"cuda"` 或 `"cuda:N"` | PyTorch 设备。 |
| `threads` | 正整数或 `None` | CPU 线程数。 |
| `nfibres` | `3` | 每个体素拟合的最大纤维数。 |
| `model` | `2` | `1` 为单扩散率，`2` 为 Gamma 扩散率。 |
| `burnin` | `1000` | burn-in 跳数。 |
| `njumps` | `1250` | burn-in 后的采样跳数。 |
| `sample_every` | `25` | 每隔多少跳保存一次；默认产生 50 个样本。 |
| `ard_weight` | `1.0` | 次要纤维的 ARD 权重。 |
| `chunk_size` | 实现默认值 | 一次并行处理的 mask 体素数，用于控制显存。 |
| `seed` | 实现默认值 | 随机种子。 |
| `overwrite` | `False` | 是否覆盖非空输出目录。 |

## Python 单被试调用

```python
from fnit.bedpostx import TorchBEDPOSTX

model = TorchBEDPOSTX(
    device="cuda:0",  # 运行设备：第一张可见 CUDA GPU
    threads=1,  # CPU 线程：输入输出和辅助计算使用 1 线程
    nfibres=3,  # 模型：每个体素最多拟合三条纤维
    model=2,  # 模型：使用多壳层 Gamma 扩散率
    burnin=1000,  # MCMC：burn-in 跳数
    njumps=1250,  # MCMC：burn-in 后的采样跳数
    sample_every=25,  # MCMC：每 25 跳保存一次
    ard_weight=1.0,  # 先验：次要纤维 ARD 权重
    chunk_size=4096,  # 资源：每批并行处理的 mask 体素数
    seed=8665904,  # 随机性：固定种子便于复现
)
result = model(
    subject_dir="/absolute/path/subject",  # 输入：含 DWI、mask、bval、bvec 的目录
    output_dir="/absolute/path/subject.bedpostX",  # 输出：后验样本目录
    overwrite=False,  # 写盘策略：不覆盖已有结果
)
```

`result.output_dir` 是输出目录，`result.nvoxels` 是参与拟合的 mask 体素数，`result.nsamples` 是每个体素保存的后验样本数。`result.elapsed_seconds` 记录完整调用墙钟时间。

## 命令行与原软件对应

FNIT 单被试命令：

```bash
fnit bedpostx \
  --subject-dir /absolute/path/subject \
  --output-dir /absolute/path/subject.bedpostX \
  --device cuda:0 \
  --nfibres 3 --model 2 \
  --burnin 1000 --njumps 1250 --sample-every 25 \
  --ard-weight 1 --chunk-size 4096 --seed 8665904 --threads 1
```

独立入口 `fnit-bedpostx` 接受同一组参数。对应的 FSL wrapper 调用为：

```bash
bedpostx /absolute/path/subject \
  --nf=3 --model=2 --fudge=1 \
  --bi=1000 --nj=1250 --se=25
```

两条命令读取同一目录结构。`--nf/--model/--fudge/--bi/--nj/--se` 分别对应 `nfibres/model/ard_weight/burnin/njumps/sample_every`。FNIT 另有 `device`、`chunk_size`、`seed`、`threads` 和显式输出目录；随机数生成器、初始化和并行归约顺序与 FSL 不同。

## 输出与结构

```text
subject.bedpostX/
├── merged_th1samples.nii.gz    # 第1条纤维的 theta 后验，[X,Y,Z,Nsample]
├── merged_ph1samples.nii.gz    # 第1条纤维的 phi 后验，[X,Y,Z,Nsample]
├── merged_f1samples.nii.gz     # 第1条纤维的分数后验，[X,Y,Z,Nsample]
├── ...                         # 按 nfibres 重复 th/ph/f 文件
├── mean_f1samples.nii.gz       # 后验平均纤维分数，[X,Y,Z]
├── dyads1.nii.gz               # 后验平均方向轴，[X,Y,Z,3]
├── mean_dsamples.nii.gz        # 平均扩散率，[X,Y,Z]
├── mean_d_stdsamples.nii.gz    # model=2 的扩散率标准差，[X,Y,Z]
├── mean_S0samples.nii.gz       # 平均基线信号，[X,Y,Z]
├── nodif_brain_mask.nii.gz     # 追踪互操作所需的脑掩膜
└── run.json                    # 参数、体素数、样本数和时间
```

各纤维按体素内后验平均分数降序排列。方向是轴而非有向向量，比较时应使用符号不变夹角。文件名与 FSL `probtrackx2` 读取的 BEDPOSTX 后验一致；轨迹计数受 seed 体素数和每体素样本数影响，不能直接解释成解剖连接概率。

## 与 FSL 的真实数据对照

当前报告使用一例真实 UK Biobank dMRI 的两个诊断裁剪：14 个弱纤维体素，以及从 FSL 后验图中富集得到的 64 个交叉纤维体素。标准设置为 model 2、三纤维、ARD=1、1000/1250 跳、每 25 跳保存一次和 seed 8665904。FSL 6.0.7.22 在 CPU 运行 `xfibres --cnonlinear`；当前 Torch 代码分别在单线程 CPU 和一张 H100 上运行。源码 hash 与报告一致。

| 当前 GPU 对 FSL CPU | MAE | Pearson r | 其他结果 |
|---|---:|---:|---|
| 第一纤维平均分数 | 0.01088 | 0.99918 | 主轴夹角中位数 0.66° |
| 第二纤维平均分数 | 0.01282 | 0.52640 | 双方均无分数 ≥0.1 的共同支持体素 |
| 第三纤维平均分数 | 0.004538 | 0.58033 | 双方均无分数 ≥0.1 的共同支持体素 |
| 平均扩散率 | 0.0001170 mm²/s | 0.96080 | — |
| 扩散率标准差 | 0.0002166 mm²/s | 0.52974 | 短链估计不稳定 |

| 同机 14 体素完整调用 | 时间 |
|---|---:|
| FSL CPU | 10.99 s |
| Torch CPU，1 线程 | 22.54 s |
| Torch H100 GPU | 39.91 s |

在 64 体素富集裁剪中，Torch GPU 对 fresh FSL 的 f2/f3 全体素相关为 0.741/0.572；双方共同满足分数 ≥0.1 时为 0.726（61 体素）/0.801（25 体素），主轴夹角中位数为 5.44°/8.30°。长链控制表明接近零的次要分数和 `d_std` 对初始化、随机种子和链长敏感；这些裁剪不是无偏全脑精度估计。

![真实 UK Biobank dMRI 诊断 ROI 中 FSL 与 FNIT 的第二、第三纤维方向轴](figures/bedpostx_real_ukb_direction_axes.png)

图中每行分别为 f2 和 f3。左列是 FSL `xfibres`，中列是当前 FNIT H100 输出，颜色表示后验平均纤维分数；右列按体素显示符号不变的锐角差。方向轴没有正负之分，因此绘图前将 FNIT 轴翻转到与 FSL 同一半球。图中只显示双方分数均 ≥0.1 的体素，f2 为 61 个、f3 为 25 个；对应夹角中位数 5.44° 和 8.30°，与机器报告逐项一致。

仓库已公开这份 64 体素裁剪的去标识方向轴图，原始 DWI 和完整后验仍保留在授权服务器。该裁剪按 FSL 后验估计富集，只用于检查受支持的次要纤维；它不能代表全脑无偏精度。机器可读指标和这一适用范围见 [`report.public.json`](../../validation/bedpostx/report.public.json)。

## 合成回归示例

[`synthetic_example.py`](synthetic_example.py) 生成 8×8×1 的已知双交叉纤维信号；[`synthetic_example.png`](synthetic_example.png) 只用于检查模型和画图流程。64 个模拟体素中，Torch GPU 对真值的 f1/f2 MAE 为 0.004268/0.004290，对 FSL 的 MAE 为 0.000955/0.001042，运行 30.81 s。该结果不能替代真实数据验收。

## 来源与限制

该实现复现已发表模型和采样算法，并非 FSL `xfibres` C++ 的逐句翻译。tensor 初始化、随机数流、浮点归约和 proposal history 均不同，因此不声明后验体积逐元素等价。方法与许可见 [FSL BEDPOSTX 文档](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/bedpostx.html)、Behrens et al. (NeuroImage, 2007)、Jbabdi et al. (MRM, 2012) 以及 [FSL 软件许可](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)。
