# 公开配对数据的 GMWMI 播种与 ACT 追踪对照

本阶段使用 OpenNeuro ds004666 同一例配对 T1w/DWI。两种实现读取相同的 MRtrix 归一化 WM FOD（`104×104×72×45`）、FreeSurfer 派生 5TT（`256³×5`）和 `5tt2gmwmi` 播种图（`256³`）。5TT/GMWMI 的体素约 1 mm，FOD 体素约 2 mm。参考 `.mif` 仅用 `mrconvert` 转为 NIfTI，没有重采样。[追踪报告](tracking_act_arc16_batch8192.public.json)和[播种报告](tracking_act_seedtest.public.json)记录输入 SHA-256、参数和指标。

## 输入、输出与用法

`sample_gmwmi_seeds` 接收同一 T1 网格的 float32 GMWMI `[A,B,C]`、ACT 顺序（cGM、sGM、WM、CSF、path）的 float32 5TT `[A,B,C,5]`、体素中心至 RAS 毫米的 `4×4` 仿射矩阵、播种尝试数和 PyTorch 随机数生成器；输出同设备 float32 `[N,3]` 的 RAS 毫米种子点。对应原程序 `tckgen -algorithm Seedtest -seed_gmwmi gmwmi.mif -act 5tt.mif -seeds 10000 -select 0 -output_seeds seeds.txt fod.mif seedtest.tck`。

`probabilistic_tractography` 接收 float32 WM SH `[X,Y,Z,C]`（`lmax=8` 时 `C=45`）及其仿射、5TT 及其仿射、GMWMI 和尝试次数。FOD 与解剖图可有不同网格；可选 FA 须在 FOD 网格。输出 `Tractogram`：`paths` 是长度可变的 `[Pi,3]` RAS 毫米轨迹元组；`endpoints` 为 `[T,2,3]`，`lengths_mm`、可选 `mean_fa` 为 `[T]`，`accepted_seeds` 为 `[T,3]`，`seeds_attempted` 为尝试次数。对应原命令：

```bash
tckgen -algorithm iFOD2 -seed_gmwmi gmwmi.mif -act 5tt.mif \
  -seeds 10000 -select 0 -maxlength 250 -cutoff 0.1 -samples 3 \
  -power 0.5 fod.mif tracks.tck
```

以下示例要求调用方先用 nibabel 读取 NIfTI，再将图像与仿射放到同一 PyTorch 设备。每个参数都以名称传递；`torch_generator` 必须在相同设备上创建。

```python
import torch
from fnit.connectome.tracking import sample_gmwmi_seeds, probabilistic_tractography

# fod_sh: [X,Y,Z,45]，归一化 WM SH，float32。
# fod_vox2ras: [4,4]，FOD 体素中心到 RAS 毫米。
# act_5tt: [A,B,C,5]，cGM/sGM/WM/CSF/path，float32。
# act_vox2ras: [4,4]，5TT 体素中心到 RAS 毫米。
# seed_map: [A,B,C]，与 5TT 同网格的 GMWMI 权重，float32。
# fa_fod: [X,Y,Z]，可选 FA，float32；无该图时传 None。
torch_generator = torch.Generator(device=fod_sh.device).manual_seed(0)
seed_points = sample_gmwmi_seeds(
    gmwmi=seed_map,                 # 输入：GMWMI 播种权重
    five_tissue=act_5tt,           # 输入：ACT 五组织图
    affine=act_vox2ras,            # 输入：解剖网格仿射，单位 mm
    n_seeds=10_000,                # 输入：尝试的种子数
    generator=torch_generator,     # 输入：同设备 PyTorch 随机数流
)
tracks = probabilistic_tractography(
    wm_sh=fod_sh,                  # 输入：WM SH 系数
    fod_affine=fod_vox2ras,        # 输入：FOD 网格仿射
    five_tissue=act_5tt,           # 输入：ACT 五组织图
    five_tissue_affine=act_vox2ras,# 输入：解剖网格仿射
    gmwmi=seed_map,                # 输入：GMWMI 播种权重
    n_seeds=10_000,                # 输入：尝试次数
    lmax=8,                       # 输入：WM SH 的最高偶数阶
    fa=fa_fod,                    # 输入：可选 FOD 网格 FA
    seed=0,                       # 输入：PyTorch 随机种子
    batch_size=8192,              # 输入：每批最多处理的种子数
    sphere_samples=128,           # 输入：初始方向离散球面采样数
    arc_proposals=16,             # 输入：每步弧线方向候选数
    max_length_mm=250.0,          # 输入：流线最大长度，mm
    min_length_mm=None,           # 输入：None 使用两倍最小 FOD 体素尺寸
    step_mm=None,                 # 输入：None 使用半个最小 FOD 体素尺寸
    max_angle_degrees=45.0,       # 输入：每步最大偏转角，度
    cutoff=0.1,                   # 输入：FOD 终止阈值
    power=0.5,                    # 输入：FOD 权重指数
)
# seed_points: [10000,3]；tracks.paths/endpoints/lengths_mm 等见上文。
```

CUDA 默认允许 TF32；几何中的高精度矩阵计算使用 float64，未用 float16。此例 PyTorch 追踪峰值已分配显存 0.74 GiB。播种函数按 GMWMI 权重抽样，把候选点投影到连续 5TT 的灰白质边界，再沿切向扰动并重新投影。追踪函数在 45° 锥内抽样方向，以弧线中点和端点的 FOD 评分；本次使用每步 16 个候选、128 个初始球面方向。[MRtrix 原 iFOD2 源码](https://github.com/MRtrix3/mrtrix3/blob/3.0.3/src/dwi/tractography/algorithms/iFOD2.h)使用校准拒绝抽样。当前实现的有限候选采样与 sGM 终止规则仍有差异；后续[同输入 sGM 单项对照](tracking_act_sgm_ab_20260927.md)记录了改动和结果。

## 真实数据：播种

相同输入下，MRtrix `Seedtest` 两次均尝试 10,000 次，分别输出 9,999、9,998 个点；8 CPU 线程耗时 0.38、0.41 s。为做等样本量分布比较，PyTorch 函数本次显式设置 `n_seeds=9999`，输出 9,999 个点，H100 用时 0.75 s、峰值已分配显存 0.425 GiB；这项计时不是同为 10,000 次尝试的比较。不同程序的随机序列和点的顺序不同，因此比较空间分布。

| 指标 | PyTorch 对 MRtrix | MRtrix 两次独立运行 |
|---|---:|---:|
| 占据区域并集上的 8 mm 空间分箱 Pearson r | 0.4660 | 0.4664 |
| 最近点距离中位数 / 第 90 百分位 | 2.07 / 3.56 mm | 2.09 / 3.57 mm |
| 种子处 GM–WM 分数绝对差中位数 / 第 90 百分位 | 0.000377 / 0.00416 | 0.000277 / 0.003998（首轮 MRtrix） |

这些播种分布指标落在 MRtrix 自身重复的波动范围内；坐标并不逐点相同。

## 真实数据：独立追踪

官方 MRtrix 3.0.3 以 `-nthreads 8` 在 10,000 次尝试中接受 2,767 条流线，用时 2.96 s，最大 RSS 222,832 KiB。PyTorch 设置 `lmax=8`、`batch_size=8192`、16 个连续锥形候选、随机种子 0，接受 2,950 条，用时 58.45 s；接受数高 6.6%，运行时间为官方的 19.7 倍。两者均在同一节点计时，但 PyTorch 包含内存中播种与追踪，MRtrix 包含图像读取和 TCK 写出，计时边界不完全相同。

| 指标 | MRtrix | PyTorch |
|---|---:|---:|
| 长度中位数 / 第 90 百分位 | 26.42 / 102.66 mm | 22.75 / 96.92 mm |
| 长度分布 KS 统计量 | — | 0.0577 |
| 到另一组轨迹端点的最近距离，中位数 / 第 90 百分位 | 2.43 / 4.72 mm（MRtrix→PyTorch） | 2.46 / 4.66 mm（PyTorch→MRtrix） |

这是两组独立随机轨迹；连接数、长度与端点差异可改变矩阵的连接支持。58.45 s 的本次追踪时间明显偏慢，弧线插值与 SH 评分适合进一步评估 Conda 内编译的 C++/CUDA 核心。当前结果尚未匹配官方 iFOD2 或完整 connectome。

![真实数据同输入的轴位轨迹密度和流线长度比较](tracking_act_density.png)

## 复跑

上述 JSON 保留输入哈希、软件版本、参数与指标。使用相同的 NIfTI 和 MRtrix `-output_seeds` 文本运行：

```bash
python tools/benchmark_connectome_tracking_act_seeds.py \
  --five-tissue five_tissue.nii.gz --gmwmi gmwmi.nii.gz \
  --reference-seeds official_seedtest_seeds.txt \
  --reference-repeat-seeds official_seedtest_seeds_repeat.txt \
  --output tracking_act_seedtest.public.json --device cuda:0
python tools/benchmark_connectome_tracking_act.py \
  --fod wm_fod_norm.nii.gz --five-tissue five_tissue.nii.gz \
  --gmwmi gmwmi.nii.gz --reference official_tracks.tck \
  --reference-seeds official_successful_seeds.txt \
  --output tracking_act_arc16_batch8192.public.json \
  --figure tracking_act_density.png --device cuda:0
```
