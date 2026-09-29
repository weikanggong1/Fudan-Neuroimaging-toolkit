# ACT 种子检查与单向播种：真实 5TT 对照

## 输入和本次修改

使用公开 ds004666 `sub-01/ses-2mm` 的校正 DWI 所生成的归一化 WM FOD、配对官方 `recon-all` 分割生成的 5TT，以及固定的 10,000 个 GMWMI 世界毫米坐标。FOD、5TT、坐标文件的 SHA-256 分别是 `32461e2e2281f60a15dfd546ad91129d7fe68582844c9d983a7602f23f7c59c6`、`32a27dabd9fa7e6094beafda8c90515642f733f1c245530b183846bf40cc5153`、`41f036eae125b83dd2b6fc7f3c916dd5da8c20915c2990df3f65cce8fb45ffe6`。影像仍在验证服务器；仓库归档种子坐标、逐点结果、矩阵、图和报告。[校正输入的来源与假定读出时间](corrected_input_provenance.public.json)及[原始图像切面](../../../docs/connectome/figures/ds004666_t1_raw_vs_topup_eddy_atlas.png)另有记录。

MRtrix3 固定源码提交 `eeab681d3e0cb004cf1d1d31579d3892197ef5b6` 的 `ACT/method.h::check_seed()`、`seed_is_unidirectional()` 规定：无效组织拒绝播种；皮层灰质侧的灰白质界面只向白质方向传播；亚皮层灰质种子仍可双向传播。方向由种子两侧各 0.001 mm 的 GM−WM 值判定。FNIT 现在实现这两个种子函数，并把单向标志用于流线拼接。后续圆弧已改为校准拒绝采样；12,600 个真实 5TT 路径采样点的状态见当前报告。

`_five_tissue_mrtrix(five_tissue, points, inverse_affine)` 接收 float32 `[X,Y,Z,5]` 五组织图、RAS 世界毫米 `[B,3]` 坐标和世界→体素 `[4,4]` 变换；返回 float32 `[B,5]`，通道顺序是皮层灰质、亚皮层灰质、白质、脑脊液、病理。超出图像、最近体素为背景时返回零；三线性权重小于 `1e-6` 时置零。MRtrix 根据 NIfTI 头文件体素尺寸规范化 sform 的方向列；仅用 nibabel 读出的浮点 sform 列长度，种子分类会在界面附近产生错误。因此直接调用公开追踪函数时，应传 `five_tissue_spacing_mm=(sx,sy,sz)`，单位 mm；`UKBConnectome` 自动从官方分割头文件传入。省略该参数时采用 affine 列长度。

`_act_seed_direction(five_tissue, seeds, directions, inverse_affine)` 接收同一 5TT、世界毫米种子 `[B,3]`、单位初始方向 `[B,3]` 及上述逆变换；返回 bool `valid[B]`、bool `one_way[B]` 和 float32 `oriented[B,3]`。`valid` 表示可播种，`one_way` 表示只沿 `oriented` 向白质传播。公开 `probabilistic_tractography(...)` 仍返回 `Tractogram`：逐流线 `[Pi,3]` 世界毫米点、`endpoints[T,2,3]`、`lengths_mm[T]`、`accepted_seeds[T,3]` 和尝试数；详细输入、输出及其他参数见[函数文档](../../../docs/connectome/README.md)。GMWMI 加权体素选择和传播候选选择改为 float64 累积分布加显式随机抽样，修复同一 GPU、同一种子跨进程矩阵偶发不同的问题。

直接调用示例；所有图像张量及仿射均已载入同一个 CUDA 设备：

```python
from fnit.connectome.tracking import probabilistic_tractography

tracks = probabilistic_tractography(
    wm_sh=wm_sh,                              # 输入：归一化 WM FOD，float32 [XF,YF,ZF,45]
    fod_affine=fod_affine,                    # 输入：FOD 体素中心→RAS 毫米，[4,4]
    five_tissue=five_tissue,                  # 输入：cGM/sGM/WM/CSF/病理 5TT，[XA,YA,ZA,5]
    five_tissue_affine=five_tissue_affine,    # 输入：5TT 体素中心→RAS 毫米，[4,4]
    gmwmi=gmwmi,                              # 输入：5TT 网格 GMWMI 播种权重，[XA,YA,ZA]
    n_seeds=10_000,                           # 输入：尝试的种子数，不等于接受流线数
    lmax=8,                                   # 输入：WM 球谐最高阶；45 个系数
    five_tissue_spacing_mm=(1.0, 1.0, 1.0),  # 输入：5TT NIfTI 头文件三轴体素尺寸，mm
    fa=None,                                  # 输入：可选同 FOD 网格 FA；正式均值另行精确采样
    seed=0,                                   # 输入：PyTorch 随机种子
    batch_size=8192,                          # 输入：每批种子数
    arc_proposals=16,                         # 输入：每轮并行计算的候选数；每弧最多拒绝采样 1000 次
    max_length_mm=250.0,                      # 输入：流线上限，mm
    min_length_mm=None,                       # 输入：None 时为 FOD 体素几何平均边长两倍
    step_mm=None,                             # 输入：None 时为 FOD 体素几何平均边长一半
    max_angle_degrees=45.0,                   # 输入：每步最大转角
    cutoff=0.1,                               # 输入：FOD 截止值
    power=0.5,                                # 输入：圆弧 FOD 概率幂次
)
# 输出：tracks.paths、endpoints、lengths_mm、accepted_seeds、seeds_attempted；
# 当 fa=None 时 tracks.mean_fa=None。
```

## 官方同输入检查

标准追踪参考命令如下。`--` 后的 `tckgen` 参数与本例参考相同；官方随机序列与 PyTorch 不相同，因此最终轨迹只比较分布和矩阵。

```bash
MRTRIX_RNG_SEED=0 tckgen wm_fod_norm.nii.gz tracks.tck \
  -algorithm iFOD2 -seed_gmwmi gmwmi.mif -act five_tissue.mif \
  -seeds 10000 -select 0 -maxlength 250 -cutoff 0.1 -power 0.5 -samples 3 -nthreads 0
```

标准 `tckgen` 不写出逐点种子分类。[仅用于 benchmark 的官方源码观测程序](../../../tools/reference/ifod2_act_seed_oracle.cpp)在该固定提交的 MRtrix 源码树中编译：复制到 `cmd/`，执行 `NUMBER_OF_PROCESSORS=4 ./build bin/ifod2_act_seed_oracle`，再运行 `bin/ifod2_act_seed_oracle wm_fod_norm.nii.gz five_tissue.mif frozen_seeds.txt official_seed0.txt > official_act_seed0_final.txt`。四个位置参数依次为 FOD、5TT、三列 RAS 毫米种子、五列官方初始方向；重定向文件写出十一列结果，依次为有效标志、单向标志、定向后的 xyz、五个组织分数和用于翻转的梯度。此观测程序不进入 FNIT 包或运行路径。PyTorch 基准命令：

```bash
python tools/benchmark_connectome_ifod2_act_seed.py \
  --five-tissue five_tissue.nii.gz \
  --seeds frozen_seeds.txt \
  --initial official_seed0.txt \
  --official official_act_seed0_final.txt \
  --output fnit_act_seed0_final.json --device cuda:0
```

`--five-tissue` 是真实五通道 NIfTI；`--seeds` 是冻结坐标；`--initial` 为同一坐标的官方初始方向；`--official` 为十一列 ACT 输出；`--output` 写汇总 JSON，同名 CSV 写逐种子的有效/单向/定向结果；`--device` 是 PyTorch 设备。采用同一官方初始方向排除了两种随机生成器的差异。结果见[逐种子报告](ifod2_act_seed_20260929/fnit_act_seed0_final.json)和[官方逐点值](ifod2_act_seed_20260929/official_act_seed0_final.txt)。

| 同一 10,000 个真实位置与初始方向 | MRtrix | FNIT |
|---|---:|---:|
| 初始方向有效 | 9,662 | 固定使用同一 9,662 个 |
| ACT 种子有效 | 9,659 | 9,659；XOR 0 |
| 皮层单向种子 | 8,430 | 8,430；XOR 0 |
| 五组织分数绝对误差 | 参考 | MAE 7.70×10⁻⁹；最大 1.79×10⁻⁷ |
| 定向不一致 | 参考 | 1 / 9,659；该点官方梯度 −2.98×10⁻⁸ |
| 已载入图像的种子检查核心 | CPU 0.0488 s | H100 0.1687 s；Torch 峰值 0.346 GiB |

官方独立进程完整墙钟 1.28 s、最大 RSS 622,912 KiB。两侧硬件与计时边界不同；本次不声称 GPU 加速。定向的单点分歧来自接近零的梯度数值符号，没有在报告中掩盖。

## 当前组合

此页只保留 ACT 种子函数的同输入检查。当前校准拒绝采样、ACT 逐点状态、随机流线和四矩阵三种子对照见[追踪报告](ifod2_rejection_20260929.md)。
