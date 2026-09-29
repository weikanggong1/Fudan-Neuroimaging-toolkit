# EPI→T1 边界配准（BBR）

`register_bbr` 将一张 3D EPI 参考影像配准到同被试的 3D T1。函数在 T1 白质边界的内外各 2 mm 处采样 EPI 强度，优化 6 自由度刚体变换。边界目标采用 FSL FLIRT 的有符号 `tanh` 代价；无场图时不估计或校正 EPI 畸变。[FSL BBR 说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/bbr.html)解释了白质边界与 EPI 灰白质强度对比的要求；[epi_reg 说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/epi_reg.html)列出 T1、去颅骨 T1、白质分割等输入。

## 输入、返回值与坐标

| 参数 | 作用 |
|---|---|
| `epi` | 待配准的 **3D** EPI/SBRef NIfTI 路径或 NiBabel 影像；不接收 4D BOLD。输出变换的起点。 |
| `t1` | 同被试的 **3D** T1 NIfTI 路径或 NiBabel 影像；定义输出网格。建议先用 FNIT SynthStrip 去颅骨。 |
| `wmseg` | T1 网格的 3D 白质二值分割；形状及 affine 须与 `t1` 一致。可用 FNIT `TorchFAST` 的白质部分容积图 `pve_wm≥0.5` 生成。 |
| `init` | 可选的 EPI→T1、**FLIRT scaled-mm** 4×4 初始矩阵，可传文本文件路径或 NumPy 数组。`None` 时调用现有 FNIT `TorchFLIRT` 6 自由度 normmi 求初始矩阵，此时 `epi`、`t1` 必须是文件路径。 |
| `device` | `"cuda:0"`、`"cpu"` 等 PyTorch 设备；`None` 时优先 CUDA。GPU 默认允许 TF32，未使用 float16。 |
| `grid_search` | 是否先做 FSL `bbr.sch` 风格的粗网格搜索；默认 `True`。 |

返回 `BBRResult`，包含 `moved`（T1 网格、float32 的 3D NiBabel NIfTI）、`matrix`（EPI→T1 的 FLIRT scaled-mm 4×4）、`moving_to_fixed_world`（同一变换的 RAS world 4×4）、`initial_cost`、`final_cost`、`boundary_points` 和 `runtime_seconds`。`save(output=..., omat=...)` 分别写配准后的 NIfTI 和 FLIRT 格式文本矩阵。`matrix` 可直接交给接受 FLIRT `.mat` 的 FNIT 重采样函数；不能把它当作 NIfTI affine 或 RAS world 矩阵使用。应用到 4D BOLD 时，应把 BBR 与每帧运动矩阵及后续空间形变组合后只重采样一次。

```python
from fnit.fmri.bbr import register_bbr

result = register_bbr(
    epi="/absolute/path/example_func.nii.gz",   # 3D EPI/SBRef，待配准影像
    t1="/absolute/path/T1_brain.nii.gz",         # 3D 去颅骨 T1，目标空间与网格
    wmseg="/absolute/path/T1_wmseg.nii.gz",      # T1 网格白质二值掩膜
    init=None,                                    # EPI→T1 FLIRT 4×4 初始矩阵；None 自动估计
    device="cuda:0",                             # PyTorch 设备；无 GPU 可用 "cpu"
    grid_search=True,                             # 做粗网格搜索后再优化边界代价
)
result.save(
    output="/absolute/path/example_func2T1.nii.gz",  # T1 网格中的 3D EPI 输出文件
    omat="/absolute/path/example_func2T1.mat",        # EPI→T1 FLIRT scaled-mm 矩阵文件
)
print(result.moving_to_fixed_world)               # EPI→T1 RAS world 4×4，供 RAS 变换链使用
print(result.final_cost)                          # 配准后的 BBR 代价值；越低越好
```

该函数只做刚体 BBR，不包括 SynthStrip、FAST、B0 场图校正、GDC 或 T1→MNI 的非线性配准。上游若已有 T1 白质分割和初始矩阵，显式传入即可复用，不需再次估计。

## 官方同输入命令

以下命令与上例使用同一张 EPI、去颅骨 T1 和白质边界；`fast` 与 `fslmaths` 两步只用于生成官方对照的分割。`bbr.sch` 来自所安装 FSL。该命令**不传场图，也不执行 GDC**。

```bash
fast -o t1_fast T1_brain.nii.gz
fslmaths t1_fast_pve_2 -thr 0.5 -bin T1_wmseg.nii.gz
flirt -in example_func.nii.gz -ref T1_brain.nii.gz \
  -dof 6 -cost normmi -omat init.mat
flirt -in example_func.nii.gz -ref T1_brain.nii.gz \
  -dof 6 -cost bbr -wmseg T1_wmseg.nii.gz \
  -init init.mat -schedule "$FSLDIR/etc/flirtsch/bbr.sch" \
  -omat example_func2T1.mat -out example_func2T1.nii.gz
```

官方 [FLIRT BBR](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/bbr.html)使用白质边界法线两侧的 EPI 强度，默认采样距离为 2 mm。FNIT 的边界点生成、粗搜索、局部 Powell 优化和三线性重采样独立实现；优化轨迹、边界法线的离散计算及边缘插值不保证逐值等同于 FSL。

## 真实 UKB 数据对照

使用同一次真实静息态采集的 3D EPI 参考影像与同被试去颅骨 T1；两套实现均跳过 GDC 和 B0 场图校正。比较只包含这一例。计时均在 gpucw1 进行；FSL 使用 CPU，FNIT 使用 CUDA。下表第一组固定**完全相同的 FSL FAST 白质掩膜和 FSL normmi 初始矩阵**，因此单独比较 BBR 步骤。矩阵误差为在 T1 脑内采样点，把 T1 点逆变换到 EPI 后的两套坐标距离。影像指标在 T1 脑掩膜内计算，强度单位为原 EPI 值。

| 指标 | FNIT BBR | FSL FLIRT BBR |
|---|---:|---:|
| BBR 耗时 | 3.73 s | 66.84 s |
| 峰值内存 | GPU 0.060 GB | CPU RSS 0.260 GB |
| 变换位移差 | 中位 0.180 mm；95 百分位 0.301 mm；最大 0.361 mm | 参照 |
| 配准后影像 | 与 FSL Pearson r=0.998683；平均绝对差 168.94 | 参照 |
| FNIT 边界代价 | 0.33050 | FSL 矩阵代入 FNIT 代价为 0.33074 |

默认组织分割链另以同一 T1 比较：FNIT `TorchFAST` 的白质部分容积阈值 `≥0.5` 与 FSL FAST 白质掩膜 Dice=0.995177。FNIT FAST 耗时 1.995 s，随后 FNIT BBR 耗时 2.011 s，合计峰值 GPU 显存 1.325 GB；FSL FAST 耗时 211.51 s。此组**仍共用 FSL normmi 初始矩阵**，FNIT FAST+BBR 对 FSL FAST+BBR 最终变换的位移差中位 0.075 mm、95 百分位 0.107 mm。FSL normmi 初始化另耗时 10.61 s；这些 FAST/BBR 时间均不包含 EPI 的运动估计、AROMA、MNI 变换或文件解压。

先前的完整配准链用 **FNIT 自身的 FAST 白质分割与 TorchFLIRT normmi 初始矩阵**测量：FNIT FAST 2.582 s、TorchFLIRT 初始化 47.604 s、BBR 4.756 s，三步串行合计 54.94 s；官方对应三步合计 288.96 s。两套初始矩阵的逆变换位移差中位 0.848 mm、95 百分位 1.372 mm；经 BBR 后，对官方最终矩阵的差异降至中位 0.099 mm、95 百分位 0.148 mm。T1 脑掩膜内配准影像 Pearson r=0.999605、平均绝对差 96.41 原强度单位。整条 FNIT 链的 GPU 显存峰值为已分配 1.325 GB、已预留 1.904 GB。该链测量时 `flirt/core.py` 的 SHA256 为 `d590686ce9970b67cf29c8f762d3cdc19113d2c6a2af8b41e9a6382d2b9ee042`；本次 FLIRT 修订后的 6-DOF 求解结果见[当前独立配准对照](../flirt/README.md)，**上述 BBR 整链数值尚未用修订后的初始矩阵重跑**。这次旧链计时未预热且 GPU 非独占；FSL 计时来自另一次 CPU 运行，不能计算稳定加速比。

原始场图缺失，无法用这组数据评价带畸变校正的 BBR；去颅骨 T1 也不能代表原始 T1 采集。FNIT 与 FSL 的代价函数和优化结果接近，但未达到逐值相同。所有公开数值仅为汇总标量，见 [`bbr_summary.json`](../../validation/fmri/bbr_summary.json)。
