# PyTorch ProbtrackX：概率纤维束追踪

`fnit.probtrackx.TorchProbtrackX` 从 BEDPOSTX 方向后验生成两类体积空间结果：单个 seed 掩膜到全脑的 `fdt_paths.nii.gz`，以及多个 ROI 之间的有向 `fdt_network_matrix`。运行时不调用 FSL。输入必须已处于同一个扩散影像网格；本功能不替用户配准图谱。

## Python 调用

```python
from fnit import TorchProbtrackX

tracker = TorchProbtrackX(device="cuda:0", nsamples=5000, nsteps=2000)
seed_result = tracker.run(
    "/absolute/path/subject.bedpostX",
    "/absolute/path/seed_to_voxel",
    seed="/absolute/path/seed.nii.gz",
)
network_result = tracker.run(
    "/absolute/path/subject.bedpostX",
    "/absolute/path/network",
    regions=["/absolute/path/roi_01.nii.gz", "/absolute/path/roi_02.nii.gz"],
)
print(seed_result.paths, network_result.network_matrix)
```

`run()` 恰好接收 `seed` 或 `regions` 之一；`regions` 至少含两个互不重叠的 ROI。每次只处理一个被试。`samples_dir` 需要 FSL BEDPOSTX 格式的 `nodif_brain_mask.nii.gz` 与每条纤维对应的 `merged_th<i>samples.nii.gz`、`merged_ph<i>samples.nii.gz`、`merged_f<i>samples.nii.gz`。所有 ROI 的三维形状和 affine 都必须与后验 mask 一致。可以使用本包 `TorchBEDPOSTX` 或原版 FSL bedpostx 输出的这些文件。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `device` | `cpu` | `cpu` 或 `cuda:0` 等 PyTorch 设备 |
| `nsamples` | 5000 | 每个 seed 体素的 Monte Carlo 轨迹数 |
| `nsteps` | 2000 | 双向总步数，各方向最多一半；必须为偶数 |
| `steplength` | 0.5 mm | 每步的物理长度 |
| `cthr` | 0.2 | 相邻方向绝对点积的下限 |
| `fibthresh` | 0.01 | 次要纤维后验体积分数阈值 |
| `batch_size` | 256 | 同时追踪的轨迹数 |
| `seed` | 12345 | PyTorch 随机数种子 |

返回的 `ProbTrackXResult` 提供输出路径、seed 体素数、接受的轨迹数和包含读写的总运行时间。若输出文件已存在，`run()` 默认报错；明确传 `overwrite=True` 才覆盖。GPU 路径使用 float32，允许 TF32 matmul 和 cuDNN。

## 命令行

```bash
fnit probtrackx \
  --samples-dir /absolute/path/subject.bedpostX \
  --seed /absolute/path/seed.nii.gz \
  --output-dir /absolute/path/seed_to_voxel \
  --device cuda:0
```

网络模式把 ROI 路径每行写入一个 UTF-8 文本文件，行顺序就是矩阵行列顺序；相对路径相对于列表文件解析：

```bash
fnit probtrackx \
  --samples-dir /absolute/path/subject.bedpostX \
  --roi-list /absolute/path/roi_list.txt \
  --output-dir /absolute/path/network \
  --device cuda:0
```

独立入口 `fnit-probtrackx` 接受相同参数。`waytotal` 在单 seed 模式是一个整数，在网络模式是每个源 ROI 一行的整数。`fdt_paths.nii.gz` 对每条接受的双向轨迹经过的每个体素计数一次。网络矩阵第 *i* 行第 *j* 列记录从 ROI *i* 发出的轨迹中到达 ROI *j* 的条数；同一轨迹多次进入 ROI *j* 仍只计一次，主对角为零。原始计数随 ROI 体素数和 `nsamples` 增长，不能直接解释为标准化连接概率。

## 与 FSL 对应的算法范围

参考实现是 FSL `probtrackx2` 的体积 seed、`--opd` 和 `--network` 默认路径：在连续轨迹位置按三线性权重随机抽相邻体素，再随机抽一个 BEDPOSTX 后验样本；初始使用第一纤维，后续在高于 `fibthresh` 的纤维中选择与当前方向最接近者；在 mask 外或曲率不合格时停止，并对两个方向合并计数。FSL 默认 5000 条/seed 体素、0.5 mm 步长、2000 总步数和 0.2 曲率阈值。不同随机数发生器与并行采样顺序使两次运行不应逐体素完全相同；验证应比较空间分布、ROI 计数、几何和时间。

当前只支持同一扩散网格内的体积掩膜及默认简单 Euler 追踪。FSL 的 surface seed、跨空间变换、waypoint/avoid/stop、随机初始纤维、`--usef`、长度加权和 modified Euler 选项尚未实现，不能拿这些设置的 FSL 输出作同参数对照。

## 与原版 FSL 的配对 benchmark

在 gpucw1 上，用一例 UK Biobank dMRI 已有的 FSL BEDPOSTX 后验作为**共同输入**，比较 FSL 6.0.7.22 `probtrackx2` CPU 与本实现 CPU。原始后验的生成日志不可得，因此本实验验证追踪阶段，不能证明 BEDPOSTX 阶段一致。两边均对每个 seed 体素抽样 200 条轨迹、总步数 400、随机种子 20260927；本实现 `batch_size=256`、CPU 8 线程。所用参数低于默认的 5000 条与 2000 步，属于匹配的功能与数值检查。ROI 均在原生扩散网格，每个 7 体素。计时包含读写；随机数实现不同，不要求逐条轨迹一致。

| 任务 / 指标 | FSL CPU | FNIT CPU |
| --- | ---: | ---: |
| seed-to-voxel `waytotal` | 1400 | 1400 |
| seed-to-voxel 密度总和 | 108889 | 109448 |
| seed-to-voxel 非零体素 | 10310 | 10699 |
| seed-to-voxel 耗时 | 13.65 s | 29.03 s |
| region-to-region `waytotal`，ROI 1 / ROI 2 | 1207 / 1073 | 1181 / 1088 |
| region-to-region 密度总和 | 102384 | 102637 |
| region-to-region 耗时 | 14.71 s | 24.62 s |

seed-to-voxel 密度图在两者非零体素并集上的 Pearson *r* = 0.9950；网络密度图 *r* = 0.9970。网络矩阵的两个非零元素分别是 FSL 1207/1073 与 FNIT 1181/1088，即相差 2.15%/1.40%。两种密度图的非零支持 Dice 分别只有 0.644/0.652，反映低计数尾部的 Monte Carlo 差别；最高计数十分位的 Dice 分别为 0.903/0.917。完整指标见 [CPU 机器可读报告](../../validation/probtrackx/report.public.json)，实验选点和解释见 [验证记录](../../validation/probtrackx/README.md)。本例 CPU 耗时没有显示加速优势。

同一例数据在 H100 GPU1 上另做匹配测试：FSL `probtrackx2_gpu` 与 FNIT 的 seed 耗时分别为 36.75/31.34 秒，网络模式为 12.42/44.05 秒；网络矩阵分别为 `[[0,1194],[1061,0]]` 与 `[[0,1190],[1067,0]]`，密度图相关 *r*=0.9968。该 GPU 有其他任务占用，且两模式计时差异方向相反，不能由此宣称稳定的 GPU 提速。[GPU 完整指标](../../validation/probtrackx/report.gpu.public.json)。

## 合成数据可视化

![合成纤维场上 FSL 与 FNIT 的 seed-to-voxel 和网络路径密度](../../validation/probtrackx/fsl_fnit_synthetic_comparison.png)

该图由 40×40×40 的管状单纤维 posterior（每体素 20 帧）生成，不含 UK Biobank 影像。两边均采用每 seed 体素 200 条轨迹、240 总步数及随机种子 20260927。seed-to-voxel 图在非零体素并集上的 Pearson *r*=0.9875；FSL 的 ROI 1→2/2→1 计数为 1369/1368，FNIT 为 1366/1369。合成数据的 [生成脚本](../../validation/probtrackx/generate_synthetic_posterior.py)、[复现命令](../../validation/probtrackx/run_synthetic_comparison.sh)、[完整指标](../../validation/probtrackx/synthetic_reference.public.json) 和 [验证记录](../../validation/probtrackx/README.md) 均可公开复现。上面的真实数据实验仅发布汇总指标。

## 来源

- [FSL ProbtrackX 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/probtrackx.html)
- [FSL ptx2 官方源码及许可证](https://git.fmrib.ox.ac.uk/fsl/ptx2)
- [FSL 软件许可证](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)
