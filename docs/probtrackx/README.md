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

## 与原版 FSL 的真实 DWI 配对 benchmark

在 `gpucw1` 上，对一例真实 UK Biobank DWI 使用**同一份原版 FSL BEDPOSTX 三纤维后验**（104×104×72，50 个抽样）比较 FSL 6.0.7.22 与 FNIT 的追踪阶段。将 JHU 白质图谱用已有 MNI→DWI 变换最近邻重采样，再在胼胝体膝部、左右皮质脊髓束、左右上纵束各选 7 个位于后验 mask 内且 FA≥0.3 的 seed 体素；选点不依赖任一追踪结果。两边均用每 seed 体素 200 条轨迹、总步数 400、0.5 mm 步长、0.2 曲率阈值、0.01 次要纤维阈值和随机种子 20260927。FNIT `batch_size=256`、CPU 8 线程，GPU 使用 float32/TF32。计时包括载入后验与写盘。

| Seed 区域 | CPU 图相关 *r* | CPU 高密度前 10% Dice | CPU 耗时 FSL / FNIT | GPU 耗时 FSL / FNIT |
| --- | ---: | ---: | ---: | ---: |
| 胼胝体膝部 | 0.9938 | 0.8902 | 14.38 / 23.30 s | 13.63 / 46.55 s |
| 右皮质脊髓束 | 0.9961 | 0.8728 | 12.66 / 24.13 s | 12.75 / 31.90 s |
| 左皮质脊髓束 | 0.9967 | 0.9020 | 12.42 / 23.20 s | 12.99 / 47.54 s |
| 右上纵束 | 0.9954 | 0.8631 | 12.36 / 23.84 s | 12.22 / 44.07 s |
| 左上纵束 | 0.9970 | 0.9416 | 12.18 / 22.25 s | 12.03 / 41.37 s |

五个 seed 的两边 `waytotal` 均为 1400。CPU 路径密度总和的相对差异不超过 1.18%；GPU 图相关 *r*=0.9939–0.9972。对胼胝体 seed 将 FSL 自身随机种子改为 20260928 重跑，所得相关 *r*=0.9939、top-10% Dice=0.9012，可作为此设置下的抽样波动参照。低计数尾部的非零支持 Dice 仅为 0.658–0.697（CPU），因此也报告高密度区域重叠。当前实现没有速度优势：CPU 五次计时中位数为 FSL 12.42 s、FNIT 23.30 s；GPU1 有其他进程占用，GPU 墙钟时间只描述这次共享环境的观测。

同一批独立图谱 ROI 的 5×5 有向矩阵在每体素 200 条轨迹时跨区命中只有个位数，因此另以 **2000 条/体素**重跑 CPU 网络。右皮质脊髓束→左皮质脊髓束的计数为 FSL 53、FNIT 65；FSL 换随机种子重跑为 67。反向为 8、8、11；其余边多为零或个位数。FSL–FNIT 网络密度图相关 *r*=0.9477、高密度 Dice=0.7788；FSL 两次运行分别为 0.9262、0.7611。耗时 FSL 31.50 s、FNIT 111.62 s。矩阵中低计数边仍不稳定，不能据此解释生物连接强度。

这是一例后验共输入的追踪对照，不能检验 BEDPOSTX 拟合阶段或逐条轨迹相同。参数低于 FSL 默认的 5000 条与 2000 步；[完整逐脑区 CPU/GPU 与网络指标](../../validation/probtrackx/report.multiregion.public.json)、[复现方法和数据边界](../../validation/probtrackx/README.md)均在验证目录。真实被试图像和对照图只保存在授权服务器；下方公开示例使用合成数据。

## 合成数据可视化

![合成纤维场上 FSL 与 FNIT 的 seed-to-voxel 和网络路径密度](../../validation/probtrackx/fsl_fnit_synthetic_comparison.png)

该图由 40×40×40 的管状单纤维 posterior（每体素 20 帧）生成，不含 UK Biobank 影像。两边均采用每 seed 体素 200 条轨迹、240 总步数及随机种子 20260927。seed-to-voxel 图在非零体素并集上的 Pearson *r*=0.9875；FSL 的 ROI 1→2/2→1 计数为 1369/1368，FNIT 为 1366/1369。合成数据的 [生成脚本](../../validation/probtrackx/generate_synthetic_posterior.py)、[复现命令](../../validation/probtrackx/run_synthetic_comparison.sh)、[完整指标](../../validation/probtrackx/synthetic_reference.public.json) 和 [验证记录](../../validation/probtrackx/README.md) 均可公开复现。上面的真实数据实验仅发布汇总指标。

## 来源

- [FSL ProbtrackX 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/probtrackx.html)
- [FSL ptx2 官方源码及许可证](https://git.fmrib.ox.ac.uk/fsl/ptx2)
- [FSL 软件许可证](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)
