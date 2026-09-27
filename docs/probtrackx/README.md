# ProbtrackX 概率纤维束追踪

`TorchProbtrackX` 读取 FSL BEDPOSTX 的方向后验，独立计算体积 seed 到体素的 `fdt_paths.nii.gz`，或多个 ROI 的有向 `fdt_network_matrix`。CPU 使用 PyTorch，CUDA 使用 Triton 融合步进；轨迹和影像张量保持 float32，GPU 默认允许 TF32，独立同源码真实 DWI 网络测量的峰值已分配显存为 2.61 GiB。每次调用处理一个被试，不调用 FSL 程序。

当前实现覆盖**同一扩散网格的体积追踪路径**。表面 seed、跨空间变换、全部稀疏矩阵等官方选项仍未覆盖；完整清单见下文。所有 seed、ROI、追踪和约束掩膜须与 BEDPOSTX 后验的三维形状及 affine 一致。

## Python 调用

```python
from fnit import TorchProbtrackX

tracker = TorchProbtrackX(
    device="cuda:0", nsamples=5000, nsteps=2000, batch_size=2048,
    steplength=0.5, cthr=0.2, fibthresh=0.01, seed=12345,
)
seed_result = tracker.run(
    "/absolute/path/subject.bedpostX", "/absolute/path/seed_result",
    seed="/absolute/path/seed.nii.gz",
)
network_result = tracker.run(
    "/absolute/path/subject.bedpostX", "/absolute/path/network_result",
    regions=["/absolute/path/roi_01.nii.gz", "/absolute/path/roi_02.nii.gz"],
)
print(seed_result.paths, network_result.network_matrix)
```

`run()` 恰好接收 `seed` 或 `regions` 之一；`regions` 至少包含两个互不重叠的 ROI。`samples_dir` 需要 `nodif_brain_mask.nii.gz` 和每条纤维的 `merged_th<i>samples.nii.gz`、`merged_ph<i>samples.nii.gz`、`merged_f<i>samples.nii.gz`。已有输出默认报错，指定 `overwrite=True` 才覆盖。返回值包含路径、seed 体素数、接受的轨迹数，以及含载入和写盘的耗时。

| 构造参数 | 默认值 | 作用及 FSL 对应 |
| --- | ---: | --- |
| `device`、`batch_size` | `cpu`、2048 | 计算设备及同时处理的轨迹数；FSL 无对应批量参数 |
| `nsamples`、`nsteps` | 5000、2000 | 每 seed 体素轨迹数、双向总步数；`-P`、`-S`，`nsteps` 必须为偶数 |
| `steplength`、`cthr`、`fibthresh` | 0.5 mm、0.2、0.01 | 步长、最小方向点积、纤维分数阈值 |
| `seed` | 12345 | 随机种子；`--rseed`。PyTorch/Triton 与 FSL 的随机流不同 |
| `distthresh` | 0 mm | 每条半轨迹的最小物理长度；`--distthresh` |
| `sampvox` | 0 mm | seed 中心周围球内随机抖动；`--sampvox` |
| `fibst`、`randfib` | `None`、0 | 1 起始编号的固定纤维 `--fibst`；`--randfib` 模式 0–3。显式 `fibst` 优先 |
| `usef` | `False` | 按后验纤维体积分数随机终止；`--usef` |

`run()` 的 `mask=` 对应 FSL `-m`，默认使用 BEDPOSTX 的 `nodif_brain_mask.nii.gz`；`avoid=`、`stop=` 分别对应体积 `--avoid`、`--stop`，`forcefirststep=True` 对应 `--forcefirststep`。这些参数都传入同一网格上的 NIfTI 路径。`waytotal` 在单 seed 模式为一个整数，在网络模式为每个源 ROI 一行。每条接受轨迹对经过的每个体素最多计数一次；网络矩阵第 *i* 行第 *j* 列为从 ROI *i* 到达 ROI *j* 的轨迹数，主对角为零。

## 命令行和官方对应

```bash
fnit probtrackx --samples-dir /absolute/path/subject.bedpostX \
  --seed /absolute/path/seed.nii.gz --output-dir /absolute/path/fnit_seed \
  --device cuda:0 --nsamples 5000 --nsteps 2000
fnit probtrackx --samples-dir /absolute/path/subject.bedpostX \
  --roi-list /absolute/path/roi_list.txt --output-dir /absolute/path/fnit_network \
  --device cuda:0 --nsamples 5000 --nsteps 2000
```

`roi_list.txt` 每行一个 ROI 路径，行顺序就是矩阵行列顺序；相对路径相对于列表文件解析。独立命令 `fnit-probtrackx` 接受相同参数。体积约束示例：`--mask tracking.nii.gz --avoid exclusion.nii.gz --stop termination.nii.gz --forcefirststep --distthresh 5 --sampvox 1 --randfib 2 --usef`。

同一参数下的 FSL 命令为：

```bash
"$FSLDIR/bin/probtrackx2" -s /absolute/path/subject.bedpostX/merged \
  -m /absolute/path/subject.bedpostX/nodif_brain_mask.nii.gz \
  -x /absolute/path/seed.nii.gz --dir=/absolute/path/fsl_seed \
  --forcedir --opd -P 5000 -S 2000 --steplength=0.5 \
  --cthr=0.2 --fibthresh=0.01 --rseed=12345
"$FSLDIR/bin/probtrackx2" -s /absolute/path/subject.bedpostX/merged \
  -m /absolute/path/subject.bedpostX/nodif_brain_mask.nii.gz \
  -x /absolute/path/roi_list.txt --network --dir=/absolute/path/fsl_network \
  --forcedir --opd -P 5000 -S 2000 --steplength=0.5 \
  --cthr=0.2 --fibthresh=0.01 --rseed=12345
```

需要约束时，在对应 FSL 命令后添加 `--avoid=... --stop=... --forcefirststep --distthresh=5 --sampvox=1 --randfib=2 --usef`；固定从第二纤维起始则用 `--fibst=2`。FSL `--opd` 对应 FNIT 当前的路径密度输出。

## 官方选项覆盖

依据 [FSL ProbtrackX 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/probtrackx.html)和 gpucw1 上 FSL 6.0.7.22 的 `probtrackx2 --help`，当前状态如下。这里的“实现”仅指同网格体积输入；没有把 ROI×ROI 的 `fdt_network_matrix` 误称为官方 `--omatrix3` 体素×体素稀疏矩阵。

| 类别 | 已实现 | 尚未实现 |
| --- | --- | --- |
| 输入和空间 | BEDPOSTX 后验、体积 seed/ROI、独立 `-m`、`-P`、`-S`、`--sampvox` | ASCII `--simple`、表面 seed、`--seedref`、`--meshspace`、`--xfm`、`--invxfm` |
| 步进和纤维 | Euler 双向追踪、`--steplength`、`--cthr`、`--fibthresh`、`--randfib` 0–3、`--fibst`、`--usef`、`--rseed` | `--modeuler`、`--loopcheck` 及局部纤维/曲率控制 |
| 约束 | 后验 mask 边界、体积 `--avoid`、`--stop`、`--forcefirststep`、`--distthresh` | 表面约束、`--waypoints`/`--waycond`/`--wayorder`/`--onewaycondition`、`--wtstop` |
| 输出 | `--opd` 密度图、`waytotal`、`--network` 有向 ROI 矩阵 | `--pd`、`--ompl`、`--fopd`、`--targetmasks`/`--os2t`、`--s2tastext`、`--opathdir`、`--omatrix1`–`--omatrix4` 及其目标/阈值选项 |

## 与原版 FSL 的配对 benchmark

**真实 DWI。** gpucw1 上的一例 UK Biobank DWI 使用同一份 FSL BEDPOSTX 三纤维后验（104×104×72，每体素 50 帧）。胼胝体膝部单 seed 含 7 个体素；五区网络使用胼胝体膝部、左右皮质脊髓束和左右上纵束的 7 体素 seed。双方均为 400 总步、0.5 mm、`cthr=0.2`、`fibthresh=0.01`、`rseed=20260927`；单 seed 每体素 200 条，网络每体素 2000 条。FNIT 批大小 2048、CPU 线程数 8。墙钟时间均含进程启动、后验载入、追踪和写盘。

| 任务 | FSL / FNIT 时间 | 密度图 Pearson *r* | 高密度前 10% Dice | `waytotal` FSL / FNIT |
| --- | ---: | ---: | ---: | --- |
| 单 seed CPU | 11.80 / 12.74 s | 0.9938 | 0.8881 | 1400 / 1400 |
| 单 seed GPU | 11.52 / 13.20 s | 0.9939 | 0.8842 | 1400 / 1400 |
| 五区网络 CPU | 27.64 / 44.17 s | 0.9390 | 0.8230 | 65 / 76 |
| 五区网络 GPU | 11.06 / 14.59 s | 0.9575 | 0.7756 | 85 / 76 |

五区网络计数较稀疏。例如右皮质脊髓束→左皮质脊髓束为 FSL CPU 53、FNIT CPU 66，FSL GPU 73、FNIT GPU 67；不能据一次运行推断低计数边的稳定生物学差异。FSL 与 FNIT 的随机数流不同，密度图不要求逐体素相同。这组验证只比较追踪阶段，不比较 BEDPOSTX 拟合。完整矩阵、支持 Dice、密度总和和源码 SHA-256 见[当前真实 DWI 报告](../../validation/probtrackx/report.current.public.json)；独立同源码的[显存记录](../../validation/probtrackx/memory.current.public.json)及[复现步骤](../../validation/probtrackx/README.md)另列。真实被试影像留在授权服务器。

**合成示例。** 40³ 单纤维管状后验（20 帧）上，双方均为每 seed 体素 200 条、240 总步。单 seed 图相关 0.9996，网络图相关 0.9998；双 ROI 矩阵的非零计数 FSL 为 1369/1368，FNIT 为 1365/1362。FSL/FNIT 墙钟分别为单 seed 0.33/4.35 秒、网络 0.62/5.26 秒，小数据启动开销占主导。在 9×5×5 直线纤维场，追踪 mask、`distthresh`、`fibst`、`usef`，以及八种体积 `stop`/`avoid`/`forcefirststep` 情况的密度图与 `waytotal` 逐体素匹配 FSL；CUDA/CPU 回归测试共 20 项通过。

![当前 FNIT 与 FSL 的合成 seed 和网络密度图](../../validation/probtrackx/fsl_fnit_synthetic_comparison.png)

合成图不含 UK Biobank 被试影像。[生成脚本](../../validation/probtrackx/generate_synthetic_posterior.py)、[运行脚本](../../validation/probtrackx/run_synthetic_comparison.sh)和[机器可读指标](../../validation/probtrackx/synthetic_reference.public.json)可复现该图。

## 来源

- [FSL ProbtrackX 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/probtrackx.html)
- [FSL ptx2 源码与许可证](https://git.fmrib.ox.ac.uk/fsl/ptx2)
- [FSL 软件许可证](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)
