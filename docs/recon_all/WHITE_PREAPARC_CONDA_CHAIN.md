# Conda white.preaparc：真实 T1 保存阶段的验证

`fnit.recon_all.white_preaparc_conda.run_white_preaparc(subject_dir, hemi, binary, assets_dir, threads=4)` 先用 Python 计算灰白质阈值，再调用 Conda 内编译的 `mris_place_surface` 放置表面。`H` 取 `lh` 或 `rh`。

| 输入 | 用途 |
| --- | --- |
| `mri/brain.finalsurfs.mgz` | 掩膜及编辑后的强度图 |
| `mri/wm.mgz` | 白质分割 |
| `mri/aseg.presurf.mgz` | 表面生成前的解剖分割 |
| `surf/H.orig.premesh` | 灰白质阈值采样表面 |
| `surf/H.orig` | 拓扑修复后的放置输入 |
| `binary` | Conda 内编译的 FreeSurfer 8.2 `mris_place_surface` |
| `assets_dir` | 外置 FreeSurfer 数据目录 |

输出为 `surf/autodet.gw.stats.H.dat`、`surf/H.white.preaparc` 和 `mri/mrisps.wpa.mgz`。两侧共用 `mri/mrisps.wpa.mgz`，第二次调用会覆盖第一次的诊断图，与 recon-all 相同。该函数不调用 FreeSurfer Python 脚本，也不要求安装 FreeSurfer 运行环境，但必须在 Conda 内编译安装原生可执行程序。测试所用程序的 SHA-256 为 `9a42f5d7b70a066daf12e67fb6a0924048b4778186ea772e8976722b226b35d5`，对应 FreeSurfer 源码提交 `d932c45b7941662ea380a05efef580568b98d41a`。如程序要求许可证，需在环境中设置 `FS_LICENSE`。阈值计算使用 Python/Numba，表面放置使用 CPU C++。

Python 调用：

```python
from fnit.recon_all.white_preaparc_conda import run_white_preaparc

result = run_white_preaparc(
    subject_dir="/path/to/subjects/sub01",  # 含 mri/surf 的被试目录
    hemi="lh",  # 左半球；右半球用 rh
    binary="/path/to/conda-native/bin/mris_place_surface",  # Conda 编译的程序
    assets_dir="/path/to/fnit-assets",  # 外置 FreeSurfer 数据目录
    threads=4,  # CPU 线程数
)
```

`result` 含 `stats`（`surf/autodet.gw.stats.H.dat` 路径）、`output`（`surf/H.white.preaparc` 路径）、`outvol`（`mri/mrisps.wpa.mgz` 路径）、`stats_seconds`（阈值计算墙钟秒数）和 `place_seconds`（放置墙钟秒数）。

命令行调用：

```bash
python -m fnit.recon_all.white_preaparc_conda \
  /path/to/subjects/sub01 lh \
  --binary /path/to/conda-native/bin/mris_place_surface \
  --assets-dir /path/to/fnit-assets --threads 4
```

命令行前两个位置参数为 `subject_dir`（被试目录）和 `hemi`（`lh`/`rh`）；`--binary` 指定 Conda 程序，`--assets-dir` 指定外置数据目录，`--threads` 指定 CPU 线程数（默认 4）。命令会打印上述返回字典。

从被试的 `mri` 目录运行时，对应的 FreeSurfer 8.2 命令为：

```bash
mris_autodet_gwstats --o ../surf/autodet.gw.stats.lh.dat \
  --i brain.finalsurfs.mgz --wm wm.mgz --surf ../surf/lh.orig.premesh
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat \
  --wm wm.mgz --threads 4 --invol brain.finalsurfs.mgz --lh \
  --i ../surf/lh.orig --o ../surf/lh.white.preaparc --white \
  --seg aseg.presurf.mgz --restore-255 --nsmooth 5 \
  --rip-bg-no-annot --rip-bg --rip-bg-lof --restore-255 \
  --outvol mrisps.wpa.mgz
```

右侧将 `lh` 改为 `rh`。

## 较早的双侧混合输入试验

真实 T1 为 `examples/data/sub-01_T1w.nii.gz`，SHA-256 为 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`。本试验的候选 `orig` 和 `orig.premesh` 来自 FNIT Python sphere、修补后的 Conda 拓扑 GA 和 Python remesh。保存该拓扑结果的临时被试仍将**官方 `brain`、`wm`、`filled`、`norm` MRI 文件**符号链接为输入；其中 `brain`、`wm` 由拓扑 GA 读取。v5 另外生成的相应 MRI 与官方逐体素一致，但本试验没有检验其元数据对 GA 的影响。就这例 T1 而言，两个候选表面的顶点顺序、坐标和面与 FreeSurfer 8.2 一致。放置阶段使用的 v5 `wm`、`aseg.presurf` 与官方均有 0 个差异体素。

本试验的候选 `brain.finalsurfs` 使用 FNIT 的 `brain`、`brainmask`、`entowm`、`aseg.presurf`，同时**借用了官方 `mca-dura` 与 `vsinus` 标签**。它与官方 finalsurfs 的 16,777,216 个体素值一致。借用的两张标签是本次放置试验的上游边界，因此这组结果不能视作从 T1 独立运行到表面。后来另有试验生成了候选辅助标签，但没有将它们用于本节的放置调用。

在上述输入上，Python 阈值文件与官方逐字节一致；Conda C++ 放置输出的有序面也一致。下表按相同顶点索引计算候选 `H.white.preaparc` 与保存的官方文件之间的**三维欧氏位移**。基线为较早的 exact-orig 试验：它平滑 `orig` 后直接复制为 `H.white.preaparc`，因此基线与放置结果的顶点索引和文件名可对应。

| 半球 | 有序顶点 / 面 | 近似基线均值 / P99，mm | 放置后均值 / P99，mm | 放置后最大值，mm | >0.1 mm 顶点数：基线 → 放置后 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LH | 106,622 / 213,240 | 0.802314 / 3.141148 | **0.000420574 / 0.006998961** | 0.630004 | 101,587 → **50** |
| RH | 105,541 / 211,078 | 0.795679 / 3.147371 | **0.000387517 / 0.005624803** | 0.638907 | 100,299 → **50** |

两侧的诊断图 `mrisps.wpa.mgz` 与官方各有 0 个差异体素；文件哈希因非体素字节而不同。使用 4 个 CPU 线程时，单次墙钟耗时如下：

| 阶段 | FNIT LH / RH，秒 | 已归档官方 LH / RH，秒 |
| --- | ---: | ---: |
| 灰白质阈值 | 6.15 / 5.14 | 3.98 / 4.29 |
| white.preaparc 放置 | 251.11 / 233.60 | 242.65 / 250.39 |

每侧仅运行一次，且共享节点负载不同；候选 RH 运行时另有 CPU 模型任务并行。较早以**完全冻结的官方输入**重放 Conda LH 放置，得到相同的均值、P99 和最大误差，但在另一天耗时 149.46 秒。这些时间不能用于断言等价重建的速度优势。

原始 [LH](../../validation/recon_all/python_gpu_port/white_candidate_chain_20260927/lh.json) 和 [RH](../../validation/recon_all/python_gpu_port/white_candidate_chain_20260927/rh.json) 记录包含有序坐标相等数、输入哈希、阈值、诊断图体素和耗时；同阶段的 [基线 LH](../../validation/recon_all/python_gpu_port/white_candidate_chain_20260927/lh_stage_baseline.json) 与 [基线 RH](../../validation/recon_all/python_gpu_port/white_candidate_chain_20260927/rh_stage_baseline.json) 记录保留原有误差分布。

另一次较早的最终 `white` 测量中，相对官方最终 `white` 的**单坐标分量绝对误差均值**约为 0.44 mm；它处于后续阶段，统计量也不同，不能与上表的 preaparc 三维位移直接相除。Conda 与官方放置优化器在相同初始平滑之后出现浮点差异，每侧留下上述 50 个离群顶点。[源码级首差分析](../../validation/recon_all/python_gpu_port/native_cpp_conda_20260927/WHITE_PREAPARC_FIRST_DIVERGENCE_20260927.md) 将第一处可见的受力差异定位在第 6 次接受步之前；现有编译选项试验没有消除该差异。

## 标准路径与历史全候选保存阶段试验

标准 runner 在 CPU MNI/辅助分割/finalsurfs 和拓扑阶段之后调用本函数，再以 Python 在 CPU 上平滑 3 轮。随后执行皮层标签、球面注释和最终 white 放置；自产上游连续整例仍需验收。

[保存阶段 LH 连通检查](../../validation/recon_all/python_gpu_port/white_connected_prefix_20260927/README.md) 分别记录了较早的混合 MRI 重放，以及后来一次**全候选输入的 LH 重放**。后者使用 FNIT 生成的 MNI152 LTA、MCA/dura、vsinus、finalsurfs、v5 `brain/wm/filled/norm` 和独立生成的 Python nofix 表面。该次 LH 放置结果仍有 50 个顶点位移 >0.1 mm；3 轮平滑后为 18 个。历史试验曾复制该 `smoothwm` 作为最终 `white` 时，相对官方最终 `white` 的同索引三维顶点位移均值为 0.288 mm。现版标准流程已改为真实的最终 white 放置。这些是同一 T1 上拼接保存阶段的检验，并非新的一次进程从原始 T1 完整运行 recon-all；本节双侧混合输入试验的数字也不能充当 RH 全候选结果。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
