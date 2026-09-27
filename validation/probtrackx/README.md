# ProbtrackX 真实 DWI 多脑区配对验证

[功能和参数](../../docs/probtrackx/README.md) · [当前优化汇总 JSON](report.optimization.public.json) · [原始五脑区 CPU/GPU 汇总 JSON](report.multiregion.public.json) · [聚合指标脚本](../../benchmark/probtrackx_multiregion.py) · [seed 选取脚本](prepare_multiregion_seeds.py)

## 当前加速实现与复验

在相同后验和 5 个 seed 上，当前实现并行解压 9 个后验 NIfTI，CPU 默认每批 2048 条轨迹；GPU 用 Triton 3.1.0 融合 float32 步进。缺少 Triton 时 GPU 自动使用原 PyTorch 路径。相同批大小 256 的载入改写在 CPU/GPU 上均与原输出逐体素一致；融合 GPU 使用独立随机流，按 FSL 分布对照。整合后的正式包入口在真实 DWI 上复跑，与隔离内核原型的 seed 图和网络图逐体素一致。显存峰值 2.61 GiB。

| 真实 DWI 运行 | FSL | 旧 FNIT | 当前 FNIT | 当前 FNIT 对 FSL 图相关 |
| --- | ---: | ---: | ---: | ---: |
| 五区网络 CPU，2000 条/体素 | 31.50 s | 111.62 s | 50.52 s | 0.9390 |
| 五区网络 GPU，2000 条/体素 | 11.66 s | 128.75 s | 17.84 s | 0.9575 |

五个 200 条/体素的 CPU seed 配对用时和精度、GPU 胼胝体 seed、矩阵原始计数、源码 SHA-256 均见[当前机器可读报告](report.optimization.public.json)。配对取同一份 FSL BEDPOSTX 后验；旧 GPU 与当前 GPU 均为批大小 2048，旧 CPU 为 256。GPU1 由其他任务共享。所有时间包含启动、载入与写盘。

在可读取这些原始运行目录的授权服务器上重新生成汇总：

```bash
python benchmark/probtrackx_optimization.py \
  --baseline-dir /absolute/path/fnit_probtrackx_multiregion_20260927 \
  --optimized-dir /absolute/path/fnit_probtrackx_opt_20260927 \
  --source-dir /absolute/path/repository/src/fnit/probtrackx \
  --output-json /absolute/path/report.optimization.public.json
```

下面记录的是优化前版本的原始五脑区验证协议，供追溯比较。

## 输入、选点与配对

- 2026-09-27 在 `gpucw1` 上运行。使用一例真实 UK Biobank DWI 已有的原版 FSL BEDPOSTX 全脑后验：104×104×72，50 帧，三纤维。FSL 为 6.0.7.22 的 `probtrackx2` CPU / `probtrackx2_gpu`；FNIT `src/fnit/probtrackx/pipeline.py` SHA-256 为 `87ae878bc076329712f23bad0980fc14610ecd73f659aa220050397fdafb9afb`。所有配对运行读取相同的后验和 mask；原始后验生成日志不可得，故仅验证追踪阶段。
- FSL 自带 JHU-ICBM 白质标签图通过已有 `MNI_to_dti_FA_warp.nii.gz` 最近邻重采样到后验 mask 网格。图谱标签 3、7、8、41、42 分别代表胼胝体膝部、右/左皮质脊髓束、右/左上纵束。每个区域在标签∩mask∩FA≥0.3 中选最大内部距离处的 7 体素十字形 seed；5 个 seed 均通过同网格、非空与 FA 检查。选点不使用 FSL/FNIT 路径图。单被试 seed 坐标与影像留在服务器。
- Seed-to-voxel：两边均为每 seed 体素 200 条轨迹、400 总步、0.5 mm 步长、0.2 曲率阈值、0.01 次要纤维阈值、随机种子 20260927。FNIT `batch_size=256`、`OMP_NUM_THREADS=8`、`MKL_NUM_THREADS=8`。FNIT 在 H100 GPU1 上使用 float32 并启用 TF32；该 GPU 同时有其他进程。5×5 网络额外使用 2000 条/体素，并用 20260928 重跑 FSL；低抽样 200 条/体素的矩阵也保留在 JSON 中用于说明抽样稀疏性。
- 墙钟时间均含启动、载入、追踪与写盘。全部 26 个运行目录都有非空 `fdt_paths.nii.gz`、`waytotal`，NIfTI 形状和 affine 经比较脚本检查；12 个 FNIT 进程退出码 0。14 个 FSL 进程退出码 255，但各日志以 `finished` 或 `TOTAL TIME` 结束且输出可读取，因此单凭退出码不判失败。

## 结果判读

报告中的 Pearson 在两图非零体素并集上计算；top-10% Dice 用 FSL 非零体素数的 10% 作为两图相同的 top-*k*；support Dice 比较全部非零体素。不同随机数流不要求逐体素完全相同。

五个 seed 的 FSL/FNIT `waytotal` 全部为 1400。CPU 密度图在非零体素并集的 Pearson *r*=0.9938–0.9970，高密度前十分位 Dice=0.8631–0.9416，密度和最大相对差异 1.18%；GPU *r*=0.9939–0.9972。胼胝体 seed 的 FSL–FSL 换随机种子对照 *r*=0.9939、top Dice=0.9012。CPU 五例墙钟中位数为 FSL 12.42 s、FNIT 23.30 s；GPU 中位数 12.75 s、44.07 s。GPU 为共享环境，速度差仅适用于本次运行。

5×5 网络每体素 200 条轨迹时，FSL/FNIT 分别只有 6/10 条跨区接受轨迹，图相关 *r*=0.595，不能稳定比较。提高到 2000 条后，右皮质脊髓束→左皮质脊髓束为 FSL 53、FNIT 65，FSL 重跑为 67；反向为 8、8、11。此时 FSL–FNIT 密度图 *r*=0.9477、top Dice=0.7788，FSL–FSL 为 0.9262、0.7611；FSL/FNIT 耗时 31.50/111.62 s。其他边多数仍为零或个位数，不能据此检验低概率边的逐项一致性。所有逐 ROI 指标、矩阵、几何和计时见 [机器可读汇总](report.multiregion.public.json)。

## 复现

在有权限读取同一 DWI/BEDPOSTX 的服务器上，先设置绝对路径 `BED`、`OUT`、`FSLDIR`、`FA`、`WARP`，并使用安装了 NumPy、SciPy、NiBabel、PyTorch 的 `PYTHON`；绘图命令另需 Matplotlib。源代码目录为仓库的 `src`：

```bash
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$FSLDIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" FSLOUTPUTTYPE=NIFTI_GZ
"$FSLDIR/bin/applywarp" \
  --in="$FSLDIR/data/atlases/JHU/JHU-ICBM-labels-2mm.nii.gz" \
  --ref="$BED/nodif_brain_mask.nii.gz" --warp="$WARP" \
  --out="$OUT/jhu_labels_dwi.nii.gz" --interp=nn
"$PYTHON" validation/probtrackx/prepare_multiregion_seeds.py \
  --labels "$OUT/jhu_labels_dwi.nii.gz" \
  --mask "$BED/nodif_brain_mask.nii.gz" --fa "$FA" --out "$OUT"
bash validation/probtrackx/run_real_multiregion.sh "$OUT" "$BED" "$FSLDIR" "$PYTHON" "$PWD/src"
bash validation/probtrackx/run_real_multiregion_network.sh "$OUT" "$BED" "$FSLDIR" "$PYTHON" "$PWD/src"
python3 benchmark/probtrackx_multiregion.py --run-dir "$OUT" \
  --output-json "$OUT/report.multiregion.public.json" \
  --private-figure "$OUT/example_real.private.png"
```

脚本的 GPU 编号设为服务器上的 GPU1。FSL `applywarp` 和 `probtrackx2` 在该环境虽返回 255，仍需按输出文件和日志核验。真实 UK Biobank 输入、posterior、seed 与单被试对照图均只保存在授权服务器；仓库只发布汇总数值与合成示例。[UK Biobank 影像公开使用指引](https://community.ukbiobank.ac.uk/hc/en-gb/articles/16594178325277-Submitting-publications-and-use-of-UK-Biobank-images)要求公开展示被试影像前联系其团队。

## 可公开的合成示例

![合成 posterior 上 FSL 与 FNIT 的 seed-to-voxel 和 ROI 网络密度图](fsl_fnit_synthetic_comparison.png)

这个示例仅使用 [生成脚本](generate_synthetic_posterior.py) 构造的 40×40×40 单纤维管状 posterior（每体素 20 帧）和两个 7 体素 ROI，不含真实被试图像。[一键复现脚本](run_synthetic_comparison.sh) 以每 seed 体素 200 条轨迹、总步数 240、随机种子 20260927 运行 FSL 与 FNIT；[绘图脚本](plot_synthetic_comparison.py) 生成上图。

在 gpucw1 的这次 CPU 运行中，两边 seed-to-voxel `waytotal` 都为 1400，密度图在非零体素并集上的 Pearson *r*=0.9875。FSL 的有向矩阵非零元素为 1369/1368，FNIT 为 1366/1369；完整计数、Dice 和计时见 [合成数据指标 JSON](synthetic_reference.public.json)。小样本计时受程序启动开销影响，不代表大规模速度。

```bash
FSLDIR=/absolute/path/to/fsl bash validation/probtrackx/run_synthetic_comparison.sh /absolute/output/dir
```

运行目录须是新目录；脚本会生成 posterior、运行两套追踪、检查 FSL 输出并写出 JSON 与图。此处图像与真实 UK Biobank 数值验证分别报告。
