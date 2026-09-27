# ProbtrackX 当前实现验证

[功能、参数及官方对应命令](../../docs/probtrackx/README.md) · [真实 DWI 汇总](report.current.public.json) · [合成数据汇总](synthetic_reference.public.json) · [合成示例图](fsl_fnit_synthetic_comparison.png)

## 验证范围

在 gpucw1 使用 FSL 6.0.7.22 `probtrackx2` / `probtrackx2_gpu` 和同一份原版 FSL BEDPOSTX 三纤维后验比较当前 FNIT。真实 DWI 的 104×104×72 后验含 50 帧；五个 ROI 均在原始图谱、追踪 mask 和 FA≥0.3 的交集中选 7 个 seed 体素，选点不依赖两套追踪结果。JHU 标签 3、7、8、41、42 分别对应胼胝体膝部、右/左皮质脊髓束、右/左上纵束。源代码 `pipeline.py`、`_triton.py`、`cli.py` 的 SHA-256 和完整矩阵均保存在[当前报告](report.current.public.json)。

双方统一 `-S 400`、0.5 mm、`cthr=0.2`、`fibthresh=0.01`、`rseed=20260927`。单 seed 为 `-P 200`，五区网络为 `-P 2000`；FNIT `batch_size=2048`，CPU 限 8 线程。墙钟时间含进程启动、载入和写盘。独立同源码运行的[显存记录](memory.current.public.json)测得 FNIT 网络峰值已分配显存 2.61 GiB；GPU1 是共享 H100，因此时间只代表这次配对运行。

| 任务 | FSL / FNIT 墙钟 | 图相关 *r* | 前 10% Dice | 图非零支持 Dice |
| --- | ---: | ---: | ---: | ---: |
| 胼胝体膝部单 seed CPU | 11.80 / 12.74 s | 0.9938 | 0.8881 | 0.6971 |
| 胼胝体膝部单 seed GPU | 11.52 / 13.20 s | 0.9939 | 0.8842 | 0.6967 |
| 五区有向网络 CPU | 27.64 / 44.17 s | 0.9390 | 0.8230 | 0.5351 |
| 五区有向网络 GPU | 11.06 / 14.59 s | 0.9575 | 0.7756 | 0.5283 |

图相关在两图非零体素并集上计算；前 10% Dice 对双方取相同的 FSL 非零体素数十分之一作为 top-*k*。随机数流不同，网络中个位数的边容易波动。当前报告只检验同一后验上的体积追踪，不能外推到 BEDPOSTX 拟合或未实现的 ProbtrackX 选项。真实被试影像、后验、seed 和对照图只留在授权服务器；仓库只发布汇总数值。

## 真实 DWI 复现

在有权读取同一影像的服务器上，先把 JHU 标签最近邻重采样到扩散网格，并用[选点脚本](prepare_multiregion_seeds.py)创建五个 seed。下列 `BED` 是 FSL BEDPOSTX 输出目录，`FA` 和 `WARP` 来自该被试已有处理结果：

```bash
export BED=/absolute/path/subject.bedpostX
export FSLDIR=/absolute/path/fsl
export FA=/absolute/path/dti_FA.nii.gz
export WARP=/absolute/path/MNI_to_dti_FA_warp.nii.gz
export OUT=/absolute/path/new-benchmark-output
export PYTHON=/absolute/path/fnit-conda/bin/python
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$FSLDIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$FSLDIR/bin/applywarp" \
  --in="$FSLDIR/data/atlases/JHU/JHU-ICBM-labels-2mm.nii.gz" \
  --ref="$BED/nodif_brain_mask.nii.gz" --warp="$WARP" \
  --out="$OUT/jhu_labels_dwi.nii.gz" --interp=nn
"$PYTHON" validation/probtrackx/prepare_multiregion_seeds.py \
  --labels "$OUT/jhu_labels_dwi.nii.gz" \
  --mask "$BED/nodif_brain_mask.nii.gz" --fa "$FA" --out "$OUT"
CUDA_VISIBLE_DEVICES=1 bash validation/probtrackx/run_real_current.sh \
  "$OUT" "$BED" "$OUT" "$FSLDIR" "$PYTHON" "$PWD/src"
"$PYTHON" benchmark/probtrackx_current.py \
  --reference-dir "$OUT" --current-dir "$OUT" \
  --source-dir "$PWD/src/fnit/probtrackx" \
  --output-json "$OUT/report.current.public.json"
```

FSL 在该服务器上的追踪进程可能返回状态 255，但日志以 `finished` 结束且 `fdt_paths.nii.gz`、`waytotal` 和网络矩阵可读取；运行脚本逐项检查这些输出。CPU FNIT、GPU FNIT 均正常返回 0。

## 合成图与逐体素检查

[合成脚本](run_synthetic_comparison.sh)创建 40³ 单纤维管状后验、两个 ROI，分别运行 FSL 和 FNIT 并生成下图。每 seed 体素 200 条、240 总步、随机种子 20260927。单 seed 的图相关 *r*=0.9996、网络 *r*=0.9998；网络矩阵非零计数为 FSL 1369/1368、FNIT 1365/1362。完整密度、Dice 与墙钟见[合成汇总](synthetic_reference.public.json)。

![当前 FNIT 与 FSL 的合成 seed 和网络密度图](fsl_fnit_synthetic_comparison.png)

```bash
FSLDIR=/absolute/path/fsl bash validation/probtrackx/run_synthetic_comparison.sh /absolute/path/fresh-output
```

9×5×5 直线纤维场进一步检验追踪 mask、最小长度、指定第二纤维、体积分数终止，以及八种 `stop`/`avoid`/`forcefirststep` 组合；密度图与 `waytotal` 均逐体素匹配 FSL。仓库 `tests/probtrackx/test_tracking.py` 的 CPU/CUDA 回归测试在 gpucw1 上 20/20 通过。
