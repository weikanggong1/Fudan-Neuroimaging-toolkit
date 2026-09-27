# ProbtrackX 当前实现验证

[功能、输入输出及官方对应命令](../../docs/probtrackx/README.md) · [真实 DWI 默认计数报告](report.default.public.json) · [真实 DWI 长度加权报告](report.current.public.json) · [真实网络连接图](real_network_pd_ompl_comparison.png) · [GPU 显存记录](memory.current.public.json)

## 数据与配对设计

在 gpucw1 使用 FSL 6.0.7.22 `probtrackx2` / `probtrackx2_gpu`，对同一例真实 UK Biobank DWI 的原版 FSL BEDPOSTX 三纤维后验比较 FNIT。后验形状为 104×104×72，每体素 50 帧。五个 ROI 从 JHU 标签 3、7、8、41、42 对应的胼胝体膝部、左右皮质脊髓束、左右上纵束中选点，并与追踪 mask 和 FA≥0.3 相交；每 ROI 取 7 个 seed 体素，选点不依赖两套追踪结果。

双方统一总步数 400、步长 0.5 mm、`cthr=0.2`、`fibthresh=0.01`、`rseed=20260927`。单 seed 每体素 200 条，五区网络每体素 2000 条；FNIT 批大小 2048，CPU 8 线程。分别配对默认计数 `--opd` 和长度加权 `--opd --pd --ompl`，每种模式运行 FSL CPU/FNIT CPU、FSL GPU/FNIT GPU 的单 seed 与五区网络。墙钟时间包含进程启动、后验载入、追踪与写盘。源码 `pipeline.py`、`_triton.py`、`cli.py` 的 SHA-256 保存在两份报告中。GPU 显存单独在真实五区网络长度加权模式测量。真实影像、后验和 seed 留在授权服务器；仓库仅发布汇总数值及 ROI 连接图。

## 真实 DWI 复现

在有权访问被试数据的服务器上，以 FSL 的 `applywarp` 最近邻重采样 JHU 标签，然后用[选点脚本](prepare_multiregion_seeds.py)创建五个 seed。以下 `BED` 是 FSL BEDPOSTX 输出目录，`FA` 和 `WARP` 来自该被试已有处理结果：

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
CUDA_VISIBLE_DEVICES=0 bash validation/probtrackx/run_real_current.sh \
  "$OUT/default" "$BED" "$OUT" "$FSLDIR" "$PYTHON" "$PWD/src" default
CUDA_VISIBLE_DEVICES=0 bash validation/probtrackx/run_real_current.sh \
  "$OUT/pd_ompl" "$BED" "$OUT" "$FSLDIR" "$PYTHON" "$PWD/src" pd_ompl
"$PYTHON" benchmark/probtrackx_current.py --run-dir "$OUT/default" \
  --mode default --source-dir "$PWD/src/fnit/probtrackx" \
  --output-json "$OUT/report.default.public.json"
"$PYTHON" benchmark/probtrackx_current.py --run-dir "$OUT/pd_ompl" \
  --mode pd_ompl --source-dir "$PWD/src/fnit/probtrackx" \
  --output-json "$OUT/report.current.public.json" \
  --output-figure "$OUT/real_network_pd_ompl_comparison.png"
"$PYTHON" validation/probtrackx/measure_gpu_current.py \
  --samples-dir "$BED" --roi-list "$OUT/pd_ompl/seed_list.txt" \
  --output-dir "$OUT/gpu_memory_network" \
  --source-dir "$PWD/src/fnit/probtrackx" \
  --output-json "$OUT/memory.current.public.json"
```

绘图需要 `matplotlib`，主页 Conda 环境包含该依赖。FSL 在本服务器上有时返回状态 255，但日志结束于 `finished` 且结果可读取；运行脚本记录状态码并逐项检查需要的输出。FNIT CPU/GPU 均正常返回 0。

## 指标解释与规则回归

密度图相关在双方非零体素并集计算；前 10% Dice 用相同的 FSL 非零体素数十分之一作为双方 top-*k*。平均路径长度图的相关和 MAE 在双方都非零的体素上计算，并另报支持 Dice 与并集（含单侧缺失体素）误差。ROI×ROI 矩阵报告完整值和绝对误差；网络连接稀疏，个位数命中不宜单独解释。

9×5×5 合成直线场用于计数规则回归，不列为正式 benchmark：`--pd --ompl` 的单 seed 和双 ROI 网络输出，以及单独 `--ompl` 的网络输出，均与 FSL 逐元素相同。`tests/probtrackx/test_tracking.py` 的 24 项 CPU/CUDA 回归测试在 gpucw1 全部通过。合成数据仅用于验证计数规则，不用于正式精度或耗时结论。
