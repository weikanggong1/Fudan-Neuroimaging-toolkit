# ProbtrackX 当前实现验证

[功能、输入输出及官方对应命令](../../docs/probtrackx/README.md) · [默认计数](report.default.latest.public.json) · [长度加权](report.current.latest.public.json)
[matrix1](report.matrix1.cpu.latest.public.json) · [matrix2 与网络](report.matrix2.cpu.latest.public.json) · [matrix3](report.matrix3.cpu.latest.public.json) · [seed→ROI](report.targets.cpu.latest.public.json)

## 数据与配对设计

在 gpucw1 使用 FSL 6.0.7.22 `probtrackx2` / `probtrackx2_gpu`，对同一例真实 UK Biobank DWI 的原版 FSL BEDPOSTX 三纤维后验比较 FNIT。五个预先定义的 ROI 由 JHU 标签图和被试 FA、追踪 mask 生成，选点不依赖两套追踪结果。

双方统一总步数 400、步长 0.5 mm、`cthr=0.2`、`fibthresh=0.01`、`rseed=20260927`。单 seed 每体素 200 条，五区网络每体素 2000 条；FNIT 批大小 2048，CPU 8 线程。分别配对默认计数 `--opd` 和长度加权 `--opd --pd --ompl`，每种模式运行 FSL CPU/FNIT CPU、FSL GPU/FNIT GPU 的单 seed 与五区网络。墙钟时间包含进程启动、后验载入、追踪与写盘。四个 ProbTrackX 源码文件的 SHA-256 保存在六份公开摘要中。GPU 显存单独在真实五区网络长度加权模式测量。真实影像、后验、seed 和逐体素结果留在授权服务器；仓库只保存经授权的六份标量摘要。

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
  --output-json "$OUT/report.default.private.json"
"$PYTHON" benchmark/probtrackx_current.py --run-dir "$OUT/pd_ompl" \
  --mode pd_ompl --source-dir "$PWD/src/fnit/probtrackx" \
  --output-json "$OUT/report.current.private.json" \
  --output-figure "$OUT/real_network_pd_ompl_comparison.private.png"
"$PYTHON" validation/probtrackx/measure_gpu_current.py \
  --samples-dir "$BED" --roi-list "$OUT/pd_ompl/seed_list.txt" \
  --output-dir "$OUT/gpu_memory_network" \
  --source-dir "$PWD/src/fnit/probtrackx" \
  --output-json "$OUT/memory.current.private.json"
"$PYTHON" validation/probtrackx/summarize_public.py \
  --input "$OUT/report.default.private.json" \
  --output "$OUT/report.default.latest.public.json"
"$PYTHON" validation/probtrackx/summarize_public.py \
  --input "$OUT/report.current.private.json" \
  --output "$OUT/report.current.latest.public.json"
```

绘图需要 `matplotlib`，主页 Conda 环境包含该依赖。FSL 在本服务器上有时返回状态 255，但日志结束于 `finished` 且结果可读取；运行脚本记录状态码并逐项检查需要的输出。FNIT CPU/GPU 均正常返回 0。

## 三类连接矩阵的真实 DWI 复现

同一份五区 ROI 列表以 [prepare_connectome_masks.py](prepare_connectome_masks.py) 制作并集 seed、整数标签图和 ROI 元数据。稀疏矩阵使用同一并集 seed、每体素 500 条、400 总步、0.5 mm、rseed=20260927；区域矩阵使用原 ROI 列表。这样 matrix1/2/3 不混入 --network 的跨 ROI 轨迹筛选。原版与 FNIT 各模式分别启动，墙钟时间包括读取、追踪和写盘。真实 DWI、每条边的原始明细和逐边连接图保留在 gpucw1；仓库的摘要不含原始体素或边矩阵。

~~~bash
export BED=/absolute/path/subject.bedpostX
export ROIS=/absolute/path/seed_list.txt
export FSLDIR=/absolute/path/fsl
export PYTHON=/absolute/path/fnit-conda/bin/python
export OUT=/absolute/path/private-matrix-benchmark
export FNIT_SRC="$PWD/src"
bash validation/probtrackx/run_real_matrices.sh \
  "$OUT" "$BED" "$ROIS" "$FSLDIR" "$PYTHON" \
  "$PWD/validation/probtrackx/prepare_connectome_masks.py" 500 all
bash validation/probtrackx/run_real_matrices.sh \
  "$OUT" "$BED" "$ROIS" "$FSLDIR" "$PYTHON" \
  "$PWD/validation/probtrackx/prepare_connectome_masks.py" 500 seed_to_targets_cst_right
CUDA_VISIBLE_DEVICES=0 bash validation/probtrackx/run_real_matrices.sh \
  "$OUT" "$BED" "$ROIS" "$FSLDIR" "$PYTHON" \
  "$PWD/validation/probtrackx/prepare_connectome_masks.py" 500 union_matrix1 gpu
CUDA_VISIBLE_DEVICES=0 bash validation/probtrackx/run_real_matrices.sh \
  "$OUT" "$BED" "$ROIS" "$FSLDIR" "$PYTHON" \
  "$PWD/validation/probtrackx/prepare_connectome_masks.py" 500 union_matrix3 gpu
CUDA_VISIBLE_DEVICES=0 bash validation/probtrackx/run_real_matrices.sh \
  "$OUT" "$BED" "$ROIS" "$FSLDIR" "$PYTHON" \
  "$PWD/validation/probtrackx/prepare_connectome_masks.py" 500 network gpu
bash validation/probtrackx/run_real_matrices_fnit.sh \
  "$OUT/fnit_runs" "$BED" "$ROIS" "$OUT/masks/target_union.nii.gz" \
  "$PYTHON" "$FNIT_SRC" cpu 500
CUDA_VISIBLE_DEVICES=0 bash validation/probtrackx/run_real_matrices_fnit.sh \
  "$OUT/fnit_runs" "$BED" "$ROIS" "$OUT/masks/target_union.nii.gz" \
  "$PYTHON" "$FNIT_SRC" cuda:0 500
"$PYTHON" benchmark/probtrackx_matrix_current.py \
  --fsl-dir "$OUT/fsl_union_matrix2" \
  --fnit-dir "$OUT/fnit_runs/fnit_cpu_matrix2" \
  --source-dir "$FNIT_SRC/fnit/probtrackx" \
  --output-json "$OUT/report.matrix2.cpu.json" \
  --network-fsl-dir "$OUT/fsl_network" \
  --network-fnit-dir "$OUT/fnit_runs/fnit_cpu_network" \
  --roi-list "$ROIS" --nsamples 500 \
  --detail-dir "$OUT/private_edge_differences"
bash validation/probtrackx/run_real_matrices_fnit.sh \
  "$OUT/fnit_runs" "$BED" "$ROIS" "$OUT/masks/target_union.nii.gz" \
  "$PYTHON" "$FNIT_SRC" cpu 500 targets_cst_right
"$PYTHON" benchmark/probtrackx_target_current.py \
  --fsl-dir "$OUT/fsl_seed_to_targets_cst_right" \
  --fnit-dir "$OUT/fnit_runs/fnit_cpu_targets_cst_right" \
  --seed "$(sed -n '2p' "$ROIS")" --target-list "$ROIS" \
  --source-dir "$FNIT_SRC/fnit/probtrackx" \
  --output-json "$OUT/report.targets.cpu.json"
"$PYTHON" validation/probtrackx/plot_connectome_matrices.py \
  --run-dir "$OUT" --device cpu --output "$OUT/real_voxel_matrices.png"
~~~

复现 matrix1/3 时，把上述 `benchmark/probtrackx_matrix_current.py` 命令中的两处 matrix2 目录和报告文件名替换为对应编号。以下命令从四份私有矩阵/目标报告去除逐边、逐体素数组，得到可发布的标量摘要；仓库内的 `latest` 文件是本次最终源码配对的版本。

~~~bash
for NAME in matrix1.cpu matrix2.cpu matrix3.cpu targets.cpu; do
  "$PYTHON" validation/probtrackx/summarize_public.py \
    --input "$OUT/report.$NAME.json" \
    --output "$OUT/report.$NAME.latest.public.json"
done
~~~

比较程序解析 .dot 的 1 起始索引和末尾维度行，并按配套体素坐标表对齐。指标包括非零边支持 Dice、非零并集上的 Pearson r 与 MAE、边权总和及单次运行墙钟时间。--detail-dir 写出每条实际连接的原版和 FNIT 权重，仅用于服务器私有复核。五区网络另比较原始计数、Cij/(Ni×P) 有向矩阵和双向均值；FNIT 保存的归一化文件须与公式逐元素一致。

## 真实 DWI 单 waypoint 对照

[复现脚本](run_real_waypoint.py)从已有 FSL ROI×ROI 网络矩阵选取最大的正非对角计数，按同一 ROI 列表确定 seed 与必经 ROI；选择发生在本次 FNIT 运行之前。随后在同一 BEDPOSTX 后验上分别运行 FSL CPU 与 FNIT CPU 的 `--waypoints` 计数模式，检查完成日志、输出文件和 `waytotal`，并在服务器内比较密度图支持、相关及耗时。`--selection-matrix` 必须与 `--roi-list` 使用同一 ROI 顺序；输出目录需事先不存在。

~~~bash
export BED=/absolute/path/subject.bedpostX
export ROIS=/absolute/path/seed_list.txt
export PRIOR_FSL_NETWORK=/absolute/path/prior-fsl-network/fdt_network_matrix
export FSLDIR=/absolute/path/fsl
export PYTHON=/absolute/path/fnit-conda/bin/python
export PRIVATE_WAYPOINT_OUT=/absolute/path/new-private-waypoint-run
"$PYTHON" validation/probtrackx/run_real_waypoint.py \
  --output-dir "$PRIVATE_WAYPOINT_OUT" --bedpostx "$BED" \
  --roi-list "$ROIS" --selection-matrix "$PRIOR_FSL_NETWORK" \
  --fsl-dir "$FSLDIR" --python "$PYTHON" --source-dir "$PWD/src" \
  --nsamples 2000
~~~

该真实 DWI 配对已在 gpucw1 完成，FSL 完成日志和双方文件均通过检查。逐体素输出、比较数值和 `report.private.json` 保留在授权服务器；公开前不据此声称数值等价。该运行只测试单 waypoint，未覆盖 `wtstop` 或其他约束组合。

## 指标解释与规则回归

密度图相关在双方非零体素并集计算；前 10% Dice 用相同的 FSL 非零体素数十分之一作为双方 top-*k*。平均路径长度图的相关和 MAE 在双方都非零的体素上计算，并另报支持 Dice 与并集（含单侧缺失体素）误差。完整 ROI×ROI 原始矩阵只保存在授权服务器；公开报告仅含汇总误差。网络连接稀疏，个位数命中不宜单独解释。

9×5×5 合成直线场用于计数规则回归，不列为正式 benchmark：`--pd --ompl` 的单 seed 和双 ROI 网络输出，以及单独 `--ompl` 的网络输出，均与 FSL 逐元素相同；matrix1/2/3 的 2×2 稀疏矩阵、坐标表、matrix2 lookup 和路径密度也与 FSL 逐项相同。新增 waypoint/`wtstop` 的 9 组合成场配对与 FSL 6.0.7.22 的 `waytotal` 和 `fdt_paths` 逐项相同，AND 条件下的 matrix2 `.dot` 亦逐项相同。`tests/probtrackx/` 的 40 项 CPU/CUDA 回归测试在 gpucw1 全部通过。合成数据仅用于验证计数规则，不用于正式精度或耗时结论；单 waypoint 的真实 DWI 配对已完成但指标仍为私有；`wtstop` 的真实 DWI 对照尚未完成。
