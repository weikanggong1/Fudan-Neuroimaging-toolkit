# 真实数据 100,000 次 ACT/iFOD2 播种

## 输入和计时范围

使用公开 OpenNeuro ds004666 `sub-01/ses-2mm` 的校正 DWI、配对 T1 的已完成 `recon-all` 分割，以及同一份归一化 WM FOD、5TT、GMWMI。参考程序是独立编译的 MRtrix3 3.0.3（源码提交 `eeab681d3e0cb004cf1d1d31579d3892197ef5b6`）；它只用于基准，不进入 FNIT 运行时。H100 PCIe 上运行 FNIT，Xeon Gold 6418H 上运行 MRtrix；服务器有其他作业。FNIT 的计时从三个影像载入 GPU 后开始，包含播种、追踪、返回对象整理与输入 SHA-256 统计，不写 TCK；MRtrix 墙钟包含影像读取及 TCK 写出。两者时段和硬件不同，下表只记录本次实测。

首次输入核对发现：原 FNIT T1 5TT/GMWMI 与参考 MRtrix 影像的体素值逐个一致，但参考的 T1→DWI 配准已写入 affine。两者八角点的最大世界坐标差为 `1.483 mm`；不能用那组三张原 NIfTI 做“同输入”结论。本次 FNIT 测量改用 `mrconvert` 从参考三张 MIF 导出的 NIfTI，没有重新采样。FOD 的原 FNIT NIfTI 与参考转换的 NIfTI-1 在体素和仿射上逐值一致。核查程序为 [benchmark_connectome_scale_inputs.py](../../../tools/benchmark_connectome_scale_inputs.py)，[逐图形状、差异及 SHA-256](tracking_scale_100k_20260929/input_compare.json)留档。后续[精确仿射检查](act_geometry_accepted_seeds_20260929.md)进一步发现：`mrconvert` 的 NIfTI-1 sform 舍入，使 5TT 和 FOD 世界坐标八角点相对各自 MIF 分别偏最多 `1.43×10⁻⁵ mm` 和 `1.55×10⁻⁵ mm`。本页三张转换图是逐体素一致的输入，不能视为亚体素几何逐值相同。

## 命令、参数和输出

下面的三个 MIF 是已固定的真实输入；`mrconvert` 仅改存储格式。`REFERENCE_BIN` 是独立的 MRtrix 可执行文件目录。参考命令由[脚本](../../../tools/reference/benchmark_tracking_scale_official.sh)执行：

```bash
REFERENCE_BIN=/path/to/independent-mrtrix/bin
FOD=wm_fod_norm.mif              # 归一化 WM 球谐 FOD，104×104×72×45
FIVE=5tt_dwi.mif                 # 配对 T1 的 5TT，256×256×256×5；affine 已映射到 DWI 世界空间
GMWMI=gmwmi_seed_dwi.mif         # 与 5TT 共网格的播种权重，256×256×256
OUT=benchmark/official_100k     # 独立参考输出目录
MRTRIX_BIN="$REFERENCE_BIN" bash tools/reference/benchmark_tracking_scale_official.sh \
  "$FOD" "$FIVE" "$GMWMI" 100000 "$OUT"

mrconvert "$FOD" fod_reference.nii.gz
mrconvert "$FIVE" five_reference.nii.gz
mrconvert "$GMWMI" gmwmi_reference.nii.gz
```

FNIT 的[实测脚本](../../../tools/benchmark_connectome_tracking_scale.py)只调用 `probabilistic_tractography`。每个输入变量的用途和输出结构如下；`--seed` 固定 PyTorch 的随机序列，不能对应 MRtrix 的随机数状态。`--batch-size` 控制并行播种批量；`--device` 选择计算卡，TF32 默认启用，所有图像数据保持 float32。

```bash
fnit_args=(
  --fod fod_reference.nii.gz          # 与参考相同的 45 系数 FOD，含体素至 RAS 仿射
  --five-tissue five_reference.nii.gz # 与参考相同的五组织分数，含已配准仿射
  --gmwmi gmwmi_reference.nii.gz      # 与 5TT 同网格的 GMWMI 播种权重
  --n-seeds 100000                    # 尝试的种子数，不是保留流线数
  --batch-size 8192                   # 每批 GPU 并行处理的种子数
  --seed 0                            # PyTorch 随机种子
  --device cuda:0                     # H100 GPU；CUDA_VISIBLE_DEVICES 可映射物理卡
  --output benchmark/fnit_100k.json   # 输入哈希、保留数、点数、秒数及内存峰值
)
python tools/benchmark_connectome_tracking_scale.py "${fnit_args[@]}"
```

输出 JSON 的 `input_sha256` 对应三张 FNIT NIfTI，`accepted_streamlines` 是 `Tractogram.paths` 的长度，`total_path_points` 是所有路径的点数；`tracking_seconds_with_inputs_loaded` 还包含哈希和点数统计，`peak_torch_allocated_gib` 和 `peak_torch_reserved_gib` 是 Torch CUDA 分配统计，`process_max_rss_kib` 是整个 Python 进程常驻内存峰值。脚本不写 TCK 或矩阵；四矩阵及脑图的同数据比较见[10k 逐步验证](ifod2_rejection_20260929.md)。

## 实测

| 实现 | 尝试种子 | 保留流线 | 保留率 | 用时 | 峰值内存 |
|---|---:|---:|---:|---:|---:|
| MRtrix3 CPU，包含读写 | 100,000 | 27,616 | 27.616% | 68.88 s | 0.291 GiB，进程 RSS |
| FNIT H100，影像已载入、无 TCK 写出 | 100,000 | 27,401 | 27.401% | 819.68 s | 0.973 GiB，Torch 已分配；1.061 GiB，进程 RSS |

相差 215 条，即 0.215 个百分点。本次 FNIT 用时是参考墙钟的 11.90 倍；受硬件、服务器共享负载及计时边界影响，不外推为一般加速比。FNIT 峰值低于本项目 20 GiB 上限。使用同一转换影像的 10,000 次 FNIT 复跑保留 2,795 条，耗时 115.28 s，Torch 峰值 0.675 GiB；[10k JSON](tracking_scale_100k_20260929/fnit_10k.json)、[100k JSON](tracking_scale_100k_20260929/fnit_100k.json)与[官方 JSON](tracking_scale_100k_20260929/official_100k.json)可复核。

保留率接近并不代表流线群体、矩阵或空间分布匹配。后续已用同一份影像完成[三次官方与一次 FNIT 的 100k 四矩阵、长度、端点和 TDI 对照](tracking_100k_matrices_20260929.md)，发现部分矩阵指标进入官方随机范围，轨迹分布仍有小但可测的差异。本页 `819.68 s` 是较早的纯追踪测量；后续更拥挤的共享 H100 上，全链实验追踪耗时 `1,443.91 s`，两次均保留 27,401 条，计时不能相互替代。[可选圆弧编译核的独立 100k 实测](tracking_compile_20260929.md)追踪 `782.49 s`、保留 27,353 条；共享卡负载不同，不能据此计算稳定加速比。当前返回结构逐条保留 GPU Tensor，100k 可运行，100 万及 1,000 万次播种尚无测量；下一步需要按块存储轨迹。一次在真实 ACT 路径坐标上的 `grid_sample` 替换试验改变了 3 个组织类别判定，且没有降低采样耗时，因此未替换已通过官方逐点核对的实现。
