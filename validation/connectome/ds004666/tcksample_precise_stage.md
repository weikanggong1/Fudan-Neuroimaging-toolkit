# 固定轨迹精确 FA 均值核验（ds004666）

固定 MRtrix3 `3.0.3-103-g026e850d` 生成的 2,758 条 `tracks_10000.tck` 与 `fa_corrected.mif`。表中 PyTorch 首轮使用 `mrconvert fa_corrected.mif fa_corrected.nii.gz -force` 转换后的 float32 体素值；该转换会改变存储轴方向，并将原 MIF 双精度空间变换写入 NIfTI 的单精度 sform。因此表中比较的轨迹和 FA 体素值相同，空间几何存在小量舍入差异。输入哈希见[GPU 机器报告](tcksample_precise_gpu.public.json)。官方命令为：

```bash
tcksample -precise -stat_tck mean tracks_10000.tck fa_corrected.mif streamline_mean_fa.txt -nthreads 8
```

`sample_streamline_mean_precise(streamlines, scalar_image, affine) -> torch.Tensor` 位于 `src/fnit/connectome/tcksample_precise.py`：`streamlines` 是同一设备、TCK 顺序的 float32 世界毫米坐标 `[Ni,3]` 列表；`scalar_image` 为同设备 float32 `[X,Y,Z]`；`affine[4,4]` 把 voxel center 转到世界坐标。输出同设备 float32 `[T]`，每条轨迹一个 FA 均值。没有落入图像体素的轨迹输出 NaN，和 MRtrix 一致。实现复用 SIFT2 的 Hermite 精确角落体素边界搜索，但 `tcksample -precise` 的上采样比例固定为 **1**；每条轨迹按体素内实际弦长加权 FA，再除以总弦长。[MRtrix tcksample.cpp](https://github.com/MRtrix3/mrtrix3/blob/026e850d/cmd/tcksample.cpp) 是逐语义对照源。

| 固定轨迹、FA 体素值（转换 NII affine） | MRtrix 原 MIF | PyTorch CPU（8 线程） | PyTorch H100 GPU0 |
| --- | ---: | ---: | ---: |
| 有限的逐轨值 | 2,758 | 2,758 | 2,758 |
| 平均 FA | 0.431210374 | 0.431210226 | 0.431210226 |
| 逐轨 Pearson r | — | 0.9999999902 | 0.9999999902 |
| 逐轨 MAE | — | 1.5311e-6 | 1.5310e-6 |
| 最大绝对误差 | — | 0.0005499 | 0.0005499 |
| 核心调用时间 | 0.03 秒命令 wall | 0.101 秒 | 1.093 秒 |
| PyTorch 峰值已分配显存 | — | — | 0.102 GB |

**原 MIF 几何复核。** 原始 MIF 的 `layout: -0,+1,+2`。用 `mrconvert fa_corrected.mif fa_corrected_positive_strides.nii.gz -strides +1,+2,+3 -force` 将数组保存为原 MIF 体素索引顺序，同时从 MIF header 的三行 `transform:` 与 `vox:` 恢复双精度 voxel-to-world affine，而不读取转换后 NIfTI 的舍入 affine。固定全部 2,758 条轨迹的 PyTorch CPU 均值与原始 MIF 官方输出比较：Pearson r = **0.9999999949**、MAE = **5.5833e−7**、最大绝对差 = **0.0005499**。原转换 NII 几何下 track 1788 的约 `0.000492` FA 差值在此复核中消失；最大残差来自 track 2742。在其单轨原版 `tckmap -precise -upsample 1` 对照中，PyTorch 将约 `0.00734 mm` 轨迹长度分配到相邻体素的另一侧。将二分参数从 float32 改为与 MRtrix 相同的 float64 并未改变此残差，故未保留该试验改动。原 MIF 几何复核在 CPU 8 线程的首轮调用为 **0.124 秒**，随后 5 次暖启动中位数 **0.102 秒**；PyTorch 输入已预载入，官方 0.03 秒为命令 wall，计时边界不同。[原 MIF 几何机器报告](tcksample_precise_mif_affine_cpu.public.json)包含原 MIF、TCK、转换 NII 与官方逐轨向量的 SHA-256，以及精确 affine；[对应真实 FA 切面与逐轨图](tcksample_precise_mif_affine_cpu.png)单独展示该复核。上表 GPU 用时只对应首轮转换 NII 几何。复核调用使用同一个 `sample_streamline_mean_precise` API，传入原 MIF affine 与正 strides NIfTI 体素。

官方 0.03 秒来自 `corrected_mrtrix_fs5tt_act_adapted/stage_times.tsv`；PyTorch 时间从 TCK 与 FA 已加载到目标设备后开始，计时范围并不完全一致。GPU 版本本阶段慢于 MRtrix 和 CPU，主要包含多轮逐体素边界的小 kernel 启动；如大量轨迹批处理需要加速，可把相同几何逻辑编译为 conda 可安装的 CUDA 扩展。当前绝对调用时间为约 1 秒，数值已能用于 mean-FA 连接矩阵。CPU/GPU 三个解析测试通过，覆盖单段中间体素、世界平移和图像外轨迹。

[首轮转换 NII 几何的真实 FA 与逐轨对照图](tcksample_precise_gpu.png)同时给出 FA 脑切面、官方与 PyTorch 散点和逐轨差值。复现时以 `$B` 表示远端 `brainmri_connectome_benchmark`，设置 `PYTHONPATH=src`，调用：

```bash
python tools/benchmark_connectome_tcksample_precise.py \
  --tracks "$B/mrtrix_corrected_fs5tt_reregistered_act/tracks_10000.tck" \
  --fa "$B/sift2_mapping_validation_20260927/fa_corrected.nii.gz" \
  --reference "$B/mrtrix_corrected_fs5tt_reregistered_act/streamline_mean_fa.txt" \
  --device cuda:0 --output tcksample_precise_gpu.public.json
```
