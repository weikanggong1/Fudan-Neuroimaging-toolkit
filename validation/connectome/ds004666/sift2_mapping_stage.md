# 固定轨迹 SIFT2 映射核验（ds004666）

**目标与输入。** 固定 MRtrix3 `3.0.3-103-g026e850d` 的 2,758 条 `tracks_10000.tck`、校正后的 WM FOD 网格、官方 243,822 个 fixel 的 index/target、处理 mask 和 `-debug` 导出的 `before_tdi_fixel.msf`。PyTorch FMLS LUT 使用同一 FOD 生成；111,569 个活跃体素的 fixel count 与官方完全一致，体素内 lobe 顺序的 target MAE 为 `7.97e-9`。MRtrix 多线程分配全局 fixel 序号的顺序不同，benchmark 按“体素坐标 + 体素内 lobe 序号”建立置换，之后才逐 fixel 比较。原始文件 SHA-256 见[机器报告](sift2_mapping_fmls_cpu.public.json)。

**官方命令。** `tcksift2 tracks_10000.tck wm_fod_norm.mif weights.txt -act 5tt_dwi.mif -nthreads 8 -debug -force`。官方 24.64 秒包含 FMLS、轨迹映射和 SIFT2 优化（`corrected_mrtrix_fs5tt_act_adapted/stage_times.tsv`）；不能与下表仅映射用时直接相除。

**函数。** `map_streamlines_to_fixels(streamlines, affine, volume_shape, voxel_ids, first_fixel_index, count, lookup_table, directions, *, step_size_mm, n_fixels)` 位于 `src/fnit/connectome/sift2_mapping.py`。每条轨迹是同一设备的 float32 世界毫米坐标 `[Ni,3]`；`affine[4,4]` 将 voxel center 转到世界坐标；`voxel_ids[V]` 是 C-order 排序的活跃体素索引；`first_fixel_index[V]` 是带 dummy fixel 0 的全局首索引；`count[V]` 和 uint8 `lookup_table[V,1281]` 对应 FMLS 体素内方向索引，`count` 是无 fixel 哨兵；`directions[1281,3]` 是打包的 MRtrix 方向表。返回 `track_index[K]`、`fixel_index[K]`、8-bit 量化后的 `length_mm[K]` 和包含 dummy 0 的 `tdi_mm[F]`。稀疏 records 可直接输入 `optimize_sift2_fixels`。全程不使用 float16；CUDA 上默认 TF32，但体素坐标使用逐分量 float32 乘加，避免 TF32 矩阵乘在体素边界引入误分类。

**实现对应。** 从 TCK header 的 `step_size=1.00639975 mm` 和最小体素间距约 2 mm 得到 6 倍 Hermite tension 0.1 上采样；在每条小边上重复 `voxelise_precise` 的 Hermite 二分，直到穿过所有中间角落体素，边界精度为 `0.005 × min(voxel spacing)`；按 MRtrix z/y/x/bin 顺序聚合 dixel；1281 方向绝对点积选 bin；按 FMLS LUT 分配 fixel；以体素对角线映射到 8-bit 长度，执行同 track/fixel 的 first-fit 255-byte 打包。实现依据 [MRtrix mapper.h](https://github.com/MRtrix3/mrtrix3/blob/026e850d/src/dwi/tractography/mapping/mapper.h)、[SIFT model.h](https://github.com/MRtrix3/mrtrix3/blob/026e850d/src/dwi/tractography/SIFT/model.h) 与 [track_contribution.h](https://github.com/MRtrix3/mrtrix3/blob/026e850d/src/dwi/tractography/SIFT/track_contribution.h)。

| 同输入指标 | MRtrix | PyTorch CPU（8 线程） | PyTorch H100 GPU0 |
| --- | ---: | ---: | ---: |
| 非零 TDI fixel | 54,035 | 54,035 | 54,035 |
| TDI 总量（官方 μ 缩放） | 77,033.19378 | 77,033.13665 | 77,033.13665 |
| 处理 mask + target 算出的 μ | 0.705499 | 0.70550628 | 0.70550628 |
| 全 243,822 fixel TDI Pearson r | — | 0.9999999371 | 0.9999999371 |
| 全 fixel TDI MAE | — | 8.1456e-6 | 8.1456e-6 |
| 映射本体时间 | 纳入官方 24.64 秒全阶段 | 0.774 秒 | 1.023 秒 |
| PyTorch 峰值已分配显存 | — | — | 0.370 GB |

[GPU 机器报告](sift2_mapping_fmls_gpu.public.json)与[脑部 TDI 示例图](sift2_mapping_fmls_gpu.png)显示同一 z=41 切面及 fixelwise 散点。另有 7 个解析测试在 CPU/GPU 通过，覆盖直线跨体素、单段跨多个体素、世界平移不变性、FMLS 空映射哨兵、first-fit overflow。该节点存在间歇性 CUDA 初始化 OOM；成功的 GPU 正式运行与 CPU 数值一致，失败重试不纳入计时。

复现：以 `$B` 表示远端 `brainmri_connectome_benchmark`，先设置 `PYTHONPATH=src`，调用 `python tools/benchmark_connectome_sift2_mapping.py --device cpu --tracks "$B/mrtrix_corrected_fs5tt_reregistered_act/tracks_10000.tck" --fixel-dir "$B/sift2_stage_validation/official_fixels_nifti1" --reference-tdi "$B/sift2_stage_validation/official_tdi_fixels/tdi.nii" --processing-mask "$B/sift2_stage_validation/proc_mask.nii.gz" --mu "$B/sift2_stage_validation/official_mu.txt" --directions src/fnit/connectome/data/mrtrix_sift2_1281.npz --lookup "$B/sift2_directions_asset_20260927/fmls_lut.npz" --output sift2_mapping_fmls_cpu.public.json`。`--device cuda:0` 为 GPU 运行；可加 `--records-output records.npz` 保存官方顺序稀疏 records 供 SIFT2 优化器。
