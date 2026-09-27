# ds004666 SIFT2 FMLS 与 ACT 处理掩膜核验

使用公开 OpenNeuro ds004666 `sub-01/ses-2mm` 的校正 AP DWI、对应 WM FOD，以及已经注册到同一会话的 FreeSurfer 8.2 5TT。固定 MRtrix3 `3.0.3-103-g026e850d` 的 `tcksift2 ... -act 5tt_dwi.mif -debug` 输出作参照。该 5TT 是本仓库可运行的 FreeSurfer 8.2 适配路径；原 UKB 脚本的 FreeSurfer 7.1 和 `5ttgen ... -first` 选项在已安装 MRtrix 中没有逐项复现。两个实测阶段和输入 SHA-256 分别见 [FMLS 报告](sift2_fmls_real_fod.json)与[处理掩膜报告](sift2_proc_mask_real_fod.json)。原始影像和 LUT 留在远端，不进入 Git。

## 函数与计算规则

| PyTorch 函数 | 输入与输出 | MRtrix 对应规则 |
|---|---|---|
| [`processing_mask_from_5tt`](../../../src/fnit/connectome/sift2_proc_mask.py) | float32 WM SH `[X,Y,Z,C]`、DWI voxel→scanner affine、5TT `[A,B,C,5]`、5TT affine，输出 DWI 网格 float32 mask `[X,Y,Z]`。输入张量在同一设备。 | `tcksift2 ... -act 5tt.mif` 的处理掩膜。不同网格时，每个非零 FOD 体素取 10×10×10 子体素，三线性采样 5TT，按 ACT 五类规则投票；有效采样数必须大于 500，最后把 WM 票数比例平方。相同形状时额外核验 affine 才使用直接 WM² 路径。 |
| [`segment_fod_fixels`](../../../src/fnit/connectome/sift2_fixels.py) | float32 WM SH `[X,Y,Z,C]`、同网格处理掩膜，返回稀疏体素 ID、各体素首 fixel 索引、fixel 数、1281 方向 lookup、逐 fixel FOD 积分、稠密 count/target。 | 固定 1281 方向的 FOD lobe segmentation：按绝对振幅排序、邻接生长、峰值 Newton 修正、0.1 峰阈值、非均匀球面积分和 lookup 膨胀。方向/邻接/积分权重的来源与 MPL 许可见[方向 asset 说明](../../../src/fnit/connectome/data/README.md)。 |

实现依据同一 MRtrix commit 的 [`SIFT/proc_mask.cpp`](https://github.com/MRtrix3/mrtrix3/blob/026e850d/src/dwi/tractography/SIFT/proc_mask.cpp)、[`ACT/tissues.h`](https://github.com/MRtrix3/mrtrix3/blob/026e850d/src/dwi/tractography/ACT/tissues.h)、[`fmls.cpp`](https://github.com/MRtrix3/mrtrix3/blob/026e850d/src/dwi/fmls.cpp) 和 [`fmls.h`](https://github.com/MRtrix3/mrtrix3/blob/026e850d/src/dwi/fmls.h)。与 MRtrix 仅比较形状的直接 5TT 导入条件相比，PyTorch 还要求两个 affine 一致，避免等形状却错位的输入跳过注册。此差异不影响本次 256³ 5TT 到 104×104×72 FOD 的异网格实测。

## 同输入真实数据结果

| 阶段 | 一致性 | H100 GPU0 计算时间 | 峰值 PyTorch 显存 |
|---|---|---:|---:|
| ACT 5TT → SIFT2 处理掩膜 | 778,467/778,752 体素数值完全相同；285 个体素有边界采样差异；非零支持各 114,573 体素且 Dice=1；全图 MAE 3.46e-7，最大绝对误差 0.001975；非零体素 Pearson r=0.999999991。 | 0.645 秒 | 0.757 GB |
| WM FOD → FMLS fixels，固定官方处理掩膜 | fixel count 在 778,752/778,752 体素完全相同，总数均为 243,822；111,569 个有效体素的 FOD target 积分 MAE 1.81e-8、最大 5.96e-8，Pearson r≈1。 | 17.71 秒 | 3.529 GB |

上述时间只覆盖各 PyTorch 函数，输入已装入内存；GPU0 与其他作业共享，这些是实测单次用时。MRtrix 完整 `tcksift2` 的 24.64 秒还包含 fixel/轨迹映射与系数优化，不能直接作为这两个子阶段的加速比。FMLS 球谐与积分在 GPU 上使用 float64，5TT 采样使用 float32；未使用 float16。方向 asset 的真实 WM FOD 角度积分核验在[单独报告](sift2_direction_asset_real_fod.json)。

![公开 ds004666 同一轴位切面的 MRtrix 与 PyTorch 处理掩膜、绝对差和 fixel 数](sift2_fmls_proc_mask_comparison.png)

图示选择非零 mask 最多的轴位 `z=37`；原始数组及 SHA-256 记录在机器报告中。MRtrix 多线程存储 fixel 的全局序号与 PyTorch 顺序不同，跨程序比较按“体素坐标 + 体素内 lobe 序号”置换。[固定真实轨迹的映射核验](sift2_mapping_stage.md)证明 243,822 个局部 lobe target 的 MAE 为 7.97e-9，并进一步比较 2,758 条官方 TCK 的 fixel TDI。

## 复现

通过 [`benchmark_sift2_processing_mask.py`](../../../tools/benchmark_sift2_processing_mask.py) 提供配对的 WM FOD NIfTI、5TT NIfTI 与 MRtrix `-debug` 导出的 `proc_mask`；通过 [`benchmark_sift2_fixels.py`](../../../tools/benchmark_sift2_fixels.py) 提供相同 FOD、官方处理掩膜、`before_fixel_count` 对应的 fixel `index.nii` 以及 `before_target.nii.gz`。脚本的 `--help` 列出完整参数，输出 JSON 记录输入 SHA-256、数值误差、时间和峰值显存。6 个聚焦解析测试见 [`test_sift2_fixels.py`](../../../tests/connectome/test_sift2_fixels.py) 与 [`test_sift2_proc_mask.py`](../../../tests/connectome/test_sift2_proc_mask.py)。
