# MMORF 验证

[返回 MMORF 文档](../../docs/mmorf/README.md)

[`report.public.json`](report.public.json) 记录 TorchMMORF 正常收敛路径与 FSL MMORF 0.3.2 的真实单被试对照。两边使用相同的脑提取 T1w、FSL 六通道 tensor、模板和 FLIRT 初始化矩阵。该数值比较测于加入非有限值恢复逻辑之前，报告保留测量时的源码哈希；当前源码的首次 LBFGS 尝试保持相同参数和损失，尚无 fresh 同输入 FSL 逐体素重测。[当前源码恢复试验](recovery.real.current.json)记录固定失败输入的重试结果和另一轮完整 raw-to-standard 运行。仓库不保存源图像或 subject identifier，只保留汇总指标与此前已公开的去标识切片。

当前报告覆盖：

- warp、Jacobian 和 warped scalar 的完整 3D Pearson、MAE、RMSE；
- shape、affine、dtype、warp 方向和单位；
- 同一 sampler/affine/native map 下，仅替换官方或 FNIT warp 的九图对照；
- 固定官方 warp 的 sampler 隔离和 control-lattice 投影诊断；
- GPU compute、含写盘 wall time、九图应用时间和峰值 CUDA allocation；
- 当前源码哈希、测试结果和仍未实现的官方求解语义。

`numerically_equivalent=false`。当前主要差异是官方 sparse full/diagonal Gauss–Newton–Levenberg 与 FNIT LBFGS strong-Wolfe 的更新轨迹。

## 复现比较

[`compare_mmorf.py`](compare_mmorf.py) 对已有输出重新计算指标。`--map` 可重复九次；其中 official 参数图必须先用与 FNIT 图相同的 `apply_mmorf_warp`、affine、native input 和 interpolation 生成，才能只比较 warp estimation：

```bash
python validation/mmorf/compare_mmorf.py \
  --fnit-warp fnit/mmorf_warp.nii.gz \
  --official-warp official/warp.nii.gz \
  --fnit-jacobian fnit/mmorf_jacobian.nii.gz \
  --official-jacobian official/jacobian.nii.gz \
  --brain-mask MNI152_T1_1mm_brain.nii.gz \
  --fnit-scalar fnit/mmorf_warped_scalar.nii.gz \
  --official-scalar official/mmorf_warped_scalar.nii.gz \
  --map-mask fnit/standard/FA.nii.gz official_warp_with_fnit_sampler/FA.nii.gz \
  --map FA fnit/standard/FA.nii.gz official_warp_with_fnit_sampler/FA.nii.gz \
  --output comparison.json
```

[`plot_mmorf.py`](plot_mmorf.py) 使用 nibabel、NumPy 和 Pillow 生成文档中的真实数据切片：

```bash
python validation/mmorf/plot_mmorf.py \
  --official-scalar official_warp_with_fnit_sampler/T1.nii.gz \
  --fnit-scalar fnit/mmorf_warped_scalar.nii.gz \
  --official-fa official_warp_with_fnit_sampler/FA.nii.gz \
  --fnit-fa fnit/standard/FA.nii.gz \
  --output docs/mmorf/figures/mmorf_fsl_comparison.png
```

`SHA256SUMS` 覆盖两份公开报告、两个复现脚本和文档图。数值指标由完整 NIfTI 计算，图像不参与指标计算。
