# 组合场：真实 DWI 对照

[函数输入、输出和调用方法](../../docs/convertwarp/README.md) · [反场及掩膜对照](../invwarp/README.md)

在 gpucw1 使用同一例真实 DWI 已保存的 FNIT dMRI pipeline TBSS 和 MMORF 配准结果。TBSS 的 `dti_FA_to_MNI_warp.nii.gz` 是包含 affine 的 FNIRT 系数场，不再额外传入 `dti_FA_to_MNI_affine.mat`。FSL 6.0.7.4 与 FNIT 分别转换该场；输出均为 `182×218×182×3`、float32、相同 affine，分量 MAE 为 `1.984×10⁻⁶ mm`，最大差 `9.54×10⁻⁶ mm`。FSL 完整命令耗时 49.73 秒，FNIT GPU Python 调用耗时 10.02 秒；计时边界不同。MMORF 原场不能直接交给 FSL `convertwarp`，FNIT 将它与 pipeline 保存的 FA→MNI 矩阵转换为 FSL dense 场。该场重采样 FA 与 pipeline 原 `apply_mmorf_warp` 输出在非零并集上的 `r=0.999999999996`、MAE `2.14×10⁻⁷`；转换 Python 调用耗时 7.27 秒。后续同输入 FSL `invwarp` 与 `applywarp` 对照见[反场验证](../invwarp/README.md)。

[TBSS/FSL 机器报告](../invwarp/report.pipeline_tbss_fsl.public.json)、[MMORF/FSL 机器报告](../invwarp/report.pipeline_mmorf_fsl.public.json)和[两条 pipeline 报告](../probtrackx/README.md)记录源码哈希、输出和计时。下例中的 `TBSS`、`MMORF` 分别指同一被试两套 pipeline 的输出根目录；`OUT` 是对照输出目录。

```bash
# TBSS：FNIRT 系数已含 FA→MNI 线性配准；--ref 是 MNI 网格。
convertwarp --ref="$TBSS/registration/standard/FA.nii.gz" \
  --warp1="$TBSS/registration/dti_FA_to_MNI_warp.nii.gz" \
  --out="$OUT/fsl_tbss_diff2mni.nii.gz" --relout
fnit convertwarp --ref "$TBSS/registration/standard/FA.nii.gz" \
  --warp1 "$TBSS/registration/dti_FA_to_MNI_warp.nii.gz" \
  --out "$OUT/fnit_tbss_diff2mni.nii.gz" --relout --device cuda:0

# MMORF：--source 是生成 FLIRT 矩阵的原始 FA；--premat 是 FA→MNI 矩阵。
fnit convertwarp --ref "$MMORF/registration/standard/FA.nii.gz" \
  --source "$MMORF/native/dti_FA.nii.gz" \
  --mmorf-warp "$MMORF/registration/mmorf_warp.nii.gz" \
  --premat "$MMORF/registration/dti_FA_to_MNI_affine.mat" \
  --out "$OUT/mmorf_diff2mni_FSL.nii.gz" --device cuda:0
```

这是一例的同输入数值核对。非单位 `premat`、`postmat` 的组合顺序另由 `tests/convertwarp/test_warp_composition.py` 验证，不将该合成单元测试计入真实数据 benchmark。
