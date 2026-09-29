# MMORF 真实数据验证

[返回功能说明](../../docs/mmorf/README.md)

[`report.public.json`](report.public.json) 记录一例真实 T1w、DTI FA 和 FSL 六通道 tensor 的双标量联合注册。FNIT 和官方 MMORF 0.3.2 使用相同图像、模板、五级计划以及相同的 FSL scaled-mm 矩阵。FNIT 的第二组 FA 矩阵由内部 PyTorchFLIRT 自动估计，再导出给官方配置。两组标量权重均为 0.5。报告分别给出线性、非线性和含写盘时间；共享 GPU 测试前后的负载及 FNIT 进程显存上限单独记录。

精度比较包含 warp、Jacobian、warped T1 和 warped FA 的完整 3D Pearson、MAE、RMSE。两张 warped scalar 都使用同一个 FNIT cubic sampler，仅替换 FNIT 或官方估计的 warp，因此采样器和线性矩阵保持不变。报告中的输入路径和病例标识已去除；真实 NIfTI 不随包分发。

## FNIT 运行

以下命令中的路径对应一例真实数据。`--mov-scalar` 与 `--ref-scalar` 按出现顺序成对；`AUTO` 表示该组缺失线性矩阵，由 PyTorchFLIRT 估计。`--memory-fraction 0.24` 对一张 H100 PCIe 约为 19.1 GiB 的进程上限，脚本记录运行前后的全卡占用。

```bash
PYTHONPATH=src python validation/mmorf/measure_real_multiscalar.py \
  --mov-scalar t1_brain.nii.gz \
  --ref-scalar MNI152_T1_1mm_brain.nii.gz \
  --mov-scalar dti_FA.nii.gz \
  --ref-scalar FSL_HCP1065_FA_1mm.nii.gz \
  --mov-tensor dti_tensor.nii.gz \
  --ref-tensor FSL_HCP1065_tensor_1mm.nii.gz \
  --aff-mov-scalar t1_to_MNI.mat \
  --aff-mov-scalar AUTO \
  --aff-mov-tensor FA_to_MNI.mat \
  --output-dir fnit_multiscalar \
  --device cuda:0 \
  --memory-fraction 0.24
```

`--mov-tensor` 和 `--ref-tensor` 是 `[X,Y,Z,6]` 的 FSL 张量；`--aff-mov-tensor` 映射 moving tensor 到第一张 reference scalar。未给的 reference 模态矩阵在同网格时为 identity。脚本在目录内生成 MMORF 结果、`mmorf_report.json` 以及额外的 `mmorf_benchmark.json`。

## 官方命令与配对比较

将 `mmorf_report.json` 中 `linear_alignment.moving_scalar[1].matrix` 写为 `FA_auto.mat`，按[功能页示例](../../docs/mmorf/README.md)准备官方 `multimodal.ini`，运行：

```bash
FSLOUTPUTTYPE=NIFTI_GZ mmorf --config multimodal.ini
```

此命令仅用于验证；FNIT 运行时不调用官方 MMORF。官方产物为 `official_warp.nii.gz` 和 `official_jacobian.nii.gz`。生成逐体素比较与两个官方 warp 的匹配采样输出：

```bash
PYTHONPATH=src python validation/mmorf/compare_real_multiscalar.py \
  --fnit-dir fnit_multiscalar \
  --official-dir official_multiscalar \
  --mov-t1 t1_brain.nii.gz \
  --mov-fa dti_FA.nii.gz \
  --ref-t1 MNI152_T1_1mm_brain.nii.gz \
  --device cuda:0 \
  --memory-fraction 0.24 \
  --output comparison.json
```

输出 `comparison.json` 给出数值与 shape/affine/dtype 合同；`official_multiscalar/sampled_T1.nii.gz` 和 `sampled_FA.nii.gz` 用于重建对照图。`plot_mmorf.py` 绘制同网格、同显示范围的 T1 与 FA 切片。`SHA256SUMS` 校验当前公开报告、脚本和图。
