# dMRI 参数图验证

公开验证使用 1 例去标识化的真实 UKB 格式 AP/PA 采集数据。本仓库不保存源图像和被试标识。

[`tbss_diagnosis.public.json`](tbss_diagnosis.public.json) 让两种实现从同一组已准备好的 9 张 native DTI/NODDI 参数图开始。Candidate A 使用已发布的 TorchFLIRT + TorchFNIRT 流程。Candidate B 固定使用 FSL 官方 FLIRT 矩阵，以单独评估 TorchFNIRT。两条路径随后都使用 TorchApplyWarp 和 UKB skeleton multiplication 规则。

当前真实数据结果如下：

| 路径 | standard-map Pearson 范围 | skeleton-map Pearson 范围 | 实测时间 |
|---|---:|---:|---:|
| Candidate A | 0.995086–0.998886 | 0.996789–0.999463 | Python pipeline 106.186 s；external wall 121.48 s |
| Candidate B | 0.997483–0.999523 | 0.994425–0.999679 | Python pipeline 18.884 s |
| Official FSL/UKB | reference | reference | external wall 976.88 s |

Candidate B 不包含 affine optimization。测试节点为共享节点，未隔离其他负载。Candidate A 和 Candidate B 的全部 standard 与 skeleton 图像均与官方结果的 shape、affine 和 dtype 一致。

stage-1 oracle 记录初始状态和 level-1 的全部 10 个 LM candidate。Torch 与 FSL 的 mask voxel count 相同，接受序列均为
`accept,accept,accept,reject,reject,reject,reject,accept,reject,accept`。
完整 stage-1 coefficient Pearson 在 CPU 上为 0.999832，在 GPU 上为 0.999879；对应的 warped-FA Pearson 分别为 0.999004 和 0.999148。

已修正的语义错误来自 warped input mask。FSL 将 binary mask 重采样到 `volume<char>`，先截断 trilinear interpolation 在边界产生的小数值，再由 `Mask()` 执行 `>0.5` 判断。当前实现还与 newimage 的 1e-8 valid-FOV tolerance、raw floor/zero-padded boundary behavior、1e-16 implicit zero rule，以及 float32 mean normalization 后构建 input mask 的顺序一致。

`fsl_fnirt_numerically_equivalent` 和 `ukb_tbss_numerically_equivalent` 仍为 false。后续 sparse-BFMatrix 与 matrix-free FP64 reduction 尚未数值一致；本次运行未触发 topology projection；外部验证目前仅包含 1 例。针对性测试命令为
`pytest -q tests/fnirt tests/dmri_pipeline/test_dmri_pipeline.py`，结果为
`63 passed`。

## 复现工具

[`run_official_tbss.sh`](run_official_tbss.sh) 依次运行官方 weighted FLIRT、3 个 FNIRT stage、`applywarp` 和 skeleton multiplication。输入参数为：

```text
OUTPUT_DIR PREPROCESSED_FA INPUT_WEIGHT MAP_DIR \
FA_REFERENCE FA_SKELETON CONFIG_PREFIX
```

[`compare_tbss.py`](compare_tbss.py) 计算 full-volume Pearson、MAE、RMSE，并检查 shape、affine 和 dtype：

```bash
python validation/dmri_pipeline/compare_tbss.py \
  --fnit-dir subject_tbss \
  --fsl-dir official_tbss \
  --output comparison.json
```

需要验证 raw-to-native 时，可用 [`compare_native.py`](compare_native.py) 比较 9 张 native 参数图。[`plot_tbss.py`](plot_tbss.py) 生成 FNIRT 和 dMRI 页面使用的真实 FA 对比图：

```bash
python validation/dmri_pipeline/plot_tbss.py \
  --official official_tbss/stats/all_FA.nii.gz \
  --fnit subject_tbss/standard/FA.nii.gz \
  --output docs/fnirt/figures/fnirt_fsl_comparison.png
```

`SHA256SUMS` 覆盖当前 diagnosis 文件、4 个复现工具和文档中的真实数据图。
