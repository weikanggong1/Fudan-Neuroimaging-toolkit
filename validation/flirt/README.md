# FLIRT 验证资产

本目录保留修订后的真实数据测量。12-DOF GM 的 10 例 CPU 和 4 例 H100 报告由 `flirt/core.py` `56a934…` 生成；最终源码 `f2c530…` 只在候选自由度处增加 `min(dof, 7)` 分支。对 12-DOF 该值仍为 7，一例真实 GM 的最终 `.mat` 与测量源码逐元素一致。该核对不等于最终源码的 10 例完整重跑；详见[源码范围记录](../runtime_dependencies/flirt_profile_source_equivalence.public.json)。最终源码直接完成了一例 6-DOF b0→T1 的 CPU/GPU 对照和一例公开 12-DOF T1w→T1w 对照。

| 文件 | 内容 |
|---|---|
| `report.public.json` | 真实 GM 前 4 例的 CPU/H100 配对报告；含源码 hash、逐例矩阵与影像指标、观察时间和显存。 |
| `report.cpu.current.json` | 10 例真实 GM CPU 逐例报告，保留测量源码 hash。 |
| `report.gpu.current.json` | 前 4 例真实 GM 的 H100 默认 TF32 报告；运行时有其他 GPU 作业。 |
| `public_example.current.json` | OpenNeuro ds000114 公开 T1w 示例的来源、命令、hash、指标和时间。 |
| `applyxfm_mni.cpu.json`、`applyxfm_mni.gpu.json` | FSL 原始 MNI152 T1 模板 1 mm↔2 mm 双向重采样的 CPU/H100 精度、耗时和源码哈希；不含影像。 |
| `benchmark_applyxfm.py` | 在装有 FSL 的验证环境重跑上述对照；FNIT 运行时不使用它。 |
| `validate_real.py` | 运行候选、用 fresh FSL `-applyxfm` 输出计算指标，并合并 CPU/GPU 报告。 |
| `plot_public_example.py` | 生成公开示例 JSON 和 `docs/flirt/figures/flirt_public_current.png`。 |
| `SHA256SUMS` | 上述报告、脚本和图的 SHA-256。 |

## 参考输出为何重新生成

UKB VBM 目录中的 `T1_GM_to_template_GM.nii.gz` 已经过后续 mask 等处理，不能作为 FLIRT 直接输出。验证保留该目录中的官方 `.mat`，再用 FSL 6.0.7.4 对原始 GM 运行：

```bash
flirt \
  -in subject_GM.nii.gz \
  -ref template_GM.nii.gz \
  -applyxfm \
  -init T1_GM_to_template_GM.mat \
  -out official_apply.nii.gz
```

`official_apply.nii.gz` 是官方矩阵的直接重采样结果。候选 moved 图像必须与它比较。原先把 VBM 后处理图当作参考时得到的 Pearson 约 0.6–0.7，反映的是处理阶段不一致，不是 FLIRT 重采样精度；该错误报告已删除。

## 重新运行

以下示例中的路径由验证者按本地数据位置填写。FSL 只生成参考输出，候选命令不调用 FSL。

```bash
python validation/flirt/validate_real.py run \
  --study-root <deidentified_study_root> \
  --source-root <fnit_source_root> \
  --official-moved-root <fresh_fsl_applyxfm_root> \
  --output-root <candidate_cpu_root> \
  --reference-report <same_input_fsl_timing_report.json> \
  --report report.cpu.current.json \
  --case-count 10 \
  --device cpu \
  --threads 4 \
  --candidate-hardware "Intel Xeon Gold 6418H"
```

GPU 运行把 `--device` 改为 `cuda:0`，并把硬件写为 `NVIDIA H100 PCIe 80 GB`。CUDA 默认使用 TF32，没有使用 float16 或 bfloat16。两次运行完成后合并：

```bash
python validation/flirt/validate_real.py combine \
  --cpu-report report.cpu.current.json \
  --gpu-report report.gpu.current.json \
  --report report.public.json
```

`--reuse-existing --process-log <log>` 可在不重复优化的情况下，使用已保存的 `caseNN/candidate.nii.gz`、`candidate.mat` 和 `worker.json` 重新计算报告。日志每行格式为 `caseNN matrix_rmsdiff process_wall_seconds`。

当前矩阵门限为 `rmsdiff <= 0.05 mm`。CPU 通过 9/10 例，H100 已完成的 4 例通过 4/4；GPU 全 10 例尚无本次修订后的测量。共享节点的非同步时间不能计算加速比。最终源码的 6-DOF 同输入对照见[刚性配准报告](../connectome/original_ukb_flirt.public.json)。
