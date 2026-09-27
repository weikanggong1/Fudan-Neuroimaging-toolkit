# FLIRT 验证资产

本目录保留最新一轮真实数据测量，以及测量源码到当前源码的可核验继承关系。真实数据报告由 `flirt/core.py` `f5315f…` 生成；当前文件为 `ce375d…`。两者仅在 **12-DOF/corratio** 配置下源码等价，且没有以当前 hash 重新跑完整真实数据。证明见 [`flirt_profile_source_equivalence.public.json`](../runtime_dependencies/flirt_profile_source_equivalence.public.json)。当前 6-DOF/normmi 的 8 mm 搜索路径已经改变，不在该证明范围内。

| 文件 | 内容 |
|---|---|
| `report.public.json` | 10 例真实 GM 的 CPU/GPU 合并报告；含源码 hash、逐例矩阵与影像指标、时间和显存。 |
| `report.cpu.current.json` | `f5315f…` 测量源码的 CPU 逐例报告；保留原 source hash。 |
| `report.gpu.current.json` | `f5315f…` 测量源码的 H100 默认 TF32 逐例报告；保留原 source hash。 |
| `public_example.current.json` | OpenNeuro ds000114 公开 T1w 示例的来源、命令、hash、指标和时间。 |
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

当前矩阵门限为 `rmsdiff <= 0.05 mm`。CPU 和 GPU 均通过 9/10 例；因此报告把 `numerically_equivalent` 保持为 `false`。
