# FLIRT 验证资产

[功能与参数](../../docs/flirt/README.md) · [最新批量 GPU 报告](gpu_batch.current.public.json)

最新报告覆盖真实 6-DOF b0→T1、12-DOF T1→MNI152 及公开 T1w→T1w。批量与原串行路径的保存矩阵、体素、header、cost、评价次数和已捕获的角度候选一致。报告保留实际测量源码与工具 SHA-256；私有病例只发布汇总标量。旧报告的 hash 不改写成当前代码 hash。本次清理另外记录新源码 hash、240 项相关测试及真实公开 T1w 的 cold/warm 回归；保存矩阵、体素、header、cost 和 8,987 次求值均与清理前一致，见报告的 `cleanup_regression`。

| 文件 | 用途与范围 |
|---|---|
| `gpu_batch.current.public.json` | 当前批量路径的 cold/warm 时间、FSL 精度、阶段、完整 profile、显存和回归 gate；公开示例图的来源与 hash。 |
| `report.cpu.current.json` | 原串行 12-DOF GM 10 例 CPU/FSL 逐例精度及完整命令时间。 |
| `report.gpu.current.json` | 同一 GM 数据的前 4 例串行 H100/FSL 对照；共享 GPU 满载时的时间，不代表当前默认批量性能。 |
| `applyxfm_mni.cpu.json`、`applyxfm_mni.gpu.json` | FSL 原始 MNI152 T1 模板 1 mm↔2 mm 的 CPU/H100 双向重采样；模板、矩阵及源码 hash。 |
| [`1→2 mm 矩阵`](../../src/fnit/flirt/assets/FSL_MNI152_T1_1mm_to_2mm.mat) | 上述原始模板对应的 FSL scaled-mm 矩阵；模板影像不随包发布。 |
| `benchmark_applyxfm.py` | 在验证环境生成 FSL oracle 并重测已知矩阵的重采样。 |
| `validate_real.py` | 重跑 GM 队列；分别生成 CPU/GPU 报告，记录当前运行模块 hash、实际 execution 和日期。 |
| `plot_public_example.py` | 从已有 input、reference 和配对 moved 图像生成三视图；不重新配准或重复计算基准指标。 |
| `SHA256SUMS` | 本目录报告、脚本、公开图及当前性能工具的校验值。 |

GM 两份报告由串行 `core.py` `56a934…` 测得，随后 `f2c530…` 的 12-DOF 分支由[历史源码范围核对](../runtime_dependencies/flirt_profile_source_equivalence.public.json)和一例真实矩阵检查确认继承；该记录不是当前批量版的十例重跑。CPU 的 `rmsdiff <= 0.05 mm` 为 9/10，原串行 H100 为 4/4。[旧 b0 CPU/FSL 对照](../connectome/original_ukb_flirt.public.json)的测量源码与计时边界也按原记录保留。重复的四例合并报告、旧 CPU 公开示例和已完成的 Surfa 迁移记录已由以上独立报告取代。

## 重跑当前 GPU 对照

使用 [benchmark_flirt_gpu.py](../../tools/benchmark_flirt_gpu.py)分别运行 `--execution reference` 和 `--execution batched`，再用 [check_flirt_gpu_parity.py](../../tools/check_flirt_gpu_parity.py)检查保存输出与搜索记录。完整命令、cold/warm 与 profile 定义见[功能页](../../docs/flirt/README.md#重跑性能与精度对照)。两条 FNIT 路径读取预先保存的 FSL oracle，运行中不调用 FSL。

## 阶段与 profile

公开 RTX 3060 病例的完整 profile：kernel launch 1,304,043→114,928，stream synchronize 161,816→841，`cudaMemcpyAsync` 总调用 197,772→2,406；CUDA tensor `float()` / `bool()` 从 26,971 / 17,974 降为 10 / 0。WSL 没有 device timeline，kernel 数来自实际 runtime launch 调用，不能据此推断 GPU kernel 时间或 copy 方向。

Linux H100 的完整 device event 与 runtime launch 计数如下（串行→批量）：

| 指标 | b0→T1，6-DOF | T1→MNI152，12-DOF |
|---|---:|---:|
| kernel 数 | 923,354→75,533 | 1,082,773→103,759 |
| `cudaStreamSynchronize` | 101,092→497 | 134,348→761 |
| Host→Device copy events | 23,903→468 | 29,863→740 |
| Device→Host copy events | 77,189→458 | 104,485→720 |
| `cudaMemcpyAsync` 总调用 | 118,542→1,355 | 164,200→2,946 |

copy event 与 runtime API 是不同计数，后者还可包含其他方向。独立 profile 的矩阵、影像、header、cost 和求值次数另与无 profiler 输出核对；profile 耗时不用于速度表。

另一次无 profiler 的 H100 cold 阶段测量，单位秒（串行→批量）：

| 阶段 | b0→T1，6-DOF | T1→MNI152，12-DOF |
|---|---:|---:|
| reference pyramid | 0.188→0.219 | 0.143→0.137 |
| 角度搜索，含 8 mm refinement | 244.21→1.98 | 207.68→1.96 |
| 4 mm 局部优化 | 78.05→1.30 | 150.20→1.44 |
| 2 mm 局部优化 | 3.08→0.40 | 7.61→0.72 |
| 1 mm 局部优化 | 4.55→0.52 | 10.08→0.63 |
| 最终重采样 dispatch | 0.088→0.090 | 0.137→0.110 |

时间是 host wall scope，CUDA 异步准备工作可能在下一阶段完成。8 mm refinement 嵌套于角度搜索，不能重复相加；最后一行不含随后等待回传的时间。H100 共享负载不同，阶段测量与 cold/warm 总时间属于不同运行，不混合计算加速比。最早 H100 cold/warm 缺少工具 hash，6-DOF 当时也缺 `_nib.py` hash；报告保留缺项，新阶段测量和完整 profile 记录各自实际 hash。

## 重跑 GM 队列

UKB VBM 保存的 moved 图像包含后续 mask，不能作为 FLIRT 直接输出。先用已有官方矩阵生成独立 oracle：

```bash
flirt -in subject_GM.nii.gz -ref template_GM.nii.gz \
  -applyxfm -init T1_GM_to_template_GM.mat -out official_apply.nii.gz
```

再运行 FNIT，以下目录由验证者按本地数据填写：

```bash
python validation/flirt/validate_real.py run \
  --study-root <deidentified_study_root> --source-root <fnit_source_root> \
  --official-moved-root <fresh_fsl_applyxfm_root> \
  --output-root <candidate_cpu_root> --reference-report <fsl_timing_report.json> \
  --report <new_cpu_report.json> --case-count 10 \
  --device cpu --execution reference --threads 4 \
  --candidate-hardware "Intel Xeon Gold 6418H"
```

队列目录与参考计时记录格式由脚本定义。重跑当前 GPU 改为 `--device cuda:0 --execution batched`，硬件填写实际型号；重跑串行 GPU 对照用 `--execution reference`。CPU/GPU 分别输出报告，不将 10 例与 4 例合成同一队列。脚本每次重新运行候选，避免将旧输出绑定到当前源码 hash。

公开图重画命令：

```bash
python validation/flirt/plot_public_example.py \
  --moving examples/data/sub-02_T1w.nii.gz \
  --reference examples/data/sub-01_T1w.nii.gz \
  --fsl-moved <fsl_oracle.nii.gz> --fnit-moved <batched/cold_0.nii.gz> \
  --figure docs/flirt/figures/flirt_public_current.png
```
