# FastVBM 验证

[返回 FastVBM 文档](../../docs/fast_vbm/README.md) · [TorchFNIRT 验证](../dmri_pipeline/README.md) · [FLIRT 验证](../../docs/flirt/README.md)

## 数值运行冻结快照：真实单例

2026 年 9 月 27 日用 1 例去标识化真实临床 T1w，在 gpucw1 NVIDIA H100 PCIe 上运行数值冻结快照。两条路径从同一 raw T1w 开始，共用 SynthStrip、TorchFAST、TorchFLIRT、TorchApplyWarp、Jacobian 和 modulation，只替换 nonlinear estimator。

冻结快照的 `fast_vbm/registration.py` 为 `02da3e…`，当前文件为 `447892…`。这次只删除私有 `initial_pull` affine bypass；公开 `FastVBM.__call__` 和 `FastVBM.run` 此前没有接收或传递该参数，实测路径始终取 `initial_pull=None` 并运行 TorchFLIRT。把旧文件规范到这条生产路径后，location-free AST 与当前文件相同（两边 SHA-256 均为 `47254a…`）；gpucw1 上相关 17 个受控测试全部通过。三份报告保留实测源码 hash，并明确 `fresh_current_hash_full_real_data_rerun=false`。

冻结快照的 `flirt/core.py` 为 `552856…`，当前文件为 `ce375d…`。源码继承分两段：[`552856… → f5315f…`](../runtime_dependencies/flirt_qc_source_equivalence.public.json) 只清理 runtime QC；[`f5315f… → ce375d…`](../runtime_dependencies/flirt_profile_source_equivalence.public.json) 只在本流程使用的 12-DOF/corratio 配置下等价。报告保留原实测 hash，并标记没有 fresh current-hash 完整重跑；已改变的 6-DOF/normmi 路径不在证明范围内。

两份分支报告共用一份源码快照清单，因此 FNIRT 报告也列出了旧 `synthmorph/pipeline.py` `e680d3…`，但 FNIRT 分支没有执行该文件，报告中的结构化状态明确不主张数值继承。SynthMorph 分支实际执行 `SynthMorph.__call__` 的 deform registration 和 linear 重采样；它通过 [SynthMorph linear 源码等价证明](../runtime_dependencies/synthmorph_linear_source_equivalence.public.json)继承到当前 `pipeline.py` `70e97c…`、`spatial.py` `dab615…`，同样没有 fresh current-hash 完整重跑。该分支清单当时漏记 `spatial.py`，这一出处限制已写入报告；nearest 不使用本次结果作证。

FSL reference 采用该病例固定的 FAST/FNIRT 工件。验证发现归档的 `T1_GM_2mm_to_template_GM` 已在 VBM 脚本中被 Jacobian 原位调制；因此先用官方 coefficient 重新生成未调制 warped GM。重新相乘后的结果与归档文件逐体素完全一致。这个角色核验避免把 modulated GM 错当作 warped GM。

| 后端 | warped GM r | Jacobian r | modulated GM r | 外部 wall | 峰值 CUDA allocation |
|---|---:|---:|---:|---:|---:|
| TorchFNIRT | 0.559420 | 0.237607 | 0.487910 | 62.23 s | 12.970 GB |
| SynthMorph | 0.636498 | 0.328663 | 0.575795 | 75.36 s | 15.487 GB |

两分支的 shape、affine 和 dtype 契约全部通过。连续值相似度明显低于数值等价要求，所以 `numerical_equivalence_passed=false`。FNIT 的 raw T1w 和 FSL FAST GM 已位于不同 native 网格；结果同时反映上游 bias correction、裁剪、脑提取、组织分割、affine 和 nonlinear registration 的差异，不能只用来评价某一个配准器。

FSL 6.0.7.4 固定官方 coefficient 的 `applywarp` 为 2.28 s，随后 `fslmaths` modulation 为 0.56 s。两项都包含各自读写，但不包含上游计算；本目录不据此计算加速比。

## 文件

- [`report.real.current.json`](report.real.current.json)：两分支汇总、源码哈希、官方文件角色核验和验收状态。
- [`report.fnirt.real.current.json`](report.fnirt.real.current.json)：TorchFNIRT 分支的完整三维指标、阶段时间和输出哈希。
- [`report.synthmorph.real.current.json`](report.synthmorph.real.current.json)：SynthMorph 分支的完整三维指标、阶段时间和输出哈希。
- [`validate_real.py`](validate_real.py)：单分支复现脚本。
- [TorchFNIRT 图](../../docs/fast_vbm/figures/fast_vbm_fnirt_real.png)和 [SynthMorph 图](../../docs/fast_vbm/figures/fast_vbm_synthmorph_real.png)：同切面 FSL reference、FNIT 和绝对差。

报告不含原始图像路径和受试者标识。当前精度范围为一例真实 T1w；官方计时从固定中间结果开始，与 FNIT raw-to-VBM 不同边界，因此不计算端到端加速比。
