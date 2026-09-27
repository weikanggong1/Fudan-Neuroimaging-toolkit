# dMRI 参数图验证

[返回 dMRI pipeline 文档](../../docs/dmri_pipeline/README.md) · [MMORF 文档](../../docs/mmorf/README.md) · [FNIRT 文档](../../docs/fnirt/README.md)

两份数值报告使用源码快照 tar `f7547d0a…`，其中 FLIRT core 为 `552856…`；当前文件为 `ce375d…`。继承链包含 [QC-only 的 `552856… → f5315f…`](../runtime_dependencies/flirt_qc_source_equivalence.public.json) 和 [12-DOF/corratio 限定的 `f5315f… → ce375d…`](../runtime_dependencies/flirt_profile_source_equivalence.public.json)。两份报告保留原测量 hash，新增的 chain 对象明确 `fresh=false`。该证明不覆盖 6-DOF/normmi。
公共包入口也保留测量 hash：`__init__.py`/`cli.py` 为 `cc9aa4…`/`2d5e3d…`，
当前 0.16.0 为 `be1cab…`/`c92a3f…`。[包入口源码等价证明](../runtime_dependencies/package_entry_source_equivalence.public.json)
归一化版本号、移除独立的 fMRI 懒加载分支（含 13 个 surface API），并过滤五个主动撤下的内部实现名称后，
保留 API 的新旧 AST SHA-256 均为 `fd295c…`。这五个名称不主张 API 兼容；证明仅限
`dmri-pipeline` 动态 parser 和 handler 路径，并明确 `fresh=false`。

TBSS 与 MMORF 两份 source manifest 都没有记录 SynthMorph 文件，两条运行路径也不调用 SynthMorph；报告已加入 `synthmorph_source_status`，明确 `executed=false`、不适用 SynthMorph linear attestation，也不主张相关数值继承。

## 实测冻结快照：MMORF raw-to-standard

2026 年 9 月 28 日用 1 例去标识化真实 UKB 格式 AP/PA dMRI 和配对 T1w，在 gpucw1 H100 上跑完冻结快照的 `DMRIPipeline(registration_backend="mmorf")`：

```text
AP/PA -> TOPUP -> EDDY -> DTIFIT + AMICO-NODDI
      -> SynthStrip T1 -> two FLIRT -> MMORF -> 9 standard maps
```

运行从原始 AP/PA 开始，外部 wall 为 563.06 s，内部总计 558.100 s。九张 standard 图全部通过 shape、affine 和 dtype 合同。最大组件级 CUDA allocation 为 13.669 GB。源码快照 tar SHA-256 为 `f7547d0a39ddd9fb6ba70deb720f229ecedc6385fa72d457efb2ded78b6c173d`。

| 图 | standard Pearson r | MAE | RMSE |
|---|---:|---:|---:|
| FA | 0.559845 | 0.099869 | 0.151588 |
| MD | 0.563159 | 3.677e-4 | 5.874e-4 |
| L1 | 0.535731 | 4.042e-4 | 6.454e-4 |
| L2 | 0.563184 | 3.769e-4 | 5.930e-4 |
| L3 | 0.585998 | 3.622e-4 | 5.631e-4 |
| MO | 0.329829 | 0.370717 | 0.481165 |
| ICVF | 0.372548 | 0.146745 | 0.223835 |
| OD | 0.540236 | 0.154255 | 0.220781 |
| ISOVF | 0.528075 | 0.177561 | 0.269235 |

reference 由既有 UKB native 参数图和 FSL MMORF 0.3.2 warp 生成，传播时沿用了先前验证固定的 FNIT affine，不是全部阶段由官方软件重跑的 raw-to-standard reference。native-map r 为 -0.002070–0.634022，表明差异在配准前已存在。因此当前结论为：完整运行成功，输出合同通过，数值等价未通过。

本次启动时同卡已有其他训练，占用 22,246.8 MiB 且利用率为 100%，所以 wall time 不是隔离性能值。FSL MMORF 既有记录为 947.62 s，只含 registration；范围和负载均不同，不计算加速比。

- [`mmorf_e2e.real.current.json`](mmorf_e2e.real.current.json)：九张 native/standard 图指标、阶段时间、显存、输入与源码哈希。
- [`validate_mmorf_e2e.py`](validate_mmorf_e2e.py)：比较和作图脚本。
- [真实 FA 图](../../docs/dmri_pipeline/figures/dmri_mmorf_fa_real.png)：official reference、FNIT current 和绝对差。

## 实测冻结快照：TBSS raw-to-standard

2026 年 9 月 28 日用同一例去标识化真实 UKB 格式 AP/PA dMRI，在 gpucw1 H100 上跑完冻结快照的 `DMRIPipeline(registration_backend="tbss")`：

```text
AP/PA -> TOPUP -> EDDY -> DTIFIT + AMICO-NODDI
      -> weighted FLIRT -> three-stage FNIRT -> 9 standard maps -> 9 skeleton maps
```

实际入口为 `python -m fnit.cli dmri-pipeline`，参数包括 `--raw-dir <RAW_DIR> --output-dir <OUTPUT_DIR> --registration-backend tbss --fa-template <FMRIB58_FA_1mm.nii.gz> --fa-skeleton <FMRIB58_FA-skeleton_1mm.nii.gz> --device cuda`。源码快照 tar SHA-256 为 `f7547d0a39ddd9fb6ba70deb720f229ecedc6385fa72d457efb2ded78b6c173d`；报告还逐文件记录完整调用链哈希。

| 图 | native r | standard r | skeleton r |
|---|---:|---:|---:|
| FA | 0.586157 | 0.694611 | 0.561165 |
| MD | 0.547537 | 0.600453 | 0.447009 |
| L1 | 0.436994 | 0.535743 | 0.413255 |
| L2 | 0.567251 | 0.613368 | 0.504887 |
| L3 | 0.634022 | 0.658698 | 0.540063 |
| MO | 0.413579 | 0.464303 | 0.622653 |
| ICVF | -0.002070 | 0.313822 | 0.107073 |
| OD | 0.506862 | 0.592098 | 0.605502 |
| ISOVF | 0.608481 | 0.624363 | 0.525343 |

九张 standard 图和九张 skeleton 图均通过 shape、affine、float32 dtype 合同；逐体素数值等价未通过。native-map r 已为 -0.002070–0.634022，说明差异在 FLIRT/FNIRT 前已经存在。

候选外部 wall 为 444.93 s，内部总计 440.092 s；其中配准和九图传播为 213.520 s。最大组件级 CUDA allocation 为 13.664 GB。启动时同卡已有其他训练，占用 20,869 MiB 且利用率为 100%，所以 wall time 不是隔离性能值。FSL 6.0.7.4 reference 的 976.88 s 从 prepared FA 和九张 native 参数图开始；范围和负载均不同，不计算加速比。

- [`tbss_e2e.real.current.json`](tbss_e2e.real.current.json)：九张 native、standard、skeleton 图指标，阶段时间，显存，输入、reference 和源码哈希。
- [`validate_tbss_e2e.py`](validate_tbss_e2e.py)：当前报告与真实 FA 图的生成脚本。
- [`run_official_tbss.sh`](run_official_tbss.sh)：FSL 6.0.7.4 weighted FLIRT、三次 FNIRT、applywarp 与 skeleton reference 命令。
- [真实 FA 图](../../docs/dmri_pipeline/figures/dmri_tbss_fa_real.png)：official FSL/UKB reference、FNIT current 和绝对差。
