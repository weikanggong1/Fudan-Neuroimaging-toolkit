# fMRI volume 与 surface：验证索引

[volume 用法](../../docs/fmri/README.md) · [surface 用法](../../docs/fmri/surface.md) · [代码职责](../../src/fnit/fmri/README.md)

本页只索引当前流程与可复核报告。用法、全部参数、流程图和原软件命令集中在功能文档；不同日期、输入和源码的测量各自保留，不拼成一次新的 benchmark。

## 当前流程与验证范围

- `fMRIVolume_pipeline`：一个原始 BIDS BOLD run，同时生成 preproc 与 clean；支持 SynthMorph/FNIRT 配准。默认关闭 STC，当前完整报告未执行 SDC/GDC。
- `fMRISurface_pipeline`：同一 run 的 volume 输出和已有同源 recon-all，生成 fsLR32k/91k；默认 preproc，可显式选择 clean。完整 surface 测量包含独立双侧 MSM、投影、QC 和保存，不包含 volume/recon-all。
- 官方程序只用于独立参照。旧 FNIT 回归、固定变换/球面算子控制、独立官方整链是不同验收层。

## 最近已完成的真实测量

| 测量与冻结版本 | 起点 / 范围 | 精度与耗时摘要 | 完整记录 |
|---|---|---|---|
| Volume 公共采样入口，2026-10-02，`81f1bb3` 发布 | 同一真实 490 帧 raw BIDS；两个后端分别与冻结旧 FNIT 比较；解剖缓存关闭 | FNIRT / SynthMorph 全输出逐位相同；API 含保存 542.12 / 461.54 s；allocated 6.50 / 13.31 GB | [public resamplers](public_resamplers_20261002/README.md) |
| FNIRT 完整 pipeline，2026-10-02 上一轮 | FastVBM、volume、dMRI 各自实际执行 FNIRT | 固定算子与完整链结果、分步骤时间分别记录 | [三条完整链](../registration_lossless_20261002/README.md) |
| Surface 执行优化，`9f9f63e` | 同一完整 490 帧 volume 与同源重建；旧串行、新串行、新并行各自估计 MSM | 旧/新 FNIT 球面及完整时序相同；API 436.245 / 343.056 / 243.695 s；独立官方皮层差异仍在 | [surface GPU/并行](surface_gpu_parallel/README.md) |
| Surface 共同输入官方投影，`ca3df003` | 同一完整 volume、white/pial/graymid、球面、面积面和 ROI | 双侧 32k GIFTI 与 91k CIFTI 全部 490 帧逐值相同 | [固定投影](fmriprep/surface_stcoff_ca3df003_projection_paired.public.json) |

这些时间保留各报告的计时边界、共享负载及实际设备。当前清理没有新测完整 pipeline，不把旧结果改标为本轮测量。

## 尚未关闭的官方对照

| 项目 | 已记录结果与结论 | 证据 |
|---|---|---|
| clean FSL 样条机制 | FNIT Periodic 与 FSL Constant/extraslice 不同；最近邻 World 舍入也不同。固定 warp 高相关不能证明机制相同 | [重采样审计](resampling_audit_20261002/README.md)、[历史固定采样](resampling.md) |
| 官方实际 MNI preproc 节点 | 固定实际变换的完整 490 帧，T1w 门通过；MNI 12 帧无法由同输入串行源码回放复现，实际产物门仍失败，原因未确定 | [实际节点](fmriprep/actual_node_interpolation_full490.public.json)、[失败诊断](fmriprep/actual_node_mni_replay_failure.public.json) |
| 独立官方 surface | CIFTI 时间 r 均值 0.977911、relative RMSE 2.2418%；球面 L/R mean 角差 0.221050° / 0.319595°；19 个皮层下结构逐值相同 | [独立完整 surface](surface_gpu_parallel/parallel_vs_official_strict1.public.json) |
| 显式开启 STC | 真实 490 帧 max 0.021484375，高于原门槛 0.01；默认关闭 STC 的测量不包含此项 | [真实 STC 控制](fmriprep/stc_real_full490.public.json) |
| 独立 raw-BIDS volume | 两方独立 HMC、BBR、解剖及 MNI 估计仍产生非零差异；FNIT SynthMorph 与官方 ANTs 的变换不同 | [历史独立 MNI](fmriprep/independent_mni_stcoff_50eb098.public.json) |

## 历史记录与专项证据

| 记录 | 用途 |
|---|---|
| [2026-09-30](HISTORY_20260930.md)、[2026-10-01 clean](HISTORY_20261001_clean.md) | 早期 clean 与固定配准测量，按原源码冻结 |
| [STC 开启历史](HISTORY_20261001_STCON_PREPROC.md)、[STC 关闭历史](HISTORY_20261001_STCOFF_PREPROC.md) | 旧 preproc/surface、官方结构升级、固定节点与独立流程差异 |
| [MCFLIRT 完整 volume](mcflirt_optimization.md)、[原同步骤 clean](matched_native.md) | 先前完整 FNIRT volume/clean 的对照、分段计时及来源 |
| [旧 volume 修复](volume_fixed.md)、[固定 warp](resampling.md) | 混杂设计、边界、组织采样与同 warp 插值控制 |
| [Surface 独立完整链](surface_e2e/README.md)、[共同球面控制](surface_fixed_sphere.public.json) | 独立球面和共同输入投影的不同验收层 |
| [AROMA 空间控制](aroma_spatial_control.public.json)、[固定 ICA](ica_fixed_input.public.json)、[组织采样](tissue_sampling_cross.public.json) | 已实现子函数的固定输入定位 |
| [2026-10-01 文件管理检查](organization_20261001.public.json)、[DeepPrep](deepprep/README.md) | 较短 run 覆盖检查，以及其他处理协议的独立参照 |

报告和失败门仍承担复核作用。历史命令在各自冻结源码下执行；当前命令从下一节进入。

## 复测

在主页 Conda 环境运行当前驱动；全部参数见功能页：

```bash
# 输入原始 BIDS，生成 preproc 与 clean；完整参数见 volume 文档。
python validation/fmri/benchmark_bids.py volume \
  --bids-root /data/bids --derivatives-root /results/fnit \
  --subject 0001 --mni-template /templates/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz \
  --source-root /path/to/Fudan-Neuroimaging-toolkit \
  --source-revision YOUR_COMMIT --report-out /private/volume.json \
  --device cuda:0

# 同一 run、已有同源 recon-all 与已校验资源；默认读取 preproc。
python validation/fmri/benchmark_bids.py surface \
  --bids-root /data/bids --derivatives-root /results/fnit \
  --subject 0001 --recon-all /data/recon-all/sub-0001 \
  --hcp-assets-dir /templates/hcp_surface_assets \
  --source-root /path/to/Fudan-Neuroimaging-toolkit \
  --source-revision YOUR_COMMIT --report-out /private/surface.json \
  --device cuda:0

# 当前 BIDS、volume/surface、数值采样及 benchmark 捕获回归。
python -m pytest -q tests/test_fmri*.py \
  tests/test_surface_parallel_execution.py tests/test_msmall_surface_composition.py
```

公共采样入口旧/新控制使用 [validate_fmri_resampler_routing.py](../../tools/validate_fmri_resampler_routing.py)，私有 case、冻结源码和门禁要求见 [public resamplers](public_resamplers_20261002/README.md)。独立 surface 官方对照使用 [benchmark_surface_e2e.py](benchmark_surface_e2e.py)，不能用提供球面的调用代替默认 MSM 完整链。

真实影像、被试路径、许可证和完整日志保留在私有验证目录。公开工件仅保留匿名角色、SHA、数值汇总及已确认可发布的脑图。

## 清理记录

2026-10-02 按用户要求清理旧代码与说明，未修改重采样算法：删除抽取后遗留的私有 helper 别名；benchmark 改为捕获实际 final 编排入口，删除默认 volume 不会调用的旧捕获分支；当前功能页去重并链接历史报告。测试与源码绑定见 [cleanup_20261002](cleanup_20261002/README.md)。

## 原实现、命令与文献

集中见 [volume](../../docs/fmri/README.md#原软件调用)、[surface](../../docs/fmri/surface.md#原软件调用)、[FEAT](../../docs/fmri/feat.md#原软件调用)、[normalization](../../docs/fmri/normalization.md#参考文献与原实现) 和 [重采样审计](resampling_audit_20261002/README.md#7-原实现与证据链接)。
