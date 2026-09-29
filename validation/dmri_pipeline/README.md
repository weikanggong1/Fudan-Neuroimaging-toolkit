# dMRI 参数图流程验证

[功能和调用方式](../../docs/dmri_pipeline/README.md) · [TBSS 当前机器报告](tbss_e2e.real.current.json) · [九图比较脚本](compare_current_eddy_pipeline.py) · [官方 TBSS 参考脚本](run_official_tbss.sh)

2026 年 9 月 29 日在 gpucw1 用一例真实 UKB 格式 AP/PA 数据，从原始图像分别运行 TBSS 和 MMORF 分支。两个分支均经过当前公开的 TorchEDDY 实现，且使用同一例被试的原始 dMRI。结果图像仍留在计算节点；仓库只保存汇总数值。

TBSS 的第一个参考固定 TOPUP 系数、掩膜和其余输入，改用 FSL `eddy_cuda10.2` 校正图，再由相同 FNIT DTIFIT、NODDI、TBSS 代码生成九图；该配对比较隔离 EDDY 输入的影响。九张 native 图 r 为 0.978640–0.999882，standard 图 r 为 0.940917–0.988332，skeleton 图 r 为 0.949470–0.991235。shape 和 affine 全部匹配。

第二个参考由先前准备的官方 UKB native 参数图开始，经 FSL 6.0.7.4 weighted FLIRT、三阶段 FNIRT 和 applywarp 生成 standard/skeleton 图。其输入边界与本次 raw-to-standard 流程不同：standard 九图 r 为 0.358268–0.699381，skeleton 九图 r 为 0.098878–0.651267。这些差异不能单独定位到 EDDY 或 FNIRT，也不能将 FSL 的配准计时与完整 FNIT 计时相除。各图 MAE、RMSE 和有效体素数见[机器报告](tbss_e2e.real.current.json)。

本次 TBSS 完整进程 wall 为 14:06.89；EDDY 阶段 673.42 s，配准、九图传播和 skeleton 阶段 58.02 s。EDDY 进程内 CUDA allocation 峰值为 4.54 GiB；全流程最大组件峰值为 NODDI 的 12.87 GiB。共享 GPU 未隔离。

比较脚本对两份相同网格的结果计算非零并集上的 Pearson r、MAE、RMSE，并检查 shape 与 affine。运行前需分别准备 FNIT 候选目录、匹配 FSL EDDY 的下游目录、官方 TBSS `stats` 目录：

```bash
python validation/dmri_pipeline/compare_current_eddy_pipeline.py \
  --candidate-root subject_tbss \
  --reference-root fsl_eddy_with_fnit_downstream \
  --reference-label matched_FSL_EDDY_plus_FNIT_downstream \
  --official-standard official_tbss/stats \
  --official-skeleton official_tbss/stats \
  --branch tbss \
  --output tbss_comparison.json
```

`--candidate-root` 和 `--reference-root` 均为含 `native/`、`registration/standard/` 的单被试结果根；TBSS 还读取 `registration/skeleton/`。`--official-standard` 和 `--official-skeleton` 接收 FSL `stats/` 目录。脚本输出 JSON，不改写图像。

## MMORF 分支

同一例 raw AP/PA 加配对 T1w 的完整 MMORF 分支成功生成九张标准空间图，wall 为 14:47.18。阶段时间为 TOPUP/输入准备 61.31 s、EDDY 657.63 s、DTIFIT 16.57 s、NODDI 22.14 s、SynthStrip/FLIRT/MMORF/传播 124.41 s。MMORF 求解本身 46.22 s，CUDA allocation 峰值 11.45 GiB。共享 GPU 未隔离。

与同一 FSL EDDY 输入、同一 FNIT DTIFIT/NODDI 生成的 native 九图相比，Pearson r 为 0.980262–0.999882。与既有官方 MMORF warp 派生的 standard 九图相比，r 为 0.323959–0.565496；后者使用先前准备的官方 native 图和固定 affine，与本次 raw-to-standard 输入边界不同，不能据此单独评判 TorchMMORF 的求解精度。官方 NODDI 三图为单帧 4D，本包为 3D；[机器报告](mmorf_e2e.real.current.json)逐项标记原始 shape 差异，指标只压去官方末尾的单帧维度。

以下命令只计算九图指标，不改写原始图或结果图。`--matched-native-root` 是匹配 FSL EDDY 的 FNIT 下游结果根，`--official-standard` 是既有官方 MMORF warp 派生九图目录：

```bash
python validation/dmri_pipeline/compare_current_eddy_pipeline.py \
  --candidate-root subject_mmorf \
  --matched-native-root fsl_eddy_with_fnit_downstream \
  --official-standard official_mmorf/derived_standard \
  --branch mmorf \
  --output mmorf_comparison.json
```

该次运行的每图数值、参考边界、阶段时间和限制见[MMORF 当前机器报告](mmorf_e2e.real.current.json)。
