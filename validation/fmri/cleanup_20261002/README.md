# fMRI 旧代码与说明清理，2026-10-02

本轮按用户要求清理 volume/surface 的旧入口与重复说明，保留当前重采样算法。基线为 `81f1bb3cccf520b736507baad69b73b7fef97fb5`；本目录记录清理差异与回归，不新增完整 MRI benchmark。

## 清理内容

| 位置 | 改动与原因 |
|---|---|
| `fnit.fmri.normalization` | 删除抽取共享采样模块后留下的 4 个私有 helper 别名。数值控制直接从 `fnit._world_resampling` 导入实际实现。 |
| `validation/fmri/benchmark_bids.py` | 旧 `end_to_end.resample_world` 捕获入口已不存在，改为捕获实际 `_resample_final_volume`，原样转发全部参数。删除默认 volume 不会进入的 standalone motion-warp 捕获分支；MCFLIRT 捕获保留。 |
| 当前功能说明与验证索引 | 用法、参数和流程集中在 `docs/fmri`；历史耗时、原始失败门和专项报告保留在验证目录，当前页链接这些证据。纠正一处旧文档锚点和 ICA 许可声明中的旧代码路径。 |

`resample_world()` 仍被 AROMA 与 surface clean 使用；`run_feat_core(spatial_warp/postmat)` 是有调用和测试的公开选项；两者保留。FNIRT、SynthMorph、preproc/clean、外部已配准球面和独立 MSM 等有效路径均保留。

## 验证

- 11 个核心采样与流程文件相对基线逐字节相同，包括共享 World 采样、ApplyWarp、SynthMorph apply、4D spline、volume/surface 编排和 MCFLIRT 采样。
- `normalization.py` 与 `pipeline.py` 的全部函数及类，经去除说明文字后 AST 相同；本轮只移除私有导入别名并更新说明。
- 服务器现有主页 Conda 环境、单 GPU 可见的相关测试：**298 passed，11 skipped，0 failed，0 errors**，pytest 汇总 34.23 s。11 项需要显式设置 `FNIT_TEST_FMRI_SURFACE_ASSETS`，本次未设置，跳过原因逐项记录。
- 新增两个 benchmark 捕获回归覆盖 FNIRT/SynthMorph、四种 final 输出，检查位置参数及关键字参数原样转发。原数值断言未更改。

[源码一致性与 SHA-256](source_comparison.public.json) · [测试命令、结果、跳过项及测试源码 SHA-256](tests.public.json)

```bash
# 在仓库根目录、主页 Conda 环境执行本轮相关回归。
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python -m pytest -q \
  tests/test_fmri*.py tests/test_surface_parallel_execution.py \
  tests/test_msmall_surface_composition.py tests/test_msmsulc_benchmark.py
```

## 保留的结论与后续工作

本轮没有修改边界条件、插值、坐标舍入、变换链、精度或科学输出，也没有重新计时完整 pipeline。最近真实 490 帧测量、脑图和分步骤耗时继续使用各自冻结报告：[volume 公共采样入口](../public_resamplers_20261002/README.md)、[surface 完整执行](../surface_gpu_parallel/README.md)。

FSL clean 样条边界、World 最近邻舍入、SynthMorph 官方线性采样、独立 MSM 及官方实际 MNI 节点的差异仍保留，见 [重采样审计](../resampling_audit_20261002/README.md) 与 [验证索引](../README.md#尚未关闭的官方对照)。它们属于后续修复范围。

## 文档入口

[Volume](../../../docs/fmri/README.md) · [Surface](../../../docs/fmri/surface.md) · [代码职责](../../../src/fnit/fmri/README.md) · [原软件调用与文献](../README.md#原实现命令与文献)
