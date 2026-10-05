# SynthSeg单层CPU 64MiB slab有限诊断（2026-10-06）

## 1. 目标与状态

这是尚未运行的私有候选，生产代码未改。仅降低普通33类网络`up[3].conv0`的slab输入cap：256→64MiB，继续用已有CPU非oneDNN后端、FP32、NCDHW、同权重/bias/kernel顺序及halo。

流程分两次派发：目标Torch2.5.1短合同→只读审核边界→单层旧/新ABBA。任何合同或真实pre-ELU值非逐位一致，立即停止后续计算。合同通过不自动启动真实层，既有GEMS完整作业按共同锁先运行。

## 2. 输入与输出

- 合同读取已核验的`synthseg_2.0.h5`，使用当前真实24×72×3³权重及24项bias；输入是固定FP32种子的有界测试张量，不作影像benchmark。
- 真层阶段仅复用已保存的skip/value检查点，当前CPU join输出为1×72×192×224×256；SHA与旧逐位join证明绑定。每arm只执行一个`up[3].conv0`，输出1×24×192×224×256的pre-ELU FP32张量。
- A1保存私有参考`.npy`及文件/value SHA；B1/B2/A2对全部264,241,152个值作实际uint32逐位比较。A1仅生成参考，其自然0不计为比较。
- 没有encoder、其他decoder、ELU/BN、likelihood/softmax、完整CNN、官方或GPU计算。原始影像/中间大数组保持私密；公开报告仅标量与SHA。

## 3. 调用、参数与预算

各阶段读取同一冻结[PLAN.json](PLAN.json)，源码14SHA、producer六文件与检查点/权重SHA固定。私有workspace、run和source路径由调度参数给定，不作为公共FNIT API。

```bash
# 第一次派发只运行合同，不自动进入MRI单层。
python run_trial.py --root "$fnit_root" --workspace "$frozen_trial_workspace" \
  --run "$private_trial_run" --source "$frozen_fnit_source" \
  --checkpoint "$verified_decoder_checkpoint" --phase contracts

# 仅在合同通过、只读审核完成后再次派发。
python run_trial.py --root "$fnit_root" --workspace "$frozen_trial_workspace" \
  --run "$private_trial_run" --source "$frozen_fnit_source" \
  --checkpoint "$verified_decoder_checkpoint" --phase ABBA
```

两阶段共用首个enqueue时间+23000秒的总界限，外层watchdog只可终止PID identity匹配的专属process group。合同worker≤120秒，各单层worker≤180秒；同8物理核、8线程、共同CPU锁内串行，CUDA不可见。run只创建一次，旧文件不覆盖。

`candidate.forward`的默认cap为64MiB；合同用4逻辑输入平面的cap复现小shape下的depth2，旧合同cap为16平面/depth14，不是搜索更多真实budget。只有CPU、FP32、72→24指定kernel、eval、no-grad、无autocast、非oneDNN且无hooks的调用进入候选；GPU、训练、梯度、dtype及观察器条件调用原layer。

## 4. 原步骤与数值风险

这是`mri_synthseg`网络内部卷积，没有独立官方CLI。本轮不会重跑官方命令。几何推导和[PyTorch2.5.1 Slow3d来源](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp#L27)详见[静态方案](../seg_cpu_conv_slab_plan_20261006/README.md)：depth14→2，调用14→96，最大im2col6.243→0.892GB，累计列矩阵项不减少。

空间M/leading dimension改变可能改变BLAS的FP32舍入；相同kernel邻域不等于逐位相同。更多halo复制与调用也可能抵消页面压力收益。

## 5. 预声明验收与计时

合同覆盖深度1/2/7/31/33、内部和真实边缘halo、strided输入、混合符号与大幅抵消、零和signed-zero；都使用真实层weight/bias。任一bit不同即写失败标量并停止剩余合同和真层。精度门不放宽。

后续真层门包括完整pre-ELU bit差0、shape/stride/dtype/finite一致、输入/权重/bias不变、输出不别名、参考文件SHA、14源码和全局flags不变、RSS≤32,000,000,000字节。没有Module hooks或ATen profiler；记录操作wall、进程user/system和minor/major faults。fault计数只能反映整个单层段，不能直接归因于GEMM。加载、join、hash、保存与比较另列worker总墙钟。

**尚无当前实测结果。** 不把有界合同称真实benchmark，也不把单层clock称完整MRI提速；图/CSV/完整GPU门未验收。

## 6. 历史与剩余工作

- [旧block11 blur试验](../seg_cpu_blur_trial_20261006/README.md)逐位一致、RSS下降45.7%，未显示速度收益，未接入默认。
- 本轮只准备单一64MiB候选与合同/ABBA调度；不复跑oneDNN、低精度或其他budget搜索，不撤回1×1/低内存防崩保护。
- 小合同非exact时不运行真实层；全部单层门通过后仍需协调者另行决定完整T1/图/CSV及GPU回归与生产接入。

## 7. 来源

- [当前完整33类CPU profile](../seg_cpu_profile_20261006/README.md)与[单层几何方案](../seg_cpu_conv_slab_plan_20261006/README.md)。
- [SynthSeg官方代码](https://github.com/BBillot/SynthSeg)；Billot et al., Medical Image Analysis, 2023。
- [PyTorch2.5.1 Slow3d官方实现](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp#L27)。
