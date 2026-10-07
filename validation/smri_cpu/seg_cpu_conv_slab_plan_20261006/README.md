# SynthSeg单层CPU slab64MiB静态计划（2026-10-06）

## 1. 范围与状态

只提议修改普通33类CPU的`SegmentUNet.up[3].conv0`单次调用：继续用现有`convolution_slabs`、FP32、NCDHW和非oneDNN后端，把该层`maximum_slab_bytes`由256MiB减至64MiB。生产代码未改，没有实现新的调度，没有执行合同、真实层、CNN、GPU或官方程序。

这是基于[唯一真实末端profile](../seg_cpu_blur_trial_20261006/README.md)的单一内存策略。该层14次Slow3d调用wall16.279秒，user55.202秒、system41.681秒；整段CPU秒/wall约5.951。高system占用支持检查列矩阵分配与页面开销，但trace没有分离unfold、GEMM和页面操作，**不能据此确定变慢原因或预测提速**。

## 2. 输入、输出和唯一参数

输入来源仍是已绑定的末端skip/value检查点。重join后形状为`1×72×192×224×256`；权重为`24×72×3×3×3`、bias为24项，输出为`1×24×192×224×256`的pre-ELU FP32值。输入、权重和每个输出的1944个邻域项顺序不变。

唯一变化是已有helper的`maximum_slab_bytes`。它约束slab输入几何，不是进程RSS或im2col工作区上限；不能把64MiB称为该层只用64MiB内存。halo、真实边缘padding、groups=1、stride/dilation=1、bias顺序及输出copy方式均保留。

输入和输出完整缓冲各需3,170,893,824和1,056,964,608字节。模型、BLAS工作区、allocator缓存、前一slab临时输出的重叠生命周期仍占内存；下表是单调用几何推导，不是RSS测量。

| 项目 | 原256MiB | 候选64MiB |
| --- | ---: | ---: |
| slab输出深度 | 14 | 2 |
| 全层调用次数 | 14 | 96 |
| 最大im2col字节 | 6,242,697,216 | 891,813,888 |
| 最大连续chunk字节 | 264,241,152 | 66,060,288 |
| 最大临时输出字节 | 77,070,336 | 11,010,048 |
| ATen GEMM M/N/K | 802816/24/1944 | 114688/24/1944 |
| 含真实边缘零pad的累计chunk深度 | 220 | 384 |

每输入平面为16,515,072字节，现公式`min(32, cap//plane_bytes-2)`机械得到14和2。单调用列矩阵变小85.7%，全层列矩阵累计展开项数仍为21,403,533,312；调用次数增为约6.86倍、重复halo增多。减少大块分配可能降低页面压力，也可能被额外调用和复制抵消。

## 3. 静态重建

```bash
# 仅标准库，核对14个冻结源SHA和已有真实标量记录；不导入Torch。
python validation/smri_cpu/seg_cpu_conv_slab_plan_20261006/build_plan.py
```

输出[PLAN.json](PLAN.json)包括全部几何、源码/权重/检查点SHA、已有profile及未执行的合同。它绑定`46eead65`生产源、CPU join`196a2c05…`和真实结果`7631b641…`，不依赖移动中的main HEAD。

## 4. 对应原步骤与舍入风险

本计划没有单独官方命令；它位于普通`mri_synthseg`网络最后decoder卷积。完整官方命令和验收仍见SynthSeg功能页，本轮未重跑。

实际trace已确认走Slow3d；[PyTorch2.5.1官方源码](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp#L27)定义了连续输入、列矩阵和GEMM维度。cap变小保留邻域顺序，但改变空间M及leading dimension，BLAS可能选择不同packing或FP32归约路径。没有数值合同前不能称无损。

## 5. 预声明的有限验收

**以下均未执行。** 先测试72→24通道、3³ kernel+bias的有界shape，固定FP32种子，包含混合符号、抵消项、零和signed-zero。深度1/2/7/31/33、H13/W17，使用16和4逻辑输入平面的cap，以复现14→2的slab差异；另覆盖strided输入和边缘/内部halo。

首门为全部pre-ELU输出uint32位相同，另核shape/stride/dtype/finite、输入/权重/bias不变和独立输出。后续若增加调度，只限CPU、eval、no-grad、无autocast的指定层，训练、CUDA及其他层保留原调用与全局flags。

小合同通过之后，需要root另行授权同检查点和同权重的旧/新**单层**真实对照：只重join一次，在8物理核共锁内比较全部pre-ELU位值、RSS和实际CPU/时钟。任何一位不同先停止，不用硬标签一致代替。完整T1图/CSV和GPU保护门在本计划之外，均未通过。

## 6. 已有结果与本轮记录

- [block11平滑试验](../seg_cpu_blur_trial_20261006/README.md)：真实posterior逐位相同，blur worker RSS下降45.7%，旧/新操作速度相当；保持生产默认不变。
- 本轮只完成源码绑定和workspace推导，未开始真实层或多参数搜索；不会复跑已拒oneDNN或改变低内存/1×1防崩保护。
- 当前完整普通33类CPU速度仍未达官方约55秒的门，单层内存推导不改变这一结论。

## 7. 参考

- [当前33类完整profile](../seg_cpu_profile_20261006/README.md)。
- [PyTorch2.5.1 Slow3d官方源码](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp#L27)。
- [SynthSeg官方代码](https://github.com/BBillot/SynthSeg)；Billot et al., *SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining*, Medical Image Analysis, 2023。
