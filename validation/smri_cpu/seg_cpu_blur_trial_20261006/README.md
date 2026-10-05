# 普通33类SynthSeg CPU高斯平滑候选（2026-10-06）

## 1. 功能与当前状态

这是独立诊断原型，**尚未接入生产、尚未通过真实posterior验收**。只考察普通33类CPU路径：在原后端、原kernel和原slab几何下，把每11个独立通道视作11个单通道batch。CUDA、parc/fast的oneDNN路径及训练/梯度路径沿用原函数。本叶没有新增依赖，没有修改共享helpers或生产源码。

已完成的真实[CPU分步报告](../seg_cpu_profile_20261006/README.md)显示两次blur共13.136秒，其中12次group33 `F.conv3d`共12.624秒；两次CNN仍占81.716秒。这里的blur候选即使成功，也不能单独填补FNIT112.952秒与官方正常55.046秒的完整墙钟差距。

## 2. 诊断输入、输出和参数

- 真实来源是[已验收decoder检查点](../seg_memory_20261005/CPU_JOIN_STAGE.public.json)的`skip.npy`和`value.npy`，不是硬分割图反推的概率。它们来自公开CC0原始T1的FNIT预处理，在最后decoder的conv0前停止；大小和SHA见[PLAN.json](PLAN.json)。
- 当前14个生产文件仍绑定`46eead65`/CPU join`196a2c05…`。检查点producer的六文件逐个对应`4ec078cb`；当前`_Block`、SegmentUNet构造/权重读取、likelihood→softmax末端AST相同，`cpu_conv.py`整文件SHA相同。已保存join逐位证明也绑定在plan内。
- 后续只恢复现存末端：join→两次conv/ELU→BN→likelihood→softmax，保存一份`1×33×192×224×256`的float32 preblur概率检查点。**该恢复还没有执行**，不重新跑前面的encoder/decoder或官方软件，不称完整T1 benchmark。
- 原型`channel_block=11`固定；完整33逻辑通道决定原32层slab，块大小不能重新决定slab深度。零padding仍只出现在真实图像边缘，内部保留邻平面halo。每个输出通道仍消费按原轴/核顺序排列的27个FP32值。
- 原型返回独立、连续的同shape/stride FP32 CPU张量；不原位改输入，不设置dtype、线程、oneDNN或CUDA全局精度。符合条件之外只调用原blur一次。

本叶的`channel_batch.py`只供诊断worker直接导入，FNIT生产代码不导入它。

## 3. 调用与合同

先机械重建源码/几何plan，标准库即可，不执行模型：

```bash
python validation/smri_cpu/seg_cpu_blur_trial_20261006/build_plan.py
```

小合同使用原生产卷积作逐位oracle，覆盖真边缘、强制多slab、单深度、strided输入、32768个输出位置、输入不变和独立存储；grad/oneDNN/autocast/非33通道/非CPU/非FP32条件回退，异常直接传播且全局flags保留。

```bash
# 仅有界合同；src是冻结FNIT源码，不执行真实MRI模型。
python validation/smri_cpu/seg_cpu_blur_trial_20261006/check_contracts.py \
  --source src \
  --output validation/smri_cpu/seg_cpu_blur_trial_20261006/LOCAL_CONTRACTS.json
```

[LOCAL_CONTRACTS.json](LOCAL_CONTRACTS.json)已记录本地**Torch2.4.1**五个有界shape逐位相同及7个回退；CUDA未初始化。它不替代目标Torch2.5.1合同或真实posterior门，不是速度benchmark。真实worker和回放将在同8物理核共用CPU锁内运行，排在现有GEMS完整作业之后。

实际已派发controller **103736**，receipt为`waiting_for_common_CPU_lock`、`jobs=[]`；科学worker尚未开始。独立外层watchdog **106165** 从初始排队时间计23000秒，校验controller的PID starttime和专属process group后才可限时终止；没有重启controller或改变五个科学文件/PLAN。此处仅是派发记录，不是计算完成记录。

## 4. 对应原步骤与后端推断

这是`mri_synthseg`普通33类概率平滑的内部步骤，没有独立官方CLI。整体官方命令仍是功能页中的`mri_synthseg --i ... --o ... --vol ... --threads 8`；本轮不重新运行该命令。

根据Torch2.5.1 Slow3d源码，原group33和channel-as-batch均构造相同kernel邻域，逐通道矩阵为**M=1,835,008、N=1、K=27**；batch路径会改变GEMM调用分支。相同M/N/K与相同邻域不能提前证明相同FP32舍入。[Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp#L27)，[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp#L551)。

原Unfold已经按通道/27项展开并行，batch33不代表首次引入并行。其外层grain20可能限制有效任务数，内层并行也可能被已有并行域抑制。先测batch11；实际后端/线程利用率仍要记录。[Unfold3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/Unfold3d.cpp#L224)，[OpenMP分配](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/ParallelOpenMP.h#L16)。

## 5. 内存与真实验收门

原型保留原slab M。按源码列矩阵形状推导：

| 每次slab调用 | 列矩阵字节 | 连续输入字节 | 临时输出字节 |
| --- | ---: | ---: | ---: |
| 原group33 | 6,539,968,512 | 257,359,872 | 242,221,056 |
| 一次完整batch33（未批准测试） | 6,539,968,512 | 257,359,872 | 242,221,056 |
| 本轮batch11 | 2,179,989,504 | 85,786,624 | 80,740,352 |

batch11依次处理三个块，累计展开数据量没有减少；这里只降低单调用列矩阵上限，不是实测RSS或速度。完整posterior缓冲仍为1,453,326,336字节，另有BLAS工作区及allocator缓存。

后续只批准一份真实preblur检查点的一组旧/新ABBA回放：全部FP32posterior逐位相同、shape/stride/dtype/finite相同、输入不变/不别名、flags和GPU原回退不变、实测RSS≤32,000,000,000字节。任一数值不exact先停止；不放宽门、不从一次原始pass宣称完整图/CSV已经通过。

唯一末端恢复可对`up[3].conv0`使用一次CPU ATen profiler，不开record_shapes/stack/memory，不用Module hooks；记录实际user/system时间与墙钟。Unfold/GEMM的C++调用可能包含在Slow3d内而没有单独ATen事件，缺少事件不说明没有相应计算。恢复和回放时间均为诊断观察，不替代正式端到端时钟。

## 6. 更新与未执行项

- 本轮：源码/producer/权重/末端AST机械绑定完成，本地Torch2.4.1小合同完成；2.5.1合同、唯一末端恢复、真实posterior ABBA、RSS及ATen trace**待执行**。
- 已拒路线仍保留记录：oneDNN slab和混合中层在case02增加相对官方错误标签，不复跑、不撤回现有低内存和1×1防崩保护。
- 不合并两次网络pass的blur，不增加batch2全概率堆叠，不切换可分离三轴归约或低精度，不开始CNN多参数搜索。

## 7. 参考

- [FNIT真实当前CPU profile与聚合](../seg_cpu_profile_20261006/README.md)。
- [原阶段输入/producer/join逐位证据](../seg_memory_20261005/README.md)。
- [SynthSeg原代码](https://github.com/BBillot/SynthSeg)；Billot et al., *SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining*, Medical Image Analysis, 2023。
- PyTorch2.5.1官方实现链接见第4节；本叶仅记录推导，没有复制上游源代码。
