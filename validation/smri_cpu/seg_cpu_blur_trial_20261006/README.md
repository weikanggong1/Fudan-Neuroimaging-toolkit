# 普通33类SynthSeg CPU高斯平滑候选（2026-10-06）

## 1. 功能与当前状态

这是已完成的独立诊断原型，**真实posterior逐位与内存门通过，未显示稳定提速，未接入生产**。只考察普通33类CPU路径：在原后端、原kernel和原slab几何下，把每11个独立通道视作11个单通道batch。CUDA、parc/fast的oneDNN路径及训练/梯度路径沿用原函数。本叶没有新增依赖，没有修改共享helpers或生产源码。结果在[RESULT.public.json](RESULT.public.json)，冻结的[PLAN.json](PLAN.json)仍保留运行前声明，不用完成状态覆盖历史计划。

已完成的真实[CPU分步报告](../seg_cpu_profile_20261006/README.md)显示两次blur共13.136秒，其中12次group33 `F.conv3d`共12.624秒；两次CNN仍占81.716秒。这里的blur候选即使成功，也不能单独填补FNIT112.952秒与官方正常55.046秒的完整墙钟差距。

## 2. 诊断输入、输出和参数

- 真实来源是[已验收decoder检查点](../seg_memory_20261005/CPU_JOIN_STAGE.public.json)的`skip.npy`和`value.npy`，不是硬分割图反推的概率。它们来自公开CC0原始T1的FNIT预处理，在最后decoder的conv0前停止；大小和SHA见[PLAN.json](PLAN.json)。
- 当前14个生产文件仍绑定`46eead65`/CPU join`196a2c05…`。检查点producer的六文件逐个对应`4ec078cb`；当前`_Block`、SegmentUNet构造/权重读取、likelihood→softmax末端AST相同，`cpu_conv.py`整文件SHA相同。已保存join逐位证明也绑定在plan内。
- 已只恢复现存末端一次：join→两次conv/ELU→BN→likelihood→softmax，保存一份`1×33×192×224×256`的float32 preblur概率检查点。preblur文件SHA为`6c885edc…`，全部概率value SHA为`711102a4…`；完整哈希见结果。没有重新跑前面的encoder/decoder或官方软件，不称完整T1 benchmark。
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

[LOCAL_CONTRACTS.json](LOCAL_CONTRACTS.json)保留本地**Torch2.4.1**五个有界shape逐位相同及7个回退。目标节点**Torch2.5.1**已重新执行这五个小合同与7个回退并通过，结果见`RESULT.public.json.contracts`；两者均未初始化CUDA。小合同不作为速度benchmark。真实worker与回放已在同8物理核、8线程和共用CPU锁内串行完成。

最初controller **103736** 为`waiting_for_common_CPU_lock`、`jobs=[]`。GEMS owner安全停止其旧作业并释放锁后，本队列自然完成合同、一次末端恢复与A1/B1/B2/A2四次blur，六个worker全部exit0。独立watchdog **106165** 从初始排队时间计23000秒，校验controller PID starttime和专属process group；本次正常退出，没有发送终止信号。没有重启controller或改变五个科学文件/PLAN。结果保留各实际PID、顺序、时间边界和watchdog状态。

只读标量收集器不导入Torch或读取影像像素作计算；它复核已完成报告，并对A1参考文件测量SHA：

```bash
python validation/smri_cpu/seg_cpu_blur_trial_20261006/collect_public.py \
  --run "$private_completed_run_directory" \
  --plan "$frozen_trial_workspace/PLAN.json" \
  --output RESULT.public.json
```

## 4. 对应原步骤与后端推断

这是`mri_synthseg`普通33类概率平滑的内部步骤，没有独立官方CLI。整体官方命令仍是功能页中的`mri_synthseg --i ... --o ... --vol ... --threads 8`；本轮不重新运行该命令。

根据Torch2.5.1 Slow3d源码，原group33和channel-as-batch均构造相同kernel邻域，逐通道矩阵为**M=1,835,008、N=1、K=27**；batch路径会改变GEMM调用分支。相同M/N/K与相同邻域不能提前证明相同FP32舍入。[Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp#L27)，[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp#L551)。

原Unfold已经按通道/27项展开并行，batch33不代表首次引入并行。其外层grain20可能限制有效任务数，内层并行也可能被已有并行域抑制。先测batch11；实际后端/线程利用率仍要记录。[Unfold3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/Unfold3d.cpp#L224)，[OpenMP分配](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/ParallelOpenMP.h#L16)。

## 5. 内存、逐位结果和阶段耗时

原型保留原slab M。按源码列矩阵形状推导：

| 每次slab调用 | 列矩阵字节 | 连续输入字节 | 临时输出字节 |
| --- | ---: | ---: | ---: |
| 原group33 | 6,539,968,512 | 257,359,872 | 242,221,056 |
| 一次完整batch33（未批准测试） | 6,539,968,512 | 257,359,872 | 242,221,056 |
| 本轮batch11 | 2,179,989,504 | 85,786,624 | 80,740,352 |

batch11依次处理三个块，累计展开数据量没有减少；这里只降低单调用列矩阵上限，不是实测RSS或速度。完整posterior缓冲仍为1,453,326,336字节，另有BLAS工作区及allocator缓存。

实际完成一份真实preblur检查点的一组旧/新ABBA回放：

| arm | 路径 | blur操作秒 | 进程最大RSS（十进制GB） | 逐位比较 |
| --- | --- | ---: | ---: | --- |
| A1 | 原group33 | 7.177799 | 10.294600 | 仅生成参考；`comparison_executed=False` |
| B1 | batch11 | 7.005766 | 5.583524 | 对A1全363,331,584个FP32值，bit差0 |
| B2 | batch11 | 6.950124 | 5.592076 | 对A1全363,331,584个FP32值，bit差0 |
| A2 | 原group33 | 6.748973 | 10.288185 | 对A1全363,331,584个FP32值，bit差0 |

三个实际比较arm最大绝对差均为0，输出value SHA均为`6251c11f…`。参考文件SHA在收集阶段测量，完整哈希、shape/stride/dtype/finite、输入不变/不别名门见结果。所有14个生产文件、五个科学文件、线程和全局flags保持不变；CUDA未初始化，所有进程RSS小于32,000,000,000字节。A1自然记录的bit差0不计为旧/新比较。

两旧arm的操作中位数为6.963386秒，两新为6.977945秒，候选增加0.209%；A1冷边界较长，共享CPU只有每路径两次观察，**没有证据说明速度提高**。操作计时包含原型eligibility和kernel构造，不包含概率读入、SHA检查、保存和逐位比较。候选最大RSS较旧最大RSS降低45.68%，这是本次完整blur worker的实测结果，不能外推为完整CNN或API的峰值降幅。

唯一末端恢复对`up[3].conv0`使用了一次CPU ATen profiler，不开record_shapes/stack/memory，不用Module hooks。该段wall为16.278799秒，进程user为55.201523秒、system为41.681274秒；整个段总CPU秒/wall为5.951，不代表GEMM单独的核占用。trace含14次`aten::slow_conv3d_forward`，其inclusive为16.054000秒、self为15.701270秒；12次contiguous为0.285410秒，42次copy为0.433221秒（嵌套事件不能相加）。未见独立unfold或GEMM ATen事件，它们可能仍包含在C++ Slow3d内部，不能据此认定未调用或归因于其中某一步。末端恢复的最大RSS为11.395170GB。恢复和回放时间均为诊断观察，不替代正式端到端时钟。

本次没有重新生成最终标签图、headers或CSV；**完整T1输出与完整GPU实测仍未验收**。由于阶段没有速度收益，保持生产默认不变，不追加CNN或官方计算。

## 6. 更新与剩余项

- `08d6f882`：冻结运行前计划、原型和本地Torch2.4.1合同，明确当时真实门待执行。
- 本轮完成：Torch2.5.1合同、唯一末端恢复、三个实际全posterior逐位门、RSS及单层ATen trace；旧/新阶段速度相当，内存下降。标量收集脚本和完整SHA保留以供机械核对。
- 未接入生产，完整T1/CSV/GPU门未执行；CPU全功能速度仍未达到官方。后续先用已有trace提出单一内存策略的保序风险与合同计划，不开展参数搜索。
- 已拒路线仍保留记录：oneDNN slab和混合中层在case02增加相对官方错误标签，不复跑、不撤回现有低内存和1×1防崩保护。
- 不合并两次网络pass的blur，不增加batch2全概率堆叠，不切换可分离三轴归约或低精度，不开始CNN多参数搜索。

## 7. 参考

- [FNIT真实当前CPU profile与聚合](../seg_cpu_profile_20261006/README.md)。
- [原阶段输入/producer/join逐位证据](../seg_memory_20261005/README.md)。
- [SynthSeg原代码](https://github.com/BBillot/SynthSeg)；Billot et al., *SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining*, Medical Image Analysis, 2023。
- PyTorch2.5.1官方实现链接见第4节；本叶仅记录推导，没有复制上游源代码。
