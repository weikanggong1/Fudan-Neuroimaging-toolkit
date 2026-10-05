# 普通33类SynthSeg CPU真实分步计时（2026-10-06）

为补齐 [只读热点审计](../seg_cpu_hotspot_audit_20261006/README.md) 缺少的当前分步数据，本轮只执行**一例原始T1的完整FNIT普通33类API**。使用已验收 `46eead65` 对应的冻结源码及CPU join `196a2c05…`，没有更改生产源码、卷积实现、布局、dtype、精度或模型参数。没有运行官方CNN、parc、fast或GPU。

## 1. 来源、运行边界与输出门

- 公开输入：OpenNeuro **ds003138 v1.0.1 / CC0** 的 `case02`；输入SHA `73e3866d…`，实际两个网络输入均为 **1×1×192×224×256**。五项权重逐个校验大小/SHA，不读取官方中间数据进行计算。
- CPU：同一正式对照节点，8线程、8物理核 `[32,36,40,44,48,52,56,60]`，沿用共用CPU锁；CUDA不可见，CUDA在前后均未初始化。
- 冻结worker SHA `836305b0…`；[PLAN.json](PLAN.json) 在运行前声明原始输入、当前14文件SHA、权重、线程、旧候选输出SHA与300秒硬上限。
- `profile_one.py` 的Python函数/方法包装只计时并调用原方法一次，不设置Module hooks。用已保存的有限合同以及同节点6项短合同检查它没有关闭CPU join、没有改变小输入slab的数值/stride/别名语义，并能在异常后恢复包装与全局标志。有限数据不作为MRI benchmark。
- 完整影像、CSV留在服务器私密运行目录；公开报告只含标量、shape、计时与SHA。原始T1前向结束并保存结果后才执行严格比较，**没有复制参考输出到结果目录**。
- 完整保存文件的压缩SHA、每个体素、全部标签Dice、dtype/affine/header/extensions与CSV数值全部和已验收候选相同：**0个不同体素、每区Dice=1、CSV差=0**。仍保留候选相对官方的既有**1个不同体素**，本轮没有修复数学误差或重新运行官方。
- 当前14个生产文件运行前后SHA相同，8次CPU join、两次真实网络前向、无hooks与调用方精度恢复门全部通过。

完整记录 [PROFILE.public.json](PROFILE.public.json) SHA：`0d4d28557ed7954328a67f4aade3bfac142dd7b3abd07557124e6cc1b16b1dbf`。聚合记录见 [SUMMARY.json](SUMMARY.json)。源码、输入、资源及工具全部绑定在上述文件内。

## 2. 本例时间主要花在卷积与平滑

观察器内API **106.377秒**，外层冷进程 **108.979秒**。这是带计时包装的一次诊断；共享CPU负载及观察器都能改变时间，**不能据此把旧112.952秒到108.979秒当作新提速**。正式未插桩的FNIT112.952秒/官方正常55.046秒仍保留，CPU官方速度门仍未通过。

下表按API的106.377秒作本例观察比例；CNN、后处理的子项已包含在父项中，不重复累加。

| API内步骤 | 本例时间（s） | API比例 | 实际范围 |
| --- | ---: | ---: | --- |
| T1预处理 | 4.362 | 4.10% | 解码、1mm重采样、方向、归一化及padding |
| 两次CNN | **81.716** | **76.82%** | original41.629 + flipped40.087；含卷积、ELU/BN、pool、join、softmax等 |
| 两次33通道高斯blur | **13.136** | **12.35%** | 完整3×3×3卷积，未更改为可分离滤波 |
| posterior其余部分 | 0.802 | 0.75% | 翻转、索引、集成及未单独观察的调用边界 |
| 后处理 | 4.710 | 4.43% | 24次LCC、概率处理、归一化、普通argmax等 |
| near-tie专用argmax | 1.074 | 1.01% | amax、阈值比较、uint8转换和argmax，沿用原规则 |
| 官方顺序软体积求和 | 0.342 | 0.32% | C-contiguous channels-last复制及原方向/NumPy归约 |
| 其他API边界 | 约0.234 | 约0.22% | header及未包装Python边界；不是一个已定位的内核 |

API以外：预校验/导入1.793秒，模型构造/加载0.297秒，保存0.164秒。加载和保存远小于主要卷积开销。

### 预处理与后处理子项

| 子项（嵌套，不能再加到父项上） | 秒 |
| --- | ---: |
| T1数据解码 | 0.414 |
| 1mm重采样 | 3.756 |
| 方向对齐 | 0.00033 |
| 百分位计算 | 0.106 |
| 24次SciPy六邻域LCC | 3.150 |
| posterior clone | 0.074 |
| posterior两次sum | 0.039 |
| 归一化的原inplace除法 | 0.027 |
| 普通posterior argmax | 0.942 |

当前SciPy LCC有实测3.150秒；此前旧profile的11.863秒已不适用于这一版。观察器按原Tensor descriptor执行sum/argmax/clone/inplace除法，不修改计算顺序。

## 3. 具体的卷积热点与可节省上限

CNN的38次模块卷积合计 **76.399秒**；各层含自己的slab/pad/copy，按两次网络合计：

| 卷积层 | 秒 | API比例 |
| --- | ---: | ---: |
| 最后decoder `up.3.conv0` | **30.586** | **28.75%** |
| 首级encoder `down.0.conv1` | **12.661** | **11.90%** |
| 最后decoder `up.3.conv1` | **10.882** | **10.23%** |
| 次高分辨率decoder `up.2.conv0` | 8.044 | 7.56% |

全部142次实际 `F.conv3d` 合计 **86.519秒**（其中网络groups=1的130次73.895秒，blur groups=33的12次12.624秒）；每次实际CPU、FP32、mkldnn禁用，与普通33类源码的保护作用域一致。

slab的显式结果copy合计只有 **0.927秒**；首末halo padding0.445秒；slab函数exclusive残余1.639秒。这里的residual包含分配、索引、循环/调用边界和观察器开销，不是纯Python循环时长。即使完全去掉这些边界，仍解决不了约58秒正式墙钟差距。`F.conv3d`内部的连续化、unfold/临时列矩阵和GEMM没有各自的operator时钟；不能把其86.519秒全部解释为算术，也不能凭这份报告断言copy完全无关。

当前两次CNN的8个join合计0.632秒。此前最后一级join局部1.836倍的计时受独立回放、内存驻留及共享负载影响，不能代替这一例8次join的完整实际总和。

## 4. 下一步应先解决什么

1. **优先限定在高分辨率原后端卷积和group33 blur。** 已有真实输入/权重检查点可做有限同状态对照，先获取这几项实际ATen算子trace/连续化与列矩阵构造份额，再评估scratch复用或分块策略。仍调用相同后端、保留halo与通道/归约顺序的改动须逐值证明；修改slab形状也可能改变BLAS内部执行，不能事先宣布无损。不得撤回低内存与1×1防崩措施。
2. **单纯channels-last布局现在降低优先级。** PyTorch2.5.1源码中，CPU非dilated3D卷积在禁用MKLDNN后进入Slow3d；其实现会连续化输入/权重，然后unfold与GEMM。这是依据已观察的参数和上游源码的推断，本轮没有ATen operator trace；不能宣称已经测到所有内部份额。仅转换布局可能先被重新转换回来，需先有限证明路径和数值，不照搬SynthSR的收益。[PyTorch分发源码](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/Convolution.cpp#L1175)，[Slow3d源码](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp#L27)。
3. **把blur视为单独待优化组件。** 当前13.136秒已量化，优先度高于全部LCC和I/O。不能为了加速把27项3D归约换成三轴可分离算法；任何新的CPU实现都应在真实posterior checkpoint上核查逐值、near-tie标签、CSV和完整结果，CUDA原调用保持不变。
4. **后处理是次级目标。** LCC3.150秒、普通argmax0.942秒与tie分支1.074秒可以独立评估buffer/扫描；避免更改六邻域、等大小连通分量选择、near-tie规则及软体积归约。即使这些都降低，也不能单独达到完整官方速度目标。

不再重复已拒的oneDNN slab/混合中层路线；它们已增加相对官方的错误标签。本轮没有新候选、没有改CPU/GPU默认策略，也没有给parc/fast填写从33类推算的百分比。parc/fast会保留oneDNN并新增69-channel网络/blur；其各自当前份额仍缺测，只有在后续候选影响它们时才安排必要的一例观察。

## 5. 计时观测的影响与可复现性

- 真实记录的观察器 bookkeeping 下界为0.002985秒；它不包含全部dispatch、metadata准备和Tensor方法包装开销，不能把这个值当成精确校正量。
- 同节点有限合同的10,000个空span平均5.20微秒，仅用于确认观察器数量级；它不代表真实函数/metadata成本，不是MRI速度benchmark。
- exclusive事件总和与API parent106.377067秒完全相等，聚合不重复计算子项。
- 保留正式未插桩冷进程时钟；不要从一次profile推新的加速倍率。正式官方TensorFlow逐算子后端仍未记录，本次只证明FNIT自身实际参数与时间分布。
- [profile_one.py](profile_one.py) 是冻结私有观察器；[run_one.py](run_one.py) 只派发有限合同和一个新CPU进程，在其退出后核验保存输出。执行参数从已现场核对的FNIT固定索引和 [PLAN.json](PLAN.json) 获取；不移动source/env，不创建Notebook_code根目录任务。
- 原图和模型文件不发布；公开JSON不含影像体素/后验数组、凭据、许可证或外部地址。

用标准库脚本从保存的events机械重算聚合，不执行模型：

```bash
python validation/smri_cpu/seg_cpu_profile_20261006/build_summary.py
```

[MANIFEST.json](MANIFEST.json) 只列该叶已跟踪的源码/报告文件及SHA，不含pycache或影像；manifest自身不递归计算自己的SHA。
