# SynthSeg 单层列缓冲复用：真实 CPU/GPU 接入验收（2026-10-06）

## 1. 功能与状态

该版本把 FNIT 自有 copy/SGEMM 胶水接入 33 类网络 `up[3].conv0`。输入仍按原14层 slab 展开，M/N/K、紧凑 leading dimensions、通道/核顺序、FP32、bias预填和已加载 Torch 的 LP64 SGEMM 保持。6.243 GB 列缓冲仅在本层调用内复用，返回后释放；不改变成熟通用slab、低内存防崩代码和GPU数学。

**同输入阶段与CPU/GPU整例验收通过；整合源码绑定及发布记录见 [ROOT_REVIEW.json](ROOT_REVIEW.json) 和 [总表](../README.md)。** 新独立Conda环境安装尚未执行；不能把已有Conda/GCC编译通过写成全新环境安装通过。完整官方CPU速度目标仍未通过。

实际源码是 `70ad6537`；准备worker是 `41ede608`。原预声明 [PLAN](PLAN.json) SHA `6daa84b1…` 和35项 [准备清单](MANIFEST.json) 保留为执行前记录，不用新结果覆盖旧声明。最新结果由 [build_summary.py](build_summary.py) 机械重算 [RESULTS.json](RESULTS.json)，不导入模型/影像软件，不运行新科学计算。09facd3e的原88项清单保留为 `RESULTS_MANIFEST_09facd3e.json`；当前清单见 `RESULTS_MANIFEST.json`。本次收尾只更新三份说明文字，计算17源、PLAN与原科学报告保持原字节；范围见 `DOCUMENTATION_FINALIZATION.json`。

## 2. 输入、输出、调用与参数

用户完整Python/CLI/参数见 [七节功能说明](../../../docs/synthseg/CPU_COLUMNS.md)。本次没有新算法参数。普通CPU SynthSeg关闭oneDNN，因此本例原图和翻转两pass均满足资格并激活。默认Plus/parc/fast CPU保持oneDNN=True，沿成熟路径；其完整重复测试未新增。调用者关闭oneDNN时，仅已验证权重、形状和CPU预算的共同33类末层可检查资格；ParcUNet没有标记。普通 `--parc` 不等于尚未实现的robust SynthSeg+。

窄条件是严格 `CPUInferenceConv3d`、普通Tensor/Parameter、CPU FP32输入 `[1,72,192,224,256]`、连续参数 `[24,72,3,3,3]`/`[24]` 且实际值SHA相同、8线程、eval/no-grad、无autocast/hooks/forward-AD/懒negative或conjugate视图、原groups/stride/dilation/padding。未知条件在数学前继续旧CPU路径；候选数学开始后的异常传播，不暗中重算。CUDA提前返回，不import CPU loader、不核模型SHA、不编译。

真实输入是CC0 OpenNeuro ds003138 v1.0.1 case02原T1，SHA `73e3866d…`；分割H5 `f190bfd7…` 和标签/名称/拓扑资源均在PLAN绑定。正式原/新17或14生产源、common4、输入/模型/配置 before/after SHA 恒定。完整结果shape `[180,224,225]`，int32、约1-mm网格，整数分割和按区软体积CSV。

## 3. 实际运行、缓存与合同

运行资源为nodecw7同8物理核，CPU affinity `32,36,40,44,48,52,56,60`、共同CPU锁、每完整arm 600s/32e9 AS与RSS上限。GPU默认True普通33类运行在同H100 UUID，两arm各300s/20e9，派发前可用显存84.09 GB；共同GPU锁。每arm完成即保存原输出/退出receipt并严格核同名gzip+CSV SHA，首差停止后续。无新官方运行或未影响模式重跑。

1. [phase1原始receipt](phase1_results/phase1/QUEUE.json)：已有Conda Python3.11/Torch2.5.1/GCC11真实构建加载1次；全新短合同cache。metadata阶段copy/SGEMM为0。artifact17,640 B、SHA `74678cc0…`；源码4385 B保持已验收旧C++字节。随后一个新进程复用artifact，6个数值case、13个copy/poison/signed-zero oracle、23个fallback/异常case全部通过，12copy/12SGEMM，所有uint32位差0、shape/stride/参数/lifetime/flags门通过。短worker RSS0.506 GB。
2. [CPU完整队列](whole_results/whole_cpu/QUEUE.json)：A1原、B1独立cold cache、B2相同cache的新warm进程、A2原。B1编译1次、B2编译0次；两candidate各2真实forward、2命中层、28copy/28SGEMM。short cache保留，没有删除以制造cold。实际 [CPU_EXIT](whole_results/CPU_EXIT.json) rc0。
3. [GPU完整队列](whole_results/whole_gpu/QUEUE.json)：原/新AB均实际2forward、FP32/default TF32、autocast关闭；optional CPU模块import为空，compile/copy/SGEMM/构建输入/probe均0。实际 [GPU_EXIT](whole_results/GPU_EXIT.json) rc0。

本地39项guard/cache合同以及4项队列控制mock是独立软件合同，不能当作真实GPU或MRI验收。phase1两个child原RC0/complete controller状态保存；未单独采样原phase1外层OS EXIT，这一点保留。whole CPU/GPU另有原外层退出receipt。

## 4. 官方对照与时钟边界

```bash
# 独立官方验证环境；FNIT运行时不调用这个命令。
mri_synthseg --i /data/example/T1w.nii.gz \
  --o /data/reference/segmentation.nii.gz --vol /data/reference/volumes.csv \
  --cpu --threads 8
```

同例、同nodecw7、同8核官方输出/时钟复用 [正式记录](../seg_memory_20261005/OFFICIAL_NODE7_FAST_BA_REPEAT.public.json)，正常冷进程55.046403秒；原373秒异常保留且不用于加速分母。官方边界是module/CLI import、完整推理、主图和CSV；本轮FNIT冷进程包含资源/源码preflight、imports、构造、API、图/CSV保存和收尾。API包含合资格层每次runtime/header/provider/权重SHA、compiler probe及首次compile。两种边界并列，不把API单独与CLI当同一时间。

唯一新posthoc只读取已保存的A1/B2/官方图与CSV，使用既有比较器。先原子保存全scalar/逐标签/CSV/完整header JSON，再用NumPy+stdlib生成新离散PNG。它是保存结果的数值评分与绘图，**不是模型benchmark，也不是metadata-only**；0新CNN/native/GPU。posthoc v1因目标环境无Matplotlib退出，原日志/EXIT保留；一次授权v2成功，无安装依赖或模型重跑。

## 5. 最新完整精度、时间与内存

| CPU arm | 冷进程秒 | API秒 | 构造秒 | 保存秒 | 最大RSS GB | compile |
|---|---:|---:|---:|---:|---:|---:|
| A1原 | 107.5875 | 104.7752 | 0.1595 | 0.1707 | 13.5211 | 0 |
| B1 cold | 90.7195 | 88.0436 | 0.1535 | 0.1505 | 13.3990 | 1 |
| B2 warm | 89.5184 | 86.7436 | 0.1588 | 0.1517 | 13.4457 | 0 |
| A2原 | 108.2877 | 105.5134 | 0.1685 | 0.1543 | 13.4589 | 0 |

两原arm中位数冷进程107.9376秒、API105.1443秒；candidate cold/warm两arm中位数90.1189秒、API87.3936秒，本例观察耗时分别下降16.51%/16.88%。只有1个T1、2个候选arm，不扩大成总体或更多shape的性能结论。候选完整冷进程仍慢于官方55.0464秒，速度门false。分步骤新whole只量preflight/构造/API/保存；没有新计时把整个CNN/blur拆开。已测单层15.39→5.72秒是 [独立单层结果](../seg_columns_real_layer_v2_20261006/README.md)，不能替代整例计时。

四个CPU完整gzip和CSV SHA完全相同，新旧9,072,000体素差0、各label Dice1、硬体积差0、软CSV差0；完整NIfTI所有struct字段及13个常规geometry/header/extension控制逐值或字节相同。输出图SHA `1d679731…`、CSV `6276e3ea…`。

同官方：差1体素，CSF(label24)少1、背景多1；CSF Dice0.99999856858、其余前景Dice1，硬体积差约1mm³。soft CSV列名/顺序相同，最大绝对差0.8mm³（TIV），CSF0.53、右皮层0.08、左白质0.06，其余非零≤0.004mm³。所有header字段/13控制相同、affine差0；官方gzip/CSV SHA不同，不能称官方位一致。候选新增/移除官方错误均0，保留的1voxel原错误标签也未变。33数值列逐项见 [完整posthoc](whole_results/posthoc_v2/POSTHOC.json)，浮点双精度仅用于离线误差计量，没有改变推理dtype。

| GPU arm | 冷进程秒（观察） | API秒（观察） | Torch allocated/reserved GB |
|---|---:|---:|---:|
| 原 | 10.4420 | 3.7742 | 10.7125 / 14.6151 |
| 新 | 7.5517 | 3.6127 | 10.7125 / 14.6151 |

GPU旧新gzip `a879ab92…`、CSV `00870e28…` SHA相同；前向/参数FP32和TF32作用域/恢复不变。allocated/reserved逐字节相同。driver本人进程树采样最大15.1771GB，两arm一致；两arm各9个非零点、最大gap1.423/0.547秒，没有失败采样，**不能称绝对峰值**。共享GPU单对墙钟只列观察，不给速度倍率；两个进程preflight明显不同。

![当前完整CPU标签对照](case02_current_cpu_labels.png)

图为本轮实际A1原、B2candidate、已有官方，从上到下3行；从左到右矢状/冠状/轴位3列。RAS显示索引89/98/129，离散色和整数2倍显示，不做插值/重采样。新PNG52,305B，SHA `30ef9e7e…`；不是拷贝旧版本图。原MRI/数组/动态库留private。

## 6. 版本记录、失败及剩余门

- 4385B自有胶水先通过编译/短合同与真实同层三次完整preELU位门，见 [真实层v2](../seg_columns_real_layer_v2_20261006/README.md)。旧64Mi改变slab的非exact拒绝和group33无加速拒绝均保留，未采纳。
- `70ad6537`仅窄CPU接入：39项local合同通过；`41ede608`冻结metadata/短worker/whole，source17和common4按实际SHA绑定，GPU未换后端。本次实际2次独立cache编译（short1、wholecold1）。无其它budget/布局/shape搜索。
- 原posthoc v1无Matplotlib导致退出，完整模型先前已全部通过；v2新增atomic先保存数字及自有PNG，4正例/3负例PNG字节/CRC合同通过。原6臂/源/出口63项SHA在posthoc前后不变。构建summary首次只读schema断言按真实phase状态/资源字段修正，0新科学调用。
- 完整CPU/GPU门通过，整合源码绑定及发布记录见ROOT_REVIEW.json和CPU总表；新独立Conda install仍未测。当前已有Conda/GCC真实compile/load通过。未知运行库/编译器/权重/shape继续成熟fallback。
- 33、parc、fast整体同线程官方速度门仍未通过。本轮未重跑默认parc/fast无影响路径，既有55.68/48.30秒与42–43/33.78秒对照不覆盖为新结果。robust SynthSeg+不在本验收。
- 下一项可研究其它groups1/k3层保持原矩阵与SGEMM的工作区复用；本轮不泛化接入，不以单层结果推算其收益。

## 7. 参考、源码和许可

新copy/C++/cache是FNIT自有代码，未复制发布原软件实现体；只使用已有Torch头文件/库和公共SGEMM，不再分发Torch/MKL/GCC/libgomp。见 [归属说明](../../../src/fnit/synthseg_parc/CPU_COLUMNS_NOTICE.md)、[PyTorch2.5.1 Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp)、[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)、[SynthSeg](https://github.com/BBillot/SynthSeg)。Billot et al., *Medical Image Analysis* (2023), [doi:10.1016/j.media.2023.102789](https://doi.org/10.1016/j.media.2023.102789)。模型继续按原资源条款外置；本例2D图来自CC0数据。
