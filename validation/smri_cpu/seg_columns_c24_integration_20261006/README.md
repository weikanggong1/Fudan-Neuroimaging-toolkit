# SynthSeg C24 窄生产接入：完整 CPU/GPU 验收（2026-10-06）

## 1. 功能与状态

本版本仅接入普通 SynthSeg 33 类网络的 `down[0].conv1`，复用相同 provider、原 32 平面 slab、列顺序和 FP32 SGEMM。独立 C24 符号与缓存保留旧 C72 数值、资格和缓存键。

**metadata 冷/暖、CPU ABBA 四臂、GPU AB 两臂和一次保存输出评分全部通过。** 运行源码固定为 `2e503082`，基线是 `7ff215ee` 的 17 源；后续 main 的说明提交没有改变运行来源。完整官方 CPU 速度目标仍未通过。

```mermaid
flowchart LR
    A[已核验源码和资源] --> B[C24 metadata 冷编译和暖复用]
    B --> C[CPU 原冷暖原四臂]
    C --> D[GPU 原新两臂]
    D --> E[保存输出评分]
```

每阶段通过完整回执后进入下一阶段，原冻结 PLAN、5 个 harness、源码和运行文件保留；无自动重试。实际编译共 2 次，完整 forward 共 12 次，新官方或原生调用 0。

## 2. 输入、输出、参数与调用

输入为同一 CC0 OpenNeuro ds003138 v1.0.1 case02 三维原 T1，使用相同外置 H5 和标签/名称/拓扑资源。完整输出为 `[180,224,225]`、int32、约 1-mm RAS 网格的分割图及按脑区 mm³ 软体积 CSV。

公共 Python、CLI、所有参数、输入与输出详见[七节功能说明](../../../docs/synthseg/CPU_COLUMNS_C24.md)。没有新公共算法参数：`weights` 选已核验资源目录，`device` 选 CPU/CUDA，`threads` 为线程预算，`cudnn_tf32` 保持原 CUDA 策略；`keep_geometry` 控制原网格恢复，`color_lut` 为色表。示例使用完整变量名，沿原 `SynthSeg` 接口。

内部优化仅接受已验证权重、`[1,24,192,224,256]` CPU FP32、intra8、eval/no-grad、oneDNNFalse 和原卷积参数；未知条件在计算前继续成熟路径。CUDA 提前旁路，不导入或构建 CPU helper。

## 3. CLI、编译、缓存与有限资源

```bash
fnit synthseg --i /data/example/T1w.nii.gz \
  --o /data/output/segmentation.nii.gz --csv-vols /data/output/volumes.csv \
  --weights /data/fnit-weights --device cpu --threads 8
```

`--i` 是原 T1，`--o` 是完整分割，`--csv-vols` 是体积 CSV；其余选项与第二节对应。该示例已按真实 parser 核对，本轮真实验收运行完整 Python API，未增加 CLI 推理。

metadata 使用两个新进程及独立缓存，AS8 GB、worker/child RSS 各4 GB、每臂180秒，编译次数1/0。实际线程 intra8、interop96，前后相同且没有 setter；已加载 ABI10404/provider/DSO门通过。CPU 同8物理核，四臂各600秒/32GB AS和RSS，共用CPU锁；C72 warm cache身份保持，新C24 whole cache cold/warm共用且与metadata缓存分离。GPU两臂各300秒、allocated/reserved和本人树采样均20GB以内，共同GPU锁。每组有限 supervisor 检查实际存活树和锁，三个controller均exit0，观测子树均退出，两个共同锁已释放。

新增依赖0，Conda已有GCC11/ABI0及成熟flags/linker/rpath复用。metadata cold编译1次，CPU B1另编译1次（实测compiler0.653401秒），B2编译0；旧C72编译0。实际sdist的C24 CPP与两Python helper成员逐字节检查通过；完整wheel及全新Conda安装未测。

## 4. 原软件、保存评分与时钟边界

```bash
# 独立官方验证环境；FNIT运行时不调用此命令。
mri_synthseg --i /data/example/T1w.nii.gz \
  --o /data/reference/segmentation.nii.gz --vol /data/reference/volumes.csv \
  --cpu --threads 8
```

官方复用同例、同节点、同8核的[已冻结正式记录](../seg_memory_20261005/OFFICIAL_NODE7_FAST_BA_REPEAT.public.json)，冷CLI55.046403秒。该边界包括CLI/import、推理和主图/CSV保存。本轮API含预处理、完整推理和内部资格/构建检查；worker还包括源码/资源核验和保存，冷进程另含进程启动与完整worker。55秒CLI与80秒API并列记录，不作为同scope精确倍率。

保存输出评分单独获授权后，仅一次读取新CPU A1/B2和已冻结官方图/CSV，原样调用成熟scorer/比较器的全label、软体积CSV、全部NIfTI struct字段与13项geometry/header/extension比较。数字原子保存后typed退出跳过旧scorer的绘图段；六个输入前后SHA保持，复用PNG及来源整SHA核验。评分5.384760秒、冷进程5.526865秒、RSS0.206189GB，120秒/AS8GB/RSS4GB上限内；这是离线误差评分时钟，0新CNN/native/GPU/绘图。

## 5. 完整精度、逐臂时钟与资源

| CPU臂 | API秒 | 构造秒 | 保存秒 | worker秒 | 冷进程秒 | RSS GB | C24编译 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1_baseline | 85.610560 | 0.272221 | 0.171145 | 108.243408 | 108.589227 | 13.401440 | 0 |
| B1_cold | 81.072970 | 0.166816 | 0.150918 | 83.403509 | 83.761533 | 13.419356 | 1 |
| B2_warm | 80.168915 | 0.157999 | 0.150479 | 82.507799 | 82.859582 | 13.419954 | 0 |
| A2_baseline | 85.388194 | 0.179907 | 0.151301 | 87.737165 | 88.115167 | 13.398995 | 0 |

CPU API中位 **85.499377→80.620943秒，减少5.705812%**。worker中位97.990286→82.955654秒；冷进程中位98.352197→83.310557秒（观察减少15.293649%）。A1未细分的核验/收尾时间为20.286242秒，其余约0.125秒；该检查时间差异单列，原因尚未定位，不算作卷积计算改善。只有一个T1、四个CPU进程，本轮没有拆分所有CNN/blur计时。

四CPU整分割gzip与CSV逐文件SHA相同；每臂2个完整forward，C72共8层命中/112copy与SGEMM，C24共4层命中/24copy与SGEMM。实际输入/输出和参数device/dtype、前向TF32/cuDNN/autocast、资源/源码/旧C72缓存、全部恢复门通过。最大RSS13.419954GB。

保存评分实测：新旧9,072,000体素差0、所有前景label Dice1、软CSV差0；完整NIfTI全部struct字段与13项geometry/header/extension控制一致。同官方差1体素，最小前景Dice0.9999985685800458，最大软CSV差0.8000000000465661mm³；完整header/geometry仍一致。候选新增/移除官方错误0/0，保留1，保留错误标签未变；官方并非位一致。

| GPU臂 | API秒（观察） | worker秒 | 冷进程秒 | allocated/reserved GB |
| --- | ---: | ---: | ---: | ---: |
| A_baseline | 5.041465 | 9.898679 | 11.868985 | 10.712467 / 14.615052 |
| B_candidate | 4.640109 | 7.882412 | 8.758892 | 10.712467 / 14.615052 |

GPU整图/CSV完全相同，sameUUID及allocated/reserved字节相同；4个可选CPU模块未import，compile/copy/SGEMM/build/probe全0，实际forward4次。FP32、原默认TF32/cuDNN和autocast策略保持及恢复。本人树driver采样峰值两臂均15.177089GB，样本18/16、非零12/11、失败0、最大gap1.674325/0.566286秒；采样不代表绝对峰值。共享GPU单对时钟只列观察，无稳定加速倍率。

![复用的CC0完整标签对照](../seg_columns_integration_20261006/case02_current_cpu_labels.png)

图复用旧C72已评分PNG，52,305 B、SHA `30ef9e7e…`；原baseline/candidate完整输出SHA与本轮新CPU A1/B2相同。图从上到下为baseline/candidate/官方，列为矢状/冠状/轴位；RAS索引89/98/129，离散2倍显示，无重新绘图或插值。完整来源与SHA在[公开清单](manifest.public.json)，原MRI/私有路径/动态库不发布。

## 6. 更新、失败与剩余门

- [C24短合同](../seg_columns_c24_contracts_20261006/README.md)与[真实层ABBA](../seg_columns_c24_real_results_20261006/README.md)通过，单层2.177倍与整例API5.706%分开。原型CPP字节与数值AST保持，不重复合成数学。
- `2e503082`冻结窄生产接入：旧C72三源逐字节保留，独立C24缓存，56项guard/cache与8项队列/资源软件合同通过。独立审查补齐每forward实际device/precision和helper前后SHA、锁FD传递、有限清理与整个controller deadline。
- 本轮metadata2、CPU4、GPU2全部首次exit0，0科学重试；首差停止门保持。20份原始JSON回执长度/SHA已校验，保留私有；最终71个上传代码/metadata文件、6资产、旧C72两个cache文件及12个完整保存文件逐条保持；源码/资产及运行结果不被报告覆盖。
- 文档CLI另作docs-only修正为真实 `fnit synthseg --csv-vols`，官方命令的 `--vol`保持。保存输出评分只执行一次并通过，复用旧CC0 PNG，完整wheel/全新Conda install未测。
- 普通33类完整官方CPU速度目标仍未过；默认parc/fast/Plus以及其他权重、shape、层不扩展，不把本轮计时覆盖为这些模式的新结果。

## 7. 来源、许可与参考

C24 CPP/helper/cache为FNIT自有代码，复用已安装Torch headers、库和同provider SGEMM，不分发Torch/MKL/GCC动态库或外置权重。图来自已核验CC0数据、复用既有PNG；模型资源沿原条款外置。

- [功能说明](../../../docs/synthseg/CPU_COLUMNS_C24.md)、[冻结准备](../seg_columns_c24_integration_prepare_20261006/README.md)、[实际指标与代码清单](manifest.public.json)。
- [SynthSeg](https://github.com/BBillot/SynthSeg)，Billot et al., *Medical Image Analysis* (2023)，[doi:10.1016/j.media.2023.102789](https://doi.org/10.1016/j.media.2023.102789)。
- PyTorch2.5.1 [ConvolutionMM3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp)、[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)，[胶水归属](../../../src/fnit/synthseg_parc/CPU_COLUMNS_NOTICE.md)。
