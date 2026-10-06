# SynthSeg C24 CPU 列缓冲：短合同结果（2026-10-06）

## 1. 功能与状态

独立原型只优化 `SegmentUNet.down[0].conv1` 的 C24→24、3×3×3 卷积，保留成熟 CPU 的 32 平面 slab、实际矩阵 M/N/K、通道/核顺序、偏置预填及同一已加载 SGEMM 提供者。本轮一次现有 Conda 编译/接口检查和一次短合同 worker **通过**。生产源码、旧 C72 代码/二进制和 GPU 路径未改。

六组实际模型权重的合成输入输出逐位一致，复制/回退/异常恢复门通过。**真实 C24 层、整例 MRI、官方精度、GPU 和速度提升尚未测试。** 本结果不替代真实 benchmark。原准备叶 [PLAN](../seg_columns_c24_prepare_20261006/PLAN.json) 和 [freeze](../seg_columns_c24_prepare_20261006/freeze.public.json) 保持冻结字节；本叶只添加实际结果。

## 2. Python 调用、输入与输出

本原型不是新增用户公共函数；完整 SynthSeg 调用保持 [功能说明](../../../docs/synthseg/README.md)。独立原型调用示例：

```python
from pathlib import Path
from prototype import ColumnsC24

columns_library_path = Path("/private/contract_run/compile/columns_c24.so")  # 本轮独立编译产物
columns_provider_sha256 = plan["provider_sha256"]  # 已加载 Torch 的固定 LP64 SGEMM 文件哈希
columns_helper = ColumnsC24(
    library=columns_library_path,
    provider_sha256=columns_provider_sha256,
    allow_compute=True,              # 仅打开本轮已批准的独立短合同
    allow_bounded_contracts=True,    # 允许本轮六种短尺寸，真实层另行审查
)
layer_output = columns_helper.forward(
    layer=segmentation_model.down[0].conv1,  # 同一 HDF5 的真实 C24 权重及偏置
    image=layer_input,                       # CPU float32 NCDHW，无梯度
)
```

参数逐条见 [准备说明](../seg_columns_c24_prepare_20261006/README.md)：`library` 为新独立 DSO；`provider_sha256` 固定已有提供者；`allow_compute` 默认 False；`allow_bounded_contracts` 默认 False；`layer` 必须是 eval `CPUInferenceConv3d`、kernel `[24,24,3,3,3]`/bias `[24]` 且值 SHA 一致；`image` 为 CPU FP32 普通 Tensor，batch=1/channel=24。输出为同尺寸连续 FP32 pre-ELU，不别名输入或参数。本次输入为种子 20261006 的有限合成张量，权重为实际 `synthseg_2.0.h5`，没有读取 MRI。

## 3. 命令、资源与执行边界

原派发命令对应 [冻结队列](../seg_columns_c24_prepare_20261006/run_prepared.py)：

```bash
python run_prepared.py   --root /cwStorage/home/gongwk/Notebook_code/FNIT   --workspace /cwStorage/home/gongwk/Notebook_code/FNIT/workspaces/smri_cpu_20261004/remaining_20261006/seg-columns-c24-contract-v1   --run /cwStorage/home/gongwk/Notebook_code/FNIT/runs/smri_cpu_20261004/remaining_20261006/seg-columns-c24-contract-v1   --approved-contracts
```

这是已执行记录；相同 `run` 已存在，队列拒绝重复派发。源码目录是本阶段冻结准备文件，输出目录保存原 QUEUE/COMPILE/CONTRACTS/log；`approved-contracts` 是本次短合同授权开关。恢复后核对了已存在的 13 个文件并全部复用，新增上传为 0。生产 17 源、旧 C72 四项、当前 compiler/provider/headers/模型哈希现场匹配。六个 INDEX 锁共同 25 秒上限，队列使用共用 CPU8 锁；完成后锁已释放。

CPU affinity 为 `32,36,40,44,48,52,56,60`；数值 intra/inter8，OMP/MKL/OpenBLAS/Numba8，CUDA不可见且未初始化。编译 metadata worker 保留原 intra8/inter192，只做接口元数据，0张量分配/复制/SGEMM；没有把 inter192 当作其实际计算核数。资源门：共用锁120秒、外层480秒、编译/短worker各180秒、编译子进程120秒、编译AS4GB/数值AS8GB/RSS≤32GB；首差停止、无自动重试。

## 4. 原软件与成熟参考

本轮参考是同一实际权重、同 slab 和同提供者的成熟 `convolution_slabs`/`F.conv3d`、oneDNN关闭。没有独立官方单卷积 CLI；完整原软件命令仅对应如下，未在本轮执行：

```bash
mri_synthseg --i input_T1w.nii.gz --o reference_labels.nii.gz   --vol reference_volumes.csv --cpu --threads 8
```

本次 Conda GCC11.2/Torch2.5.1 ABI0 只编译 FNIT 自有 glue，动态使用现有 Torch/MKL，不调用 FreeSurfer。新 DSO 17,656 B、SHA `b8fd9829d92619fcd031b0c1af5e0fb9e04dee1209909211aa04a6133210d4e7` 留私密目录；未发布模型、动态库或上游实现体。全新独立 Conda 环境未测试。

## 5. 实测精度、时间与内存

| 实际短阶段 | 退出/验收 | worker秒 | 最大RSS B |
|---|---|---:|---:|
| 编译/接口，无张量数学 | RC0 / PASS | 2.912187 | 342,417,408 |
| 一次数值合同 | RC0 / PASS | 2.737028 | 468,205,568 |

时间是含 imports/校验/收尾的观察值；没有同尺寸参考时钟或真实层耗时，不计算提速倍率。compiler子进程最大RSS342,413,312 B。外层 QUEUE 自然完成且 valid_queue=True；**未另存外层 OS 退出码**，两个 child RC0 有原始记录，不事后补造 controller RC0。

| 输入深度，H=13/W=17 | 特征 | 完整FP32位差 | max_abs |
|---:|---|---:|---:|
| 1 | 单平面/边界 | 0 | 0 |
| 2 | 两平面 | 0 | 0 |
| 7 | 内部halo | 0 | 0 |
| 65 | 多32平面slab和末块 | 0 | 0 |
| 67 | 非连续输入/奇数末块 | 0 | 0 |
| 65 | 正负零/取消误差输入 | 0 | 0 |

全部shape/stride/dtype/finite、输入参数不变、输出独立和正常释放门通过。13个copy oracle均bit0，包括H/W单点和poison尾部；实际candidate copy/SGEMM各12。30个回退状态不启动candidate数学；默认compute关闭门、旧fallback错误/provider错误传播、注入copy错误1次与异常buffer释放/flags恢复通过。前后源/模型/二进制/全局hooks/精度状态一致。外部设备仅用meta sentinel核提前回退，**本轮没有实际GPU测试**。

原始 QUEUE SHA `628b8236…`、COMPILE `8703f61f…`、CONTRACTS `f3d07a42…`；完整数值和字段见 [RESULTS.json](RESULTS.json)，原记录仅路径脱敏见 [receipts](receipts/run/CONTRACTS.json)，独立重哈希见 [POST_BINDINGS.json](POST_BINDINGS.json)。当前真实普通33类完整耗时90.119秒/同8核官方55.046秒属于上一轮 C72验收，仍保留 [既有脑图与整例数字](../seg_columns_integration_20261006/README.md)；不把它写成新的C24结果。此次没有新增影像图，因为没有新MRI或分割输出。

## 6. 更新记录与剩余工作

- `fd52712d`、`5ee75b68`：冻结准备与实际权重只读绑定；调整metadata线程声明，0数学。本叶不覆盖准备声明。
- 本次：复用已上传文件，唯一编译→数值队列完成；原receipt保存，两个worker RC0，所有短门通过，无生产接入。
- 先前 C72：整例CPU/GPU输出门通过，观察冷进程约下降16.51%，完整官方CPU速度门未过；见 [报告](../seg_columns_integration_20261006/README.md)。group33-batch无稳定提升、64Mi改变slab导致位差路线仍拒绝。
- 下一步 [真实层预案](NEXT_REAL_LAYER_PLAN.json) 只准备：CC0 ds003138 case02 exact T1，early-stop采集 original/flipped 的 down0.conv0+ELU 输出，停止在 conv1 前，然后同input层 ABBA；须另行冻结worker及审查批准后才能执行。短合同通过不代表真实层通过，更不代表整例或官方速度达标。

公开清单 [RESULTS_MANIFEST.json](RESULTS_MANIFEST.json) 只包含本叶的源码/标量/文档。原始影像、特征数组、权重、DSO和许可证不提交。

## 7. 来源、许可与参考

新胶水来自 FNIT 已验收 C72 自有代码；不复制再发布 Torch/MKL 实现体。本轮权重依既有 FNIT 资产配置合法外置，仅记录大小/SHA；未来T1数据为OpenNeuro ds003138 v1.0.1 / CC0。有限张量不涉及被试影像。

- [SynthSeg源码](https://github.com/BBillot/SynthSeg)，Billot et al., *Medical Image Analysis* 2023，[doi:10.1016/j.media.2023.102789](https://doi.org/10.1016/j.media.2023.102789)。
- [PyTorch2.5.1 Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp)、[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)。
- [OpenNeuro ds003138](https://openneuro.org/datasets/ds003138/versions/1.0.1)；真实图例沿用上一轮已验收报告，不新增或合成脑图。
