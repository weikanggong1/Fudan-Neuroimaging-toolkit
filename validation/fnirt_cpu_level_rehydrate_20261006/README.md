# FNIRT CPU：一次真实 level 状态重建

## 1. 功能与结论

在真实 GM 配准保存的第二 accepted 参数点（`solve3`）上，当前 FNIT 进行一次 `linearize`，内部一次 `evaluate`。count、SSD、bending energy、覆写前 λ/cost，以及全部 1177 个 full gradient 和独立 diagonal 元素逐位等于既有 current assembly。任务 exit0，允许保存后续诊断使用的私有状态。

这是单点状态桥接通过；matrix-free action、求解及完整非线性配准尚未验证，生产源码未改。流程为：保存输入/资源及源码绑定 → 一次当前评价和线性化 → scalar/full-vector 精确门 → 私有 checkpoint → 前后绑定与精度状态检查。结果见 [ACCEPTANCE.public.json](ACCEPTANCE.public.json) 和 [原始 summary](run/stage1/summary.public.json)。

## 2. Python 调用、输入与输出

本目录是验证工具，不提供新的生产 API。科学调用在 [stage1_linearize.py](source/stage1_linearize.py)：`state, gradient, matvec, diagonal = system.linearize(coefficients, scale)`；仅构造 `matvec`，没有调用它。full gradient/diagonal 使用 `2 * gradient`、`2 * diagonal`，与已保存 current assembly 的单位相同。

| 输入 | 格式、意义及来源 |
| --- | --- |
| solve3 参数 | 小端 F64，1177 个数；3×7×8×7 个位移系数及一个全局强度比例，混合单位；SHA `21d339a4…`，不是 solve2 |
| smoothed moving / initial fixed | 原保存 F32 `.npy`；moving 为 224×288×288，fixed level 为 24×28×24；只复用不依赖系数的预处理数组 |
| GM、template、affine、mask | 已授权公开真实 ds003138-derived GM 与声明资源；NIfTI 头信息、体素大小、几何及 affine 复原当前 level；首层 mask 按 config 不启用，只核对其几何 |
| 当前来源 | 17 个 FNIRT 文件加现有 `flirt/coordinates.py`；registration SHA `caab8ffc…`、optimizer `903d5031…`；全部输入及来源见 [expected](source/expected.public.json) |
| 固定有效 λ | `9049.463427795125`；只修改本实例 evaluate 返回的状态，与已有 assembly 条件一致 |

旧输入的 registration SHA 是 `e67c71a6…`，其第一 accepted 参数等于 solve2。当前 optimized CPU blur 来源不同，本次没有重新平滑；上述 full gradient、diagonal 和标量桥接实际通过，支持这一次 solve3 状态重建，不能推及全部当前预处理。

输出 summary 含 scalar、哈希、形状/dtype/stride、config、计数与时钟。私有 `level_state.private.npz` 为 2,647,508 字节、68 个数组，SHA `78abee8589e6e026fede946136c9e78589817b01a1d55d2085aee19758ceae0e`；文件0600、目录0700。数组保持在服务器，未下载或公开。它保存 F32 影像状态、bool mask、F64 系数/basis/weights/gradient、bending Gram/diagonal 和几何；moving 数组仍按原路径/SHA 引用。

checkpoint 保存的是新建数组与 metadata，未持久化旧 Python 对象身份。summary 的 stride 来自写出前的 Tensor 视图，不表示 NPZ 加载会自动保留原 stride；本轮没有回读数组或验证布局恢复。当前 `SpatialNormalCPU` 的 packed layout/scratch 均为 None、copy bytes0；未来恢复与 callback 需要独立验证。

## 3. 命令行与复核

只读复核公开报告，无 NumPy/Torch 导入，不启动科学 worker：

```bash
python validation/fnirt_cpu_level_rehydrate_20261006/verify_report.py
```

当次 runner 是 [run_stage1.sh](source/run_stage1.sh)，科学 worker 参数完整为：

```bash
# fnit_root：现场已读 README/INDEX 的统一 FNIT 根目录
# stage1_workspace：该次冻结源码/expected/PLAN 所在目录
# stage1_output：新建私有输出目录，已存在时拒绝覆盖
python stage1_linearize.py --root "$fnit_root" \
  --workspace "$stage1_workspace" --output "$stage1_output" --approved-stage1
```

`--approved-stage1` 是防误调用参数，不能替代任务授权。原任务已执行一次，无需重跑。nodecw7 使用八个 physical 核（32,36,…,60），线程8、interop1，共用 CPU outer lock 后取本模块锁；child180s、controller19600s、20,000,000,000 字节 address-space cap。调度器明确等待已排 Seg126298 原身份结束才进入共锁。Seg 自身合同失败不改变本任务参数；本任务只等待它释放资源。实际 enqueue、调度和最终索引见 [launch](run/launch.public.json)、[scheduler](source/SCHEDULER.public.json)、[completed INDEX](run/index_completed.public.json)。

## 4. 对应原软件

这是 FNIRT 内部一次 level 线性化，没有独立官方命令。本轮未执行 FSL、未重跑原生配准；比较对象是已绑定的 **当前 FNIT assembly**，不是把旧 native 输出当作当前 evaluate oracle。完整 FNIRT 命令及既有真实原软件比较见 [FNIRT 功能说明](../../docs/fnirt/README.md)。

读取声明 mask 资源不等于链接或运行 FSL。进程开始/结束 FSL DSO 映射快照均空；它们只证明两个记录时刻的映射状态，不是全程系统调用审计。

## 5. 真实精度及耗时

| 当前 saved assembly 桥接门 | 结果 |
| --- | --- |
| count | 14341，准确相等 |
| SSD | 60.329756040354546，F64 bits 相等 |
| bending energy | 2.3604429240341207，F64 bits 相等 |
| 覆写前 effective λ / cost | 9049.463406053183 / 61.81924365370954，F64 bits 相等 |
| full gradient（1177） | 0 different bits；max/relative L2 均0；值 SHA `79e5d87f…` |
| full independent diagonal（1177） | 0 different bits；max/relative L2 均0；值 SHA `833b3d6c…` |
| 29 个源码/输入绑定 | 11 个输入/资源/既有结果身份 +17 FNIRT +1 coordinates；前后相同 |
| 实际调用 | evaluate1、linearize1；callback/solver/native0；未物化 H |

| 时钟边界 | 秒 |
| --- | ---: |
| 导入及开始绑定 | 1.493669 |
| 数组恢复与几何构造 | 0.076581 |
| linearize（包括 evaluate） | 2.103898 |
| 其中 evaluate | 2.088256 |
| scalar/vector 桥接 | 0.000728 |
| checkpoint 写出 | 0.086889 |
| worker 开始至后置绑定、summary 写出前 | 3.850832 |

上述不是配对性能 benchmark，evaluate 已包含于 linearize，不能重复相加。maxRSS526548KiB；address-space cap 不代表实际物理内存峰值20GB。CUDA 未初始化，cuDNN TF32=True、matmul TF32=False、默认 dtypeF32、grad_enabled=True，前后完全保持。不存在本轮 GPU 数值或速度验收。

这是数值状态诊断，没有新输出配准脑图；既有真实完整配准图不能作为本次 callback/solver 验收。checkpoint gate 不代表完整配准精度或收敛恢复。

## 6. 更新记录与文件绑定

1. 旧 freeze `89465e49…` 只供准备审查，未上传或执行。root 要求输出隐私保护后冻结本次 `774a1273…`：worker umask077、目录0700，并增加实际 affinity/env 断言。
2. controller141432 按原 Seg126298 身份排队，只执行一次科学 worker，exit0。冻结科学源码与独立调度器身份分别保留在 [freeze](source/freeze.public.json) 和 [SCHEDULER](source/SCHEDULER.public.json)。
3. 只读 queued 状态采集假设遇到自然完成而 AssertionError；未改变科学状态或重新启动。旧计划和调度 snapshot 的 not-started 字段不是最终状态，解释见 [STATUS_CONTEXT](STATUS_CONTEXT.public.json)，原记录不覆写。
4. [COLLECTION_BINDINGS](COLLECTION_BINDINGS.public.json) 核对23个 as-run 源码/metadata 文件；完整私有 checkpoint 只收身份和 schema。最终 INDEX 在六锁下重读、保留他项变化及权限后原子更新；本轮未修改生产源码。

[MANIFEST.public.json](MANIFEST.public.json) 绑定公开叶全部文件，自身除外。先前 current 系统 old/new 求解及跨组装系统参考限制见 [current replay](../fnirt_cpu_current_replay_20261006/README.md)。

## 7. 参考及下一边界

FNIT 当前实现：[registration](../../src/fnit/fnirt/registration.py)、[spline](../../src/fnit/fnirt/spline.py)、[CPU normal](../../src/fnit/fnirt/_normal_cpu.py)。原实现与方法：[FSL FNIRT](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt.html)；Andersson、Jenkinson、Smith，*Non-linear registration, aka spatial normalisation*，FMRIB Technical Report TR07JA2（2007）。参考链接只提供出处，本轮没有新原生运行。

后续仅准备从本 accepted checkpoint 恢复 callback 的单独计划；不得重新 evaluate/linearize/assembly/PCG。原 H 的逐单位列构造与任意方向的浮点执行次序不同，列一致性不能推广为任意方向逐位一致；观察到舍入差异也不能直接定性为算法错误。stage2 尚未执行，也未接入 CPU runtime。
