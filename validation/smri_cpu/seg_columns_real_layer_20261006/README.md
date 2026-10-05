# SynthSeg：单层列缓冲复用的真实输入 ABBA 计划（2026-10-06）

## 1. 功能与阶段

**目前仅准备、静态冻结；0上传、0INDEX登记、0worker、0数值。** [v2短合同](../seg_columns_reuse_v2_20261006/README.md)已通过：六个真实权重小输入的完整 FP32 pre-ELU 输出逐位相同，13复制门、23回退守卫和异常恢复通过。小尺寸 M 不代表真实层 M 的结果，本计划等待协调者另行审查授权。

候选只复用原14-plane slab的列缓冲，保留K1944、N24、原M/leading dimensions、C/kD/kH/kW排列、bias预填及同一已加载LP64 SGEMM。最大列缓冲仍为6,242,697,216B，不减少总展开量。它是自有copy加公开SGEMM胶水，不是原隐藏ATen wrapper。生产、GPU、原dtype、后端和旧冻结目录不修改；不重编译v1 `.so`。

## 2. Python接口与真实输入输出

私有程序复用已经审查的v2 `ColumnsReuse`，本轮不复制或修订该类：

```python
# 只有获得本阶段审查授权的单层worker才显式开启。
columns_reuse = ColumnsReuse(
    library=verified_previous_library_path,      # 同一17640B v1 .so
    provider_sha256=verified_provider_sha256,    # 当前Torch已加载LP64库
    allow_compute=True,                          # 仅这次被批准的层调用
    allow_bounded_contracts=False,               # 这里使用真实大shape
)
pre_elu_values = columns_reuse.forward(last_decoder_layer, joined_input_values)
```

| 输入 | 格式与意义 |
| --- | --- |
| 原始来源 | OpenNeuro ds003138 v1.0.1 / CC0，同一个case02 T1；只用已保存检查点，不重新读T1或运行上游CNN。 |
| `skip.npy` | FP32 `[1,24,192,224,256]`，最后一级skip；1,056,964,736B，SHA固定。 |
| `value.npy` | FP32 `[1,48,96,112,128]`，最后一级上采样源；264,241,280B，SHA固定。 |
| `joined_input_values` | 成熟 `join_nearest_cpu` 重建 `[1,72,192,224,256]`，完整值SHA必须等于既有 `38457e8127bddb7d39d3e36435f4b632046eee46bec64abc10e69e79f7e61288`。 |
| 权重与bias | 当前SynthSeg HDF5中的 `SegmentUNet.up[3].conv0`，权重 `[24,72,3,3,3]`、bias `[24]`；文件大小/SHA及参数值前后核对。 |

每个新进程只调用这一层一次，输出为FP32 `[1,24,192,224,256]` 的完整pre-ELU张量，不执行ELU、BN、下一层、head、softmax、分割和统计。A1保存私有 `baseline_preELU.private.npy`；B1/B2/A2只流式比较，不发布数组。公开输出只有各臂scalar `report.json` 和队列记录，包含来源、完整bit计数、shape/stride、内存及独立时钟边界。

joined输入占3,170,893,824B，仅在内存使用，不另存。新私有磁盘只增加A1一个pre-ELU数组：数值1,056,964,608B，加至多4096B header，上限1,056,968,704B（约1.057GB），另有小标量/日志文件；其余臂不保存大型输出。既有1.321GB skip/value不复制。

旧producer六文件SHA、当前14文件SHA、六个终端AST/file证明、权重/manifest/输入文件、v1 CPP/compile/binary和v2五源码/原receipts均固定在 [PLAN.json](PLAN.json)。[READONLY_TARGET.json](READONLY_TARGET.json)为实际只读relay stdout标量记录，checkpoint manifest为1922B/SHA `6976c489eab3724608e6d86503f911a34adccb7321a61274a74f8b559d87a2a3`；当前检查只读过manifest和文件size，未额外导入Torch或读取数组值。未来每臂执行前后才流式核对所有文件完整SHA。

## 3. 私有命令行与参数

以下入口已准备，**尚未执行，必须另获本阶段授权**：

```bash
# deadline从首次排队开始，不因等待或arm切换重置。
timeout --signal=TERM --kill-after=30 23000 \
  python run_abba.py \
    --root "$fnit_server_root" \
    --workspace "$frozen_layer_workspace" \
    --run "$new_layer_run_directory" \
    --approved-real-layer
```

`--root`是固定FNIT入口；`--workspace`固定 `seg-columns-real-layer-v1` 新冻结源码；`--run`必须不存在，保留所有已写receipt、不重跑；`--approved-real-layer`只是被批准阶段的显式CLI门，不能代替协调者授权。四臂固定A1→B1→B2→A2，四个新进程，各180秒硬界。共同CPU锁覆盖这一个有限四臂组；8物理核 `32,36,40,44,48,52,56,60`，OMP/MKL/OpenBLAS/Numba/Torch8、interop8。队列预留750秒供四臂与收尾，外层总界23000秒；不抢占已有任务。

worker地址空间cap及RSS门均32,000,000,000B。清除loader/PYTHONPATH/core overrides，CUDA不可见。没有Module hooks、ATen profiler或其他数值臂。`build_plan.py`是原本地Git工作树的静态复现程序，不在服务器运行；runner和worker只消费冻结PLAN。

## 4. 原步骤及比较方向

没有独立原软件CLI对应这一个内部CNN层。本轮旧方为当前成熟 `CPUInferenceConv3d.forward`，保持原默认256MiB input-slab预算，机械得到depth14/14calls/末次10；新方仅调用字节相同的v2 helper与v1 `.so`。

两方均FP32 / inference_mode / oneDNN=False，原flags恢复；不写TF32/autocast策略。两方在同进程加载当前Torch2.5.1、相同TorchCPU binary和8个installed headers，所有臂都加载同一helper/binary做metadata/provider gate；只有B1/B2通过该胶水计算，A1/A2仍执行成熟Torch路径。公开SGEMM地址/SHA/参数相同仍需真实全输出门，不能仅据ABI宣称等价。

## 5. 预声明验收与计时

| 臂 | 数学与角色 | 真实比较门 |
| --- | --- | --- |
| A1 baseline | 一次原卷积，有限值和来源门通过后保存唯一参考 | `comparison_executed=False`，不算旧新bit门。 |
| B1 candidate | 一次候选，预期14copy/14SGEMM | A1完整pre-ELU FP32 bit差0。 |
| B2 candidate | 另一个新进程候选，14copy/14SGEMM | 同一A1完整bit差0。 |
| A2 baseline | 另一个新进程原卷积 | 同一A1完整bit差0，检查原路径重复性。 |

每臂前后核全部source/runtime/header/provider/weight/input文件；joined值和参数不变，输出finite、FP32、同shape/stride且不别名，正常/异常flags恢复、CUDA未初始化，RSS≤32GB。参考文件SHA在比较前后不变。首个arm失败或非exact即保存其记录并停止剩余臂，不改容差、不重跑、不启动后续模型。

层时钟只包一次真实层调用，包含候选资格/provider保护和私有copy/SGEMM计数wrapper开销；计数不装Module hooks、不改变C++输入。另报process user/system、page faults、RSS及整个worker加载/重join/hash/save/比较时钟。只记录这组ABBA的配对观察，尚不称完整CPU/官方速度收益；缓存和共享机器负载仍可能影响单层时钟。

源/输入hash、加载、重join、保存和比较位于层操作时钟之外。报告前后系统loadavg三个标量，不收集进程名或地址；它们只是共享负载背景，不能证明本层实际使用八个核。

本次计划尚无实际结果、脑图、完整map/CSV/官方误差或GPU输出。即使单层三bit门通过，也不能宣称完整SynthSeg等价或把旧stage 1.836倍拼接比当CNN提升。

## 6. 记录与下一边界

- v1编译/metadata加载通过，未数值；v2独立strict Tensor/forward-AD守卫和短合同通过；两阶段原freeze/receipt/binary保留。
- 此计划复用旧真实skip/value，无新CNN capture；六个producer/current终端证明由本地实际 `git show 4ec078cb` 和当前14源机械重算，未依据Git头替代源SHA。
- [PLAN.json](PLAN.json)声明0上传/0INDEX/0worker/0数学，之后若获授权，实际状态另写receipt，不改冻结PLAN历史。
- 协调者读源码与计划后才能上传并运行这一组；本计划不授权whole CNN、官方、GPU、新cap/layout/provider/dtype或参数搜索。

## 7. 来源与依赖

自有胶水与依赖/许可证据见 [v1](../seg_columns_reuse_20261006/README.md) 和 [v2](../seg_columns_reuse_v2_20261006/README.md)。当前环境没有新增依赖，复用已编译旧库；未证明全新Conda环境的编译安装，若最终生产采用仍需独立验收。

原理依据 [PyTorch2.5.1 Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp) 和 [CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)。[SynthSeg官方仓库](https://github.com/BBillot/SynthSeg)，Billot et al., *Medical Image Analysis*, 2023。没有复制发布原软件实现、动态库、权重或图像。
