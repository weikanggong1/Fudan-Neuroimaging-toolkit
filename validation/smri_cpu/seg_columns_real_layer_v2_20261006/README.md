# SynthSeg：单层列缓冲真实输入检查程序 v2（2026-10-06）

## 1. 功能与阶段

**一次纯哈希前置合同和完整单层ABBA已自然完成，全部exit0；三实际完整FP32逐位门通过。生产尚未接入。** 本页不能作为完整SynthSeg或性能验收。v1唯一派发因参数哈希维度检查错误退出，原卷积/copy/SGEMM全部0，详见 [原失败报告](../seg_columns_real_layer_20261006/README.md)。v1源码、PLAN与原始receipt保留，不重跑旧冻结目录。

v2仅把私有 `value_sha` 的 `ndim == 5` 改成 `ndim >= 2`，保留singleton batch、C-contiguous限制以及完全相同的逐通道字节哈希。真实权重加batch后是6维，bias和图像哈希也需各自处理；image/output的独立5维shape门保持。单层 `layer_trial.py` 与v1逐字节相同，候选胶水、C++、旧 `.so`、生产14源、原FP32/后端/14-plane几何不改。

## 2. Python接口、输入与输出

```python
# 私有哈希helper，只读取CPU Tensor的连续原始字节。
weight_with_batch = last_decoder_layer.weight.view(1, *last_decoder_layer.weight.shape)
weight_value_sha256 = value_sha(weight_with_batch)  # [1,24,72,3,3,3]
```

纯哈希前置worker只复制/view固定uint32位模式，包含signed zero、infinity和NaN payload；不做浮点算术。四个正例为2D `[1,24]`、5D `[1,3,2,3,4]`、6D `[1,24,72,3,3,3]` 位模式，以及已经SHA绑定的HDF5实际权重6D。oracle直接读取整块C-contiguous原始字节，与逐通道hash独立比较；5D另与旧循环完全相同的legacy hash比较。rank1、batch2、非连续三负例必须拒绝。实际权重只构造/加载模型参数，不调用任何model/layer forward。

哈希前置worker不读取MRI数组，不导入列缓冲helper或旧动态库，不调用copy/SGEMM/卷积。公开输出为 `HASH_CONTRACTS.json`，包含每例SHA、维度、负例、source/runtime/flags和RSS；没有数组输出。

通过后才允许原计划四个新进程：复用CC0 OpenNeuro ds003138 v1.0.1同case02 T1已存末端 `skip.npy` `[1,24,192,224,256]` 与 `value.npy` `[1,48,96,112,128]`，成熟join恢复 `[1,72,192,224,256]`，只运行 `SegmentUNet.up[3].conv0`。输出是完整FP32 `[1,24,192,224,256]` pre-ELU；不执行ELU/BN/head/softmax/后续CNN、分割、统计。源/权重/模板许可及输入SHA沿用 [v1输入表](../seg_columns_real_layer_20261006/README.md) 和 [PLAN.json](PLAN.json)。

joined输入仅在内存使用3,170,893,824B；新私有大数组仅A1唯一pre-ELU参考，含header上限1,056,968,704B。B1/B2/A2不保存大型输出。单层三实际位门不能替代完整map/CSV/官方或GPU验收。

## 3. 私有命令行

```bash
# 仅在协调者独立审查并授权本v2冻结后执行一次。
timeout --signal=TERM --kill-after=30 23000 \
  python run_abba.py \
    --root "$fnit_server_root" \
    --workspace "$frozen_real_layer_v2_workspace" \
    --run "$new_real_layer_v2_run_directory" \
    --approved-real-layer
```

`--root`是固定FNIT入口；`--workspace`为独立 `seg-columns-real-layer-v2`；`--run`必须不存在；`--approved-real-layer`是显式程序门，不能代替协调者批准。先一个60秒纯hash worker，成功后A1→B1→B2→A2各180秒，CPU8公共锁覆盖这一组，固定八个物理核 `32,36,40,44,48,52,56,60`，Torch/interop/OMP/MKL/OpenBLAS/Numba8，CUDA不可见。

总deadline23000秒，从首次排队起不重置，预留810秒；地址空间/RSS各32,000,000,000B。任何preflight或层arm失败停止剩余，无重试/宽容差/参数搜索。所有科学5源码由PLAN固定；`build_plan.py`仅本地静态生成PLAN，服务器不运行它。

## 4. 原步骤

没有对应这一个内部层的官方独立CLI。A1/A2为成熟 `CPUInferenceConv3d.forward`，默认256MiB预算给原depth14、14calls/末次10；B1/B2复用已经短合同通过的v2 `ColumnsReuse` 和字节相同的旧17640B库。

K1944/N24/M802816（末次573440）、NN、lda/ldc实际紧凑M、ldb1944、C/kD/kH/kW顺序、bias预填+beta1、同Torch已加载LP64公共SGEMM保持。最大columns6,242,697,216B，生命周期仅单次层调用。该自有copy/SGEMM胶水不等于隐藏ATen wrapper；不换provider、不改allocator、生产或GPU。

## 5. 验收与时间边界

本次已通过下列原预声明门：4正例原始字节SHA exact、5D旧哈希exact、3负例拒绝和正常/异常收尾，全部source/runtime/权重/flags/RSS门。前置失败必须保证四个真实层arm0。2D NumPy scalar buffer语义已通过目标Torch2.5.1/NumPy实际合同，包含signed0与NaN payload，原始字节SHA相同。

| 臂 | 角色 | 实际比较 |
| --- | --- | --- |
| A1 | 原卷积唯一参考保存 | `comparison_executed=False`，不计为位门。 |
| B1/B2 | 两个独立候选进程 | 各完整pre-ELU uint32逐位差0。 |
| A2 | 独立原路径重复进程 | 同A1完整逐位差0。 |

真实层源/文件/provider前后SHA、joined/input参数值、shape/stride/dtype/finite/noalias、原flags恢复、CUDA未初始化和RSS≤32GB全通过才接受。保留原参考SHA，首不一致即停。

操作时钟只包单次卷积，包含候选guards/provider保护/counter开销，不含load/join/hash/save/位比较；另报完整worker时钟、user/system/pagefaults/RSS与loadavg前后。仅一组配对观察，不作官方或完整CPU速度比，不把配置8线程称实际GEMM八核占用。没有新脑图或完整map/CSV。

### 实际同层结果

[HASH_CONTRACTS.json](HASH_CONTRACTS.json)、[QUEUE.json](QUEUE.json)及四arm报告均按远端原字节/SHA收回。[SUMMARY.json](SUMMARY.json)由 [build_summary.py](build_summary.py) 机械重算；公开不含原数组、权重、动态库或参考文件。[INDEX_CLOSED.json](INDEX_CLOSED.json)另记录六锁终态、实际原receipt SHA与参考文件当前身份。

| arm | 操作秒 | user秒 | system秒 | minor faults | RSS GB | 实际比较 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A1原路径 | 15.383990 | 35.103042 | 40.044285 | 23,783,610 | 11.3949 | 唯一参考生成，不计位门。 |
| B1候选 | 5.728821 | 33.030189 | 6.758967 | 3,894,691 | 11.3175 | 完整264,241,152个FP32值逐位差0。 |
| B2候选 | 5.711896 | 33.123028 | 6.698394 | 3,707,043 | 11.3188 | 同一A1完整逐位差0。 |
| A2原路径 | 15.387470 | 35.273149 | 40.093069 | 23,842,166 | 11.3970 | 同一A1完整逐位差0。 |

四输出值SHA都是 `16d9251480386c56e8cbec6e20e406d2ef6560869799d008e007840077ffa69f`；shape、连续stride、dtype、finite和noalias相同。B1/B2各14copy/14SGEMM；A1/A2候选calls0。完整joined/权重/bias值和文件/source/runtime/header/provider/PLAN/flags前后门通过，CUDA未初始化，最大RSS11.397GB低于32GB。

纯哈希4正例、3负例通过；前置卷积/copy/SGEMM均0。A1唯一私有参考1,056,964,736B，实际文件SHA `1bec5b85dd84bbcc464bc2a18a02260fb99042418ad7fadf27928e3674cf9a94`，三比较及结束再核文件不变。新编译/wholeCNN/tail/native/GPU全0，科学重试0。v1失败记录原样保留。

这组ABBA的操作中位数15.385730→5.720358秒，描述比约2.69；只限当前同层/同机器/同8核/同输入。操作时钟含候选资格/provider/counter成本，不含load/join/hash/IO。对应worker含IO/hash观察31.352/19.629/19.764/29.724秒，外层31.754/20.041/20.192/30.153秒；不能混为单层耗时。loadavg前后记录在各arm，不能推出GEMM独占八核。

system均值40.069→6.729秒、minor fault均值23,812,888→3,800,867，user均值35.188→33.077秒。这与一次columns工作区复用减少重复缺页成本相符；没有逐次allocator/syscall trace，不能声称查明唯一原因。峰值RSS基本相同；该候选主要改善本层耗时，并未把6.243GB workspace压缩。

完整CPU官方速度门仍未通过：已保存普通33完整112.95秒与官方55.05秒的差距不是本层试验的验收范围。下一步需要CPU限定生产fallback、Conda安装编译、受影响原T1完整输出/header/CSV门和CUDA原路保护；不自动扩展layer/shape或把单层比推算整CNN。

## 6. 历史与下一步

- [v1编译加载](../seg_columns_reuse_20261006/README.md) 与 [v2短数值合同](../seg_columns_reuse_v2_20261006/README.md) 保留，旧库不重编。
- real-layer-v1准备提交a3ad5ed8，一次实际A1 metadata错误，原direct/copy/SGEMM全0，失败报告8f5d9a79。原freeze不修改、不重跑。
- 本独立v2仅修私有哈希守卫并新增纯hash前置，准备提交b7cc5f83。冻结时的 [STATIC_CHECKS.json](STATIC_CHECKS.json) 原档保持；实际结果另写原始receipts，不覆写准备历史。
- 本v2获root独立审查后一次授权，timeout183231及controller/5worker已自然退出；六共享INDEX锁闭合为 `bounded_real_layer_exact_passed_pending_production_review`。无wholeCNN、官方、GPU、低精度、provider/矩阵/layout变更或生产集成验收。

## 7. 来源、依赖与许可

无新增Python或运行库依赖，复用现有Conda环境和旧私有 `.so`。完整新Conda安装/编译尚未验收，不发布库、权重、MRI或第三方源码。

[PyTorch2.5.1 Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp)、[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)、[SynthSeg官方仓库](https://github.com/BBillot/SynthSeg)，Billot et al., *Medical Image Analysis*, 2023。
