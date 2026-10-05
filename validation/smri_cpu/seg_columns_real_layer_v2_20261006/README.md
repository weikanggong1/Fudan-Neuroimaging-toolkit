# SynthSeg：单层列缓冲真实输入检查程序 v2（2026-10-06）

## 1. 功能与阶段

**仅准备、冻结，尚无上传、INDEX登记、合同worker或真实层计算。** 本页不能作为完整SynthSeg或性能验收。v1唯一派发因参数哈希维度检查错误退出，原卷积/copy/SGEMM全部0，详见 [原失败报告](../seg_columns_real_layer_20261006/README.md)。v1源码、PLAN与原始receipt保留，不重跑旧冻结目录。

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

本次尚无实际结果。预声明门：4正例原始字节SHA exact、5D旧哈希exact、3负例拒绝和正常/异常收尾，全部source/runtime/权重/flags/RSS门。前置失败必须保证四个真实层arm0。2D NumPy scalar buffer语义目前只准备合同，尚未运行，不能称已通过。

| 臂 | 角色 | 实际比较 |
| --- | --- | --- |
| A1 | 原卷积唯一参考保存 | `comparison_executed=False`，不计为位门。 |
| B1/B2 | 两个独立候选进程 | 各完整pre-ELU uint32逐位差0。 |
| A2 | 独立原路径重复进程 | 同A1完整逐位差0。 |

真实层源/文件/provider前后SHA、joined/input参数值、shape/stride/dtype/finite/noalias、原flags恢复、CUDA未初始化和RSS≤32GB全通过才接受。保留原参考SHA，首不一致即停。

操作时钟只包单次卷积，包含候选guards/provider保护/counter开销，不含load/join/hash/save/位比较；另报完整worker时钟、user/system/pagefaults/RSS与loadavg前后。仅一组配对观察，不作官方或完整CPU速度比，不把配置8线程称实际GEMM八核占用。没有新脑图或完整map/CSV。

## 6. 历史与下一步

- [v1编译加载](../seg_columns_reuse_20261006/README.md) 与 [v2短数值合同](../seg_columns_reuse_v2_20261006/README.md) 保留，旧库不重编。
- real-layer-v1准备提交a3ad5ed8，一次实际A1 metadata错误，原direct/copy/SGEMM全0，失败报告8f5d9a79。原freeze不修改、不重跑。
- 本独立v2仅修私有哈希守卫并新增纯hash前置；[STATIC_CHECKS.json](STATIC_CHECKS.json)记录源码AST/字节检查，不能替代target worker结果。
- 本计划仍等待root审查授权；无wholeCNN、官方、GPU、低精度、provider/矩阵/layout变更或生产集成授权。

## 7. 来源、依赖与许可

无新增Python或运行库依赖，复用现有Conda环境和旧私有 `.so`。完整新Conda安装/编译尚未验收，不发布库、权重、MRI或第三方源码。

[PyTorch2.5.1 Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp)、[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)、[SynthSeg官方仓库](https://github.com/BBillot/SynthSeg)，Billot et al., *Medical Image Analysis*, 2023。
