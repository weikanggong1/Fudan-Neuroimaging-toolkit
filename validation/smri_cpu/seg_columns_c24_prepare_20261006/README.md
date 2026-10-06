# SynthSeg C24 列缓冲复用：准备阶段

## 1. 功能与当前状态

本目录只准备 `SegmentUNet.down[0].conv1` 的独立 CPU 原型。它复用已验收 C72 实现的提供者资格、复制顺序和 SGEMM 参数约束，新增 C24、最多 32 平面入口。生产代码、旧 C72 源码和二进制保持原样；CUDA 生产调用路径没有接入本原型。

当前完成的是服务器索引及实际权重的只读绑定、代码准备和 stdlib 静态检查。**尚未编译、加载新 DSO、执行数值合同、读取影像或运行 CNN；尚无 C24 精度、内存峰值或提速结果。** `PLAN.json` 声明后续合同，`PREPARATION_CHECKS.json` 只记录准备检查。

短合同的顺序是新私密目录编译及接口加载 → 六组实际权重的合成数值合同与复制/回退检查 → 首差停止。短合同通过后，才另行审查真实层输入采集与回放。合成合同不替代真实 benchmark。

## 2. Python 调用、输入和输出

这是验证原型，不是 FNIT 新公共接口。下面代码须在后续授权、接口门及资源门通过后使用；默认 `allow_compute=False`，CPU 数值调用会报错。

```python
from pathlib import Path
from prototype import ColumnsC24

columns_library_path = Path("/private/new_run/compile/columns_c24.so")  # 本阶段独立编译产物
columns_provider_sha256 = plan["provider_sha256"]  # 已加载 Torch LP64 SGEMM 提供者的固定哈希
columns_helper = ColumnsC24(
    library=columns_library_path,
    provider_sha256=columns_provider_sha256,
    allow_compute=True,                 # 后续授权合同才打开
    allow_bounded_contracts=True,       # 仅验证脚本允许小尺寸；真实层默认 False
)
layer_output = columns_helper.forward(
    layer=segmentation_model.down[0].conv1,  # 同一 HDF5 的真实 C24→24 权重及偏置
    image=layer_input,                       # CPU float32，B=1、C=24 的 NCDHW 输入
)
```

- `library`：本阶段的新 DSO 路径；不覆盖或重编旧 C72 二进制，不搜索通用 BLAS。
- `provider_sha256`：已加载 Torch 的公共 `sgemm_` 所属文件哈希；地址、路径和 SHA 不匹配即报错。
- `allow_compute`：默认 False；仅接口加载可以关闭状态执行。CPU 数值调用关闭时明确报错。
- `allow_bounded_contracts`：默认 False，仅接受真实层形状 `[1,24,192,224,256]`；True 只供已声明的合成短合同。
- `layer`：严格为 `CPUInferenceConv3d`、eval 状态；kernel `[24,24,3,3,3]`、bias `[24]`，FP32 连续参数的逐位哈希必须等于只读绑定。
- `image`：CPU FP32、普通 Tensor、无梯度；允许非连续输入，按原 halo 规则生成连续 slab。输出为同形状连续 FP32 Tensor，不别名输入或参数。
- 窄守卫拒绝训练、autocast、oneDNN、线程不是 8、hooks、subclass、forward AD、负视图、参数变更等状态，保留成熟层调用。外国设备在参数/提供者检查前返回原层调用；该短合同只用 meta sentinel 验证，不代表实际 GPU 验收。
- 偏置先预填，SGEMM `beta=1`；列顺序为 `C,kD,kH,kW`。每 slab 保持原 M/N/K 与 leading dimensions，末次 slab 使用实际 M 的紧凑前缀。两 scratch 只驻留单次层调用，不在模型或全局缓存中保留。
- 复制、SGEMM 或后置资格门发生错误后传播异常，不改用旧算法重复计算。

实际权重是 53,079,152 B 的 `synthseg_2.0.h5`，文件 SHA `f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e`。HDF5 entry `unet_conv_downarm_0_1` 的 PyTorch kernel SHA 为 `080e9e82b43ebd3eb35fa991a5d264b073c35e0c027aea9534510470e552d699`，bias SHA 为 `e5c6f0bafe28b340b599bd6b52faaec3a1c462edd6676103c0efddfbd6a73944`。只读过程未构造网络或做参数运算，记录见 `WEIGHT_BINDING.json`。

## 3. 验证命令与资源

**当前不要执行。** 后续授权并上传冻结代码、依六锁协议登记索引后，唯一队列命令如下：

```bash
python run_prepared.py \
  --root /cwStorage/home/gongwk/Notebook_code/FNIT \
  --workspace /cwStorage/home/gongwk/Notebook_code/FNIT/workspaces/smri_cpu_20261004/remaining_20261006/seg-columns-c24-contract-v1 \
  --run /cwStorage/home/gongwk/Notebook_code/FNIT/runs/smri_cpu_20261004/remaining_20261006/seg-columns-c24-contract-v1 \
  --approved-contracts
```

`root` 为 FNIT 固定入口；`workspace` 为冻结源码；`run` 必须不存在，保存 `QUEUE.json`、编译日志与 `COMPILE.json`、`CONTRACTS.json`；`approved-contracts` 是后来明确批准本阶段的执行开关，当前仅有准备授权。队列不上传、修改 INDEX 或重试。

CPU 亲和性为 `[32,36,40,44,48,52,56,60]`，数值 worker 的 Torch intra/inter 为 8，OMP/MKL/OpenBLAS/Numba 限定 8。编译 metadata worker 无张量运算，记录并保留 Torch 原线程状态；不把默认 interop 状态算作已用核数。共用锁最多等待 120 秒，外层 480 秒；编译 worker 180 秒、编译子进程 120 秒、短合同 worker 180 秒。编译地址空间上限 4 GB，短合同 8 GB，RSS 门 32 GB。源、旧 C72 文件、参数、精度 flags、hooks 和 CUDA 未初始化均核对前后；首次失败保存实际退出及日志并停止剩余项。

Conda GCC 11.2.0、Torch 2.5.1 ABI0、现有 Torch headers/libtorch/MKL 的哈希全部固定。只动态使用既有已加载提供者，不 `dlopen` 另一套 BLAS，不改 allocator 或全局环境。不增加依赖，不发布编译产物或权重。新 C24 编译和新建 Conda 环境均尚未测。

## 4. 原软件对应

目标层是 SynthSeg 神经网络内部卷积，没有独立官方命令。完整分割参考为 `mri_synthseg --i input.nii.gz --o labels.nii.gz`；本准备阶段不调用官方程序，也不比较新分割输出。生产 CLI/API 未变。

PyTorch 成熟参考仍为同一 `convolution_slabs`/`F.conv3d`、oneDNN 关闭。小合同明确给 `maximum_slab_bytes=34*C*H*W*4`，由原公式得到 `min(32,34-2)=32`；不会把 C72 的 14 平面规则借用到 C24。C++ 是 FNIT 自有复制胶水，未嵌入 PyTorch/MKL 源码。

## 5. 精度、时间与内存证据

本 C24 原型未运行，所有新增结果为 **pending**。计划六种输入深度 1、2、7、65、67、65，含内部 halo、多次 32 平面分块、非连续末轴、正负零与取消误差输入；全部使用实际 HDF5 权重。预计 12 次 candidate copy/SGEMM、13 次独立 copy oracle（额外 H/W=1），30 个不启动数学的回退守卫及一次注入复制错误。逐个 FP32 `uint32` 位、shape/stride/finite、输入参数保持、poison 尾部和释放生命周期门均须通过；首差停止。

旧真实 profile 的 `down.0.conv1` 两前向合计 12.661 秒来自优化前、带 instrumentation 的 API。它证明应考察的位置，**不是当前 C24 baseline，也不能据此推算收益**。当前已验收 C72 接入的普通 33 类冷 worker ABBA 中位数约 90.119 秒；复用同节点同 8 核官方 CLI 55.046 秒。两个时钟边界不同，尚未通过整体官方速度门。本原型不重跑官方、不放宽既有分割/CSV 门。

真实 C24 几何机械推导：M=1,835,008，N=24，K=648，每次 32 平面，6 slabs。columns 4,756,340,736 B，scratch output 176,160,768 B，halo chunk 187,170,816 B，整层输入和输出各 1,056,964,608 B，简单和约 7.23 GB。该和不含网络常驻张量、BLAS、allocator 和采集缓冲，不能当作 RSS；真实层后续仍须实测 ≤32 GB。

后续若短合同通过，先另批原病例 original/flipped 输入的 early-stop capture：只跑到 `down0.conv0 + ELU`，在 `down0.conv1` 前停止；再做唯一真实层 ABBA，A1 仅 reference、B1/B2/A2 三次完整 preELU 位比较。当前未批准或准备这些科学执行，不以单层结果替代整例验收。

## 6. 更新与已拒路线

- 当前：新增独立 C24/32 入口及有限计划，实际权重只读绑定完成；0 编译/数值/CNN/上传。
- 前次 C72：同矩阵列缓冲复用已通过真实层和 CPU/GPU 整例门，见 `../seg_columns_integration_20261006/`；本阶段不改旧源或 binary，也不迁用其单层倍率。
- group33→batch11：逐位和内存门通过，但无稳定速度提升，不重复该路线。
- 64 MiB slab：真实参数短合同因改变 slab M 出现位差，未做真实层，不降低 C24 slab 深度。
- 首次只读权重查询因 CPU 节点没有 Git 可执行程序在权重读取前停止；后续仅将 HEAD 查询改为 stdlib 读取。见 `PREPARATION_HISTORY.json`，不是数值失败或科学重跑。

## 7. 代码来源与参考

自有原型来源是已验收 `_columns_reuse.cpp` 的 C24 特化，精确来源 SHA 和最小改动静态验证见 `PREPARATION_CHECKS.json`；不再分发 Torch headers、MKL、权重或 DSO。权重位置复用当前 FNIT 资产清单。

- [SynthSeg 官方代码](https://github.com/BBillot/SynthSeg)：Billot et al., *NeuroImage* 2023，SynthSeg。
- [PyTorch 2.5.1 ConvolutionMM3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp) 与 [CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)：对应现有成熟 F32 CPU 路径的接口依据，实际数学匹配须以合同证明。
- 本项目 [已验收 C72 接入报告](../seg_columns_integration_20261006/README.md)、[原真实 profile](../seg_cpu_profile_20261006/README.md)、[blur 否定结果](../seg_cpu_blur_trial_20261006/README.md)。
