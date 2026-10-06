# SynthSeg C24 真实层结果（2026-10-06）

## 1. 功能与结果

本次验证 `SegmentUNet.down[0].conv1` 的 CPU 列缓冲复用。复用与成熟卷积相同的 32 平面分块、列顺序、偏置预填和已加载 LP64 SGEMM 提供者，仅减少重复分配。真实 T1 的原图与翻转图都通过完整 FP32 位比较；两次前向的层时钟中位数由 **10.889 秒降至 5.003 秒（2.177 倍，减少 54.06%）**。

这次完成一次前缀采集和四臂 ABBA，生产接入尚未实施。完整分割、CSV、CPU/GPU 整例与原软件性能比较沿用各自既有记录，不能由本层结果推算。

```mermaid
flowchart LR
    A[真实 T1] --> B[成熟预处理一次]
    B --> C[原图与翻转图分别 conv0 加 ELU]
    C --> D[保存两个输入，在 conv1 前停止]
    D --> E[A1 成熟层：保存两个参考]
    E --> F[B1 候选：完整逐位比较]
    F --> G[B2 候选：完整逐位比较]
    G --> H[A2 成熟层：完整逐位比较]
```

## 2. Python 调用、输入与输出

这组结果来自独立验证脚本，未新增公共推理参数。以下例子读取公开指标，无影像读取或模型运行：

```python
import json
from pathlib import Path

c24_result_directory = Path("validation/smri_cpu/seg_columns_c24_real_results_20261006")
c24_result_manifest = json.loads((c24_result_directory / "manifest.public.json").read_text())
c24_layer_shape = c24_result_manifest["layer_input_shape"]  # N、C、D、H、W
c24_baseline_pair_seconds = c24_result_manifest["timing"]["baseline_pair_median_seconds"]
c24_candidate_pair_seconds = c24_result_manifest["timing"]["candidate_pair_median_seconds"]
c24_maximum_rss_bytes = c24_result_manifest["resources"]["maximum_RSS_bytes"]
```

- 原始输入为真实 3-D T1 NIfTI；网格、预处理、翻转及参数转换复用冻结的成熟实现。公开报告保留指标与代码身份；原始影像、特征数组、参考数组及其身份信息留在私有验证记录中。
- 层输入为 CPU、连续 FP32 的 `[1,24,192,224,256]`，轴为 NCDHW。翻转严格沿网络输入第 2 轴，原图和翻转图分别执行。
- `layer` 为 `CPUInferenceConv3d` 的 `down[0].conv1`，eval、无梯度；kernel `[24,24,3,3,3]`、bias `[24]`，使用同一实际 HDF5 的参数位序。stride/dilation 均为 1，padding 为 1，groups 为 1。
- 层输出为同形状、连续 FP32 的 preELU 张量。stride 为 `[264241152,11010048,57344,256,1]`；输出不与输入或参数共享存储。随后 ELU、BN、pool 和其他网络层不在本次运行范围。
- `library`、`provider_sha256`、`allow_compute`、`allow_bounded_contracts` 属于[独立原型参数](../seg_columns_c24_prepare_20261006/README.md)。真实运行使用 `allow_compute=True`、`allow_bounded_contracts=False`，未增加生产 API 参数。
- A1 保存两个完整参考；B1/B2/A2 只保存标量报告。逐个 FP32 uint32 比较全部输出，不采用抽样、统计容差或模拟数据。

## 3. 命令行与复核

该层没有独立 FNIT 公共 CLI。已经结束的唯一队列不再派发；可用下例查看公开结果：

```bash
python -m json.tool validation/smri_cpu/seg_columns_c24_real_results_20261006/manifest.public.json
```

冻结验证脚本参数含 `root`（统一入口）、`workspace`（冻结源目录）、`run`（唯一私有输出目录）和 `approved-real-layer`（本次批准开关），完整定义见[准备文档](../seg_columns_c24_real_prepare_20261006/README.md)。准备文档描述准备时点；本页记录实际执行后的结果，冻结源不改标为新版本。

CPU 使用同一 8 个物理核，Torch intra/inter、OMP/MKL/OpenBLAS/Numba 各 8。数值作用域 oneDNN 关闭、FP32、无 autocast，CUDA 未初始化；进入与退出时 flags、线程、源码及提供者一致。实际运行未编译新 DSO，复用已通过短合同的 C24 产物。

## 4. 原软件对应

这是 SynthSeg 内部卷积，没有独立官方命令。完整 SynthSeg 的对应参考命令为：

```bash
mri_synthseg --i input_T1w.nii.gz --o labels.nii.gz --vol volumes.csv --cpu --threads 8
```

`--i` 为原始 T1，`--o` 为分割图，`--vol` 为脑区体积 CSV，`--cpu` 指定 CPU，`--threads 8` 为线程预算。这次没有调用原软件；成熟 FNIT 的 `convolution_slabs/F.conv3d` 是本层数学与位序参考。官方[独立 Python 命令和参数](https://github.com/BBillot/SynthSeg#try-it-in-one-command-)同样对应完整模型。

## 5. 真实精度、耗时与资源

| 臂 | 原图层秒 | 翻转层秒 | 两次合计秒 | 冷 worker 秒 | 最大 RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 成熟层 | 5.442066 | 5.419593 | 10.861659 | 35.209871 | 8.706695 |
| B1 列复用 | 2.491528 | 2.522443 | 5.013972 | 23.848563 | 9.765122 |
| B2 列复用 | 2.491223 | 2.499855 | 4.991077 | 23.397481 | 9.769492 |
| A2 成熟层 | 5.473928 | 5.442470 | 10.916398 | 29.403498 | 9.757446 |

层时钟包含候选资格检查和复制/SGEMM 计数开销。冷 worker 包含启动、导入、模型读取、输入与参考哈希、位比较、报告 I/O；A1 生成参考，其余臂读取参考，因此冷 worker 分列观察，不据此给出匹配的整例加速结论。

六次比较（B1/B2/A2 的原图与翻转图）均为 **不同位元素 0，最大绝对差 0，全部有限，shape/stride/dtype 一致**。全部输入、参数、参考、捕获文件及源码身份前后不变，所有后置条件通过。

前缀预处理 4.318054 秒；前缀完整 worker 21.761034 秒，外部冷 worker 22.095596 秒，RSS 2.637062 GB。五阶段 controller 134.868886 秒、supervisor 135.121064 秒，退出码 0，无科学重试。

实际计数：MRI 解码 1、preprocess 1、conv0 2、conv1 8（候选 4）、candidate copy 24、SGEMM 24；完整 model.forward、原软件、GPU、编译均为 0。每个候选 pass 有 6 个 32 平面 slab。

最大实际 RSS 为 **9,769,492,480 B = 9.77 GB / 9.10 GiB**，低于 32 GB 门槛；地址空间限制单列为 32 GB。四个私有数组合计 4,227,858,944 B，报告与日志合计 111,500 B。CPU 共用锁在 controller 完成后释放，监督进程与科学子进程均已结束。

本层没有新分割图，脑区 Dice、CSV 与脑图不适用。完整接口的既有脑图及各自精度/时间记录见[已验收 C72 接入报告](../seg_columns_integration_20261006/README.md)，其版本与时钟边界保持原记录。

## 6. 更新与 benchmark 记录

- 本次真实层：基线源码 `7ff215ee86c49414b2fa6156fdf6769aa54d00f8`，冻结准备提交 `cf649e54a6425215ff14fba2e9b71ec04f69ad82`；唯一前缀与 ABBA 五阶段全部通过。此时生产未改变。
- [C24 短合同](../seg_columns_c24_contracts_20261006/README.md)：一次独立编译、六组实际权重合同 bit0，复制/回退/释放门通过；合成合同只承担前置检查。
- [C24 准备](../seg_columns_c24_prepare_20261006/README.md)及[真实层准备](../seg_columns_c24_real_prepare_20261006/README.md)：保留冻结时点的代码、参数和检查记录。
- 已验收 C72 路径及其源码、二进制、提供者、cache 身份在本次前后均一致。旧 C72 的整例观察与未过官方 CPU 速度门的结论保持原记录。
- 后续只准备 `down[0].conv1` 实际权重、完整 shape 的窄生产接入及有限 CPU/GPU 整例计划，由根任务审查一次执行；未扩大层、权重、shape 或低精度范围。

## 7. 来源、许可与参考

MRI 预处理、读写及网络复用 FNIT 的 PyTorch/nibabel 实现；C++ 为自有复制胶水，连接当前环境 Torch 已加载提供者。未新增依赖、未复制发布上游实现、动态库、MRI 或权重。权重复用现场核验的既有资产；新增安装与生产接入另列验证。

- Billot et al. *SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining*. **Medical Image Analysis** (2023)，[官方代码与文献](https://github.com/BBillot/SynthSeg)。
- PyTorch 2.5.1 [ConvolutionMM3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp) 和 [CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)：成熟 FP32 CPU 参考接口。
- [公开数据集页面](https://openneuro.org/datasets/ds003138/versions/1.0.1)：原始数据从上游取得，公开报告只保留本次指标和代码。
- [manifest.public.json](manifest.public.json)：完整标量、实际调用计数与代码 SHA；原始逐字节回执由根任务独立核验，保存在私有记录。
