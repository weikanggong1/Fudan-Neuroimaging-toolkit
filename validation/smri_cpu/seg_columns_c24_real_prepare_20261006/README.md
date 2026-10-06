# SynthSeg C24 真实层：冻结准备（2026-10-06）

## 1. 功能与状态

[短合同](../seg_columns_c24_contracts_20261006/README.md) 已通过：实际模型权重六组逐位一致、独立编译加载和回退/释放门通过。本目录准备一次真实T1的前缀采集及 `down[0].conv1` 的 original/flipped 两pass ABBA。**本目录尚未上传、未解码MRI、未执行采集/实层；真实层精度、内存峰值和速度均待测。** 生产代码、旧C72和GPU未接入此原型。

流程：成熟 T1 preprocess → 原图/翻转图各 `conv0+ELU`，在 `conv1` 前停止并保存两个private输入 → A1原层保存两个参考 → B1/B2候选与A2原层分别逐位比较两个pass。一个capture及四个arm序列执行，首差停止。

## 2. Python、输入与输出

这是独立验证的冻结源码，运行开关另行授权。仅查看计划和已冻结参数的完整例子如下，不启动科学计算：

```python
import json
from pathlib import Path

c24_validation_directory = Path("validation/smri_cpu/seg_columns_c24_real_prepare_20261006")
c24_real_plan = json.loads((c24_validation_directory / "PLAN.json").read_text())
c24_layer_shape = c24_real_plan["layer_input_shape"]  # [1,24,192,224,256]
c24_original_input_sha256 = c24_real_plan["assets"]["raw_t1"]["sha256"]
c24_same_provider_sha256 = c24_real_plan["provider_sha256"]
c24_layer_weight_sha256 = c24_real_plan["layer"]["kernel_value_sha256"]
```

- `root`：现场FNIT固定入口，引用现有repo/环境/资产，不移动原prefix。
- `workspace`：本次新冻结源码目录，PLAN/worker/hash必须匹配。
- `run`：全新且不存在的私密运行目录；同名存在则拒绝，先核已运行状态。
- `approved-real-layer`：必须另获批准后传入的开关；本准备不隐含执行批准。
- 输入：CC0 OpenNeuro ds003138 v1.0.1 case02原T1，22,870,097B，SHA `73e3866d4e54f9cb253868daab4bf90303a97bc193e8bda21e2e60c53a5dea21`，模型H5 `f190bfd7…`。原图SHA仅现场重哈希，未解码。六项既有input/weight/labels/names/topology/parc资源保持源清单，未下载或发布。
- capture输出：`capture/original.private.npy`、`flipped.private.npy`，各[1,24,192,224,256]/连续float32，来自 mature `preprocess_t1(...cpu,min_pad=128)` 和 `F.elu(model.down[0].conv0(x))`。翻转输入严格 `torch.flip(x,(2,))`。无BN/pool/conv1/后续encoder/decoder/head/blur/标签。
- A1输出：两份private完整preELU参考、report；B1/B2/A2只保存标量report/log，不保存新大数组。每个报告含真实时钟、完整uint32位差、有限性/shape/stride/输入参数不变/资源/flags/source门。

模型加载沿 `SegmentUNet().load_h5(...).eval()`，同HDF5转置/复制。手工prefix序列按 `_Block.forward` 的第一句、`posterior` 输入/翻转、普通SynthSeg preprocess/oneDNN作用域绑定源码AST。不会安装Module hooks或调用 `model.forward`，因此不会误走整个CNN。

## 3. 命令及有限资源

**以下为准备好的命令，当前不要执行。** 上传冻结清单、六INDEX锁登记、根审查批准后才运行一次：

```bash
python run_real_abba.py   --root /cwStorage/home/gongwk/Notebook_code/FNIT   --workspace /cwStorage/home/gongwk/Notebook_code/FNIT/workspaces/smri_cpu_20261004/remaining_20261006/seg-columns-c24-real-layer-v1   --run /cwStorage/home/gongwk/Notebook_code/FNIT/runs/smri_cpu_20261004/remaining_20261006/seg-columns-c24-real-layer-v1   --approved-real-layer
```

CPU8同物理核 `[32,36,40,44,48,52,56,60]`，intra/inter8，环境线程8，CUDA不可见。共用锁最多120秒；capture180秒，每个arm180秒，外层1020秒，首失败停止/无重试/不放宽bit0。地址空间和最终实际`ru_maxrss`各≤32GB，二者分开记录；AS cap不是实测RSS。前置剩余磁盘≥6GB，再核新数组总字节≤4,227,860,000。每次只驻留一个pass input/output，比较和哈希使用≤8MiB片段；scratch仅单层调用，不缓存跨pass。

C24复用已通过的新binary `b8fd9829…`，无需编译或安装依赖。source17加三公共import源、原C72四项、当前Torch/provider/header、新C24prototype/DSO、短合同原receipt、input/assets均按大小/SHA核前后；全部private数组/参考文件/报告在各arm前后核同一身份。

## 4. 原软件与计算参考

原层是成熟 `CPUInferenceConv3d`/`convolution_slabs`，默认256MiB cap对应32平面、真实K648/N24/M1,835,008；候选保持同矩阵、copy顺序、bias预填与同一公共LP64 SGEMM。完整SynthSeg官方命令仅对应如下，本阶段不运行：

```bash
mri_synthseg --i input_T1w.nii.gz --o labels.nii.gz --vol volumes.csv --cpu --threads 8
```

无完整分割图/CSV，所以不核新脑区Dice或软体积。实际precision门FP32/noautocast、conv scope oneDNNFalse，CPU保持调用方CUDA TF32设置；恢复正常及异常后的原flags。不改变GPU默认、旧slab或原C72。

## 5. 尚未测的数值、时间与内存

本阶段所有真实结果pending。A1两真实reference生成，无比较；B1/B2/A2逐uint32位核所有两个pass输出，必须bit0/max0/finite/shape/stride等一致。每arm2次C24层，总8次，其中候选4次、24copy和24SGEMM；capture2次conv0，无model.forward/newcompile/native/GPU/fullCNN。

单层clock只包含层调用及candidate实际资格/计数开销；另列imports/construct/输入读取/哈希/比较与完整workerclock。ABBA中位数是同一病例层观察，不估计完整pipeline提升。C24 columns4.756GB、output scratch0.176GB、halo0.187GB、input/output各1.057GB合计约7.23GB，仅结构预算，真实RSS必须测。文件各data1,056,964,608B+header≤256，两个capture和两个A1共约4.23GB。既有C72整例CPU/GPU结果及脑图见 [已验收报告](../seg_columns_integration_20261006/README.md)，不会替换为本阶段未测数字。

## 6. 更新与后续门

- 当前：仅准备4个独立worker、源码AST/实际资产/合同DSO身份、有限PLAN与静态检查，0科学执行/上传。服务器只读核对source20、assets6、dependencies11及五项成熟AST全部exact，目标工作区/运行目录未存在，见 [LIVE_BINDINGS](LIVE_BINDINGS.json)。
- 短合同：一次编译及六case bit0；见 [原receipt报告](../seg_columns_c24_contracts_20261006/README.md)。短通过并不自动批准真实层。
- 原C72整例：冷进程观察约下降16.51%，同官方CPU速度门仍未过。其他不同slab/blur的否定结果保留，不复跑。
- 真实层通过后仍需另行审查窄生产接入与必要整例CPU/GPU输出门；未通过的候选不接入默认。未知shape或运行库回退成熟实现。

## 7. 来源、许可和文献

成熟MRI读写/网络/预处理复用FNIT自有PyTorch/nibabel代码，不调用FreeSurfer/FSL或封装包。新C++为FNIT自有glue，运行依赖现有Torch/MKL，不复制发布库、模型、MRI或上游实现体。CC0数据仅用于授权后有限真实测试。

- [SynthSeg](https://github.com/BBillot/SynthSeg)，Billot et al., *Medical Image Analysis* 2023，[doi:10.1016/j.media.2023.102789](https://doi.org/10.1016/j.media.2023.102789)。
- [PyTorch2.5.1 Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp)、[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)。
- [OpenNeuro ds003138 v1.0.1](https://openneuro.org/datasets/ds003138/versions/1.0.1) / CC0。
