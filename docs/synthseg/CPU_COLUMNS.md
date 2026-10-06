# SynthSeg 可选 CPU 列缓冲复用（候选，2026-10-06）

## 1. 功能简介

该候选在 SynthSeg 33 类网络最后一级 `SegmentUNet.up[3].conv0` 复用一次调用内的列缓冲。输入仍按原来的14层slab展开，矩阵尺寸、通道和卷积核顺序、FP32、bias预填以及当前Torch已加载的LP64 SGEMM保持。缓冲在该层返回后释放，不缓存大张量、不改变全局allocator。

只有已验收权重、`[1,72,192,224,256]` 输入、72→24通道、3×3×3卷积、8个CPU线程、eval/no-grad、oneDNN关闭且无autocast/hooks/forward-AD时启用。未知shape、权重、运行库或编译环境继续使用成熟CPU分块卷积。Tensor子类、懒negative/conjugate视图也继续原路径。CUDA、训练和通用slab代码不使用这个候选。

**当前仅完成私有同层实测及候选本地守卫/缓存合同。新生产编译缓存、完整CPU/GPU和安装门待审查后执行；尚未接入main。** 该候选不是新的分割模型，也不改变公共API。

## 2. Python调用、输入与输出

```python
from pathlib import Path
import nibabel as nib
from fnit import SynthSeg, SynthSegPlus

input_t1_path = Path('/data/example/T1w.nii.gz')  # 单幅3-D T1；保留原affine/header。
external_weights_directory = Path('/data/fnit-weights')  # 已校验的H5和标签数组。
output_segmentation_path = Path('/data/output/segmentation.nii.gz')
output_volumes_path = Path('/data/output/volumes.csv')

segmenter = SynthSeg(
    weights=external_weights_directory,  # 默认None使用FNIT已声明资源目录。
    device='cpu',                       # CUDA仍使用既有PyTorch路径。
    threads=8,                          # 本候选仅接受8；其他预算继续原CPU路径。
    cudnn_tf32=True,                     # 原默认不变；只作用于CUDA前向。
)
segmentation_result = segmenter(
    input_t1_path,
    keep_geometry=False,                # False输出约1-mm RAS网格；True恢复输入网格。
    color_lut=None,                      # 可传已有色表路径，不改变标签数值。
)
nib.save(segmentation_result.segmentation, output_segmentation_path)
segmentation_result.write_volumes_csv(input_t1_path, output_volumes_path)

# 公共--parc接口示例。默认CPU oneDNN开启时保持原路径，不强行切换后端。
parcellator = SynthSegPlus(
    weights=external_weights_directory,
    parc_weights=external_weights_directory,
    device='cpu',
    cudnn_tf32=True,
)
parcel_result = parcellator(
    input_t1_path,
    keep_geometry=True,  # Plus默认输出原T1网格。
    fast=False,          # True省去左右翻转集成，保留既有fast后处理。
    min_pad=128,         # 预处理最小padding；实际网格还需满足网络32倍数。
    volumes=True,        # 返回软体积，允许write_volumes_csv。
)
```

输入仍是NIfTI路径或普通SynthSeg接受的nibabel影像；模型通过外置原始H5和NPY加载。输出仍是整数分割影像、按标签的mm³软体积、总颅内容积、标签名及真实前向precision记录。Plus另返回皮层脑区和合并图，字段和顺序不变。完整参数与输出表见 [SynthSeg](README.md)、[Plus](../synthseg_plus/README.md) 和 [已有精度策略](../../validation/smri_cpu/seg_tf32_20261005/README.md)。本候选没有新的用户算法参数。

普通33类 `SynthSeg.__call__` 的CPU调用关闭oneDNN，同形状原图和翻转两pass均可资格检查。`SynthSegPlus` / `run_synthseg_parc_t1` 共用相同33类网络：调用方oneDNN=False时，普通模式两pass、fast一pass可检查；当前公开CPU默认oneDNN=True时继续成熟路径。皮层 `ParcUNet` 没有标记，不启用候选。未知权重即使网络结构相同也不启用。

## 3. 命令行及Conda缓存

```bash
# 公共命令保持不变，候选无需新开关。
fnit-synthseg --i /data/example/T1w.nii.gz \
  --o /data/output/segmentation.nii.gz --vol /data/output/volumes.csv \
  --weights /data/fnit-weights --device cpu --threads 8
fnit-synthseg --i /data/example/T1w.nii.gz \
  --o /data/output/combined.nii.gz --parc --fast \
  --weights /data/fnit-weights --device cpu --threads 8

# 主页环境已包含GCC/G++11，不增加依赖；从仓库根创建环境。
conda env create -f environment.yml
conda activate fnit
fnit-setup-weights --model synthseg --dest /data/fnit-weights
fnit-setup-weights --model synthseg --dest /data/fnit-weights --verify-only
```

仅首次合资格CPU层调用才查已有Torch头文件/库、provider和GCC11，编译FNIT自有4,385B胶水。可用环境变量：`CXX`指定一个GCC11编译器可执行文件，不能附参数；`FNIT_SYNTHSEG_CPU_CACHE`指定本人拥有的0700缓存目录；未指定时使用`$XDG_CACHE_HOME/fnit/synthseg_columns`，或`~/.cache/fnit/synthseg_columns`。锁和产物为0600，按源码、运行库、头文件、provider、编译器、ABI和flags生成键，原子发布。无编译器或不匹配库时数学开始前回退原卷积；开始候选数值后异常传播，不暗中重复卷积。

CPU目标是Linux x86_64、Torch2.5.1/ABI0、已验收运行库SHA及MKL LP64 provider；其他平台/版本先回退。动态库不打包，weight和MRI也不进入wheel。编译子进程最多120秒、退出收尾5秒，锁等待15秒；仅编译子进程CUDA不可见，不写调用方环境、线程、TF32或autocast。已运行合同只mock了新缓存的编译/加载，**全新同prefix Conda安装与此新构建实际编译尚未测试**。

## 4. 原软件调用与原步骤

```bash
# 仅独立官方benchmark环境运行；FNIT生产不会调用该命令。
mri_synthseg --i /data/example/T1w.nii.gz \
  --o /data/reference/segmentation.nii.gz --vol /data/reference/volumes.csv \
  --cpu --threads 8
```

原软件的这一个内部卷积层没有独立CLI。候选保持原成熟CPU路径的14-plane slab，K=1944、N=24、M=802816（最后10层M=573440），NN、lda/ldc=实际紧凑M、ldb=1944、alpha=beta=1。自己的copy胶水调用的是已经加载的公共SGEMM，不是隐藏的ATen CPUBlas/Unfold3d wrapper；不加载其他BLAS。

## 5. 最新精度、时间及内存

真实输入为CC0 OpenNeuro ds003138 v1.0.1 case02已保存的自产decoder skip/value，仅恢复这一个层的输入；没有完整CNN或官方新运行。[单层ABBA报告](../../validation/smri_cpu/seg_columns_real_layer_v2_20261006/README.md) 保留原输入/权重/运行库SHA、四个新进程和原退出receipt。

| 单层臂 | 操作秒 | 实际完整pre-ELU比较 | RSS GB |
|---|---:|---|---:|
| A1原路径 | 15.383990 | 唯一参考生成；不算比较门 | 11.3949 |
| B1候选 | 5.728821 | 264,241,152个FP32值逐位差0 | 11.3175 |
| B2候选 | 5.711896 | 同A1逐位差0 | 11.3188 |
| A2原路径 | 15.387470 | 同A1逐位差0 | 11.3970 |

该操作时钟含私有候选守卫/provider/counter，不含load/join/hash/IO；两臂中位数15.385730→5.720358秒只描述本层。新生产逐次参数SHA、缓存校验/首次编译成本尚未在整例计时。6.243GB列缓冲仍然存在，生命周期仅单次层调用；RSS基本相同。缺页和system时间下降与缓冲复用相符，没有allocator trace，不能归因于唯一机制。

候选本地39个守卫/缓存/异常合同通过，来源、精度状态、CUDA未初始化前后门通过；正例资格的实际权重SHA仅在该本地测试中mock，真实单层权重另有上述既存合同。新缓存没有真实编译/copy/SGEMM调用，不能标为完整验收。

已完成旧完整版本CPU耗时普通33约112.95秒、parc55.68秒、fast约42–43秒；同节点/同8核官方正常冷进程55.05/48.30/33.78秒。API与官方CLI的输出/时钟边界分别见 [正式记录](../../validation/smri_cpu/seg_memory_20261005/README.md)。**完整官方CPU速度门仍未通过，不能以本层比推算整例加速。** 本轮不重新绘制单层脑图；既有完整脑图版本见 [逐标签图](../../validation/smri_cpu/seg_memory_20261005/case02_cpu_labels.png)，不能标为新候选完整结果。

## 6. 版本与待验收

- 4385B自有胶水经 [编译加载v1](../../validation/smri_cpu/seg_columns_reuse_20261006/README.md)、[短合同v2](../../validation/smri_cpu/seg_columns_reuse_v2_20261006/README.md) 和私有真实同层ABBA验收。原库不发布。
- v1真实层因纯哈希rank守卫错误在卷积前退出，记录 [原失败](../../validation/smri_cpu/seg_columns_real_layer_20261006/README.md)；独立v2仅修哈希并通过前置合同后运行，原失败不删除。
- 本次新增lazy CPU构建/缓存及严格资格守卫，标记仅落在33类末级层。39本地合同通过；38合同旧receipt保留，随后只补懒negative/conjugate视图守卫。
- [接入计划与状态](../../validation/smri_cpu/seg_columns_integration_20261006/README.md)：新库metadata/真实短合同→CPU33同原T1完整ABBA→parc/fast及GPU原路保护。每阶段先审查，首差停止；不自动扩大shape/layer、重跑官方或放宽逐位门。

## 7. 参考、源码与许可

- [SynthSeg官方代码](https://github.com/BBillot/SynthSeg)；Billot et al., *Medical Image Analysis* (2023), [doi:10.1016/j.media.2023.102789](https://doi.org/10.1016/j.media.2023.102789)。权重继续按FreeSurfer资源条款外置。
- [PyTorch2.5.1 Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp) 与 [CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)。本轮只阅读接口与矩阵定义；自有copy没有复制其实现体。
- [PyTorch BSD许可](https://github.com/pytorch/pytorch/blob/v2.5.1/LICENSE)；MKL及GCC/libgomp沿用用户已有Conda运行库的许可证。FNIT不再分发这些库，详见 [本胶水归属](../../src/fnit/synthseg_parc/CPU_COLUMNS_NOTICE.md) 和 [既有第三方说明](../../THIRD_PARTY_NOTICES.md)。
