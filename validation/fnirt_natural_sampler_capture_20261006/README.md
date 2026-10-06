# FNIRT 自然采样 capture（2026-10-06）

## 1. 功能简介

本轮在原 FNIRT 自然缓存前缀中记录 plain 与三个 partial 的实际输入快照、原返回值和执行阶段。两次自有源码编译、一次 native capture、一次整数/字节校验均已成功。生产代码修改为 0。

流程：原自然调用 → 自有记录 shim → 原 DSO delegate 一次 → 保存原返回位模式 → 整数/字节门核验。报告截至 capture；后续同输入数学控制的结果不在此报告中。

## 2. Python 调用与输入输出

输入是冻结源码、既有原对象、既有前缀输入和原输出摘要；实际 operands 保持私密。记录按阶段、坐标位模式和完整输入快照的多重集关联，线程顺序不作为目标体素顺序。每条记录包含 floor、fraction、8-corner 快照、raw return、三个导数及额外的 valid 诊断。

输出为本目录的公共 manifest，仅列计数、门状态和允许公开的源码/二进制摘要；不含坐标、数据、SDK 代码或二进制文件。读取结果示例：

```python
import json
from pathlib import Path

# validation_directory：公共报告目录。
validation_directory = Path("validation/fnirt_natural_sampler_capture_20261006")
# validation_manifest：本次已完成 capture 的结构化摘要。
validation_manifest = json.loads(
    (validation_directory / "manifest.public.json").read_text(encoding="utf-8")
)
print(validation_manifest["execution"]["native_exit_code"])  # 0
```

## 3. 命令行调用

本次没有新增生产 CLI。以下命令仅格式化公共结果；参数为 manifest 路径：

```bash
python -m json.tool validation/fnirt_natural_sampler_capture_20261006/manifest.public.json
```

## 4. 原软件对应

观察点为原 `Robj` 的 plain 调用、grad 阶段已缓存的 `Robj`，以及原 `RobjDeriv` 的三个 partial 调用。每次被记录的调用只 delegate 原方法一次，没有额外原采样调用或缓存 getter。实际 `RTLD_NEXT` delegate 已核验为原 DSO，SHA-256 为 `ea2f45b4bff49dea10f073ba6dff7710b3c9027c5c7b16d3d805fcb36a463e9f`。

本轮未执行完整 FNIRT 配准命令或 SSD/gradient/solver。原缓存前缀范围见[已有公共结果](../fnirt_native_cache_prefix_result_20261006/README.md)，原实现见 [FSL FNIRT](https://git.fmrib.ox.ac.uk/fsl/fnirt)。SDK 源码语义与实际 DSO 的数学构建来源仍分别记录。

## 5. 实际结果、精度与耗时

| 阶段 | 自然调用 | 记录数 | 唯一坐标键 | 重复键 |
| --- | --- | ---: | ---: | ---: |
| 1 | cf 的 plain | 16,128 | 16,128 | 0 |
| 2 | grad 的 cached Robj | 0 | 0 | 0 |
| 3 | grad 的三个 partial | 16,128 | 16,128 | 0 |

plain 与 partial 的实际坐标位模式、完整输入快照多重集及 volume 对象一致；原 raw return 多重集与各自自然保存输出一致。原自然 outputs/counters 与既有前缀完全相同，丢失或溢出记录为 0。moving 共 **18,579,456 voxels**，四次端点字节摘要一致。

编译时两个产物分别记录 **965／989** 个实际依赖 header、**19／22** 个静态依赖 DSO；这些是编译元数据。实际运行时为 **23 个 DSO**，全部与绑定身份一致。

| 诊断范围 | 耗时 |
| --- | ---: |
| 两次编译的 worker | 21.483 s |
| instrumented native capture | 41.56213565 s |
| capture controller | 54.510 s |
| outer 控制 | 54.678 s |

owned tree 峰值 RSS 为 **496,029,696 bytes**。这些时间包含观察开销，只是前缀诊断；完整 MRI benchmark 为 0，不能作为加速结果。

corners 是自有快照，不能称原 DSO 内部 load trace；moving 只验证端点字节。derived valid 是额外诊断，与自然 datamask 分列。FNIT 输入对应、逐目标体素对应及公式差异原因尚未证明。整数/字节校验器没有 Float 解码或候选公式求值；本轮新增 SSD/gradient/solver、GPU 操作为 0。图像输出为 **0**，不能构造脑图。

## 6. 更新与 benchmark 记录

- 既有原缓存前缀输出、计数及源码保持原样。
- 2026-10-06：完成 2 次自有编译、1 次自然 capture 和 1 次整数/字节校验，无重试；14 份编译文本、15 份 capture 文本逐项 SHA 核验，另核实际 alias ELF 收据。源码冻结与执行前后绑定一致，所有受控进程已退出、资源释放门通过。
- 本阶段产品修改为 0；后续同输入数学控制继续单独记录，当前没有公式修复或精度通过结论。

## 7. 参考文献与原实现

- [FSL FNIRT 原代码库](https://git.fmrib.ox.ac.uk/fsl/fnirt)。
- Andersson, Jenkinson & Smith. *Non-linear registration, aka spatial normalisation*. FMRIB Technical Report TR07JA2 (2007), [原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。
