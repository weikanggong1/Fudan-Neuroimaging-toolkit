# FNIRT saved-warped 字节分类控制（2026-10-06）

## 1. 功能简介

本控制检查已保存的 warped 数据在改为 X-fast 存储后，是否与既有原缓存 pre 或 post 的 SHA-256 一致。实际控制已完成，分类为 `neither`：两者都不同。

流程：已保存的 XYZ C 顺序字节 → 原样拷贝每个 4-byte 词为 X-fast → 计算 SHA-256 → 对照已有 pre/post 摘要。仅整理存储字节，没有 Float 解码或浮点数学运算。

## 2. Python 调用与输入输出

输入的数据描述为 `<f4`、XYZ 轴、C 顺序；输入比较依据为已有 pre/post SHA。每个位置的原始 4-byte 词完整拷贝，不改变 XYZ 轴或方向。源字节偏移为 `4 * ((x_index * number_y_voxels + y_index) * number_z_voxels + z_index)`；目标偏移为 `4 * ((z_index * number_y_voxels + y_index) * number_x_voxels + x_index)`。三个 `number_*_voxels` 表示对应轴的长度。

输出是 `manifest.public.json`，包含三个摘要、分类、执行计数、源码摘要和诊断资源记录；不包含输入数据或图像。本次没有新增 FNIT Python 接口。读取公共结果的示例：

```python
import json
from pathlib import Path

# validation_directory：公共报告目录，不是数据目录。
validation_directory = Path("validation/fnirt_saved_warped_reference_20261006")
# validation_manifest：读取本次控制的结构化结果。
validation_manifest = json.loads(
    (validation_directory / "manifest.public.json").read_text(encoding="utf-8")
)
print(validation_manifest["comparison"]["phase_classification"])  # neither
```

## 3. 命令行调用

本次没有新增公共诊断命令。以下命令只读取并格式化结果，其中参数是公共 manifest 的路径：

```bash
python -m json.tool validation/fnirt_saved_warped_reference_20261006/manifest.public.json
```

## 4. 原软件对应

这项字节分类没有等价的 FNIRT 配准命令。pre/post 摘要来自已有原缓存观察记录；本次原软件调用为 0，没有重跑配准或采样。原实现见 [FSL FNIRT 代码库](https://git.fmrib.ox.ac.uk/fsl/fnirt)。

## 5. 实际结果、精度与耗时

| 字节对象 | SHA-256 |
| --- | --- |
| saved-warped，X-fast | `c5f722324dd63a25fd9e434f4b535fe72092f2d414615884d10e9b0708074a27` |
| 已有原缓存 pre | `939fe0e67b29ac11e1bef3e8fec88e742b0387b7c8e9441cb94fcdf2cea2d4ed` |
| 已有原缓存 post | `7df95160f76a48bb2cff09c3da8b651b0d769d81cc641d658d2d6b6a35f198a4` |

saved-warped 重排后为 64,512 bytes，其 SHA 与 pre、post 均不相等。控制只比较摘要，没有重新计算误差指标。FNIT 与原采样的 moving、coordinates 尚未证明相同，因此 `neither` 不能确定公式错误的原因，也不能证明精度等价。

worker 耗时为 **0.017085092 s**，峰值 RSS 为 **17,293,312 bytes**。这是字节整理和 SHA 分类的诊断记录；没有本轮端到端或分步骤配准性能比较，不能作为 speedup。

本次 sampler、native、solver、GPU 调用均为 0；生产运行时代码和 GPU 代码修改为 0。图像输出为 **0**，没有脑图，不能用字节摘要构造脑图示例。

## 6. 更新与 benchmark 记录

- 既有 pre/post 缓存观察及 post 比较记录保持原样，本次没有重算 post 指标。
- 2026-10-06：完成一次 saved-warped 字节分类，worker exit0，无自动重试；10 份原始文本的大小和 SHA 均核验一致，运行前后源码 SHA 相同，本地冻结 worker 与实际执行源码匹配。
- 本轮新增公共报告，精度与性能结论仍限于上述字节诊断。

## 7. 参考文献与原实现

- [FSL FNIRT 原实现代码库](https://git.fmrib.ox.ac.uk/fsl/fnirt)。
- Andersson, Jenkinson & Smith. *Non-linear registration, aka spatial normalisation*. FMRIB Technical Report TR07JA2 (2007), [原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。
