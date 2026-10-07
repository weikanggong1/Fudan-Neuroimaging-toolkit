# FNIRT 同记录操作数 CPU 公式控制（2026-10-06）

本报告截至记录操作数控制；随后独立真实体积API已执行，结果见[CPU采样候选](../fnirt_cpu_sampler_case_20261006/README.md)。本报告原控制结果与调用计数保持。

## 1. 功能简介

在已有自然采样捕获的同一组 16,128 个操作数上，各执行一次 SDK plain、SDK partial 和 FNIT 源码参考公式。SDK 两条参考与原 delegate 返回值逐位一致；FNIT 源码参考存在下表所列差异。本轮没有修改产品或 GPU 路径。

这是已记录操作数的精度诊断。成熟 FNIT volume API、目标网格逐点对应、完整 MRI 配准和历史 2,425 个差异字的原因仍未完成验证。

## 2. Python 调用、输入与输出

本轮没有新增公共计算 API。输入为此前真实采样捕获的坐标字、3 个分数、8 个角点字及原 delegate 的返回字；plain/partial 的坐标和完整输入多重集已核验一致。角点来自自有快照，不能作为 DSO 内部加载轨迹。

输出为公共聚合指标 `manifest.public.json`。partial 的四列为 `value`、`gx_voxel`、`gy_voxel`、`gz_voxel`；导数单位为体素坐标导数，尚未除以体素尺寸。私密坐标、数组、源码及二进制没有随报告发布。

```python
import json
from pathlib import Path

# 读取本报告的聚合指标，不执行采样。
report_directory = Path("validation/fnirt_recorded_operand_reference_20261006")
manifest_path = report_directory / "manifest.public.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
plain_value_metrics = manifest["metrics"]["FNIT_raw_value_vs_actual_plain"]
print(plain_value_metrics["bit_mismatches"])
```

`count` 为比较字数，`bit_mismatches` 为 float32 字级差异数，`maxabs` 为最大绝对误差，`difference_L2`/`relative_L2` 为误差 L2 及其相对值。`derivedvalid_masked` 将双方都乘以捕获记录中重新推导的有效性；它没有证明原 caller datamask 的逐目标对应，也没有执行 FNIT 自身有效性分支。

## 3. 命令行调用

在本报告目录读取结果：

```bash
python -c 'import json; from pathlib import Path; print(json.loads(Path("manifest.public.json").read_text())["status"])'
```

私密 worker 的一次执行已结束，三条公式各一遍；成熟 API、Numba JIT、native、编译、solver 和 GPU 调用均为 0。

## 4. 原软件对应与字节来源

plain/partial 对应原实现的三线性采样及其体素导数。本轮没有重跑完整 `fnirt` 配准命令，原 delegate 返回值沿用[自然采样捕获](../fnirt_natural_sampler_capture_20261006/README.md)。参考公式显式保留 float32 舍入；partial 中的 double 字面量提升与 FNIT 全 float32 表达式分别处理。SDK 源码身份和实际 delegate DSO 身份分列于清单；本轮匹配不能反推二者的构建来源相同。

另一次纯 4 字节字置换回执表明，官方 saved moving 经 X 方向倒序后，raw SHA 为 `4a776c63665e5a792eca9029b09626f97a692ef7668967238bb5fa62a6dd0ae0`，与自然捕获 moving 的端点 SHA 相同。旧 FNIT 外部 moving 的 raw SHA 为 `f4561df8400924435b899dccbff04ded239f2155aa6f3807db482736b80a0793`，与该端点不同。置换没有解码浮点载荷或执行浮点运算；这项字节关系尚未证明相同几何、坐标或目标网格的一一对应。

## 5. 实际精度、耗时与可视化

每项比较含 16,128 个 float32 字。SDK plain 的 value，以及 SDK partial 的 value/gx/gy/gz，在 raw 和 derivedvalid-masked 两种统计下均为 **0 个差异字、误差 L2=0**。

| FNIT 源码参考比较 | 差异字数 | 最大绝对误差 | 相对 L2 |
| --- | ---: | ---: | ---: |
| value 对原 plain | 2,338 | 1.52588e-5 | 6.07554e-8 |
| value 对原 partial | 813 | 1.52588e-5 | 3.76875e-8 |
| gx 对原 partial | 0 | 0 | 0 |
| gy 对原 partial | 913 | 9.53674e-7 | 3.53504e-8 |
| gz 对原 partial | 1,520 | 1.52588e-5 | 7.62207e-7 |

FNIT 参考对 partial 的 raw 与 derivedvalid-masked 指标相同；20 项统计均有限，signed-zero 差异均为 0。这里控制的是已保存角点与分数上的公式行为，尚未运行成熟采样 API，也不能把 2,338 直接对应到历史 2,425 个差异字。

| 本轮诊断计时/资源 | 实际值 |
| --- | ---: |
| 三公式循环 | 0.378596 s |
| worker 总时长 | 0.575370 s |
| supervisor 总时长 | 0.695479 s |
| outer 总时长 | 0.767069 s |
| 自有进程树峰值 RSS | 83,066,880 B |

这些是带观察和文件门控的诊断时长，不构成完整 MRI benchmark 或加速比。原始 v1/v2 文本各 13 份均通过大小、SHA 和压缩文本逐字节核验；本轮源码及输入前后相同，进程退出、共用锁释放。

本阶段脑图输出为 **0**；没有可发布的脑图，不能据聚合数字构造脑图例子。

## 6. 最近更新与 benchmark 记录

- 自然采样捕获：记录原 plain/partial 调用和相同输入多重集，见[捕获报告](../fnirt_natural_sampler_capture_20261006/README.md)。
- v1：旧外部 moving 字节不匹配，随后因 `ModuleNotFoundError: formulas` 退出；三条公式执行次数均为 0。原失败及回执完整保留。
- moving 来源核对：纯字置换证明官方 saved moving 的 X 倒序字节与捕获端点一致；没有补写 v1 的输入匹配结论。
- v2：使用显式文件模块加载，公式源码与 v1 逐字节相同；删除 moving/API/JIT 执行，只完成本报告的 recorded-operand 控制。
- 下一阶段的隔离 CPU 候选与成熟 API 控制尚未执行，本报告没有将其列为通过。

[此前 saved-warped 字节分类](../fnirt_saved_warped_reference_20261006/README.md)和[缓存前缀结果](../fnirt_native_cache_prefix_result_20261006/README.md)具有各自范围。本轮产品改动为 0，GPU 路径改动为 0。

## 7. 参考文献与原实现

- [FSL FNIRT 原代码库](https://git.fmrib.ox.ac.uk/fsl/fnirt)。
- Andersson, Jenkinson & Smith. *Non-linear registration, aka spatial normalisation*. FMRIB Technical Report TR07JA2 (2007)，[原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。
