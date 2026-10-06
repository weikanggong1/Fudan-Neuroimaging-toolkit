# FNIRT CPU 同目标坐标与 RHS 隔离控制（affine v3）

## 1. 功能与流程

本轮把已有真实检查点的公共 NIfTI 坐标帧转换为 NEWIMAGE 内部源体素帧，证明采样坐标与实际原 delegate 记录对应，再比较当前及候选 CPU 采样对梯度前缀的影响。全部输入门通过；候选采样值及三轴导数逐位一致，RHS 仍有尾差。候选尚未接入默认实现，生产代码、测试及 GPU 路径未改。

```mermaid
flowchart LR
    A[保存检查点与固定源码] --> B[3×4 affine 源定义转换]
    B --> C[先保存完整16 word诊断]
    C --> D[消费12 word逐位门]
    D --> E[一次成熟坐标构建]
    E --> F[XX与目标顺序及角点逐位门]
    F --> G[当前与候选各一次采样]
    G --> H[固定lambda下两种梯度前缀]
```

原第4行的字节保留且不参与坐标计算。转换依据原函数消费的前三行 affine 和隐式 `w=1`，没有反射已舍入坐标，也没有按采样输出寻找匹配。本轮是新的受控诊断；源码几何规则与安装 DSO 的内部矩阵读取 trace 分列。

## 2. Python 调用、输入与输出

这是验证报告，没有新增公共采样 API。以下示例只读取去敏结果：

```python
import json
from pathlib import Path

# 在 FNIT 仓库根目录运行；不读取图像或重跑诊断。
report_path = Path(
    "validation/fnirt_cpu_sampler_case_20261006/affine_coordinate_v3/manifest.public.json"
)
report = json.loads(report_path.read_text(encoding="utf-8"))
print(report["matrix_word_summary"]["consumed_bit_mismatches"])
print(report["metrics"]["candidate_FSLorder_vs_native_positive_g"]["full"])
```

| 输入或控制 | 格式及作用 |
| --- | --- |
| 已接受检查点 | 68个数组成员；坐标阶段只恢复6个，前门通过后总计恢复32个 |
| 保存变形场、affine grid及源矩阵 | 按原 dtype/shape/stride 恢复，复用成熟坐标构建函数，不重算 grid 或 spline |
| 实际采样记录 | 两个自然阶段各16,128个唯一 XX；原 raw value/voxel derivative、floor、fraction、角点及 valid 分列 |
| 已保存官方 moving | 仅隔离对照；按原始 word 恢复内部存储方向，先与实际 moving 身份相同；不用于生产读取 |
| 固定及 scaled fixed | fixed 的原 rawRef 身份与保存参数末8字节相同后，一次保存状态乘法；不重算 scale reduction |
| gradient/RHS | 3×392个系数和1个全局 scale；正梯度 `g=2×FNIT half-gradient`，负 RHS 为 `-g` |
| lambda | 复用原保存的 `9049.463427795125`，不从本轮新 residual/SSD 重算 |
| 执行边界 | CPU8；AS8GB/RSS4GB；锁等待60s、子进程120s、工作180s、OS240s及清理10s；唯一attempt、retry0 |

私密输出保存坐标、两组 warped/gradient/residual、两种正梯度及负 RHS、完整16 word矩阵诊断和监督回执。公开文件只含统计及源码/回执 SHA，不含坐标、矩阵原word、被试路径或数组。

## 3. 命令行调用

在仓库根目录查看结果：

```bash
python -m json.tool validation/fnirt_cpu_sampler_case_20261006/affine_coordinate_v3/manifest.public.json
```

本轮没有新增生产 CLI，也没有重新调用完整配准入口。

## 4. 原软件对应

对应 NEWIMAGE 内部坐标构造、`interpolate` / `interp3partial` 和 FNIRT 梯度前缀。复用[实际自然捕获](../../fnirt_natural_sampler_capture_20261006/README.md)的原 delegate 结果，本轮新原软件调用为0。原完整入口示例如下，本报告不执行该命令：

```bash
# moving_volume/reference_volume/affine_matrix/config_file 是使用者自己的输入路径。
fnirt --in="$moving_volume" --ref="$reference_volume" \
  --aff="$affine_matrix" --config="$config_file" --iout="$registered_volume"
```

当前与候选共享本轮内部源坐标帧及投影矩阵，采样算术差异在这个匹配状态上隔离。与旧公共 NIfTI 源矩阵相比，本轮私密投影矩阵也转换至内部源帧；这项诊断转换没有修改生产 geometry。官方体积仅用于验证，未来实现必须从原始输入自行构造一致的内部帧、mask 和 geometry。

## 5. 真实精度、耗时与可视化

### 输入及矩阵门

| 门 | 实际结果 |
| --- | --- |
| 完整16 words诊断 | 原矩阵、转换矩阵、header参考全部保存 |
| 实际消费3×4矩阵 | 12/12 words逐位一致，包含符号零 |
| 全4×4诊断 | 1个word不同，位于未消费第4行；原第4行字节保留，不声称全矩阵逐位一致 |
| 重建坐标与实际XX | 两侧各16,128个唯一键，交集16,128，缺失/新增0 |
| 目标顺序 | 按 logical XYZ 坐标键关联后，原 plain / partial / natural mask 全部逐位一致 |
| 固定与 moving身份 | 固定 rawRef 及 moving 内部 word 身份门通过 |
| floor/fraction/valid/有效角点 | 129,024个角点逐位一致；13,652个出界角点及 padding 符号零一致 |

目标关联同时由坐标位键、原网格顺序输出和固定几何门约束，没有使用并行调用序号当作 target index。它证明本轮新控制的目标对应；不把新的输入记录倒写成历史运行的内部 trace。

### 当前与候选采样

| 与本轮实际原 partial 输出比较 | 当前不同words | 候选不同words |
| --- | ---: | ---: |
| value | 813 | 0 |
| gx | 0 | 0 |
| gy | 913 | 0 |
| gz | 1,520 | 0 |

raw 与 valid-masked 两套统计均保存，表内两种统计的不同word数相同；候选均逐位一致，mask亦相同。本轮只各调用一次 `derivatives=True`；plain 的目标顺序门使用已保存原结果。

### 同状态梯度前缀

| 对保存原正梯度比较 | 不同words | maxabs | relative L2 |
| --- | ---: | ---: | ---: |
| current，LM序 | 1,177 | 4.3575014e−6 | 4.1715173e−7 |
| candidate，LM序 | 1,177 | 3.7817380e−7 | 3.6229266e−8 |
| current，FSL累计序 | 1,177 | 4.3454812e−6 | 4.1597449e−7 |
| candidate，FSL累计序 | 1,177 | 2.2594031e−9 | 8.7144863e−10 |

候选减少了两种累计顺序的误差，但1177个Float64 words仍非逐位相同。FSL累计序候选的全局 scale maxabs 为5.5067062e−14；三个系数块的relative L2分别为9.0680748e−8、3.2102025e−8、3.1622046e−8。完整逐块/raw/masked/符号零统计见manifest。

与旧保存状态相比，current/candidate residual分别有2,064/2,396个不同words，投影梯度有9,319/10,057个不同words。这些旧状态比较保留；本轮同输入改善没有证明历史2,425word差异完全由采样算术造成。官方 `cf` plain到 `grad` partial可能改变缓存的 warped 状态，后续 dynamic lambda/cost 必须核原缓存语义，不能用本轮新 SSD 替代。

| 本次时钟范围 | 秒 |
| --- | ---: |
| 一次成熟坐标构建 | 0.000373 |
| current 首次 True API | 1.685188 |
| candidate 首次 True API | 1.514359 |
| worker / supervisor / outer | 6.413145 / 6.880635 / 6.974342 |

两个API首调用可能包含JIT；导入/JIT没有独立细分计时，没有额外warm调用或ABBA。本轮不提供完整速度收益。owned tree峰值RSS614,522,880B；Torch intra8、interop96及TF32 flags前后保持原值，CUDA未初始化，没有加载原软件DSO。

本轮复用已有真实保存状态，没有新完整配准或脑图；[原完整CPU benchmark及脑图](../../../docs/fnirt/README.md#5-最新精度和运行时间)仍绑定原版本。16份原始文本逐项size/SHA核验，源码/输入/保存及派生状态/输出字节闭合，树退出、CPU8锁释放，六索引锁关闭仅本任务状态且保留所有peer。

## 6. 更新及 benchmark 记录

- 原 coordinate v1复现旧坐标SHA后与实际XX交集0，未执行sampler/RHS；原部分JSON失败保留，补录只读已保存整数位模式。
- frame v2在全4×4矩阵门停止，coordinates/API/RHS均0；原FAIL保留。此前两个键路径问题在部署前修复，属于未执行准备包。
- 2026-10-06 affine v3：依据实际3×4消费者，先保存16word诊断，再执行严格12word门。一次坐标与全部输入门通过，当前/候选各一次采样及两种RHS累计完成；native/blur/cost/H/diag/PCG/full/GPU均0。
- 下一步先读同状态H/diag和动态lambda/cost来源，再批准限定控制；完整原始输入CPU配准、Tensor支持域与fallback兼容、GPU路径保护仍待验收。默认实现保持原版本。

## 7. 原实现、许可与参考文献

- [FSL NEWIMAGE源码](https://git.fmrib.ox.ac.uk/fsl/newimage)、[FSL FNIRT源码](https://git.fmrib.ox.ac.uk/fsl/fnirt)。原SDK文本仅私密审查，本报告不发布原代码；FNIT改写遵循[FSL 6.0许可](../../../licenses/FSL-6.0.txt)。
- Andersson, Jenkinson & Smith. *Non-linear registration, aka spatial normalisation*. FMRIB Technical Report TR07JA2 (2007), [原文](https://www.fmrib.ox.ac.uk/datasets/techrep/tr07ja2/tr07ja2.pdf)。
