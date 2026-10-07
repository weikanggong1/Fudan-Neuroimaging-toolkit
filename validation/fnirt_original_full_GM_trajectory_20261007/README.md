# 原版 FNIRT 完整 GM 自然参数轨迹（2026-10-07）

## 1. 功能简介

在原版 FSL 6.0.7.4 FNIRT 的四级 GM 配准中，被动记录自然成本、梯度、Hessian 调用及参数边界。每次代理只转发一次，不刷新成本缓存，不增加配准或 PCG 求解。原版运行返回0；保存的 coefficient、dense field 和 nonlinear Jacobian 三个完整文件均与既有官方结果相同，轨迹可用于后续完整链首差定位。

原控制器曾把 `NIter` 当作已接受步数而误报；原失败记录保留，另一个只读保存资格组通过。本报告不改变 FNIT 默认实现，也不验收此前完整 CPU 候选的精度。

```mermaid
flowchart LR
  A[原输入与原 GM 配置] --> B[一次原版四级 FNIRT]
  B --> C[自然调用与私密参数快照]
  B --> D[三项保存输出]
  C --> E[独立只读保存资格]
  D --> E
```

## 2. Python 调用、输入与输出

本目录公开安全聚合结果，不分发原软件、SDK 或私密参数。读取示例：

```python
import json
from pathlib import Path

# summary_path：本目录公开的聚合记录；不含影像或参数数组。
summary_path = Path(
    "validation/fnirt_original_full_GM_trajectory_20261007/summary.json"
)
with summary_path.open(encoding="utf-8") as summary_stream:
    trajectory_summary = json.load(summary_stream)
# qualification_passed：保存轨迹及三项完整文件的独立资格结果。
qualification_passed = trajectory_summary["separate_saved_metadata_qualification"]["qualified"]
# level_records：每级参数维数、原 NIter、存储成本和最终阻尼。
level_records = trajectory_summary["levels"]
print(qualification_passed, level_records)
```

| 输入或参数 | 含义与格式 |
| --- | --- |
| moving GM | 三维 Float32 NIfTI 灰质概率图，来自既有官方处理链。 |
| reference GM | 三维 NIfTI 灰质模板，使用原空间与头信息。 |
| initial affine | 原 FLIRT 格式4×4文本矩阵，保持原版坐标约定。 |
| reference mask | 原三维参考空间 mask；配置仍只在末级应用。 |
| GM config | 原 `GM_2_MNI152GM_2mm`，最大迭代5/5/10/5，前三层估计全局尺度、末层固定尺度。 |
| output destinations | 新独立 coefficient、dense field、nonlinear Jacobian 文件；不覆盖历史结果。 |

公开输出为 [summary.json](summary.json)。29份原 Double 参数快照和222条自然事件留在私密运行目录；第1至4级参数维数分别为1,177、6,049、31,753、31,752。末级固定 scale 不属于参数向量。

## 3. 命令行调用

公开聚合记录可直接检查：

```bash
python -m json.tool validation/fnirt_original_full_GM_trajectory_20261007/summary.json
```

观察器没有产品 CLI；本轮未改生产入口，也没有新增依赖。

## 4. 原软件调用

下面是本轮原输入配置的对应命令结构，变量须由使用者填入有权读取的文件。观察器实际调用保持这些算法参数，仅三项输出落点更新：

```bash
# moving_gm_path/reference_gm_path：原三维 GM 图；initial_affine_path：原 FLIRT 矩阵。
# reference_mask_path：参考空间mask；以下三个输出路径必须是新落点。
fnirt --in="$moving_gm_path" --ref="$reference_gm_path" \
  --aff="$initial_affine_path" --config=GM_2_MNI152GM_2mm \
  --refmask="$reference_mask_path" --cout="$coefficient_output_path" \
  --fout="$dense_field_output_path" --jout="$jacobian_output_path"
```

本轮未添加 `--iout`、`applywarp` 或 `fslmaths`。既有官方 moved/modulated 图由后续重采样和调制产生，故不属于本次三端点资格。原版 PCG 位于不可被本观察器截获的可执行程序符号中，PCG 迭代轨迹记为 NA。

## 5. 真实精度、耗时与可视化

| 结果 | 实际观察 |
| --- | ---: |
| 原版完整调用 / 自然级数 | 1 / 4 |
| 自然 cf / grad / hess / sf | 29 / 25 / 25 / 0 |
| 事件 / 私密参数快照 | 222 / 29 |
| 私密参数总字节 | 4,665,208 |
| coefficient / dense field / nonlinear Jacobian | 三项完整文件均与历史官方文件完全相同 |
| moved / modulated / PCG 迭代轨迹 | NA |

| 级别 | MaxIter | 原 NIter | 原存储成本 | 最终阻尼 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5 | 6 | 59.89472075220734 | 1.0000000000000002e−6 |
| 2 | 5 | 6 | 92.40713924765691 | 1.0000000000000002e−6 |
| 3 | 10 | 11 | 138.8418854399147 | 1.0000000000000003e−11 |
| 4 | 5 | 6 | 440.65964752174665 | 1.0000000000000002e−6 |

原 `NextIter(success)` 在成功分支的终止比较中仍递增计数，正常耗尽预算时返回 `MaxIter+1`；失败试步不递增。因此 NIter 不等于直接记录的接受次数或试步总数。原校验器要求 `NIter≤MaxIter`，在原版成功退出之后产生误报。独立保存资格按源码定义核对该字段，同时保持原配置预算不变。

| 时钟范围 | 秒 |
| --- | ---: |
| 被观察的原版进程 | 276.569501 |
| worker（含原失败验证器） | 276.763567 |
| supervisor / 独立 OS wrapper | 276.864737 / 276.929601 |
| 只读保存资格 / 其 controller | 0.061804 / 0.191965 |
| 既有官方记录 | 326.096913 |

这些时钟互相嵌套。历史官方与本轮 CPU 亲和性不同，且本轮增加观察；不计算速度倍率。独立保存资格没有解码影像或参数浮点数组，以三项完整文件身份核验头信息和所有数据 words。本轮没有新增脑图；公开参数数组和影像不随报告发布。

原控制保留 `complete=false`；独立保存资格34门、26项后验身份全部通过。原进程树5个 PID/start 实例、资格组3个实例均退出，共同锁释放、六索引 peer 保留，两个阶段执行授权均已关闭。

## 6. 版本与 benchmark 记录

- **原完整自然观察**：一次原版四级运行与三项端点成功保存；原 NIter 语义验证器误报保留。
- **只读保存资格**：不重跑原版、编译、CF、PCG 或 GPU；源码定义的计数范围与三项完整文件身份均通过。
- **后续候选轨迹**：另行准备未改数学的完整 CPU 参数观察；需先核六项既有候选输出透明性，再比较真实自然边界参数。成本或 PCG 次数相近不证明参数点相同。

此前 bounded solve3 与候选完整链的 coarse solve3 未证明同参数点；本次原自然轨迹将用于消除这一对照缺口。配对结果尚未产生，不推断完整候选误差原因。

## 7. 参考文献与原实现

- [FSL FNIRT 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt.html)。
- [FSL miscmaths 原代码库](https://git.fmrib.ox.ac.uk/fsl/miscmaths)：非线性优化调用与迭代计数语义。
- [FSL fnirt 原代码库](https://git.fmrib.ox.ac.uk/fsl/fnirt)。本轮使用已安装6.0.7.4；源码证据与实际安装二进制身份分列，不主张重建原二进制。
- [CPU 预条件器三臂控制](../fnirt_cpu_preconditioner_ablation_20261007/README.md)：固定保存算子下的有限精度敏感性范围。
- [FNIT CPU 采样候选](../fnirt_cpu_sampler_case_20261006/README.md)：原采样公式与真实输入控制。

原 SDK 和软件按原许可证使用，本报告不再分发其代码或运行库。
