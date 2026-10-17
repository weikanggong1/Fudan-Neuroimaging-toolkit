# 两例整例配对的公开计时账

## 1. 功能与输入输出

[`export_whole_timing.py`](export_whole_timing.py) 使用标准库读取已有 JSON
元数据，不导入 FNIT、Torch 或影像库，不启动重建、GPU 或事后评估。
它将本轮四次原始 T1 空目录整例的 SHA 绑定证据转换为公开 JSON 和 CSV。
流程是：私有绑定清单 → 原文件 SHA/完整性校验 → 字段白名单 → 计时账。

`extract(evidence=...)` 接收普通字典，返回可序列化字典，不写文件。输入
格式如下。所有 `sha256` 必须是对应现存文件的真实 SHA-256，不能使用
示意占位值执行；四个 `runs` 按实际运行顺序提供，不能重排。

```json
{
  "schema": "fnit-whole-timing-inputs-v1",
  "pair_report": {"path": "/data/private/speed_pair.private.json", "sha256": "<实际SHA>"},
  "runs": [
    {
      "execution_index": 0,
      "benchmark": {"path": "/data/private/run0.benchmark.json", "sha256": "<实际SHA>"},
      "runtime": {"path": "/data/private/run0.runtime.json", "sha256": "<实际SHA>"},
      "supervisor": {"path": "/data/private/run0.supervisor.json", "sha256": "<实际SHA>"}
    }
  ],
  "source_manifests": {
    "candidate": {"path": "/data/private/candidate.archive_manifest.json", "sha256": "<实际SHA>"}
  }
}
```

示意中的 `runs` 实际必须有四项：病例0基线、病例0候选、病例1候选、
病例1基线。各项分别绑定 benchmark、运行报告和单病例监督器报告。
`pair_report` 内保存的三份子报告 SHA、输入 SHA、源码版本和重复计时字段
须与实际文件一致。缺失、哈希变化、未完成、非零退出、缺输出、网格门失败、
未知后端或阶段结构直接抛异常；不写成部分成功。

`source_manifests` 可省略，或提供某臂已有的归档清单。未提供的一臂记为
`not_provided`，不创建或推断新来源证明。本轮候选归档清单有2117项；
pair 的运行前源码映射每臂944项；benchmark 的 FNIT Python 源码映射每臂
639项。这三种范围分别记录数量、规范 JSON 映射摘要和实际清单文件 SHA，
不把它们称作同一清单。639项必须逐项匹配 pair 记录；有归档清单时还须
逐项匹配对应归档记录。公开内容只列五个实际变更的科学源码及其两臂 SHA。

输出 [`reports/whole_timing.public.json`](reports/whole_timing.public.json) 包含：

- 四次运行各自的输入、源码、三份报告 SHA、实际后端、缓存先后、六种
  墙钟、59条父阶段及九组互斥计时账。
- 主机、CPU、亲和性、线程、逻辑设备、声明环境、资源、原生程序、N4
  依赖的相同预算核对；只导出核对布尔值、数量和 SHA。H100 PCIe 型号
  来自实际 GCA/WM 运行记录；物理 GPU 的一致性由32个半球 worker 的
  实际身份核对，仅导出一致性，不输出 UUID。
- 两例各自的旧新时间变化及两例 CLI 总和变化；失败采样只导出数量，
  完整自有同期显存峰值未知时保持 `null`，20 GB验收保持未评估。

[`reports/whole_timing.csv`](reports/whole_timing.csv) 每行是同病例的一个
时间对照，附两臂 benchmark SHA。`scope` 区分独立时钟、互斥父组和组内
单条父阶段；不同 scope 是同一运行的不同表示，不能全表相加。
时间单位均为秒，没有坐标空间转换或影像数组输出。

## 2. Python 与命令行调用

```python
import importlib.util
import json
from pathlib import Path

script_path = Path("/data/fnit/validation/recon_all/optimizations/20261017_sphere_normals/export_whole_timing.py")
module_spec = importlib.util.spec_from_file_location("whole_timing_export", script_path)
timing_module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(timing_module)
evidence_path = Path("/data/private/timing_evidence.json")  # 四次原报告与SHA的私有绑定
evidence = json.loads(evidence_path.read_text())  # 不在公开目录保存路径清单
public_timing = timing_module.extract(evidence=evidence)  # 返回公开计时字典，不写文件
```

```bash
python validation/recon_all/optimizations/20261017_sphere_normals/export_whole_timing.py \
  --evidence-json /data/private/timing_evidence.json \
  --output-json /data/public/new_whole_timing.public.json \
  --output-csv /data/public/new_whole_timing.csv
```

三个参数均必填且无默认值：`evidence-json` 是上述私有清单；`output-json`
和 `output-csv` 是两个必须尚不存在的输出文件，父目录须存在。工具拒绝
覆盖原结果，成功后只打印两个输出 SHA。清单可引用原文件或内容相同的
私有副本，SHA 必须不变。JSON 中 `evidence_bindings_sha256` 是私有清单
字典的规范编码摘要；它包含路径绑定的摘要，不等于原报告 SHA 或源码 SHA。
输入路径改变会改变该摘要，原科学数字和报告 SHA 不变。

## 3. 时钟和阶段账

| 字段 | 计时边界 |
| --- | --- |
| `CLI` | 实际重建子解释器启动至退出，包含校验、模型加载、传输、计算和输出读写。 |
| `full_harness` | benchmark 的 main 入口至已有报告写入，包含参数/输入校验与源码、资源哈希；不含该解释器 main 前导入及最后一次元数据发布。 |
| `API` | 重建 API 入口至校验、线程恢复和输出写入完成；不含 CLI 解释器启动及最终公开元数据写入。 |
| `supervisor_case_outer` | 监督器启动 benchmark 子解释器前至其退出，包含 benchmark 导入和最终发布。 |
| `supervisor_API` | 单病例监督器函数全程，含原始输入校验和已有报告写入；不含该解释器导入及最后发布。 |
| `pair_supervisor_outer` | 配对 driver 启动监督器子解释器前至退出，含其导入和最终发布。 |

这些时钟有包含关系，不能相加。每例使用其当次实际时钟，没有拿历史官方
耗时代替本轮基线；本工具不提供官方同环境速度结论。

互斥父组如下：到 `surface_hemisphere_group` 之前的阶段为 `volume_prefix`；
surface/register/annotation/finish_surface 四个并行组各取其唯一父阶段；
`stats_*` 汇为统计组；`mni_mesh_parallel` 单列；剩余父阶段归为其他串行
步骤。`CLI_residual = CLI - 59条父阶段之和`，保留 CLI 导入、校验、调度
和未落入这些计时器的开销。九组之和还原 CLI，不能另加左右半球时间。
每条父阶段都核对 `profiling_nesting_depth=0` 和目标 CUDA 同步记录。
原 `algorithm_substep_seconds`、worker span、左右脑和重叠区间不加入账本。

## 4. 本轮实测与解释范围

绑定的最终 pair 报告 SHA 为
`f12b6c8f64f25730349bf176beecca26aef19ece54a1929e24969a2179fd465c`。
基线源码为 `5ca07f999375d02281ab60589cb2840e7b488790`，候选源码为
`71cc91899bdf9713690cddbe32054386d8f58d63`。四次均完整退出0、138/138，
生产网格门通过。精度和官方对照由本轮事后评估报告单独给出。

两臂每例总预算4个 CPU线程、四核亲和，双半球 worker 各用有效2个
Torch/Numba线程、interop为1。该预算不表示每个内部 CPU算子都使用4线程。
两臂保留 TF32 默认和既有 FP32阶段例外，无半精度；surface worker启用
分配缓存，其余三个半球组继承关闭策略，两臂相同。实际策略见 JSON；
除球面法向 `numba→torch` 外后端选择相同。

Numba/Triton 缓存按臂独立，配对开始各臂使用新目录，病例0是各臂第一例，
病例1复用各臂已产生的缓存。病例0按 A→B、病例1按 B→A。没有清除系统
页缓存或宣称每个内部调用都是冷调用，不能把两例不同输入当作同臂重复。

CLI 从2398.175→2299.843秒、2198.098→2131.620秒，分别观察缩短4.10%和
3.02%；两例 CLI 总和缩短3.59%。这是共享 GPU 上各一对的观察值，不是
稳定吞吐估计，10分钟目标未达到。非目标步骤也有时间漂移，例如第一例
stats 和第二例 `finish_metrics_lh`；不把这些变化全部归功于本轮优化。

目标卡采样包含其他作业。四份 benchmark 的完整自有同期显存均未归属，
即使采样失败列表为空，也不能将卡上峰值、Torch零计数或部分进程下界当作
完整自有峰值。本表保持 `owned_simultaneous_peak_bytes=null` 和20 GB
`not_assessed`；独立外层监测的范围和下界由其原报告另列，不混入本表。

## 5. 验证与原软件对应

2026-10-17 使用四份当前 SHA 绑定报告重新提取，核对完整59父阶段、九组
还原 CLI、六时钟、运行顺序、源文件映射、资源及缓存契约。只读元数据
回归四项均通过，另检查哈希变化、嵌套阶段和未知策略被拒绝；不替代真实数据 benchmark。
提取器没有对应的 FreeSurfer CLI；被计时流程对应 `recon-all -i T1 -all`。
本工具不运行该命令，也不读取或复制官方参考结果。相关原始阶段验证见
[本目录阶段说明](README.md)及[配对 driver](../../../../docs/recon_all/RECON_NATIVE_SPEED_PAIR.md)。

### 可复现的真实元数据测试

专属入口为 [`test_whole_timing_export.py`](../../../../tests/recon_all/test_whole_timing_export.py)，
仅使用标准库 unittest 和本提取器。默认未指定私有证据时，四项明确 skip，
不生成模拟 benchmark，不把 skip 记作真实回归通过：

```bash
python3 tests/recon_all/test_whole_timing_export.py --verbose
```

真实证据目录必须含 `evidence.private.json` 及其引用的 SHA 绑定文件。设置
`FNIT_WHOLE_TIMING_EVIDENCE_DIR` 指定该目录，并以
`FNIT_WHOLE_TIMING_EVIDENCE_SHA256` 明确绑定清单内容；指定目录后，清单
不存在、未提供 SHA 或摘要不符均为错误。下面 SHA 是本轮实际私有清单的
摘要；重建不同路径的清单时须用它实际的新 SHA 替换：

```bash
FNIT_WHOLE_TIMING_EVIDENCE_DIR=/data/private/timing \
FNIT_WHOLE_TIMING_EVIDENCE_SHA256=d10878b2d7d16501e90de9b19f0acb08d0b874aada249b913986cde287c4e533 \
python3 tests/recon_all/test_whole_timing_export.py \
  --receipt-json /data/public/new_whole_timing_validation.public.json \
  --verbose
```

两个环境变量是私有输入绑定，不写入公开回执。`--receipt-json` 可选、
无默认值，若指定则须为尚不存在的新 JSON；`--verbose` 可选，默认简略
unittest 输出。测试失败返回非零；缺证据的回执状态为 `not_assessed`。

四项检查使用原数据的内存副本：完整提取与隐私排除、错误 SHA 拒绝、
嵌套父阶段拒绝、未知策略拒绝。第一项逐例核对原 CLI 值逐位保留、59条
父阶段各出现一次、九组定义及 residual 公式与原记录准确一致；重组浮点
和的闭合仅容许两倍该 CLI 值的 float64 ULP，回执报告实际闭合误差。这是
计时加法的舍入检查，不是影像或指标容差。

本轮四项真实元数据回归均通过，脱敏回执为
[`whole_timing_validation.public.json`](reports/whole_timing_validation.public.json)。
回执绑定测试、导出器、实际证据清单及原 pair 的 SHA，保留执行/跳过/失败
计数和闭合检查，不保存私有路径、身份、异常正文或原报告内容。这些测试
不重跑重建，也不更改整例、官方比较或原公开计时文件。
