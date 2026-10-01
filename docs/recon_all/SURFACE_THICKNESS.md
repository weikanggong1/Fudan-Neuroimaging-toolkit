# 厚度搜索的完整空间候选

[返回表面指标](SURFACE_METRICS.md)

recon-all 的 CUDA 分支已经调用已有 PyTorch 厚度函数。本次优化其搜索和可达性，不另写厚度定义。两例双侧同输入 GPU 回归通过后，完整空间候选算法已设为 `thickness_map` 默认；`thickness_map_indexed` 是同一函数的兼容名称。旧密集实现仅保留在验证目录，继续作为冻结同输入基线，不参与生产计算。

原 CUDA 搜索为每个 1024 顶点块计算对全部目标顶点的 `cdist`，之后逐顶点执行 Python 20 跳搜索。新函数复用原法向和方向判断，复用 SciPy `cKDTree` 查询完整的局部候选，并在 PyTorch 当前设备上批量计算候选距离与方向。CPU 的 Numba 内核共享一次双向图搜索，仍检查最多 20 跳。没有降低精度或改变面积、厚度、标注及网格。

## 输入、输出和参数

`thickness_map(white_file, pial_file, output_file, *, device="cuda:0") -> dict`：

| 参数 | 类型、空间及限制 |
| --- | --- |
| `white_file` | `str` 或 `Path`，最终 white 三角表面；顶点 `(N,3)`、面 `(F,3)`，surface RAS，mm |
| `pial_file` | `str` 或 `Path`，最终 pial；必须与 white 有相同顶点数和有序面，坐标有限 |
| `output_file` | `str` 或 `Path`，写出的 float32 FreeSurfer morph 文件；`(N,)`、mm、同一顶点顺序；自动建立父目录 |
| `device` | PyTorch 设备字符串，默认 `cuda:0`，也支持 `cpu`；空间索引和图可达性位于 CPU，距离、法向和方向筛选使用此设备 |

输入读取优先 nibabel。输入不对应、坐标非有限、设备不可用或读写错误时抛出异常，不返回近似厚度。函数不启用 FP16/BF16，也不改变调用者的 TF32 设置。GPU 阶段前后同步；分步和总时间包括函数的读取、传输、计算及写出，不包括调用前导入。

CPU 空间查询使用 `min(4, torch.get_num_threads())` 个工作线程：调用者设置一个 PyTorch 线程时，KDTree 也使用一个；四个或更多时，KDTree 最多使用四个。调用期间须保持 PyTorch 线程设置不变。该预算约束空间查询，不代表整个进程及所有库的线程数上限。

返回字段为 `vertices`（顶点数）、`device`（实际设备）、`setup_seconds`（读取、法向、CSR 和索引）、`compute_seconds`（完整候选和可达性，包含冷启动 JIT）、`total_seconds`（包含写出）、`candidate_pairs`（双向总候选对）、`peak_candidate_pairs`（单块双向候选数）、`kdtree_workers`（本次空间查询工作线程数）、`candidate_search="complete-radius"`、`radius_guard_mm=0.0001`、`maximum_hops=20` 和 `maximum_direction_thickness_mm=5.0`。

```python
from pathlib import Path
from fnit.recon_all.surface_thickness_gpu import thickness_map

report = thickness_map(
    white_file=Path("/data/subjects/sub01/surf/lh.white"),  # 最终 white，surface RAS、mm
    pial_file=Path("/data/subjects/sub01/surf/lh.pial"),  # 同序、同面的最终 pial
    output_file=Path("/data/diagnostics/sub01-lh.thickness"),  # 隔离诊断输出，float32、mm
    device="cuda:0",  # 使用已明确映射的目标 GPU；不启用半精度
)
```

## 完整候选和图规则

每方向初值为同索引 white/pial 的直接距离。候选必须同时满足位移方向、pial 法向方向和距离小于初值，且从当前顶点最多 20 跳可达。最后将两个方向分别截断至 5 mm，再取平均。

能够改变截断后输出的候选必定位于 `min(直接距离, 5 mm)` 范围。空间索引查询该完整球域，并沿用 0.0001 mm 浮点边界保护；不截成最近 256 点，也不遗漏距离相同的候选。每个候选的最终距离和方向仍由原 float32 PyTorch 运算计算。图搜索发现双向最近合法候选均已可达后可停止；否则扩展到 20 跳并在全部可达候选中取最短距离。相同距离候选中的任一个可达点给出相同标量厚度。

内部 `_adjacency_csr` 返回排序的对称 CSR 图（`indptr`、`indices`）；`_radius_candidates` 返回每行候选偏移、int32 顶点编号、bool 合法标记和 float32 距离；`_reachable_distances` 返回 `(B,2)` float32 双向距离。`_clip_mean` 保留原截断、float32 求和和写出转换顺序。这些是厚度函数的内部步骤，没有独立官方 CLI。

搜索取消了对全部顶点的密集距离矩阵。常见局部候选情况下工作量随完整局部候选数增长；极密集或重叠网格仍可能产生大量候选，因此不声称所有输入具有固定线性复杂度或固定显存上界。每块沿用 1024 顶点，报告候选数，不通过削减候选数量控制内存。

## 同输入回归

优化的预定义门槛为旧/新厚度 **绝对差 ≤ 0.000001 mm、相对项 0、无越界顶点**。官方同输入验证继续保留项目已有厚度门槛 **0.005 mm + 0.001 × |reference|**。两个门槛都由复现脚本固定，不能根据结果修改。整例指标等效另行评价，不由阶段测试推断。

2026-10-01 在 gpucw1、物理 GPU1（UUID `GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba`，逻辑 cuda:0）、四个 PyTorch 线程、默认 TF32、float32、分配缓存关闭的同一环境，对 FNIT 两例最终 white/pial 各执行两轮，交替新旧先后顺序。冻结源码标识为 `3d9856c+stage1tar0444db72`，计算候选 SHA-256 为 `fb3b665150a2c78c0a8a3a5813f29648e71b74efc7474f4b6c318489862e0136`。报告记录实际输入、旧/新源码和执行脚本 SHA-256，不把历史耗时当成本轮结果。

| 冻结 FNIT 网格 | 顶点数 | 旧密集函数，两轮 | 完整空间候选，两轮 | 同序差异 |
| --- | ---: | ---: | ---: | --- |
| [sub-01 LH](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/thickness_lh/report.json) | 105,539 | 38.42 / 34.45 s | 8.86 / 9.57 s | 两轮均 0 |
| [sub-01 RH](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/thickness_rh/report.json) | 104,864 | 33.05 / 27.52 s | 6.69 / 6.46 s | 两轮均 0 |
| [sub-02 LH](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/sub02_thickness_lh/report.json) | 119,363 | 48.57 / 42.24 s | 13.40 / 13.45 s | 两轮均 0 |
| [sub-02 RH](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/sub02_thickness_rh/report.json) | 118,303 | 39.78 / 41.15 s | 9.09 / 9.25 s | 两轮均 0 |

八次厚度图全部逐值一致，最大/P99/平均误差为 0，没有超出预定义门槛的顶点。四个半球两轮平均的阶段耗时分别缩短 74.70%、78.30%、70.44%、77.33%；完整逐轮结果在[机器摘要](../../validation/recon_all/python_gpu_port/performance_20261001/thickness_summary.json)。每块双向最多候选为 42,170 / 30,930 / 31,363 / 29,966 对，而不是全目标网格矩阵。耗时包含影像加载、CPU/GPU 传输、计算及写出，并同步目标 GPU；共享设备观察值不代替整例提速。

缓存关闭，allocated/reserved 报告为不可用，不能解释为零。四个**完整新旧配对进程**的显存采样最大值分别为 1,308,622,848 / 1,304,428,544 / 1,365,245,952 / 1,361,051,648 字节，不能单独归于新函数；采样间隔最大 2.25–2.30 s、查询失败 0，连续峰值仍未验证。20 GB 整例预算另由整例监控验收。该配对未读取官方厚度，不宣称新的官方逐值复现结果。

真实回归通过后，默认函数更名和旧生产代码清理不改上述计算顺序；清理快照 SHA-256 为 `c5d586b58f8b57af1ecc5b61fcf0e33c980ee5eaca681fe7cc6c25be57083f48`。随后修正 KDTree 不遵守单线程预算的问题，源码 SHA-256 为 `ab019eb7d413c7b08f01ad6b2e499c406ab513cb531d723142c2db2d6c12e961`：四线程下仍使用四个查询工作线程，候选规则、批大小与浮点运算顺序未改。新源码的 GPU 同输入复核和原始 T1 整例报告须绑定实际版本；上面八轮数据仍属于 `fb3...` 阶段快照，不能改标为新源码的测试结果。

### 线程预算修正后的独立 GPU 复核

2026-10-01 03:11 UTC，用相同 sub-01 LH 冻结 FNIT white/pial，在同一物理 GPU UUID、四线程、TF32/float32 和关闭分配缓存的环境，只执行一次 `ab019...` 新函数；没有修改冻结 `1b8c36d` 快照或整例目录。[报告](../../validation/recon_all/python_gpu_port/performance_20261001/thickness_workers_ab019/result/report.json)绑定 105,539 个顶点、211,074 个有序面及原输入 SHA-256。

同步函数耗时 **13.706 s**，包括读取、传输、计算和写出；包含脚本启动、校验与报告的完整命令墙钟为 **17.961 s**。返回 `kdtree_workers=4`，双向候选 2,398,404 对、单块最多 42,170 对。新图与保存的 `1b/c5d...` 图数值和文件 SHA-256 都相同，差异顶点、最大/P99 差及越 1e-6 mm 容差顶点均为 0。与既有同输入 Conda 图最大差 4.768e-7 mm、P99 2.384e-7 mm，10,622 个非零尾差、越既有门槛顶点 0；这些尾差与旧图相同。这不是新运行的官方参考。

[单进程监控](../../validation/recon_all/python_gpu_port/performance_20261001/thickness_workers_ab019/monitor/monitor.json)共 9 个样本，同次查询合计父子进程显存最大 **562,036,736 字节**。采样请求间隔 2 s，最大间隔 2.256 s，查询失败 0；allocated/reserved 为 null，连续峰值未验证。独立的全 GPU 查询观察到 33,843–46,463 MiB、利用率均为 100%，不归入本进程占用。该共享设备单次观察不构成 worker 修改的配对提速实验，也不代表 `ab019...` 已运行原始 T1 整例。阶段 1 的八轮量值与哈希保持原归属；本次结果单独保存在 [workers_summary.json](../../validation/recon_all/python_gpu_port/performance_20261001/workers_summary.json)。

`benchmark_thickness_workers.py` 校验候选源码、white/pial、保存的旧图及 Conda 图 SHA-256，并只调用一次新函数。`--pair-report` 指向冻结配对报告及旁边的旧厚度图；`--conda-binary` 仅记录已经生成参考图的源码构建程序哈希，不执行它；`--output` 必须是新目录。输出新厚度图及 JSON；结构、哈希或设备不匹配时抛异常，数值未通过时保存报告并退出 1。该脚本没有独立官方 CLI，对应厚度阶段的命令仍为下文的 `mris_place_surface --thickness`。

```bash
gpu_uuid=GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba # 与冻结测试相同的物理 GPU
export CUDA_VISIBLE_DEVICES="$gpu_uuid" PYTORCH_NO_CUDA_MEMORY_CACHING=1 # 初始化前设置
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 # 线程预算
python validation/recon_all/python_gpu_port/benchmark_thickness_workers.py \
  --candidate-source-file /data/diagnostics/ab019/surface_thickness_gpu.py \
  --expected-source-sha256 ab019eb7d413c7b08f01ad6b2e499c406ab513cb531d723142c2db2d6c12e961 \
  --pair-report /data/diagnostics/final_1b8c36d/thickness/report.json \
  --white /data/subjects/sub01/surf/lh.white --pial /data/subjects/sub01/surf/lh.pial \
  --conda-map /data/diagnostics/lh.cpp.thickness \
  --conda-binary /opt/conda/envs/fnit/bin/mris_place_surface \
  --output /data/diagnostics/ab019/new-result \
  --code-version 1b8c36d25a68e253a1e59b6d02114890afa467de+thickness-ab019eb \
  --gpu-uuid "$gpu_uuid" --device cuda:0 --threads 4
# candidate-source-file 与 expected-source-sha256 绑定实际候选源码。
# pair-report、white、pial 是同一冻结网格；conda-map 为已生成的同输入厚度。
# conda-binary 只记录版本；output 为新目录，code-version 标识代码来源。
# gpu-uuid 显式映射到 device；threads 同时约束 Torch 和 KDTree 工作预算。
```

### 新旧函数配对脚本

`benchmark_thickness_indexed.py` 读取冻结的真实 white/pial、`3d9856c` 原函数源码、新输出目录和代码标识，默认 `cuda:0`、四个 PyTorch 线程、两轮交替先后顺序。输出每轮两张厚度、JSON 差异、同步耗时、输入/源码 SHA-256；显存缓存关闭时 allocated/reserved 为不可用，不能解释为零。可传 `--reference-map` 指定完全相同冻结网格的官方厚度，参考文件仅由独立 benchmark 路径生成。失败仍保存报告并退出 1。

```bash
white_file=/data/subjects/sub01/surf/lh.white   # FNIT 最终 white，不复制官方网格
pial_file=/data/subjects/sub01/surf/lh.pial     # 对应最终 pial
baseline_source=validation/recon_all/python_gpu_port/reference_surface_thickness_dense_3d9856c.py # 冻结原 FNIT 函数
output_dir=/data/diagnostics/thickness-indexed-lh # 必须是新的隔离目录
code_commit="$(git rev-parse HEAD)"            # 报告同时绑定实际源码 SHA-256
python validation/recon_all/python_gpu_port/benchmark_thickness_indexed.py \
  --white "$white_file" --pial "$pial_file" \
  --baseline-source-file "$baseline_source" \
  --output "$output_dir" --code-commit "$code_commit" \
  --device cuda:0 --threads 4 --repeats 2
```

## 与当前 Conda 厚度图的同输入诊断

本轮独立诊断已经用同一 FNIT white/pial 和参数 `20 5` 生成当前 Conda `mris_place_surface` 厚度图。随后在 headcw 只读取既有新旧 Python 输出与这些 Conda 图，不再次执行 GPU 或原生计算。程序 SHA-256 为 `44ad3994f8b86809b447919094273d3aa591b49be935ac4e09139e66bdddfe65`；生成命令和日志见性能目录的 `fnit_gpu_fix_probes*.sh`、`diagnostics/*.cpp.thickness.log`。这不是新运行的官方参考。

| 同输入 Conda 对照 | 新旧各两轮的最大绝对差 | P99 绝对差 | 越现有容差顶点 |
| --- | ---: | ---: | ---: |
| [sub-01 LH](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/thickness_lh_vs_conda.json) | 0.000000477 mm | 0.000000238 mm | 0 |
| [sub-01 RH](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/thickness_rh_vs_conda.json) | 0.000000954 mm | 0.000000238 mm | 0 |
| [sub-02 LH](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/sub02_thickness_lh_vs_conda.json) | 0.000000477 mm | 0.000000238 mm | 0 |
| [sub-02 RH](../../validation/recon_all/python_gpu_port/performance_20261001/diagnostics/sub02_thickness_rh_vs_conda.json) | 0.000000954 mm | 0.000000238 mm | 0 |

这些 float32 尾差在旧 Python 中也存在，新算法没有增加。诊断沿用绝对 0.005 mm 加 0.001 相对项，不改变优化旧/新 0.000001 mm 的门槛。两轮共 16 张新旧图均通过同输入 Conda 对照；整体指标等效仍未判定。

`compare_thickness_conda_maps.py` 的参数都是显式文件路径：`--pair-report` 为旧/新配对 JSON；`--white`、`--pial` 必须符合它记录的输入 SHA-256；`--conda-map` 为同输入、同序、20 跳/5 mm 的 Conda 厚度；`--conda-binary` 仅记录已使用程序的 SHA-256，不运行它；`--output` 为不存在的 JSON 路径。脚本还检查每张既有 Python 图的 SHA-256，返回各轮最大/P99/平均绝对差、差异值数、越界顶点和通过状态。哈希或结构不匹配时抛异常，数值失败保存 JSON 并退出 1。它不加载 CUDA、读取原始 T1 或检查官方整例。

```bash
white_file=/data/subjects/sub01/surf/lh.white # 与配对报告哈希一致的 FNIT white
pial_file=/data/subjects/sub01/surf/lh.pial   # 对应 FNIT pial，surface RAS、mm
pair_report=/data/diagnostics/thickness_lh/report.json # 原两轮新旧配对报告及其 morph 输出
conda_binary=/opt/conda/envs/fnit/bin/mris_place_surface # 已安装的 FNIT Conda 源码构建组件
conda_map=/data/diagnostics/lh.cpp.thickness  # 已用相同输入、20/5 参数生成的图
comparison_report=/data/diagnostics/thickness_lh_vs_conda.json # 新 JSON 报告
python validation/recon_all/python_gpu_port/compare_thickness_conda_maps.py \
  --pair-report "$pair_report" --white "$white_file" --pial "$pial_file" \
  --conda-map "$conda_map" --conda-binary "$conda_binary" \
  --output "$comparison_report"
```

原生写出文件以日志中的实际路径为准：本轮第二例请求 `sub02.lh.cpp.thickness`，程序实际加了半球前缀，写成 `lh.sub02.lh.cpp.thickness`；右侧相同。不要按请求文件名误判为缺失输出。

`summarize_thickness_indexed.py` 仅整理既有 JSON。`--reports` 指向本性能目录，`--final-source` 只用于散列最终生产源码，`--output` 指定不存在的摘要文件。摘要同时保存实测 `fb3...` 与最终 `c5d...`，不把版本更名后的待测整例冒充八次冻结阶段测试；官方新参考为 `not_run`，整体等效为 `not_assessed`。

```bash
python validation/recon_all/python_gpu_port/summarize_thickness_indexed.py \
  --reports validation/recon_all/python_gpu_port/performance_20261001 \
  --final-source src/fnit/recon_all/surface_thickness_gpu.py \
  --output /data/diagnostics/thickness_summary.json
# reports 读取八轮真实阶段与监控；final-source 散列最终默认函数。
# output 是新的机器摘要，不执行 GPU 或影像算法。
```

2026-10-01 在 headcw 的主页 Conda 环境实际执行六项 CPU 单元检查，全部通过：[报告](../../validation/recon_all/python_gpu_port/thickness_indexed_20261001/unit_checks.json)。覆盖 20/21 跳边界、等距及不连通候选、工作数组复用、超过 256 候选、截断转换和 CPU 旧/新函数回归。报告绑定候选、冻结旧实现、测试和执行脚本的 SHA-256。它们是算法语义检查，不替代上面的真实数据 benchmark，尚未给出整例提速结论。

线程预算修正后，同日同一 CPU 环境执行新增检查及原六项回归，七项全部通过：[新源码报告](../../validation/recon_all/python_gpu_port/thickness_indexed_20261001/unit_checks_workers.json)。真实 KDTree 的线程 1/4 双向候选数组逐值一致；该报告绑定 `ab019...` 源码，原 `c5d...` 六项报告保留版本归属。CPU 语义检查、上述单半球真实 GPU 复核与原始 T1 整例按各自范围报告。

该 Conda 环境未安装 pytest，`run_thickness_indexed_unit_checks.py` 通过标准库 AST 选取同一测试文件的函数，并用标准库 patch 上下文替代需要 fixture 的检查。线程预算回归使用真实 `cKDTree` 和 400 个非等距候选，包含方向及法向拒绝、超过 256 候选；比较线程 1 与 4 的双向候选偏移、顶点编号、合法标记及距离逐值一致，最后恢复原 PyTorch 设置。当前脚本执行七项检查，四个参数均为显式路径；断言失败抛异常。使用具名参数复现：

```bash
python validation/recon_all/python_gpu_port/run_thickness_indexed_unit_checks.py \
  --candidate-source-file src/fnit/recon_all/surface_thickness_gpu.py \
  --baseline-source-file validation/recon_all/python_gpu_port/reference_surface_thickness_dense_3d9856c.py \
  --tests tests/recon_all/test_surface_thickness_gpu.py \
  --report /data/diagnostics/thickness-cpu-unit-checks.json
# candidate-source-file 是当前候选；baseline-source-file 是原 FNIT 函数。
# tests 指向相同单元测试源码；report 是新的报告输出，不含真实影像。
```

## 官方对应与文献

```bash
mris_place_surface --thickness surf/lh.white surf/lh.pial 20 5 surf/lh.thickness
```

官方程序只用于独立对照，不在 FNIT 生产路径调用。Numba 和 SciPy 已包含于主页 `environment.yml`，本次不新增依赖。

- [固定版本 mris_place_surface](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mris_place_surface/mris_place_surface.cpp)。
- Fischl B, Dale AM. Measuring the thickness of the human cerebral cortex from magnetic resonance images. *PNAS*. 2000;97:11050–11055. [doi:10.1073/pnas.200033797](https://doi.org/10.1073/pnas.200033797)。
