# recon-all 性能与指标比较

[重建入口](README.md) · [本轮实测](../../validation/recon_all/python_gpu_port/performance_20260930/README.md) · [比较口径](../../validation/recon_all/python_gpu_port/RELEASE_GATES.md)

性能优化先比较冻结的同输入阶段，再从原始 T1 和空目录运行整例。单阶段只证明该算子在给定输入上的表现；整例检查上游误差传播和最终脑区指标。官方结果只供独立 benchmark 读取，生产阶段不读取这些目录。

2026-10-01 起使用[可选剖析](PROFILING.md)：生产默认不插入阶段同步，剖析显式同步目标设备并记录前后等待及父子 CPU 时间。API 全程包含前置校验、线程设置/恢复、加载、传输及数据读写；外层命令墙钟另包含导入、启动和最终报告写出。`run_monitored.py` 用单调时钟记录采样间隔，分别记录 NVML 查询耗时和超时，固定物理 UUID，在同一次 compute-apps 查询中计算父子合计。历史 CSV 的 UTC 空窗不能直接解释为 GPU 计算时间；GPFS mtime 与主机墙钟也不能未经校准混合定位。最新脚本、全部参数和结果见[当前验证目录](../../validation/recon_all/python_gpu_port/performance_20261001/README.md)。

## 两种计时与显存

主调度的 `total_seconds` 包含各阶段加载模型、读写影像、传输和计算，指定设备的末尾同步计入阶段秒数；开始同步在阶段计时点之前，仍包含在函数总时间中。它从入口资产校验后开始。外层 `command_wall_nanoseconds` 从启动 Python 前计时，另包含导入、入口校验和初始化。两种时间分开报告；旧基线没有外层时间时，不编造同范围的命令总时间比。共享硬件的单次差值只作为本次观察，不能当作独占资源的稳定速度比。本次 GPU 总时间减去阶段秒数之和约 1.23 s；不能把几百秒的异常阶段耗时称为计时外的等待或从总时间中扣除。

本轮前后均设置 PyTorch 与支持该选项的原生程序为 4 线程；没有全局限制 Numba、BLAS 或 OpenMP。Numba 默认可用线程在 gpucw1 为 128、nodecw10 为 192，不能将这两例描述成整个进程仅使用四线程。硬件、库版本及运行进程的线程环境保存在逐主机 JSON 中。

[GPU 启动器](../../validation/recon_all/python_gpu_port/performance_20260930/launch_full_recon_gpu.sh)使用九个路径/资源参数和一个汇总路径：`python_bin` 为主页 Conda Python，`code_root` 为固定源码的 `src/`，`input_t1` 为原始 T1，`subject` 为不存在的输出目录，`weights`/`assets` 为已校验资源，`gpu_index` 为 NVML 物理 GPU 编号，`sample_csv` 为采样表，`run_log` 为 stdout/stderr，`run_summary` 为退出码、采样最大值、次数和外层时间。启动器查询该编号的 UUID，同时用于 CUDA 可见设备和显存采样；逻辑设备为 `cuda:0`。`FS_LICENSE` 只传许可证路径。每一行分别记录父进程、当时的子进程和两者合计字节数；不同时间的峰值不相加。采样循环睡眠 2 秒，查询耗时使实际间隔略长，连续峰值未获保证。

```bash
python_bin=/data/conda/envs/fnit/bin/python  # 已安装 FNIT 的 Conda Python
code_root=/data/fnit-frozen/src             # 绑定提交与 SHA-256 的源码快照
input_t1=/data/sub01_T1w.nii.gz             # 一幅真实原始 T1w
subject=/data/subjects/sub01-new           # 必须不存在的被试目录
weights=/data/fnit-weights                 # 按固定清单校验的权重
assets=/data/fnit-assets                   # 按固定清单校验的模板与图谱
gpu_index=1                               # nvidia-smi 的物理 GPU 编号
sample_csv=/data/reports/sub01-gpu.csv     # 同时刻父子进程显存采样
run_log=/data/reports/sub01.log            # 程序标准输出和错误
run_summary=/data/reports/sub01-summary.txt # 退出状态和外层命令墙钟
export FS_LICENSE=/private/license.txt    # 自己的许可证路径
bash validation/recon_all/python_gpu_port/performance_20260930/launch_full_recon_gpu.sh \
  "$python_bin" "$code_root" "$input_t1" "$subject" "$weights" "$assets" \
  "$gpu_index" "$sample_csv" "$run_log" "$run_summary"
```

[CPU 启动器](../../validation/recon_all/python_gpu_port/performance_20260930/launch_full_recon_cpu.sh)使用同样的路径参数，省去 `gpu_index` 和 `sample_csv`。两个启动器都在被试目录已存在时失败；重建失败仍保存外层退出码，成功不表示官方数值复现通过。运行期间不改写启动器或源码快照。

[已初始化 CUDA API 脚本](../../validation/recon_all/python_gpu_port/run_initialized_cuda_api.py)先设置 `--threads`，再在 `--device`（默认 `cuda:0`）保留一个 float32 元素并调用真实 API。两个位置参数为原始 T1 与空输出目录；`--weights-dir`、`--assets-dir` 必填，`--threads` 默认 4。完成后另存 `run-api-invocation.json`，含调用方式、计时、CUDA 预初始化状态、启动线程预算、四字节保留张量与脚本哈希；失败时抛异常，侧车仅在成功后写出。已有 CUDA 缓存的调用方保留可用 allocated/reserved 统计；关闭缓存的运行不能把统计接口返回的 0 写成零显存。非默认逻辑 `cuda:1` 的错误路径探针只验证指定设备统计，不代表整例重建。

`279e09f` 已完成的整例包装器在首次小张量后、进入 FNIT 时才设置四线程。`e036f57` 的第三次 GPU 尝试将设置提前，但 Talairach 子进程仍启动失败；随后从另一新空目录完成了显式 UUID 的整例。缓存、模块加载及小张量对照记录在[启动诊断](../../validation/recon_all/python_gpu_port/performance_20260930/startup_e036f57/README.md)。不能将失败尝试的秒数计为提速，或仅凭线程设置后的某次成功认定 CUDA OOM 已修复。

```bash
input_t1=/data/sub01_T1w.nii.gz            # 原始单幅 T1w
subject=/data/subjects/sub01-api-new       # 新的空被试目录
weights_dir=/data/fnit-weights             # 已校验权重目录
assets_dir=/data/fnit-assets               # 已校验资产目录
device=cuda:0                              # 当前可见 GPU 的逻辑编号
threads=4                                  # 初始化前即设置的 PyTorch 线程预算
python validation/recon_all/python_gpu_port/run_initialized_cuda_api.py \
  "$input_t1" "$subject" --weights-dir "$weights_dir" \
  --assets-dir "$assets_dir" --device "$device" --threads "$threads"
```

## 优化前后与官方参考

`compare_performance_pair.py` 是 FNIT 诊断工具，没有对应的独立 FreeSurfer CLI。它调用现有 138 项比较器和现有表面距离函数，另执行脑区统计及分区 Dice。参数全部必填：

| 参数 | 输入与限制 |
| --- | --- |
| `--baseline` | 优化前、同原始 T1 的完整 FNIT 被试目录。 |
| `--candidate` | 优化后完整 FNIT 被试目录；不能是手动补跑后冒充的整例。 |
| `--official` | 独立生成的同一 T1 官方参考目录。 |
| `--label-table` | 已校验的 `FreeSurferColorLUT.txt`，用于具名报告离散标签。 |
| `--output-dir` | 不存在的新诊断目录，防止覆盖其他报告。 |
| `--code-commit` | 实际候选计算源码提交；另用归档和源码 SHA-256 核实，不能填当前文档提交代替。 |

```bash
python validation/recon_all/python_gpu_port/compare_performance_pair.py \
  --baseline /data/subjects/sub01-before \
  --candidate /data/subjects/sub01-after \
  --official /data/benchmark-only/sub01-reference \
  --label-table /data/fnit-assets/FreeSurferColorLUT.txt \
  --output-dir /data/reports/sub01-pair \
  --code-commit ACTUAL_TESTED_COMMIT
```

输出为两组 `strict_vs_*.json`、`region_vs_*.json`、`dice_vs_*.json`、`surface_vs_*.json` 和 `summary.json`。严格比较不通过仍保存失败并继续计算指标；读取、几何、标签语义或工具执行出错会抛异常并停止，不能将其当作普通数值失败。`execution_status=complete` 表示比较工具完成；`overall_metric_equivalence=not_assessed` 表示尚无正式确认的总体阈值。

脑区统计按名称配对 68 个 aparc 区、45 个 aseg 结构及 70 个 wmparc 区，保存逐区有符号差、绝对差、相对差和最差区。厚度为 mm、面积为 mm²、体积为 mm³，曲率为 mm⁻¹；参考值为零的区不计算相对差，单独列出。Dice 比较七张离散分割图，核对 conform 仿射及所有体素为非负整数；允许整数值以 float32 存储，禁止非整数标签。背景 0 不进入汇总，各标签的体素数、交集和 Dice 全保留。

表面文件使用 surface RAS mm。仅在顶点数和有序面对应时报告同索引距离；否则报告两个方向的精确点到三角面距离，均含均值、P99、最大值及超过 0.1 mm 的点数，后者仅是诊断计数。另报连通分量、边界边、非流形边、Euler 特征数与 sphere/sphere.reg 向内或退化面数。球面的形状距离不能证明球面顶点的解剖对应。重建入口另检查 white/pial 各自自相交；两张表面相互穿越应另行验证，不能把自相交结果写成两表面互不穿越。

## 真实 T1 图示

`plot_recon_all_comparison.py` 读取 `--reference`、`--candidate` 完整目录，以及上述 `--region-report`、`--dice-report`；`--output-dir` 必须不存在，`--code-commit` 同样填写实际计算提交。它生成三个 PNG 和 `provenance.json`：T1 三个 conform 中心切面的 white/pial 叠加、aparc 三类指标最差十区的有符号百分比误差、最低 Dice 的 Destrieux 体积分区局部边界。蓝绿表示参考表面、橙红表示候选；局部分区边界蓝色为参考、红色为候选。三角面与切平面求交，不将不同网格的顶点按索引相减。

图示用于定位，不替代数值判定；侧车保存所有实际读取输入及脚本的 SHA-256。conform 网格或 surface RAS 变换不一致时失败。绘图使用主页环境已有的 nibabel、NumPy 和 Matplotlib，无新增依赖。

```bash
python validation/recon_all/python_gpu_port/plot_recon_all_comparison.py \
  --reference /data/benchmark-only/sub01-reference \
  --candidate /data/subjects/sub01-after \
  --region-report /data/reports/sub01-pair/region_vs_official.json \
  --dice-report /data/reports/sub01-pair/dice_vs_official.json \
  --output-dir /data/reports/sub01-pair/figures \
  --code-commit ACTUAL_TESTED_COMMIT
```

## 厚度变化与 white/gray 采样

`probe_contrast_thickness.py` 是隔离诊断，没有对应的独立官方命令：冻结优化前 FNIT 的 rawavg、orig、white 和 cortex，仅换入候选 thickness，调用既有 `contrast_percentage`。具名参数 `--baseline`、`--candidate` 指向同一真实 T1 的两套 FNIT 输出；`--output-dir` 必须不存在；`--code-commit` 绑定候选源码。它在新目录建立只读输入链接，并写出 `report.json`，逐侧保存输入哈希、CPU 四线程耗时、相对两套 w-g.pct 的不同顶点数和最大/P99 百分点差。表面坐标为 surface RAS mm、厚度为 mm，灰质采样位置是厚度的 30%；规则对应 `mri_vol2surf` 投影及 `mri_concat --paired-diff-norm --mul 100`。缺失输入或不对应的数组会抛异常；交叉输入只用于诊断。

```bash
baseline=/data/subjects/sub01-before       # 优化前的 FNIT 输出
candidate=/data/subjects/sub01-after       # 同原始 T1 的优化后输出
output_dir=/data/reports/contrast-swap     # 必须不存在的隔离目录
code_commit=ACTUAL_TESTED_COMMIT           # 实际候选计算提交
python validation/recon_all/python_gpu_port/probe_contrast_thickness.py \
  --baseline "$baseline" --candidate "$candidate" \
  --output-dir "$output_dir" --code-commit "$code_commit"
```

## 生成机器可读摘要

`summarize_performance_runs.py` 只汇总本轮固定命名的记录，不运行重建，也没有对应的官方命令。具名参数 `--reports` 指向 `performance_20260930`（完整阶段 JSON、比较目录、固定源码清单、实时资源哈希、外层退出码、GPU CSV 与 API 调用侧车）；`--output-dir` 必须不存在。`--candidate-tag` 默认 `e036f57`，`--baseline-tag` 默认 `279e09f`，选择已收集的版本文件，不控制数值阈值；旧配对使用 `--candidate-tag 279e09f --baseline-tag b8cd`。它核对完成状态、退出码和提交后，写出 `current_full_runs_20260930.json` 与 `final_metric_consistency_20260930.json`：前者保存调用方式、两种耗时、相对直接基线及最初 b8cd 的变化、慢阶段、线程和同时刻显存；后者保存 Dice、脑区偏差及优化前后距离。采样记录另列实际 UUID、最大五个空窗、超过 30 秒的间隔数和预算未验证状态；30 秒仅为诊断计数，不是新验收阈值。输入报告的 SHA-256 随结果保留。记录缺失、运行未完成、版本或资源哈希不符时抛异常，不能生成成功摘要。GPU 间隔以 CSV 的 UTC 时间计算，受整秒时间戳精度限制。

```bash
reports=/data/fnit-validation/performance_20260930  # 已收集的两例完整实测记录
output_dir=/data/fnit-validation/current-summary  # 必须不存在的摘要目录
candidate_tag=e036f57                              # 实际计算提交对应的报告版本
baseline_tag=279e09f                               # 同 T1、同设备的直接优化前基线
python validation/recon_all/python_gpu_port/summarize_performance_runs.py \
  --reports "$reports" --output-dir "$output_dir" \
  --candidate-tag "$candidate_tag" --baseline-tag "$baseline_tag"
```

## 版本核验与异常阶段重放

以下脚本只复现本轮已授权服务器的固定实验，路径、提交和输出名在脚本中明确声明，没有隐含参数或独立官方等价命令；通用重建与算子入口仍使用上文的具名参数。输出目录或 JSON 必须不存在，输入缺失、版本不符或回归失败时抛异常。

| 脚本 | 全部输入及输出结构 |
| --- | --- |
| `performance_20260930/fingerprint_e036f57.py` | 读取本轮 provenance、14 程序构建清单及其声明的两个原始 T1、权重、资产、候选/参考程序；输出 `runtime_fingerprints_e036f57.json`，逐文件保存 SHA-256、字节大小与 mismatches。`fingerprint(path=...)` 返回 `{size_bytes: int, sha256: str}`，文件读取失败时抛异常。不读取许可证或参考影像。 |
| `performance_20260930/hardware_e036f57.py` | 在 gpucw1/nodecw10 的主页 Conda 环境读取版本、CPU/NVML 和本次正在运行的候选父进程 `/proc`；输出 `hardware_主机_e036f57.json`，含线程及六个声明的环境变量。只读取本人的目标进程，不初始化 CUDA；不是恰好一个本次父进程时失败，须在整例运行中执行。 |
| `performance_20260930/benchmark_wm_edit_repeat_e036f57.py` | 冻结本次 FNIT 的 entowm、aseg.presurf、wm.seg、brain（1 mm conform 网格）与 Conda WM 编辑程序，参数由 command 数组完整保存；输出两个新 WM 与日志、report.json。报告含 shape/dtype、仿射一致性、差异体素数、最大/P99 灰度差、秒数及程序/输入/脚本哈希。Linux `children_max_rss_bytes` 为该诊断父进程累计子进程最大 RSS，单位字节，不是单个子进程的独立峰值。 |
| `performance_20260930/launch_mni_e036f57_after_full.sh` | 使用本次已完成 FNIT 整例的 orig、裁剪 T1、aff.lta 和不变的权重/资产/Conda 三程序，显式调用既有 `benchmark_mni_nonlinear.py`，设备 UUID、四线程、关闭分配缓存固定；输出新的 MNI 诊断目录、四子步骤计时 JSON、日志、退出码与外层纳秒数。模型阶段沿用已验证 FP32 例外，调用后恢复 TF32。未另测本次独立阶段进程显存，allocated/reserved 为不可用，不能写成零。 |
| `performance_20260930/compare_mni_replay_e036f57.py` | 读取上述两套 FNIT MNI 输出，复用既有严格体积比较器；输出比较 JSON，含三个 NIfTI 的输入哈希、dtype、shape、仿射/头一致性及最大/P99 差。向量 warp 与强度检查图保留各自原网格，不重采样；1e-6 浮点容差沿用比较器，不改变门槛。 |

WM 编辑对应 `mri_edit_wm_with_aseg -keep-in -fix-ento-wm entowm.mgz 3 255 255 -fix-acj aseg.presurf.mgz 255 255 -fill-seg-wm -fix-scm-ha 1 wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz`；生产执行文件来自主页 Conda 固定源码构建，原实现见[固定 WM 编辑源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_edit_wm_with_aseg/mri_edit_wm_with_aseg.cpp)。MNI 输入空间、各输出形状、官方命令及原实现见[MNI 非线性链](MNI_NONLINEAR_CHAIN.md)。本轮 WM 重放计时包含原生输入/写出和日志，未建立父进程 CUDA 状态，不等价于恢复整例现场。

```python
from pathlib import Path
from fingerprint_e036f57 import fingerprint  # 固定实验脚本目录须位于 Python 搜索路径

image_fingerprint = fingerprint(
    path=Path("/data/sub01_T1w.nii.gz"),  # 可读的原始 T1 文件，仅散列字节，不载入影像
)
# image_fingerprint 的 size_bytes 为文件大小；sha256 为该文件实际字节的摘要。
```

## 官方复现与独立部署的证据范围

本轮优先复用既有同主机 N4 重放、固定输入 EM LTA 重放和表面指标三方测试；跨主机 N4 差异另列。官方参考整例尚未做完整重复运行，不能把全部 FNIT 差异归因于随机性。安装证据覆盖主页 Conda 环境、固定源码构建和资源清单；没有物理移除预装软件的干净机器，因此整例部署隔离仍未验证。PATH、ldd 和模块导入检查不能代替此验收。

## 原实现与参考文献

- [固定版本 FreeSurfer recon-all](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/scripts/recon-all)；参考生成命令为 `recon-all -i T1w.nii.gz -s subject -sd subjects -all -parallel -openmp 4 -itkthreads 1`。
- Fischl B. FreeSurfer. *NeuroImage*. 2012;62:774–781. [DOI](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
