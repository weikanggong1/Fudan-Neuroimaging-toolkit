# recon-all 性能与指标比较

[重建入口](README.md) · [本轮实测](../../validation/recon_all/python_gpu_port/performance_20260930/README.md) · [比较口径](../../validation/recon_all/python_gpu_port/RELEASE_GATES.md)

性能优化先比较冻结的同输入阶段，再从原始 T1 和空目录运行整例。单阶段只证明该算子在给定输入上的表现；整例检查上游误差传播和最终脑区指标。官方结果只供独立 benchmark 读取，生产阶段不读取这些目录。

## 两种计时与显存

主调度的 `total_seconds` 包含各阶段加载模型、读写影像、传输和计算，GPU 阶段在计时边界同步指定设备；它从入口资产校验后开始。外层 `command_wall_nanoseconds` 从启动 Python 前计时，另包含导入、入口校验和初始化。两种时间分开报告；旧基线没有外层时间时，不编造同范围的命令总时间比。共享硬件的单次差值只作为本次观察，不能当作独占资源的稳定速度比。

本轮前后均设置 PyTorch 与支持该选项的原生程序为 4 线程；没有全局限制 Numba、BLAS 或 OpenMP。Numba 默认可用线程在 gpucw1 为 128、nodecw10 为 192，不能将这两例描述成整个进程仅使用四线程。硬件、库版本及运行进程的线程环境保存在逐主机 JSON 中。

[GPU 启动器](../../validation/recon_all/python_gpu_port/performance_20260930/launch_full_recon_gpu.sh)使用九个路径/资源参数和一个汇总路径：`python_bin` 为主页 Conda Python，`code_root` 为固定源码的 `src/`，`input_t1` 为原始 T1，`subject` 为不存在的输出目录，`weights`/`assets` 为已校验资源，`gpu_index` 为物理 GPU 编号，`sample_csv` 为采样表，`run_log` 为 stdout/stderr，`run_summary` 为退出码、采样最大值、次数和外层时间。`FS_LICENSE` 只传许可证路径。指定 GPU 上每一行分别记录父进程、当时的子进程和两者合计字节数；不同时间的峰值不相加。采样循环睡眠 2 秒，查询耗时使实际间隔略长，连续峰值未获保证。

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

[已初始化 CUDA API 脚本](../../validation/recon_all/python_gpu_port/run_initialized_cuda_api.py)先在 `--device`（默认 `cuda:0`）保留一个 float32 元素，再调用真实 API。两个位置参数为原始 T1 与空输出目录；`--weights-dir`、`--assets-dir` 必填，`--threads` 默认 4。完成后另存 `run-api-invocation.json`，含调用方式、计时、CUDA 预初始化状态、四字节保留张量与脚本哈希。已有 CUDA 缓存的调用方保留可用 allocated/reserved 统计；关闭缓存的运行不能把统计接口返回的 0 写成零显存。非默认逻辑 `cuda:1` 的错误路径探针只验证指定设备统计，不代表整例重建。

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

`summarize_performance_runs.py` 只汇总本轮固定文件名的记录，不运行重建，也没有对应的官方命令。具名参数 `--reports` 指向 `performance_20260930`（完整阶段 JSON、比较目录、固定源码清单、实时资源哈希、外层退出码、GPU CSV 与 API 调用侧车）；`--output-dir` 必须不存在。它核对完成状态、退出码和提交后，写出 `current_full_runs_20260930.json` 与 `final_metric_consistency_20260930.json`：前者保存调用方式、两种耗时、慢阶段、线程和同时刻显存；后者保存 Dice、脑区偏差及优化前后距离。输入报告的 SHA-256 随结果保留。记录缺失、运行未完成、版本或资源哈希不符时抛异常，不能生成成功摘要。GPU 间隔以 CSV 的 UTC 时间计算，受整秒时间戳精度限制。

```bash
reports=/data/fnit-validation/performance_20260930  # 已收集的两例完整实测记录
output_dir=/data/fnit-validation/current-summary  # 必须不存在的摘要目录
python validation/recon_all/python_gpu_port/summarize_performance_runs.py \
  --reports "$reports" --output-dir "$output_dir"
```

## 官方复现与独立部署的证据范围

本轮优先复用既有同主机 N4 重放、固定输入 EM LTA 重放和表面指标三方测试；跨主机 N4 差异另列。官方参考整例尚未做完整重复运行，不能把全部 FNIT 差异归因于随机性。安装证据覆盖主页 Conda 环境、固定源码构建和资源清单；没有物理移除预装软件的干净机器，因此整例部署隔离仍未验证。PATH、ldd 和模块导入检查不能代替此验收。

## 原实现与参考文献

- [固定版本 FreeSurfer recon-all](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/scripts/recon-all)；参考生成命令为 `recon-all -i T1w.nii.gz -s subject -sd subjects -all -parallel -openmp 4 -itkthreads 1`。
- Fischl B. FreeSurfer. *NeuroImage*. 2012;62:774–781. [DOI](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
