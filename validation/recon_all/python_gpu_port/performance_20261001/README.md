# 2026-10-01 recon-all 性能修复：阶段实测

本页汇总已完成的真实同输入测试。当前原始 T1 整例和受控旧版基线仍待完成，整例提速、138 项严格复现、输出完整性和网格质量均待汇总；整体指标等效为 `not_assessed`。没有采用尚未正式确认的整例等效阈值。完整数值和 SHA-256 见 [stage_summary.json](stage_summary.json)。

## 版本和运行范围

| 计算版本 | 绑定与实测范围 |
|---|---|
| 阶段 1 | `3d9856c…+stage1tar0444db72`：未提交修改的冻结快照，非干净 Git commit；tar SHA-256 `0444db72c248fac01bf9385c615da4f0e9fbade94ebbe2ccd2a70942f0c2180f`。两例统计、厚度和 SynthSeg 精度诊断。 |
| 冻结 `1b8c36d` | `1b8c36d25a68e253a1e59b6d02114890afa467de`；[部署源码清单](source_1b8c36d_manifest.json)。最终默认厚度的 sub-01 LH、后验缓冲区、线程测试及当前整例。 |
| 后续剖析修复 | 基于 1b8，profiling.py SHA-256 `7eed61ea…`；headcw 8 项 CPU/模拟 CUDA 检查通过，实际命令 1.645 s，非真实 GPU 重建。完整报告 SHA 和远端路径在汇总 JSON。 |
| `12a4834` 后续入口/worker 修复 | 当前说明对应的提交；公开入口失败状态 8 项本地测试通过。厚度 worker 源码 SHA-256 `ab019eb7…` 已排队，真实 GPU 尚未测量。当前整例仍使用冻结 1b。 |

GPU 阶段在 gpucw1 的 H100 PCIe（UUID `GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba`）运行，CPU 为 Xeon Gold 6430；CPU 线程测试在 nodecw10 的 Xeon Gold 6418H 运行。Torch 2.5.1、CUDA 11.8、cuDNN 90100，Torch 线程预算 4；使用 float32，未启用 FP16/BF16。TF32 默认保留，SynthSeg 卷积的已验证 FP32 例外单独记录。输入、权重、资产、Conda 构建程序及独立参考程序的大小与 SHA-256 均在 [运行清单](runtime_fingerprints_1b8c36d.json)；[GPU](hardware_gpucw1_1b8c36d.json)和 [CPU](hardware_nodecw10_1b8c36d.json)快照记录当时资源。

`git diff e036f57..3d9856c -- src` 为空：3d9856c 只增加此前验证记录，算法源码与 e036 相同。本轮受控旧版可作为算法基线；GPU wrapper 对齐 cuDNN FP32 后才能比较性能，不能将旧实际 TF32→FP32 的标签变化归于优化。

## 六项问题的处理结果

| 项目 | 当前处理与证据 |
|---|---|
| 1. SynthSeg 精度覆盖 | 设备选择与精度策略分开；两次真实前向都记录 cuDNN TF32=False、matmul TF32=True、float32、autocast 关闭。[实际前向](final_1b8c36d/buffer_uncached_api/actual-forward.json)。 |
| 2. 剖析和墙钟范围 | 剖析模式同步显式目标 GPU，并计入阶段写出；公开入口墙钟包含前置校验和报告。父/子 CPU 时间与同步等待分别记录；后续失败/allocator 状态修复单独绑定版本。[43 项测试及 19 子测试](unit_tests_1b8c36d.json)、[入口失败测试](public_wrapper_failure_tests_local.json)。 |
| 3. 多图谱统计重复 | 按表面路径和版本缓存几何、邻接、法线、面积、主曲率和 no-th3 基础量；每个图谱批量汇总后一次回传。[两例完整报告](diagnostics/stats_summary.json)。 |
| 4. 已有 PyTorch 指标 | 当前 CUDA 分支本来已默认使用自有厚度、面积、曲率；CPU 仍默认使用 Conda 源码构建指标。优化厚度的完整空间候选与编译可达性，不重复实现指标。[厚度报告](thickness_summary.json)。 |
| 5. 分配缓存与缓冲区 | 保留低显存 no-cache；后验缓冲区复用通过同 FP32 输出回归。缓存启用仅有单阶段观察，尚不能设为默认或宣布缓冲区提速。[保存文件审计](final_1b8c36d/buffer_stored_dtype_audit.json)。 |
| 6. 线程、设备与双侧 | Torch/Numba 作用域恢复线程预算；原生子进程仍须逐项核对。保留 surface.defects.mgz 的 LH 初始化/RH 合并顺序，未机械并行或迁移 CPU 阶段。[线程回归](cpu_final_1b8c36d/thread_ca_sub01/report.json)。 |

## 多图谱统计：两例、双侧、每轮 12 份输出

输入是 FNIT 已生成的固定 white/pial、顶点图、注释、cortex 和统计前置文件，非原始 T1 整例。white.preaparc、最终 white 与 pial 分开缓存；相同顶点数不代表可共用坐标。缓存键包含解析后的路径及文件版本，覆盖同一路径更新的失效检查。

| 真实数据 | 轮次 | 旧版 s | 缓存版 s | 阶段速度比 | 完整文本相同 |
|---|---:|---:|---:|---:|---:|
| [sub-01](diagnostics/stats/report.json) | 1 | 95.864 | 16.934 | 5.66× | 12/12 |
| [sub-01](diagnostics/stats/report.json) | 2 | 73.893 | 12.444 | 5.94× | 12/12 |
| [sub-02](diagnostics/sub02_stats/report.json) | 1 | 127.219 | 45.661 | 2.79× | 12/12 |
| [sub-02](diagnostics/sub02_stats/report.json) | 2 | 162.184 | 44.968 | 3.61× | 12/12 |

四轮共 48/48 份 `.stats` 文件逐字节相同，九个数值列最大差均为 0；门槛是完整文本一致。两例整个配对命令耗时分别 203.600/384.544 s，包含启动、输入哈希与报告；采样的父子进程同时占用峰值分别 2,027,945,984/2,036,334,592 字节。统计函数时间包括读写，并在边界同步 GPU。

**体积定义保持原有语义。** `-no-th3` 脑区体积按每个三角面平均厚度乘 white/pial 面积和，再向三个顶点累加；单角贡献为 `mean(thickness[face]) × (white_area + pial_area) / 6`，以 float64 汇总。TH3 顶点四面体体积图是另一输出，不能替代它。空间为 surface RAS（mm），厚度 mm、面积 mm²、体积 mm³。接口与具名示例见 [统计缓存说明](../../../../docs/recon_all/SURFACE_STATS_CACHE.md)。

## 厚度：完整候选索引与可达性检查

相同 white/pial 输入、相同有序网格。优化回归预设绝对容差 1e-6 mm、相对容差 0；四个半球每个两轮均有 0 个不同数值，最大/P99 差为 0。空间索引保留完整半径候选与 20-hop 扩展规则。阶段 1 测量对应 `fb3b6651…` 源码，不能当作最终 `c5d586b5…` 或后续 `ab019eb7…` 的时间。

| 输入 | 顶点/面 | dense 两轮 s | indexed 两轮 s | 平均耗时降幅 | 同输入 Conda 最大差 mm |
|---|---:|---:|---:|---:|---:|
| [sub-01 lh](diagnostics/thickness_lh/report.json) | 105539/211074 | 38.420/34.447 | 8.858/9.574 | 74.70% | 4.77e-07 |
| [sub-01 rh](diagnostics/thickness_rh/report.json) | 104864/209724 | 33.052/27.522 | 6.689/6.456 | 78.30% | 9.54e-07 |
| [sub-02 lh](diagnostics/sub02_thickness_lh/report.json) | 119363/238722 | 48.566/42.241 | 13.398/13.448 | 70.44% | 4.77e-07 |
| [sub-02 rh](diagnostics/sub02_thickness_rh/report.json) | 118303/236602 | 39.775/41.151 | 9.087/9.254 | 77.33% | 9.54e-07 |

冻结 1b 最终默认版另测 sub-01 LH：dense 45.589 s，indexed 13.774 s，仍逐值相同；与同输入 Conda map 最大差 4.77e-07 mm。[最终版报告](final_1b8c36d/thickness/report.json)。既有 Conda 比较门槛是 `0.005 mm + 0.001 × abs(reference)`，不是本轮新建的整例等效标准；该参考也不是新运行的系统安装官方程序。

阶段 1 配对进程采样峰值 1,304,428,544–1,365,245,952 字节，包含 dense 和 indexed；不是两个实现各自独立峰值。函数墙钟包括网格读写、传输及同步，排除脚本导入和 CUDA 上下文初始化；整例提速仍待测。

## SynthSeg：精度修复、缓冲区和缓存分别评价

本节只有 sub-01 冻结 conformed T1。旧实际 TF32 和修正 FP32 之间有 148 个标签体素变化，最小标签 Dice 0.9996683，TIV 增加 35.125 mm³；这是精度策略变化，不能混入同精度缓冲区回归。

| 版本/策略 | API 状态 | 函数及读写 s | 整个命令 s | 采样父子同时峰值（字节） |
|---|---|---:|---:|---:|
| [阶段1，旧实际 TF32/no-cache](diagnostics/old_effective_tf32/actual-forward.json) | CLI | 32.033 | 46.721 | 19,411,238,912 |
| [阶段1，FP32/no-cache](diagnostics/corrected_uncached_api/actual-forward.json) | 已初始化 CUDA API | 70.258 | 85.570 | 12,897,484,800 |
| [阶段1，FP32/cache](diagnostics/corrected_cached_cli/actual-forward.json) | CLI | 59.367 | 73.869 | 20,352,860,160 |
| [阶段1，FP32/cache](diagnostics/corrected_cached_api/actual-forward.json) | 已初始化 CUDA API | 58.168 | 73.905 | 20,352,860,160 |
| [1b，FP32/缓冲区/no-cache](final_1b8c36d/buffer_uncached_api/actual-forward.json) | 已初始化 CUDA API | 80.795 | 96.692 | 14,508,097,536 |
| [1b，FP32/缓冲区/cache](final_1b8c36d/buffer_cached_cli/actual-forward.json) | CLI | 64.597 | 71.573 | 18,138,267,648 |

1b 的两种缓冲区输出与修正 FP32 的原缓冲区结果：标签差 0、所有标签 Dice=1、几何差 0，保存的 MGZ 和 CSV 都逐字节相同。MGZ SHA-256 为 `50f9e58f…`，CSV 为 `aef61247…`，完整摘要见 [只读保存格式审计](final_1b8c36d/buffer_stored_dtype_audit.json)。旧 probe 的 `same_dtype=False` 比较了保存 MGH 的 `>f4` 与内存 NIfTI 的 int32；不是仅字节序差，也不证明输出文件 dtype 不同。保留原报告，另将保存文件两侧重新读入。

1b 两次实际前向均为 float32、matmul TF32=True、cuDNN TF32=False、autocast 关闭，原图与翻转后验共用缓冲区。当前单次缓冲区计时未比同策略阶段1更快，不能宣布缓冲区加速。阶段1开启缓存采样超过 20,000,000,000 字节；1b 单阶段 cache 的采样值较低，但 CLI/API 和上下文状态不同，未构成配对整例缓存实验。no-cache 的 allocated/reserved 不可用，记为 null，不记 0；1b cache 实测 allocated/reserved 为 15,621,712,896/17,574,133,760 字节。详见 [精度说明](../../../../docs/recon_all/SYNTHSEG_PRECISION.md)。

## 线程预算同输入回归

| 输入/阶段 | Numba 128 s | Numba 4 s | 输出比较 |
|---|---:|---:|---|
| [sub-01 CA normalize](cpu_final_1b8c36d/thread_ca_sub01/report.json) | 20.328 | 22.838 | 数值、dtype、几何/有序面相同 |
| [sub-02 CA normalize](cpu_final_1b8c36d/thread_ca_sub02/report.json) | 20.447 | 23.015 | 数值、dtype、几何/有序面相同 |
| [sub-01 LH standard sphere](cpu_final_1b8c36d/thread_sphere_sub01_lh/report.json) | 209.395 | 218.038 | 数值、dtype、几何/有序面相同 |

Torch 两侧均为 4；Numba 初始容量/掩码 192，测试退出恢复 192。时间包含首次 JIT 和 IO，固定先 128 后 4、未重复暖机，因此没有线程提速结论。两例 norm/ctrl_pts 全部差值 0；球面顶点数、有序面一致，坐标最大/P99 差 0。原生程序的线程数不能由 Python 的 4 推断；N4 和拓扑的固定预算保留。[线程说明](../../../../docs/recon_all/THREAD_BUDGET.md)。

## 旧整例变慢的原因：仍未确定

[历史整例重新分析](slowdown_analysis.json)将 sub-01 的主要增加定位到 MNI 非线性、WM edit 和 pretess；这不是当前版本的新整例结果。固定真实输入复测 WM edit 的两种 seed，在 GDB 下约 34.502/33.728 s，颜色表检查仅 0.101/0.033 s，体素、dtype 和几何相同，未支持“随机颜色表造成十分钟延迟”。旧监测缺少子进程 CPU/IO/等待分解，且有较大采样间隔、GPFS 时间与主机时间偏差，无法判定延迟原因；新剖析分别记录这些范围。

## 当前整例与受控基线（待完成）

| 原始输入/主机 | 新版 | 同策略旧版 | 整例速度比 |
|---|---|---|---|
| sub-01 / gpucw1 | [冻结1b GPU重试](run_gpu_retry1_1b8c36d.sh)：pending | [e036受控基线](run_gpu_control_after_candidate_20261001.sh)：pending | 待完成 |
| sub-02 / nodecw10 | [冻结1b CPU](run_cpu_final_1b8c36d.sh)：pending | [e036控制日志](cpu_control_e036f57.log)：pending | 待完成 |

两侧必须从原始 T1 与空输出目录开始，线程预算 4，GPU no-cache，并对旧版真实 SynthSeg posterior 作用域应用同一 cuDNN FP32 例外。控制仅在 benchmark wrapper 内对齐精度，未改旧算法/缓冲区，也不读官方输出。首次 GPU 尝试在 Talairach 子进程报告 CUDA OOM，exit=1，56.347 s；[失败日志](final_1b8c36d/full_sub01_monitor/command.log)与[监测](final_1b8c36d/full_sub01_monitor/monitor.json)保留，不作为完成或速度结果。

待补整例墙钟（校验、加载、传输、计算、读写、报告全含）、父子同时显存、各阶段占比、138 项严格复现、输出完整性、网格质量和最终分区 Dice/双向表面距离/厚度面积体积偏差及局部异常。整体等效尚未判定，不能从统计文本或局部相关性推断。没有预装脑影像软件的干净环境整例尚未验证；运行清单的哈希核对不代替隔离部署验证。

## 复现命令与输入输出

以下为复现入口，不表示本页新增执行。使用同一个声明的 Conda Python、对应冻结源码 PYTHONPATH、已授权的真实数据目录；下列路径按报告设置，不下载影像、权重或许可证。输出独立诊断目录，失败应查看异常与 JSON，不能生成 complete 占位结果。统计/厚度输入采用同一 surface RAS 顶点顺序；SynthSeg 输入为冻结 conformed MRI 网格，输出分割沿用该网格与整数标签语义。整例输出使用原始 T1，新目录不得混入检查点。

```python
from pathlib import Path
import os
import subprocess

runtime_root = Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929")  # 已安装的声明 Conda 运行目录
diagnostic_root = runtime_root / "volume_parity_20260930/fixes_20261001"  # 已有固定代码及报告
candidate_root = diagnostic_root / "code_1b8c36d"  # 使用1b时绑定实际版本，不冒充stage1
python_executable = runtime_root / "fnit_main_env/bin/python"  # 原始报告中的同一 Python
subject_directory = runtime_root / "volume_parity_20260930/full_sub01_e036f57_uuid"  # 固定的FNIT自产输入
environment = dict(os.environ)
environment.update(PYTHONPATH=str(candidate_root / "src"),
                   CUDA_VISIBLE_DEVICES="GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba",  # 显式物理GPU
                   PYTORCH_NO_CUDA_MEMORY_CACHING="1",  # 初始化CUDA前设置低显存策略
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   NUMBA_NUM_THREADS="4")  # 明确总Python预算；原生日志另行核对

subprocess.run([str(python_executable),
    str(candidate_root / "validation/recon_all/python_gpu_port/benchmark_surface_stats_cache.py"),
    "--subject", str(subject_directory),  # white/pial、注释、顶点图和统计前置文件
    "--baseline-source", str(runtime_root / "volume_parity_20260930/performance_e036f57_code/src"),  # 已有e036冻结源码
    "--baseline-commit", "e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68",  # 旧算法版本
    "--candidate-commit", "1b8c36d25a68e253a1e59b6d02114890afa467de",  # 本次复现使用的候选版本
    "--output", str(diagnostic_root / "reproduce_stats_1b"),  # 两轮stats和机器可读报告
    "--device", "cuda:0", "--threads", "4", "--repetitions", "2",  # 同输入交替运行
], env=environment, check=True)
```

其他入口沿用同一环境，参数均为显式具名选项：

- [厚度配对脚本](../benchmark_thickness_indexed.py)：`--white`、`--pial`（同有序网格，mm），`--baseline-source-file`（旧实现文件）、`--reference-map`（独立同输入 Conda 顶点图，仅诊断）、`--output`、`--code-commit`、`--device cuda:0`、`--threads 4`、`--repeats 2`；输出逐顶点厚度图、比较 JSON 和同步读写耗时。
- [SynthSeg脚本](../benchmark_synthseg_precision.py)：`--input`（conformed MRI）、`--weights`（已声明并校验权重）、`--baseline-seg`（同FP32诊断参考）、`--output`、`--cudnn-tf32 false`、`--allocator disabled`、`--initialized-api`、`--threads 4`、`--device cuda:0`、`--code-version`；输出分割、脑区CSV和两次真实前向设置。
- [线程脚本](../benchmark_thread_budget.py)：`--subject`（冻结FNIT目录）、`--assets`（GCA资产）、`--output`、`--stage ca` 或 `--stage sphere --hemi lh`、`--masks 128 4`、`--torch-threads 4`、`--code-commit`；输出两套 norm/ctrl 或 sphere 和严格比较 JSON。
- [当前整例GPU命令](run_gpu_retry1_1b8c36d.sh)、[CPU命令](run_cpu_final_1b8c36d.sh)及[受控旧版wrapper](../run_controlled_legacy.py)包含原始输入、权重、资产、设备、线程和输出路径。[监测器](../run_monitored.py)显式 `--gpu-uuid`、`--interval 2` 和 `--query-timeout 5`，以同次查询合计父子占用，不将不同时刻峰值相加。

真实精度与时间以原始 JSON 为准，表格仅显示三位小数。采样最大间隔约 2.25–2.75 s（首次失败整例 3.85 s），未保证捕获连续峰值；不混用 GB/GiB。无新增运行依赖，现有 Torch/NumPy/SciPy/Numba/nibabel 已在主页 Conda 安装路径；Conda 独立源码构建程序继续保留。
