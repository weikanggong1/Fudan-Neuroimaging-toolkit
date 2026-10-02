# recon-all 热点优化后的原始 T1 整例

本页记录实际计算提交 `c24852054f3321c1142b1ae88fa3d2bf68329bb3`（归档 SHA-256 `098ccb63a3931a4f749710fb76dd9beb4751f7b8b14c0f130fa922f9698b708e`）与优化前 `1b8c36d25a68e253a1e59b6d02114890afa467de` 的两例真实 T1。均从原始输入和空目录运行，不读取官方中间结果。状态截至2026-10-01：两例执行完成，各66阶段、138项齐全；官方及扩展质量对照分开报告。

## 完整命令墙钟

| 原始输入／调用 | 主机、设备、线程 | 优化前 | 优化后 | 墙钟减少 |
| --- | --- | ---: | ---: | ---: |
| sub-01，已初始化 CUDA 的 Python API | gpucw1，H100 GPU1，同一UUID，4线程 | 4972.667 s（82.878 min） | 4242.884 s（70.715 min） | 729.783 s，14.676%，1.172倍 |
| sub-02，CPU CLI | nodecw10，Xeon Gold6418H，4线程 | 5295.422 s（88.257 min） | 5274.884 s（87.915 min） | 20.539 s，0.388%，1.0039倍 |

GPU比较相同 `run_monitored.py` 的 `command_wall_seconds`；CPU比较CLI子进程的墙钟。均包含解释器启动、校验、模型、传输及文件读写，不混用API计时或额外监控启动范围。两例是不同被试和硬件，不能计算二者的GPU/CPU比；共享硬件单次观察不代表稳定吞吐。详细66阶段与内部步骤见 [GPU](whole/sub01/timing_pair_summary.json)、[CPU](whole/sub02/timing_pair_summary.json)，对应CSV为各目录的table.csv。

GPU球面配准：LH 478.815→133.654 s、RH 458.313→113.826 s，合计省689.649 s，占整例净节省94.50%。CPU配准合计省250.388 s，但未修改的原生final white/pial增加197.259 s，RH white.preaparc增加65.361 s，抵消收益。Numba初始容量旧192、新4，实际掩码均4；保留此环境差异，未把原生速度变化归因于随机性或新的Python算法。

GPU第一轮归一化185.426→219.296 s、第二轮174.512→181.624 s，整例中变慢。此前同输入CPU控制点优化结果仍保留，但不能代替本次GPU阶段计时。下一轮应对GPU归一化及原生表面放置分别做相同输入重复、拆分计算/传输/读写，核线程和CPU亲和性等实际设置。

## 优化是否引入退化

| 检查 | sub-01 GPU | sub-02 CPU |
| --- | --- | --- |
| 执行与输出 | 66阶段完成，138/138齐全 | 66阶段完成，138/138齐全 |
| 相对1b严格138项诊断 | 137/138 | 138/138 |
| 同序表面坐标与面 | 完全相同 | 严格比较通过 |
| 分割 | 7类图均无差异体素，每标签Dice=1 | 同左 |
| 68区厚度、面积、体积、平均曲率 | 逐区零差异 | 逐区零差异 |
| aseg、wmparc区域统计 | 零差异 | 零差异 |
| 整体指标等效 | not_assessed，未批准整体门槛 | 同左 |

[GPU完整配对](whole/sub01/paired/summary.json)、[CPU完整配对](whole/sub02/paired/summary.json)保留原阈值；先完成的[CPU快速严格诊断](whole/sub02_fast_partial/strict_vs_baseline.json)作为采集过程记录。唯一GPU严格失败为 `stats/lh.aparc.pial.stats` 的PialSurfArea表头88330.1→88330.2 mm²，34区×9列全部相同。保存area的float64合计仅差0.0000170283 mm²；生产原顺序float32累加差0.015625 mm²，六位有效数字格式跨边界后显示为0.1 mm²。[诊断](whole/sub01/pial_stats_diagnostic/说明.md)绑定影像、标签、源码和复算结果。面积/缓存/表头实现两版未改变，尚未做GPU重复确认最初面积尾差的执行来源；没有降低阈值或关闭GPU。

## 与官方结果及局部异常

两例官方严格诊断分别6/138和2/138。这些历史差异没有被本轮性能优化消除，不将高相关性作为整体等效依据。当前68区统计见[sub-01](whole/sub01/paired/region_vs_official.json)、[sub-02](whole/sub02/paired/region_vs_official.json)：

| 相对官方的68区指标 | sub-01 | sub-02 |
| --- | ---: | ---: |
| 厚度MAE／最大绝对误差（mm） | 0.041838／0.185 | 0.021691／0.121 |
| 面积MAE（mm²） | 28.7353 | 25.6912 |
| 面积绝对相对误差中位数／P90 | 1.3504%／6.2786% | 1.0600%／3.2010% |
| 体积MAE（mm³） | 112.4706 | 75.7500 |
| 体积绝对相对误差中位数／P90 | 2.1813%／5.9506% | 1.1834%／3.2637% |

[逐标签Dice](whole/sub01/paired/dice_vs_official.json)：aparc+aseg中位0.94929、最小0.85390；aparc.a2009s+aseg中位0.91131、最小0.01408，需保留异常脑区检查。sub-02[逐标签报告](whole/sub02/paired/dice_vs_official.json)中aparc+aseg中位0.96419、最小0.87336，a2009s中位0.93520、最小0.73798。官方与候选网格顶点数/面顺序不同，使用双向点到三角面距离：[表面报告](whole/sub01/paired/surface_vs_official.json)中white均值约0.071–0.086 mm、P99 0.410–0.505 mm；pial均值0.080–0.109 mm、P99 0.541–0.670 mm，局部最大仍有数毫米。sub-02[双向表面报告](whole/sub02/paired/surface_vs_official.json)的white均值0.035–0.041 mm、P99 0.220–0.285 mm；pial均值0.060–0.073 mm、P99 0.419–0.477 mm，最大3.185 mm。

[扩展质量](whole/sub01/quality/report.json)确认单连通、Euler=2、无非流形边及异常顶点link，sphere/sphere.reg无翻折。但white/pial存在LH147、RH197组proper transverse三角面穿越，包含皮层内穿越，不能用标准mesh_validation的passed替代这一结果。相对优化前表面几何零差异；这不是本轮新增的形状变化。sub-02[扩展质量](whole/sub02/quality/report.json)同样通过连通/非流形/link/球面方向检查，但white/pial存在LH92、RH119组proper transverse穿越，双方三角面均在皮层内的分别67、75组。CPU相对优化前表面几何完全相同；这两例的局部质量问题均未因本轮优化增加，也尚未解决。

![当前sub-01真实T1表面叠加](whole/sub01/figures/t1_surface_overlay.png)

[sub-01逐脑区误差](whole/sub01/figures/region_errors.png)、[异常分区边界](whole/sub01/figures/local_region_boundary.png)；sub-02的[T1表面叠加](whole/sub02/figures/t1_surface_overlay.png)、[脑区误差](whole/sub02/figures/region_errors.png)、[异常边界](whole/sub02/figures/local_region_boundary.png)同样已生成。各图像输入/脚本/图像SHA在同目录provenance.json中；[完成记录](whole/comparison_progress.json)和[两例产物SHA](whole/artifact_sha256.json)绑定实际比较版本与传输文件。

## 资源与安装范围

GPU父子进程同时NVML采样峰值14,508,097,536字节（14.508 GB），旧版16.119 GB。新3624次样本、请求1秒、最大间隔3.451秒、查询失败0；旧请求2秒。均非连续峰值证明；CPU例没有GPU监控，不记为零显存。TF32默认与既有FP32例外保持，未启用FP16/BF16，现有低显存allocator策略未删除。程序、权重、资产的当前复核见 [指纹](runtime_fingerprints_hotspot_candidate.json)，本轮未新增依赖。干净隔离安装验收仍未完成；参考程序重现性使用已有证据，本轮没有重新运行官方命令。

## 复现与更新记录

使用 [GPU配置](sub01_whole_retry1_config.json)、[CPU配置](sub02_whole_retry1_config.json)和 [启动器](../run_full_hotspots_launcher.py)：

```bash
# 配置中的input是原始T1；output与diagnostic_root须为新目录。
python validation/recon_all/python_gpu_port/run_full_hotspots_launcher.py \
  --config /bench/sub01_whole_retry1_config.json  # 声明Python、源码、资源、许可、设备和4线程
# 只读比较器等待各例完成后读取候选、基线和官方已有输出，不写回生产。
python validation/recon_all/python_gpu_port/collect_hotspot_whole_comparison.py \
  --config /bench/whole_comparison_retry1_config.json  # 修改为新的比较目录后使用
```

两个脚本属于benchmark控制，不是独立重建算法，没有官方等价命令；标准单T1流程对应官方 `recon-all -i T1w.nii.gz -s sub01 -all -openmp 4`，官方程序仅在隔离参考路径运行。完整Python调用和所有输入/输出结构见 [重建入口](../../../../docs/recon_all/README.md)，原实现和文献见 [热点审计](../../../../docs/recon_all/HOTSPOT_ACCELERATION_AUDIT.md)。

| 版本 | 记录 |
| --- | --- |
| 1b8c36d | 旧66阶段完整运行；已有厚度空间索引、统计缓存和精度/线程修复 |
| c248520 | 本轮有序CUDA配准平均、Numba平滑/BFS、CSR、remesh/quick sphere及控制点内核；59项本地回归与真实阶段测试 |
| c8e23bc | 来源、文档、CPU标准球面剖析和可复现整例启动记录，生产算法未变化 |

最优先的性能定位是冻结相同输入复测GPU归一化与原生white/pial：本次已测得它们抵消了部分收益，尚未定位到内部算子。标准球面距离SSE/静态CSR与remesh缩边是已有剖析支持的下一组自有CPU热点。

