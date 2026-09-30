# recon-all 性能配对：真实 T1，2026-09-30

本目录只记录本轮实际运行。两例去标识 T1、官方参考、权重和资产的 SHA-256 及机器来源见[前段清单](../volume_parity_20260930/provenance.json)。基线源码为 `b8cd17bb441307d88dda78ba8f56e77695b595a8`；阶段报告另保存实际载入源码与脚本的 SHA-256。`sub-01` 的完整基线在 gpucw1，`sub-02` 在 nodecw10；GPU1 与其他任务共享，因此阶段秒数是观察值，不代表独占机器的稳定加速。官方结果只在独立比较路径读取，未参与 FNIT 阶段计算。

| 同一 FNIT 真实输入的阶段 | CPU 基线 | 候选 | 输出回归 | 资源 |
| --- | ---: | ---: | --- | --- |
| `sub-01` 第二次归一化，gpucw1 | 85.17 s | CUDA 59.15 s | `brain.mgz` 0/16,777,216 个差异体素 | 进程采样峰值 1,247,805,440 字节 |
| `sub-02` 第二次归一化，CPU nodecw10、CUDA gpucw1 | 85.96 s | CUDA 71.09 s | `brain.mgz` 0/16,777,216 个差异体素 | 同上；跨主机耗时不能作为配对提速 |
| `sub-01` MNI 非线性链，gpucw1 CPU | 339.99 s | 只返回变换 343.26 s | 前向、逆向、检查图的 SHA-256 全相同 | 共享负载，单次未见加速 |
| `sub-02` MNI 非线性链，headcw CPU | 282.33 s | 只返回变换 253.95 s | 前向、逆向、检查图的 SHA-256 全相同 | 同主机单次观察 |

第二次归一化直接复用 `normalize_t1_aseg` 的现有 PyTorch CUDA 偏置场路径；控制点仍在 CPU。两次 GPU 运行均同步计时，包含加载、传输、写出；显存 CSV 每 2 秒采样一次父子进程同一时刻用量，不能当作连续峰值。[sub-01 报告](sub01/brain_second_gpu_report.json)、[sub-02 报告](sub02/brain_second_gpu_report.json)包含输入哈希、函数分步耗时及 PyTorch allocated/reserved。

MNI 链复用现有 SynthMorph，不计算无人使用的 `moved`、`fixed_moved`。GPU recon-all 自 `279e09f` 起完整传递主设备并使用 FP32 CUDA；独立函数仍默认 CPU，调用结束恢复此前 TF32 设置。两例当前封装的同输入耗时为 [160.81 s](sub01/mni_gpu_final_gpucw1_report.json) / [156.04 s](sub02/mni_gpu_final_gpucw1_report.json)，四个子步骤计时包括模型加载、写出、warp 转换、求逆和检查图重采样。程序和函数源码哈希均保存在报告中；与先前 FP32 GPU 诊断的三个输出数组及仿射完全相同。

同一 `sub-01` 输入的 CPU 观察值为 339.99 s，TF32 CUDA 为 149.64 s。TF32 的前向/逆向 warp P99 差达 0.075/1.108 mm，检查图与 CPU 相差 220,314 体素，采样显存峰值 20,308,819,968 字节，未作为本阶段默认。FP32 两例相对 CPU 的前向/逆向 P99 差均约 0.000015/0.000031 mm；最大逆向差为 0.000473/0.000183 mm，最近邻检查图仍差 19/17 体素。这些尾差保留在[逐值诊断](sub01/mni_gpu_fp32_vs_cpu.json)和[sub-02 诊断](sub02/mni_gpu_fp32_vs_cpu.json)中。当前封装的父子进程采样峰值均为 12,996,050,944 字节，PyTorch allocated 约 9.27 GB、reserved 约 12.17 GB；未启用半精度。这里的阶段结果不代替完整重建的退化检查与总体指标评价。

当前 `279e09f` 在同一 sub-01 输入、同一 GPU 做了缓存开关配对：启用缓存 187.13 s，关闭缓存 165.38 s；模型与写出分别为 22.41/18.93 s，warp 求逆为 112.23/114.54 s。三份输出数组及仿射完全一致。[配对差异](sub01/mni_cache_pair_279_retry_comparison.json)、[缓存开启](sub01/mni_gpu_cached_279_retry_report.json)、[缓存关闭](sub01/mni_gpu_uncached_279_retry_report.json)保留源码、输入和程序哈希。该测试未采样进程级显存，只有缓存开启时可用的 PyTorch 峰值；共享 GPU 单次观测不足以归因整例耗时。此前首轮在计算前的 CUDA 同步报显存不足，没有产生阶段结果，见[失败日志](mni_cache_pair_279_initial_failure.log)。未据此更改默认缓存策略。

## 接入既有 PyTorch 表面指标

GPU 流程的 white/pial 面积、white/pial 曲率和厚度改为已有的 `area_map`、`curvature_map`、`thickness_map`；CPU 流程保留 Conda 源码构建路径。两个被试、双侧、五种图共 20 项同输入、同网格测试，官方与 Conda 数组全部一致，PyTorch 结果全部通过当前顶点图容差。厚度最大差不超过 0.000000954 mm，面积不超过 0.000000954 mm²；曲率最差图为 sub-02 右侧 pial，最大差 0.000232786，P99 为 0.000026345，超限顶点为零。详细参数、文件结构和逐项报告见[表面指标说明](../../../../docs/recon_all/SURFACE_METRICS.md)。

| 同输入厚度图 | Conda CPU | PyTorch CUDA |
| --- | ---: | ---: |
| sub-01 LH / RH | 20.23 / 20.70 s | 5.37 / 5.74 s |
| sub-02 LH / RH | 22.70 / 25.63 s | 6.54 / 6.94 s |

这些时间包含函数内读写并同步 GPU，未包含独立脚本导入和 CUDA 初始化；表面指标阶段的 PyTorch reserved 峰值约 0.57 GB，没有单独测量该阶段进程级峰值。单算子时间不解释为整例速度比。

## 原始 T1 整例配对

基线 `b8cd17b` 与中间版本 `bb28e0c` 均从空目录运行 66 阶段、生成 138 项并通过双侧网格检查。前者耗时 sub-01 GPU 5884.96 s、sub-02 CPU 6080.80 s；后者为 5951.02 / 6055.82 s，单次变化分别为慢 1.12% / 快 0.41%，未观察到稳定整例提速。中间版本只接入 CUDA 第二次归一化和跳过未使用的 SynthMorph 重采样；表面指标及 MNI CUDA 在后续版本接入。

中间版本 GPU 监控 CSV 保存了 2790 次采样，最大父子进程合计为 19,348,324,352 字节。该轮监控启动器在运行期间被更新，外层退出标志未保存；重建 JSON 和日志显示执行完成，但不补写不存在的退出码。最终版本使用固定命名、固定内容的启动器、源码归档和[逐文件源码清单](source_279e09f_manifest.json)，分别检验 CLI 与已初始化 CUDA 的 Python API。整例调用总时间与外层命令时间分别记录，后者还含导入、入口校验和初始化。

标准球面梯度平滑用当前真实 `sub-02` 网格做了线程实验：1024 轮、119,363 顶点的输出 SHA-256 在 4–128 线程间相同，时间随线程数上升从 3.20 s 降至 0.51 s。[微核记录](sub02/sphere_average_threads_head.json)。因此没有为了这个核降低 Numba 线程数；完整球面与球面配准仍是主要 CPU 热点，微核计时不能替代整例时间。

另用 `sub-01` 的同一 `lh.aparc` 输入对 GCSA 标注做了四线程 CPU/CUDA 配对：CPU [27.23 秒](sub01/annot_lh_aparc_cpu_threads4_report.json)、CUDA [28.53 秒](sub01/annot_lh_aparc_gpu_threads4_report.json)，输出 annotation SHA-256 相同。两者差距仅 1.30 秒，保留现有 CUDA 调度；完整整例中的单次标注耗时还受共享负载影响。[CUDA 进程采样](sub01/annot_lh_aparc_gpu_threads4_summary.txt)峰值为 2,006,974,464 字节。

两例中间版本与基线的严格诊断均为 138/138，通过七张分割图所有标签 Dice=1，68/45/70 区统计误差为零；同网格表面坐标差为零。[sub-01 完整配对](sub01/full_bb28e0c_pair/summary.json)、[sub-02 完整配对](sub02/full_bb28e0c_pair/summary.json)分别保留对官方参考的失败、分区 Dice、双向点到三角面距离、最差脑区和真实 T1 图示。严格复现、优化退化与整体等效分开报告，总体等效尚未判定。

`8957e07` 的两例整例在生成 filled 和 MNI 辅助图后因主调度漏传 device 退出，外层退出码均为 1；最后阶段 JSON 的 running 是未被捕获的状态，不能据此声称正在运行或完成。[失败证据](failed_8957_runs.json)保留其源码、20 个阶段和资源记录。`279e09f` 修正显式设备传递，并补齐该调度边界的失败记录；[4 项回归](wiring_regression_279e09f.json)通过，随后两例均从新空目录重启并完成。GPU 整例包含预初始化 CUDA 的 API，CPU 整例使用 CLI。

| 原始 T1、空目录连续整例 | b8cd 基线 | 279e09f | 单次观察变化 | 当前外层命令时间 |
| --- | ---: | ---: | ---: | ---: |
| sub-01，gpucw1 GPU | 5884.96 s | 6303.79 s | 慢 7.12% | 6316.84 s |
| sub-02，nodecw10 CPU | 6080.80 s | 6099.80 s | 慢 0.31% | 6107.05 s |

两例均为 66 阶段、138/138 项、网格通过、退出码 0。本轮没有观察到整例加速。GPU 较基线增加的主要阶段是第二次归一化 178.72 s、第一次归一化 87.68 s、最终表面合计 60.13 s，以及 MNI 33.35 s。第一次归一化的计算实现没有修改，独立 GPU 阶段与完整调用也处于不同共享负载；目前不能把全部耗时变化归因于算法或缓存。需在负载匹配的配对运行中确认性能，不能用独立算子快替代这张整例表。

当前 GPU 的采样最大父子进程合计为 19,411,238,912 字节（19.41 GB、18.08 GiB），共 2824 行；采样睡眠 2 s，CSV 的实际间隔中位数 2 s、最大 20 s，连续峰值未验证。当前 API 已预初始化 CUDA，仍显式关闭分配缓存；完整缓存开启的 API 尚未验证。两例同主机 PyTorch/原生线程选项为 4，Numba 默认 128/192，不能声称全进程仅四线程。

相对基线，GPU 严格诊断 133/138，CPU 138/138。GPU 五项差异为前向/逆向 MNI warp、最近邻检查图和双侧 w-g.pct；检查图有 19 个体素不同、最大 42 灰度级，其余分割图各标签 Dice 全为 1。双侧 w-g.pct 分别有 123/89 个顶点改变，最大 0.000683/0.000973 百分点，P99 均为 0；冻结 rawavg、orig、white、cortex，仅换入 GPU 厚度后完全重现候选图，见[厚度交换诊断](sub01/contrast_thickness_swap_279.json)。这是厚度尾差经采样坐标传播的实测证据，未修改严格门槛。两例 aparc/aseg/wmparc 逐区统计相对基线均无改变；对官方的历史精度差异仍保留，整体等效未判定。

复现脚本全部输入/输出、参数、具名示例、计时范围及隔离验证边界见[比较方法](../../../../docs/recon_all/BENCHMARK_METHODS.md)。

## 保留的 CPU 与 Conda 阶段

同一 sub-01 左侧 inflated/smoothwm 在 gpucw1、PyTorch 4 线程和 Numba 128 线程下做了 [cProfile 剖析](sub01/sphere_profile_279/profile.txt)：全阶段 541.31 s（含剖析开销），224 次 `initial_vertex_normals` 累计 337.03 s，其中已有法向数值内核约 33.72 s；构造面关联索引的 Python 循环占主要时间。距离 SSE 累计 98.52 s、梯度平滑 11.16 s。重跑的有序面和坐标与本轮完整流程完全一致，见[剖析回归](sub01/sphere_profile_279/report.json)。该单阶段剖析没有测 GPU 显存，也不是优化前后速度比。

复现脚本 [profile_sphere_279.py](profile_sphere_279.py) 绑定本次服务器路径和提交，读取两张 surface RAS mm 网格，写出隔离 sphere、每轮报告、`sphere.prof` 和累积前 30 项 `profile.txt`。输入须有相同有序面；参数显式使用 `inflated=...`、`smoothwm=...`、`output=...`、`finish_device="cpu"`。对应官方内部步骤为 `mris_sphere inflated sphere`；完整原实现和接口见[阶段索引](../../../../docs/recon_all/CONDA_CPP_STAGES.md)。缺失输入或重跑坐标不同会报错，不能将其标为回归通过。

| 当前阶段 | 核对到的现有实现与处理 |
| --- | --- |
| GCA EM 注册 | Python 诊断覆盖首轮搜索，尚不是完整替代；保留已验证的 Conda 程序。 |
| white / pial 放置 | 保留 Conda 固定源码路径；已有 Python pial 用于同输入诊断，现有证据没有显示更快。 |
| standard sphere / sphere.reg | 复用已验证的 Python/Numba；现有 device 选项仅影响末尾清理，不能将其描述为整段 GPU 优化。 |
| inverse GCAM | 现有 FNIT 代码为 CPU/Numba，没有成熟的 PyTorch GPU 全流程；本次实际 MNI 配对中原生求逆约 112–115 s。 |
| GCSA 标注 | 已在 CUDA；本次 CPU 27.23 s、CUDA 28.53 s，维持现有调度。 |

没有新增依赖，厚度、面积、曲率和绘图所需包均已在主页 Conda 环境中。安装产物及资源复用本轮清单，未重新编译未改动的原生程序；没有无预装软件的干净运行环境，独立部署整例仍未验证。

## 复现入口

三个脚本均在仓库根目录、已安装主页 Conda 环境和有权访问真实输入的机器上执行；每次输出请指定不存在的路径。`benchmark_mni_nonlinear.py` 的 `--source-subject` 为已生成 `orig.mgz`、裁剪 T1 和 `aff.lta` 的 FNIT 被试目录，`--output-subject` 是新的隔离诊断目录。`--weights`、`--assets` 已按固定清单校验，`--native-bin` 为当前 Conda `bin`；`--device` 选 CPU 或 CUDA，`--threads` 为 PyTorch 线程数。它写出前向/逆向/check 及含阶段、输入、程序 SHA-256 的 JSON；输入缺失或程序失败时抛出异常。

```bash
python validation/recon_all/python_gpu_port/benchmark_mni_nonlinear.py \
  --source-subject /data/subjects/sub01 \
  --output-subject /data/diagnostics/sub01-mni-cpu \
  --weights /data/fnit-weights --assets /data/fnit-assets \
  --native-bin /data/conda/envs/fnit/bin \
  --report /data/diagnostics/sub01-mni-cpu.json \
  --device cpu --threads 4 --code-commit "$(git rev-parse HEAD)"

python validation/recon_all/python_gpu_port/benchmark_second_normalize.py \
  --norm /data/subjects/sub01/mri/norm.mgz \
  --aseg /data/subjects/sub01/mri/aseg.presurf.mgz \
  --brainmask /data/subjects/sub01/mri/brainmask.mgz \
  --output /data/diagnostics/sub01-brain-cuda.mgz \
  --report /data/diagnostics/sub01-brain-cuda.json \
  --device cuda:0 --threads 4 --code-commit "$(git rev-parse HEAD)"
```

第二个脚本读取同一 1 mm conform 网格，写出 uint8 `brain.mgz` 和步骤报告；三个输入几何不一致时失败。`benchmark_sphere_average_threads.py` 读取同序 inflated/smoothwm 网格，并输出各线程数的耗时和结果哈希。官方对应命令和原实现链接见[MNI 说明](../../../../docs/recon_all/MNI_NONLINEAR_CHAIN.md)及[归一化说明](../../../../docs/recon_all/NORMALIZATION.md)。完整整例运行及严格 138 项比较仍须从原始 T1 和空目录完成，不可将这里的冻结输入阶段测试充当整例验收。
