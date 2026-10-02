# recon-all 性能配对：真实 T1，2026-09-30

本目录只记录本轮实际运行。两例去标识 T1、官方参考、权重和资产的 SHA-256 及机器来源见[前段清单](../volume_parity_20260930/provenance.json)。当前计算源码为 `e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68`，直接基线为 `279e09f0d2a166237871b3d683a6be75bd5e99b4`，最初基线为 `b8cd17bb441307d88dda78ba8f56e77695b595a8`；阶段报告另保存实际载入源码与脚本的 SHA-256。`sub-01` 的完整基线在 gpucw1，`sub-02` 在 nodecw10；GPU1 与其他任务共享，因此阶段秒数是观察值，不代表独占机器的稳定加速。官方结果只在独立比较路径读取，未参与 FNIT 阶段计算。

| 同一 FNIT 真实输入的阶段 | CPU 基线 | 候选 | 输出回归 | 资源 |
| --- | ---: | ---: | --- | --- |
| `sub-01` 第二次归一化，gpucw1 | 85.17 s | CUDA 59.15 s | `brain.mgz` 0/16,777,216 个差异体素 | 进程采样峰值 1,247,805,440 字节 |
| `sub-02` 第二次归一化，CPU nodecw10、CUDA gpucw1 | 85.96 s | CUDA 71.09 s | `brain.mgz` 0/16,777,216 个差异体素 | 同上；跨主机耗时不能作为配对提速 |
| `sub-01` MNI 非线性链，gpucw1 CPU | 339.99 s | 只返回变换 343.26 s | 前向、逆向、检查图的 SHA-256 全相同 | 共享负载，单次未见加速 |
| `sub-02` MNI 非线性链，headcw CPU | 282.33 s | 只返回变换 253.95 s | 前向、逆向、检查图的 SHA-256 全相同 | 同主机单次观察 |

第二次归一化直接复用 `normalize_t1_aseg` 的现有 PyTorch CUDA 偏置场路径；控制点仍在 CPU。两次 GPU 运行均同步计时，包含加载、传输、写出；显存 CSV 每 2 秒采样一次父子进程同一时刻用量，不能当作连续峰值。[sub-01 报告](sub01/brain_second_gpu_report.json)、[sub-02 报告](sub02/brain_second_gpu_report.json)包含输入哈希、函数分步耗时及 PyTorch allocated/reserved。

MNI 链复用现有 SynthMorph，不计算无人使用的 `moved`、`fixed_moved`。GPU recon-all 自 `279e09f` 起完整传递主设备并使用 FP32 CUDA；独立函数仍默认 CPU，调用结束恢复此前 TF32 设置。两例当前封装的同输入耗时为 [160.81 s](sub01/mni_gpu_final_gpucw1_report.json) / [156.04 s](sub02/mni_gpu_final_gpucw1_report.json)，四个子步骤计时包括模型加载、写出、warp 转换、求逆和检查图重采样。程序和函数源码哈希均保存在报告中；与先前 FP32 GPU 诊断的三个输出数组及仿射完全相同。

同一 `sub-01` 输入的 CPU 观察值为 339.99 s，TF32 CUDA 为 149.64 s。TF32 的前向/逆向 warp P99 差达 0.075/1.108 mm，检查图与 CPU 相差 220,314 体素，采样显存峰值 20,308,819,968 字节，未作为本阶段默认。FP32 两例相对 CPU 的前向/逆向 P99 差均约 0.000015/0.000031 mm；最大逆向差为 0.000473/0.000183 mm，最近邻检查图仍差 19/17 体素。这些尾差保留在[逐值诊断](sub01/mni_gpu_fp32_vs_cpu.json)和[sub-02 诊断](sub02/mni_gpu_fp32_vs_cpu.json)中。当前封装的父子进程采样峰值均为 12,996,050,944 字节，PyTorch allocated 约 9.27 GB、reserved 约 12.17 GB；未启用半精度。这里的阶段结果不代替完整重建的退化检查与总体指标评价。

直接基线 `279e09f` 在同一 sub-01 输入、同一 GPU 做了缓存开关配对：启用缓存 187.13 s，关闭缓存 165.38 s；模型与写出分别为 22.41/18.93 s，warp 求逆为 112.23/114.54 s。三份输出数组及仿射完全一致。[配对差异](sub01/mni_cache_pair_279_retry_comparison.json)、[缓存开启](sub01/mni_gpu_cached_279_retry_report.json)、[缓存关闭](sub01/mni_gpu_uncached_279_retry_report.json)保留源码、输入和程序哈希。该测试未采样进程级显存，只有缓存开启时可用的 PyTorch 峰值；共享 GPU 单次观测不足以归因整例耗时。此前首轮在计算前的 CUDA 同步报显存不足，没有产生阶段结果，见[失败日志](mni_cache_pair_279_initial_failure.log)。未据此更改默认缓存策略。

## 接入既有 PyTorch 表面指标

GPU 流程的 white/pial 面积、white/pial 曲率和厚度改为已有的 `area_map`、`curvature_map`、`thickness_map`；CPU 流程保留 Conda 源码构建路径。两个被试、双侧、五种图共 20 项同输入、同网格测试，官方与 Conda 数组全部一致，PyTorch 结果全部通过当前顶点图容差。厚度最大差不超过 0.000000954 mm，面积不超过 0.000000954 mm²；曲率最差图为 sub-02 右侧 pial，最大差 0.000232786，P99 为 0.000026345，超限顶点为零。详细参数、文件结构和逐项报告见[表面指标说明](../../../../docs/recon_all/SURFACE_METRICS.md)。

| 同输入厚度图 | Conda CPU | PyTorch CUDA |
| --- | ---: | ---: |
| sub-01 LH / RH | 20.23 / 20.70 s | 5.37 / 5.74 s |
| sub-02 LH / RH | 22.70 / 25.63 s | 6.54 / 6.94 s |

这些时间包含函数内读写并同步 GPU，未包含独立脚本导入和 CUDA 初始化；表面指标阶段的 PyTorch reserved 峰值约 0.57 GB，没有单独测量该阶段进程级峰值。单算子时间不解释为整例速度比。

## 原始 T1 整例配对

本次 `e036f57` 唯一的生产计算修改是稳定排序构建法向面关联索引；保留既有数值内核和每个顶点的累加顺序。先对八张冻结真实网格做逐元素回归，再跑冻结输入的完整双侧球面，最后两例从原始 T1、新空目录运行。没有跳过必要阶段、替换参考文件或降低门槛。

| 原始 T1、空目录连续整例 | 直接基线 279e09f | 当前 e036f57 | 本次观察变化 | 当前外层命令时间 |
| --- | ---: | ---: | ---: | ---: |
| sub-01，gpucw1 GPU | 6303.79 s | 7422.34 s | 慢 17.74% | 7437.40 s |
| sub-02，nodecw10 CPU | 6099.80 s | 5756.04 s | 快 5.64% | 5764.93 s |

两例均完成 66 阶段、138/138 项、双侧网格通过、退出码 0。函数时间包括加载、传输、计算和读写；外层时间另含导入、入口校验与初始化。GPU 使用已初始化 CUDA 的 Python API，CPU 使用 CLI；两者的 PyTorch/支持选项的原生程序为四线程，Numba 默认 128/192，没有全局限制 BLAS/OpenMP。固定源码归档 SHA-256 为 `3f3819aacac33ac63cd795628ef84260901a7e0ed247c3907c024ad272235d3b`，见[源码清单](source_e036f57_manifest.json)。共享硬件单次差值只表示本次观察。

最初 `b8cd17b` 的函数全程为 5884.96/6080.80 s；当前相对它为慢 26.12% / 快 5.34%。直接基线的[整例摘要](summary_279e09f/current_full_runs_20260930.json)及[最终指标](summary_279e09f/final_metric_consistency_20260930.json)继续承担配对参考，现版入口不再将它写成当前结果。`bb28e0c` 的第二次归一化与 transform-only 接线、`8957e07` 的设备漏传失败及 `279e09f` 的修复记录仍在本目录；这些记录按各自提交解释，不替代本次完成的整例。

### 阶段耗时变化

| 整例阶段 | sub-01 基线 → 当前 | sub-02 基线 → 当前 |
| --- | ---: | ---: |
| 左侧 surface | 1117.56 → 832.94 s | 1013.69 → 828.81 s |
| 右侧 surface | 935.27 → 735.07 s | 944.25 → 796.28 s |
| GCA EM 注册 | 652.85 → 654.43 s | 485.71 → 484.03 s |
| 左侧球面配准 | 485.53 → 500.97 s | 467.70 → 465.39 s |
| 右侧球面配准 | 462.94 → 463.18 s | 402.62 → 402.12 s |
| 左侧最终表面 | 436.50 → 410.24 s | 636.32 → 632.53 s |
| 右侧最终表面 | 408.67 → 385.27 s | 637.29 → 640.05 s |
| MNI 非线性链 | 366.03 → 1018.06 s | 318.64 → 317.73 s |
| WM 编辑 | 33.52 → 666.17 s | 28.65 → 28.67 s |
| wm_pretess | 5.53 → 286.62 s | 4.63 → 4.62 s |
| 第二次归一化 | 263.90 → 342.53 s | 85.41 → 89.42 s |

CPU 整例缩短 343.76 s；GPU 双侧 surface 合计节省 484.83 s，但前段新增耗时超过这一收益。WM 编辑、wm_pretess、MNI 不执行本次修改的关联索引，其时间异常不能由法向排序的算法变化解释。GPU 总时间减去全部阶段时间仅约 1.23 s，长等待已经包含在各阶段中，没有从总时间扣除。阶段计时包含末尾指定设备同步；现有记录不能再区分原生执行、读写、调度等待和同步各自的贡献，原因尚未确认。

### 同输入 WM 重放与启动诊断

整例结束后，使用本次 FNIT 的 entowm、aseg.presurf、wm.seg、brain，在同一 gpucw1 独立重跑相同 Conda WM 编辑程序两次：[报告](sub01/wm_edit_repeat_e036f57_retry/report.json)为 31.68/31.94 s，子进程 user+system 为 31.66/31.93 s，输出 uint8、256³、体素与仿射全部一致。计时包含子进程启动、原生加载/写出和日志，未建立父进程 CUDA 上下文；不能据此认定整例的 666 s 来自哪一种等待。首次诊断在原生程序结束后的 NumPy shape JSON 序列化失败，没有完成报告；[原失败脚本](benchmark_wm_edit_repeat_e036f57_initial_failed.py)和[日志](wm_edit_repeat_e036f57.log)独立保留，修正为 Python int 后从新的隔离目录运行，未补写首轮秒数。

再冻结本次 FNIT 的 orig、裁剪 T1、aff.lta，在相同 GPU UUID、四线程和缓存关闭策略下重放 MNI：[四子步骤报告](sub01/mni_e036f57_after_full_report.json)为 166.33 s，外层命令 170.90 s，退出码 0。模型与写出 26.66 s、warp 转换 20.69 s、变换求逆 110.46 s、检查图重采样 8.04 s；[三个输出回归](sub01/mni_e036f57_after_full_comparison.json)的数组、dtype、仿射和头均与本次整例一致，最大/P99 差为 0。该测试是新进程中的冻结阶段，没有复现原整例此前的 GPU 状态或共享负载，也没有另测进程显存；166 s 不能替代整例内的 1018 s。当前证据定位到异常耗时发生于连续整例，但尚不能确定同步、驱动、存储或调度中哪一项造成等待。

GPU 完成前的三个启动失败，以及线程、缓存和模块加载探针见[启动诊断](startup_e036f57/README.md)。六次设备映射探针中数字编号与 UUID 均指向 GPU 1，未发现错卡；本次成功整例仍显式固定 CUDA/NVML 为相同 UUID。没有凭某次探针成功宣称 CUDA OOM 原因已修复，也没有把失败尝试计为短整例。

### 显存与版本核验

当前 GPU 采样最大父子进程同时刻合计为 19,411,238,912 字节（19.41 GB、18.08 GiB），2611 行；采样循环睡眠 2 s，实际间隔中位数 2 s、最大 656 s，五个间隔超过 30 s。该诊断间隔计数不设显存门槛。采样空窗完整列于[整例摘要](../current_full_runs_20260930.json)，持续低于 20,000,000,000 字节未验证。默认关闭分配缓存，父进程 allocated/reserved 不可用；可得的 Talairach 子进程统计不代表整例峰值。缓存开启的完整 API 尚未验证。

[实时指纹](runtime_fingerprints_e036f57.json)重新读取两幅原始 T1、11 项权重、102 项资产、14 个 Conda 程序及 6 个官方参考程序，变化数 0；个人许可证没有读取或散列。逐主机版本、CPU、GPU 与候选进程线程环境见[gpucw1](hardware_gpucw1_e036f57.json)、[nodecw10](hardware_nodecw10_e036f57.json)。原生程序没有改动或重新编译。

### 严格复现、优化退化与最终指标

本次相对直接基线，两例严格比较均为 **138/138**；七张分割图全部标签 Dice=1；68 个 aparc 区面积/厚度/体积/曲率、45 个 aseg 结构及 70 个 wmparc 区统计差为 0；同网格 white/pial 坐标差为 0。双侧闭合、连通性、非流形、各自自相交以及 sphere/sphere.reg 向内/退化面检查均通过。white/pial 相互穿越尚未独立验收。[sub-01 完整配对](sub01/full_e036f57_pair/summary.json)、[sub-02 完整配对](sub02/full_e036f57_pair/summary.json)及各自 figures/ 保留实际 T1 叠加与异常脑区图。

对官方严格诊断仍为 **5/138、2/138**。主要最终指标未改变：厚度 MAE 为 0.037515/0.021691 mm；面积脑区绝对相对误差中位数为 1.439%/1.060%，灰质体积为 1.658%/1.183%；Destrieux 最低 Dice 仍为 0.072727/0.737984，局部表面最大距离为 6.339/3.185 mm。两套网格不同，不做同索引参考比较；平均相关性不代替局部检查。详细逐脑区、逐标签和双向点到三角面距离在[当前最终指标](../final_metric_consistency_20260930.json)及配对目录中。整体指标等效阈值未经确认，仍为 `not_assessed`。

直接基线 `279e09f` 相对最初 `b8cd17b` 的 GPU 五项严格差异仍完整保留：MNI warp 与检查图，以及双侧 w-g.pct。后者在冻结 rawavg、orig、white、cortex 后仅交换厚度即可重现，见[厚度传播诊断](sub01/contrast_thickness_swap_279.json)。本次没有改门槛、回退 GPU 或删除这些历史精度问题。

复现脚本输入/输出、参数、具名中文示例、计时范围及隔离验证边界见[比较方法](../../../../docs/recon_all/BENCHMARK_METHODS.md)。

## 保留的 CPU 与 Conda 阶段

同一 sub-01 左侧 inflated/smoothwm 的旧 [cProfile](sub01/sphere_profile_279/profile.txt)为 541.31 s，224 次法向累计 337.03 s；其中数值内核约 33.72 s，大部分时间花在 Python 索引循环。稳定排序后[新剖析](sub01/sphere_profile_e036f57/profile.txt)为 288.84 s、法向累计 65.93 s，最终坐标和有序面一致，观察阶段耗时减少 46.64%。同输入右侧 134 次更新、154.36 s，坐标与有序面也一致，未计算缺少同口径基线的右侧提速比。[函数完整说明、八张网格回归与具名示例](../../../../docs/recon_all/SURFACE_NORMALS.md)。距离 SSE 目前累计约 98.12 s，是下一项球面热点；保持现有算法，未仅凭剖析改写。

此前 `sub-02` 119,363 顶点、1024 轮梯度平滑的线程实验在 4–128 线程间输出 SHA-256 相同，观察值从 3.20 s 降到 0.51 s，[微核记录](sub02/sphere_average_threads_head.json)。没有为这个核降低 Numba 线程数，也没有把微核时间作为整例速度比。GCSA 同输入四线程 CPU 27.23 s、CUDA 28.53 s，annotation SHA-256 相同，保留现有 CUDA 调度：[CPU](sub01/annot_lh_aparc_cpu_threads4_report.json)、[CUDA](sub01/annot_lh_aparc_gpu_threads4_report.json)。

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

## 下一项验证

优先复测 GPU 连续整例的前段异常耗时，核对阶段调用与末尾 CUDA 同步各自的等待，并改善显存采样空窗。独立 WM/MNI 重放已经完成，重复它们不能代替连续整例状态。负载匹配后可在同一 gpucw1 用下列命令再次从原始 T1、新目录运行；本报告不含这次尚未执行的复测结果，不能用它代替本次 7422.34 s。

```bash
task_root=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929 # 本轮主页 Conda 与资源根目录
diagnostic_root="$task_root/volume_parity_20260930" # 固定源码及既有报告位置
python_bin="$task_root/fnit_main_env/bin/python"    # 相同 Conda Python
code_root="$diagnostic_root/performance_e036f57_code/src" # 本次已测计算源码
input_t1=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/examples/data/sub-01_T1w.nii.gz # 相同原始 T1
subject="$diagnostic_root/full_sub01_e036f57_uuid_repeat" # 必须不存在的输出目录
weights="$task_root/weights"                         # SHA-256 不变的权重
assets="$task_root/assets"                           # SHA-256 不变的资产
gpu_index=1                                           # 固定到原 GPU UUID
sample_csv="$diagnostic_root/full_sub01_e036f57_repeat_gpu.csv" # 新采样文件
run_log="$diagnostic_root/full_sub01_e036f57_repeat.log" # 新日志
run_summary="$diagnostic_root/full_sub01_e036f57_repeat_summary.txt" # 新退出状态与命令时间
export FS_LICENSE=/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt # 仅传已有授权许可证路径
bash validation/recon_all/python_gpu_port/performance_20260930/launch_api_gpu_e036f57_uuid.sh \
  "$python_bin" "$code_root" "$input_t1" "$subject" "$weights" "$assets" \
  "$gpu_index" "$sample_csv" "$run_log" "$run_summary"
```
