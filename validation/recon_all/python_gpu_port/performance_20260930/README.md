# recon-all 性能配对：真实 T1，2026-09-30

本目录只记录本轮实际运行。两例去标识 T1、官方参考、权重和资产的 SHA-256 及机器来源见[前段清单](../volume_parity_20260930/provenance.json)。基线源码为 `b8cd17bb441307d88dda78ba8f56e77695b595a8`；阶段报告另保存实际载入源码与脚本的 SHA-256。`sub-01` 的完整基线在 gpucw1，`sub-02` 在 nodecw10；GPU1 与其他任务共享，因此阶段秒数是观察值，不代表独占机器的稳定加速。官方结果只在独立比较路径读取，未参与 FNIT 阶段计算。

| 同一 FNIT 真实输入的阶段 | CPU 基线 | 候选 | 输出回归 | 资源 |
| --- | ---: | ---: | --- | --- |
| `sub-01` 第二次归一化，gpucw1 | 85.17 s | CUDA 59.15 s | `brain.mgz` 0/16,777,216 个差异体素 | 进程采样峰值 1,247,805,440 字节 |
| `sub-02` 第二次归一化，CPU nodecw10、CUDA gpucw1 | 85.96 s | CUDA 71.09 s | `brain.mgz` 0/16,777,216 个差异体素 | 同上；跨主机耗时不能作为配对提速 |
| `sub-01` MNI 非线性链，gpucw1 CPU | 339.99 s | 只返回变换 343.26 s | 前向、逆向、检查图的 SHA-256 全相同 | 共享负载，单次未见加速 |
| `sub-02` MNI 非线性链，headcw CPU | 282.33 s | 只返回变换 253.95 s | 前向、逆向、检查图的 SHA-256 全相同 | 同主机单次观察 |

第二次归一化直接复用 `normalize_t1_aseg` 的现有 PyTorch CUDA 偏置场路径；控制点仍在 CPU。两次 GPU 运行均同步计时，包含加载、传输、写出；显存 CSV 每 2 秒采样一次父子进程同一时刻用量，不能当作连续峰值。[sub-01 报告](sub01/brain_second_gpu_report.json)、[sub-02 报告](sub02/brain_second_gpu_report.json)包含输入哈希、函数分步耗时及 PyTorch allocated/reserved。

MNI 链继续以 CPU 为生产默认。两例全输出的 SHA-256 原样保留，跳过 SynthMorph 原先计算但未使用的 `moved`、`fixed_moved`。独立设备试验在 gpucw1 的同一 `sub-01` 输入上记录 CPU 339.99 s、TF32 CUDA 149.64 s、诊断性 FP32 CUDA 161.73 s；TF32 的前向/逆向 warp P99 差达 0.075/1.108 mm，检查图与 CPU 相差 220,314 体素，采样显存峰值 20,308,819,968 字节。FP32 的对应 P99 降至 0.000015/0.000031 mm，检查图仍差 19 体素，采样峰值 12,996,050,944 字节。两种 GPU 模式均未接入生产。[FP32 同输入误差](sub01/mni_gpu_fp32_vs_cpu.json)和[TF32 同输入误差](sub01/mni_gpu_vs_connected_cpu.json)可复查；更接近 CPU 的 FP32 试验不能掩盖未通过的逐体素结果。

标准球面梯度平滑用当前真实 `sub-02` 网格做了线程实验：1024 轮、119,363 顶点的输出 SHA-256 在 4–128 线程间相同，时间随线程数上升从 3.20 s 降至 0.51 s。[微核记录](sub02/sphere_average_threads_head.json)。因此没有为了这个核降低 Numba 线程数；完整球面与球面配准仍是主要 CPU 热点，微核计时不能替代整例时间。

另用 `sub-01` 的同一 `lh.aparc` 输入对 GCSA 标注做了四线程 CPU/CUDA 配对：CPU [27.23 秒](sub01/annot_lh_aparc_cpu_threads4_report.json)、CUDA [28.53 秒](sub01/annot_lh_aparc_gpu_threads4_report.json)，输出 annotation SHA-256 相同。两者差距仅 1.30 秒，保留现有 CUDA 调度；完整整例中的单次标注耗时还受共享负载影响。[CUDA 进程采样](sub01/annot_lh_aparc_gpu_threads4_summary.txt)峰值为 2,006,974,464 字节。

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

第二个脚本读取同一 1 mm conform 网格，写出 uint8 `brain.mgz` 和步骤报告；三个输入几何不一致时失败。`benchmark_sphere_average_threads.py` 读取同序 inflated/smoothwm 网格，并输出各线程数的耗时和结果哈希。官方对应命令和原实现链接见[MNI 说明](../../../docs/recon_all/MNI_NONLINEAR_CHAIN.md)及[归一化说明](../../../docs/recon_all/NORMALIZATION.md)。完整整例运行及严格 138 项比较仍须从原始 T1 和空目录完成，不可将这里的冻结输入阶段测试充当整例验收。
