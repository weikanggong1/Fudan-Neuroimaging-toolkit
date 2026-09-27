# ds004666：官方 MRtrix 随机波动与 FNIT 三种子对照

## 同输入范围

本阶段沿用 `sub-01/ses-2mm` 的同次 T1/DWI、TOPUP+EDDY 校正 DWI 与旋转后梯度，以及已完成的 FreeSurfer 8.2 解剖分割。固定 MRtrix 归一化 WM FOD、配准到 DWI 的 5TT/GMWMI、MRtrix FA 和同一 20 区 atlas；五项输入的 SHA-256 在 [seed 1](official_mrtrix_rng_variability/seed_1/input_sha256.txt) 与 [seed 2](official_mrtrix_rng_variability/seed_2/input_sha256.txt) 完全相同，也与 [seed 0 原始来源哈希摘录](official_mrtrix_rng_variability/seed_0_reference_input_sha256.txt)一致。此参考使用 10,000 次播种、FreeSurfer 5TT 但没有 FIRST，20 区 SynthSeg atlas；它是原 UKB 脚本的公开数据适配版，不能代表 1,000 万次播种、FIRST 和 UKB atlas 的逐字复跑。

官方 seed 0 是既有参考；[可复现脚本](../../../tools/benchmark_connectome_mrtrix_variability.sh)另以 `MRTRIX_RNG_SEED=1,2` 跑两次相同 `tckgen -algorithm iFOD2 -seed_gmwmi ... -act ... -seeds 10000 -select 0 -maxlength 250 -cutoff 0.1 -samples 3 -power 0.5 -nthreads 0`，再运行 `tcksift2`、`tckstats -dump`、`tcksample -precise -stat_tck mean` 和四次 `tck2connectome -symmetric -assignment_radial_search 4`。每次确切输入、输出、命令及时间分别保存在 [seed 1](official_mrtrix_rng_variability/seed_1/commands.txt) 和 [seed 2](official_mrtrix_rng_variability/seed_2/commands.txt) 的同名目录。三次 SIFT2 和矩阵命令均用 8 CPU 线程；此前 4 线程试跑已从本阶段的最终数值报告中排除。[4/8 线程受控记录](official_mrtrix_rng_variability/thread_control_4_vs_8.public.json)证明同一 seed 的 TCK 流线 payload 相同，四张矩阵的 190 条非对角边逐元素相同；仅加权矩阵对角线有不超过 5.12e-6 的数值差。

[MRtrix 的随机种子说明](https://mrtrix.readthedocs.io/en/latest/reference/environment_variables.html)要求固定 `MRTRIX_RNG_SEED`，并在需要重现多线程随机流程时禁用多线程。对此安装版还做了实测：再次以 seed 1 和 `-nthreads 0` 追踪，三次 TCK 的**流线二进制部分 SHA-256 完全相同**；完整文件因输出路径和 header 不同而 SHA 不同。[哈希验证](official_mrtrix_rng_variability/seed_1/rng_verification.json)给出旧条件两次追踪与最终 8 线程条件追踪的三个 payload 哈希。

| 实现与 seed | 接受流线 | atlas 双端分配 | 追踪秒 | SIFT2 秒 |
|---|---:|---:|---:|---:|
| MRtrix 0 | 2,758 | 2,740 | 16.90 | 24.64 |
| MRtrix 1 | 2,728 | 2,710 | 17.42 | 25.68 |
| MRtrix 2 | 2,727 | 2,710 | 15.18 | 25.28 |
| FNIT 0 | 2,827 | 2,827 | 266.83 全流程 | — |
| FNIT 1 | 2,828 | 2,828 | 291.19 全流程 | — |
| FNIT 2 | 2,832 | 2,832 | 332.74 全流程 | — |

FNIT 时间包含 DWI 读取、响应/FOD、追踪、SIFT2 和矩阵计算，计时在 CSV/TCK 落盘之前停止；MRtrix 时间只列复用既有 FOD/5TT/FA 后的追踪与 SIFT2，且并发负载不同，不能据此计算完整流程加速比。MRtrix 的双端分配数从对称 count 矩阵的含对角线上三角求和复核；FNIT 接受数见 [三个整例报告](current_seed_0/report.json)、[seed 1](current_seed_1/report.json)、[seed 2](current_seed_2/report.json)。最终 8 线程复跑的 seed 1 和 2 均正常退出 0；各自四张 20×20 矩阵与输出 SHA 保存在对应目录。

## 矩阵误差与随机基线

[分析脚本](../../../tools/benchmark_connectome_rng_envelope.py)读取官方 0/1/2 与 FNIT 0/1/2 的**同一批 24 张 CSV**；[机器报告](official_mrtrix_rng_variability/official_fnit_3x3.public.json)保存 24 个文件 SHA-256、官方内部 3 对、FNIT 内部 3 对和跨实现全部 9 对的逐对指标。已在本地逐一核对报告哈希与归档 CSV。Pearson 对 count/FBC 使用严格上三角全部 190 条边；对 mean length/FA 使用双方 count 均非零的边。下表的“全边相对 L1”固定为 `Σ|候选−参考| / Σ|参考|`，求和范围均为无对角线的 190 条边，与整例报告的 `relative_l1_upper` 同公式。报告另提供 `relative_l1_common_nonzero` 与旧比较脚本的 `nonzero_mean_scaled_mae`，这些量有不同分母，不能互换。

| 矩阵 | 官方内部 Pearson r | FNIT↔官方 Pearson r | 官方内部全边相对 L1 | FNIT↔官方全边相对 L1 |
|---|---:|---:|---:|---:|
| count | 0.9914–0.9937 | 0.9832–0.9916 | 0.2283–0.2583 | 0.2367–0.3100 |
| SIFT2 FBC | 0.9927–0.9933 | 0.9811–0.9904 | 0.2383–0.2982 | 0.2828–0.3541 |
| mean length | 0.7981–0.8787 | 0.8418–0.9129 | 0.6855–0.7491 | 0.6437–0.7220 |
| mean FA | 0.6970–0.8454 | 0.4335–0.6906 | 0.5335–0.5702 | 0.5938–0.6617 |

官方内部的连接支持 Dice 为 0.752–0.790，跨实现为 0.707–0.757。mean FA 在共同非零边的相对 L1 为官方内部 0.0777–0.1014、跨实现 0.1264–0.1785；跨实现 9 对的 mean FA 相关性均低于官方内部 3 对最低值，全边和共同边相对 L1 均高于官方内部最高值。count/FBC 的跨实现误差区间与官方随机波动有部分重叠，不能从单个 seed 断言追踪已精确等价。FNIT 内部 3 对的范围和全部 15 对原值见机器报告。

![三种固定种子的官方内部、FNIT 内部与跨实现全边相对 L1](official_mrtrix_rng_variability/official_fnit_rng_envelope.png)

图中每点是一对 20×20 矩阵，横线为组内中位数，蓝色虚线为官方内部三对中的最大误差。当前完整 connectome **尚未达到各输出都处于官方自身随机波动范围**：mean FA 保留可复现的跨实现差距；此基线供后续逐步修正 FA 采样、追踪路径及连接支持时复用。

## 复现

在具备所列远端固定输入的 `gpucw1` 上运行 `bash tools/benchmark_connectome_mrtrix_variability.sh "$BENCHMARK_ROOT"`（先把 `BENCHMARK_ROOT` 设为本机存放固定参考数据的目录，并在 `PATH` 中提供独立参考用的 MRtrix）。已有 `COMPLETE` 种子会跳过，未完成种子会拒绝覆盖。六组小矩阵已归档至本目录；在安装 NumPy、SciPy、Matplotlib 的环境中，从仓库根目录重算 JSON 和图：

```bash
base=validation/connectome/ds004666
python tools/benchmark_connectome_rng_envelope.py \
  --official "$base/corrected_mrtrix_fs5tt_act_adapted" \
    "$base/official_mrtrix_rng_variability/seed_1" \
    "$base/official_mrtrix_rng_variability/seed_2" \
  --fnit "$base/current_seed_0" "$base/current_seed_1" "$base/current_seed_2" \
  --output "$base/official_mrtrix_rng_variability/official_fnit_3x3.public.json" \
  --figure "$base/official_mrtrix_rng_variability/official_fnit_rng_envelope.png"
```
