# Connectome 随机重复对照工具

## 1. 功能和策略

本工具读取已有四矩阵，比较官方 MRtrix 自身重复与 FNIT—MRtrix 差异。至少提供三次官方重复，本轮计划使用五次，即种子 `0..4`、每次 **100000 次播种尝试**。`-select 0` 不固定接受轨迹数。

```mermaid
flowchart LR
    A[同输入 FOD / 5TT / GMWMI / FA / atlas] --> B[MRtrix 五次独立追踪]
    A --> C[FNIT 独立追踪]
    B --> D[每次 SIFT2 / 长度 / FA 只计算一次]
    D --> E[每套 atlas 的四矩阵]
    C --> F[已有 CLI 四矩阵]
    E --> G[官方互比的完整观察范围]
    F --> H[所有 FNIT 与官方比较]
    G --> I[误差不超过官方最大值；相似性不低于官方最小值]
    H --> I
```

误差低于官方最小值、相似性高于官方最大值均可通过；不用要求 FNIT 超过 MRtrix 重复性。原最小—最大值、每个官方互比、每个交叉比较、FNIT 自身互比仍全部保留。至少两次 FNIT 时，单列 `fnit_reproducibility_status` 按同一门槛验收其自身重复；只有一次时该项为 `not_assessed`。无法定义的指标为 `null/not_assessed`。这是有限次数的实际观察范围，**不是总体置信区间**；门槛在读取本轮真实比较结果之前确定，不根据结果事后调节。矩阵验收不代表两个 RNG 相同，也不替代解剖学和完整 pipeline 验证。

## 2. Python、输入和输出

```python
from pathlib import Path
from tools.benchmark_connectome_seven_atlas_envelope import compare

# 官方五次独立输出，每个目录包含 atlases/<名称>/{count,sift2_fbc,mean_length,mean_fa}.csv。
official_repeat_directories = [Path("reference") / f"seed-{seed}" for seed in range(5)]
# FNIT 当前 CLI 输出，无需重跑矩阵；也可提供多个 FNIT 重复目录。
fnit_connectome_directories = [Path("fnit_connectome")]
repeat_report = compare(
    official_repeat_directories,
    fnit_connectome_directories,
    dataset="OpenNeuro ds001226 sub-CON03 ses-preop",  # 实际数据标签
    n_seeds=100000,                                    # 播种尝试数
    official_seeds=[0, 1, 2, 3, 4],                    # 对应目录的真实种子
    fnit_seeds=[0],                                    # 对应 FNIT 目录
    atlases=None,                                      # 默认检查全部输出模板；可显式选择子集
)
```

输入支持以下格式，不改变节点编号、已有边的矩阵值或统计量：

- 当前 FNIT：`atlases/<名称>/connectome_count.csv`、`connectome_sift2_fbc.csv`、`connectome_mean_length.csv`、`connectome_mean_fa.csv`；同目录 `atlas_dwi.nii.gz`、`nodes.tsv`、`region_labels.csv` 提供节点和来源信息。
- 官方：同样的 profile 目录，四个 CSV 可以没有 `connectome_` 前缀；`atlas.sha256`、`nodes.tsv`、`nodes.txt` 提供来源信息。
- 旧七 atlas benchmark：`report.json` 的 `profiles` 表和 `<profile>.npz`，保留兼容。
- 单 atlas 工具还支持 `candidate_` 前缀和官方 `matrices/` 子目录。

`nodes.tsv` 的列为 `index / original_label / hemisphere / name`，必须按 `1..K` 排列并与矩阵维度一致。有节点表或 atlas SHA 时校验同一节点顺序和来源；缺少来源信息明确标为 `not_fully_available`，不伪称已验证。显式选择模板子集时在报告保留实际名称，默认要求所有重复包含相同完整模板集合。

比较工具拒绝重复的实际目录（含符号链接别名）和提供的重复种子标签。不同目录本身不能证明独立运行；独立性为调用者声明，完整官方执行由参考工具的命令和 manifest 提供证据。如果 atlas 的最后几个 canonical 节点在 DWI 网格中完全缺失，MRtrix 输出维度可小于 `K`。参考工具在明确节点表与实际标签范围验证后，保留原 CSV，另写 canonical CSV，只为这些确实不存在的末尾节点补零行/列；记录原维度、缺失节点和两种文件 SHA，不改变 atlas 或现有边的值。

输出 JSON 包括：`dataset`、种子、四矩阵摘要、节点/atlas 来源、`pairwise`、`ranges` 和 `matrix_envelope_status`。`ranges` 同时保留 `official_min_max`、`official_values`、`fnit_vs_official`、`inside_count`（描述原双侧范围）、`comparison_accepted`（真正的单侧验收）。对角线单独报告；主要边集为严格上三角。count/FBC 使用全部非对角边；FA/长度主要误差只使用两份 count 均非零的边。原工具的归一化定义保留，单 atlas 的全上三角 FA/长度 L1 另外作为包含支持集差异的指标记录。

## 3. 命令行

```bash
python tools/benchmark_connectome_seven_atlas_envelope.py \
  --official reference/seed-0 reference/seed-1 reference/seed-2 reference/seed-3 reference/seed-4 \
  --fnit fnit_connectome \
  --dataset 'OpenNeuro ds001226 sub-CON03 ses-preop' \
  --n-seeds 100000 --official-seeds 0 1 2 3 4 --fnit-seeds 0 \
  --output repeat_envelope.json --figure repeat_envelope.png
```

单 atlas 使用 `benchmark_connectome_tracking_100k_envelope.py`，将路径指向对应模板目录。多 atlas 可加 `--atlas fs-aparc aparc+tian-s1` 明确选择子集。`--figure` 为可选的矩阵/误差图，不是新的脑影像 benchmark；标题和点数随实际数据、节点数和重复次数变化。`--official-seeds`、`--fnit-seeds` 可省略，此时只标记运行序号，不猜测种子。

本轮 CON03 的独立参考计划（执行前需确认真实检查点已生成）：

```bash
fnit_repository_dir=/path/to/Fudan-Neuroimaging-toolkit  # 本轮实际代码检出/工具部署目录
benchmark_root=/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002
benchmark_python=/cwStorage/home/gongwk/Notebook_code/fnit_conda_env_956b1a9/bin/python
mrtrix_binary_directory=/public/software/apps/MRtrix3/3.0.3/bin
checkpoint_directory="$benchmark_root/root_diagnostic_CON03_v1/checkpoints"
fnit_output_directory="$benchmark_root/root_diagnostic_CON03_v1/connectome"
official_output_directory="$benchmark_root/reference_CON03_five_repeats_v1"  # 必须是新目录

# 在 nodecw10 使用独立官方软件；默认 dry-run 不运行命令、不创建输出。
"$benchmark_python" "$fnit_repository_dir/tools/reference/benchmark_connectome_repeats_official.py" \
  --mrtrix-bin "$mrtrix_binary_directory" \
  --checkpoint-dir "$checkpoint_directory" \
  --fnit-dir "$fnit_output_directory" \
  --output-dir "$official_output_directory" \
  --dataset 'OpenNeuro ds001226 sub-CON03 ses-preop' \
  --n-seeds 100000 --seeds 0 1 2 3 4 --downstream-threads 8 \
  --atlas fs-aparc aparc+tian-s1 aparc.a2009s+tian-s1 glasser+tian-s1 glasser+tian-s4 \
          schaefer200+tian-s1 schaefer500+tian-s4 schaefer1000+tian-s4 \
  --dry-run
# 核对计划后去掉 --dry-run 执行；该脚本不进入 FNIT 运行时。
```

参考工具只允许新输出目录；失败留下明确日志和失败命令。参数 `--tracking-inputs` 可覆盖默认 `checkpoint-dir/tracking_inputs.pt`；`--seeds` 默认 `0..4`、至少三次且不能重复；`--downstream-threads` 默认 8；`--atlas` 默认全部 CLI 模板。实际绑定的 `tracking_kwargs` 决定步长、最小/最大长度、转角、cutoff、power，并验证 `n_seeds` 与检查点一致。保存版本、可执行文件和输入 SHA、逐命令 argv、时间/RSS 和输出检查。主入口时间包含检查与导出，不包含 Python 导入启动。

## 4. 对应官方命令和几何

实际 nodecw10 参考程序为 MRtrix **`3.0.3-103-g026e850d`**；每次运行仍重新记录 `tckgen -version` 和程序 SHA。本轮 FOD 体素为 2.5 mm、默认步长 1.25 mm、ACT 默认最小长度 5 mm；脚本从实际 PT 仿射计算有效值，不把这些数值写死。

```bash
MRTRIX_RNG_SEED=0 tckgen -algorithm iFOD2 \
  -seed_gmwmi gmwmi.mif -act five_tissue_act.mif \
  -seeds 100000 -select 0 -maxlength 250 -minlength 5 -step 1.25 \
  -angle 45 -cutoff 0.1 -samples 3 -power 0.5 -nthreads 0 \
  wm_fod.mif tracks.tck
tcksift2 tracks.tck wm_fod.mif weights.txt -act five_tissue_sift2.mif -nthreads 8
tckstats -dump lengths.txt tracks.tck -nthreads 8
tcksample -precise -stat_tck mean tracks.tck fa.mif mean_fa.txt -nthreads 8
tck2connectome -symmetric -assignment_radial_search 4 tracks.tck atlas.nii.gz count.csv -nthreads 8
tck2connectome -symmetric -assignment_radial_search 4 -tck_weights_in weights.txt \
  tracks.tck atlas.nii.gz sift2_fbc.csv -nthreads 8
tck2connectome -symmetric -assignment_radial_search 4 -tck_weights_in weights.txt \
  -scale_file lengths.txt -stat_edge mean tracks.tck atlas.nii.gz mean_length.csv -nthreads 8
tck2connectome -symmetric -assignment_radial_search 4 -tck_weights_in weights.txt \
  -scale_file mean_fa.txt -stat_edge mean tracks.tck atlas.nii.gz mean_fa.csv -nthreads 8
```

保留最大 1000 次方向尝试与 rejection trials；`samples=3` 对应 downsample factor 2。显式 `cutoff=.1`、`power=.5` 覆盖 ACT/算法默认值，不额外乘 0.5。不添加 backtrack、crop-at-GMWMI、mask 或 `zero_diagonal`。追踪 `-nthreads 0` 保留参考 RNG 的禁多线程语义，下游 8 线程不影响播种顺序。相同整数种子不能产生与 PyTorch 逐条相同的轨迹。

精确输入来自 `tracking_inputs.pt`，不用将诊断 NIfTI-1 的舍入仿射说成精确内部几何。参考工具导出 NIfTI-2，逐值验证 FOD/5TT/GMWMI 数组和 Float64 仿射。ACT 5TT 保留原 header spacing；SIFT2 5TT 和 GMWMI 保留调用时原始仿射。FA 用检查点数组和实际 FOD 几何，atlas 用 CLI 标签数组和同一 DWI 几何。原 CLI atlas 摘要标识源文件；导出文件摘要/仿射差异单独记录。MRtrix 转换后的 transform/spacing 同样留在报告，因此不会把来源相同误称为参考文件逐字节相同。

## 5. 验证和实测状态

本次新增小数组测试只覆盖工具格式、错误处理、边集定义、单侧门槛、五次重复、几何导出和实际命令参数。它们不构成科学精度/耗时 benchmark。本轮十例整例和五次官方重复的真实结果由总控制在运行完成后写入 cohort 报告；本工具不预填通过结果，不沿用 ds004666 旧数值作为新数据结论。

2026-10-02 在实际 nodecw10 只读执行 `tckgen -version/-help`，确认上述参考版本、步长、ACT 最小长度、samples/power/trials/downsample 和 `-nthreads 0` 参数语义。CPU 工具回归与原矩阵比较测试 **19 passed、1 skipped，3.17 秒**；跳过项是可选绘图，需要项目已声明的 matplotlib。本轮共享测试环境尚缺此依赖，不改动正在运行整例所用的环境。以上耗时是工具测试时间，不是追踪或 pipeline benchmark。

## 6. 更新记录

- 2026-10-02：三次固定输入改为至少三次、默认计划五次；支持任意 FNIT 重复和当前 CLI 多 atlas CSV；去掉旧数据/20 节点硬编码；修正单侧验收，新增空值和对角报告；独立官方参考共用每次追踪的后处理，保留精确检查点几何和完整来源。
- 旧报告继续绑定旧代码/数据，历史 `inside_count` 仍表示双侧观察范围；新版 `comparison_accepted` 才用于本轮验收。没有重新运行旧结果或改写旧结论。

## 7. 参考和源码

- [MRtrix tckgen](https://mrtrix.readthedocs.io/en/latest/reference/commands/tckgen.html)、[tck2connectome](https://mrtrix.readthedocs.io/en/latest/reference/commands/tck2connectome.html)、[tcksift2](https://mrtrix.readthedocs.io/en/latest/reference/commands/tcksift2.html)、[MRtrix3 源码](https://github.com/MRtrix3/mrtrix3)。实际默认值同时核查独立参考源码的 `tracking/tractography.h`、`tracking/shared.cpp`、`algorithms/iFOD2.h`、`seeding/seeding.cpp`，并以运行程序版本/SHA 为准。
- Tournier et al. MRtrix3. *NeuroImage* (2019). [doi:10.1016/j.neuroimage.2019.116137](https://doi.org/10.1016/j.neuroimage.2019.116137)。Smith et al. ACT. *NeuroImage* (2012). [doi:10.1016/j.neuroimage.2012.06.005](https://doi.org/10.1016/j.neuroimage.2012.06.005)。Smith et al. SIFT2. *NeuroImage* (2015). [doi:10.1016/j.neuroimage.2015.06.092](https://doi.org/10.1016/j.neuroimage.2015.06.092)。
- [OpenNeuro ds001226 原仓库](https://github.com/OpenNeuroDatasets/ds001226)，本轮只使用新下载的公开原始 DWI/T1w；数据下载和输入摘要另见十例 cohort manifest。
