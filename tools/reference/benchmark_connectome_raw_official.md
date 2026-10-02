# 同一原始病例的独立官方 connectome 对照

## 1. 功能与流程

`benchmark_connectome_raw_official.py` 从本轮官方自产 DWI 和解剖合同继续执行 MRtrix 追踪、SIFT2、FA/长度采样及八套 atlas 矩阵。它是独立 benchmark 工具，不进入 FNIT 运行时，也不读取 FNIT 的 FOD、FA、atlas 或 PT 检查点。

```mermaid
flowchart TD
    R[冻结公开 raw BIDS 清单：T1 / AP / PA / 梯度 / JSON] --> P[官方 TOPUP + EDDY]
    R --> FS[本轮真实官方 recon-all]
    P --> D[官方 mask / response / MSMT-CSD / mtnormalise / FA]
    FS --> A[官方 5TT / GMWMI / rigid registration / 多 atlas]
    D --> A
    D --> V[校验已完成合同、原始 SHA、自产文件与真实 header]
    A --> V
    V --> T[官方 iFOD2 + ACT：seed0..4 / 每轮100k尝试]
    T --> S[SIFT2 + precise FA + 长度]
    S --> M[八套 atlas 各输出 count / FBC / length / FA]
    M --> C[同 raw 与同节点语义：官方10个重复组合对 FNIT 已有矩阵]
```

本轮显式选择官方 modeling 与 `official_anatomy_raw10_v1` 的实际 `completed` 合同；上游路径以本轮交接为准，不能自动回退。CON01/03 原始选帧匹配；后八例按正式 FNIT 已冻结的原始 B0 frame index 作为共同输入参数，从原始文件由 `fslroi/fslmerge` 重建。这一输入适配及逐位抽帧检查保留在 producer 报告；官方仍自产 TOPUP field、mask、EDDY、FOD、FA 与 atlas，不称两条链独立选择 B0。早先失败或尚未完成的目录保留原状，不回退消费。若上游明确启用真实 `eddy_cpu` fallback，必须显式绑定其新完成合同；报告保留 CPU solver、host 与实际时间，GPU solver 的时间和 UUID 分开记录。

## 2. Python、输入格式与输出

官方执行器支持 `main(argument_list)`，同原始病例矩阵工具支持：

```python
from pathlib import Path
from tools.benchmark_connectome_raw_cohort_envelope import compare

# 路径对应已经完成的真实输出；本函数只读矩阵和来源，不重新计算影像。
comparison_report = compare(
    official_root=Path("official_tracking/sub-CON03"),
    fnit_dirs=[Path("fnit/sub-CON03/connectome")],
    fnit_reports=[Path("fnit/sub-CON03/gpu_report.json")],
    manifest_path=Path("frozen/input_manifest.json"),
    manifest_sha256="原始清单的完整64位SHA256",
    case_id="sub-CON03",
    fnit_seeds=[0],
)
```

官方执行输入：

- **原始清单**：JSON 的 `cases[].case_id/subject/session/input_files[]`；每个原始 T1、AP/PA NIfTI、bval、bvec、JSON 均有真实路径与 SHA。数据集、snapshot 与许可由清单记录。原始文件在运行前后重新校验。
- **官方 DWI consumer contract**：`scope=official_self_produced_raw_dwi_chain`，同 case、`state=completed`。提供 `files.wm_fod_normalized`、`files.FA` 及原始官方 rawprep 报告路径与 SHA。每项文件含 path、size_bytes、sha256 和实际 grid/readback。
- **官方解剖 consumer contract**：`scope=official_self_produced_fresh_fs_anatomy_and_raw_dwi_atlases`，同 case、`state=completed`。提供 `five_tissue_dwi_world`、`gmwmi_dwi_world`、八 atlas 的 NIfTI 与 nodes.tsv，并绑定原始 T1、真实 fresh FS、prepared/complete 报告及所消费 DWI 合同 SHA。
- **已经审计的固定输入官方 manifest**：只用于核对本轮 MRtrix 二进制及命令规划器身份；其中的 FNIT 图像不作为新官方链输入。

5TT 是自身 native 结构格点上的五通道 NIfTI，GMWMI 保持其格点；world affine 映射到 DWI 世界坐标。FOD 为 `[X,Y,Z,45]`、lmax8；FA 为三维标量图；atlas 为三维非负整数标签，节点连续编号 1…K，由 `nodes.tsv` 的 index/original_label/hemisphere/name 四列定义。工具记录原始非有限值，完整消费原文件。

输出：

- `inputs/`：指向官方自产原图像的软链接，没有重导出或变更图像数值/几何。
- `seed-0/` … `seed-4/`：真实 `tracks.tck`、SIFT2 权重、原始长度与 precise FA 文本；`atlases/<profile>/` 各有四 CSV、原节点表和来源摘要。
- `reference_manifest.json`：全部计划与实际命令、returncode、log/time 文件、每项 reader 几何、原始/自产来源 SHA、每轮接受数及标量非有限值 QC。耗时只包括本工具的追踪/下游阶段，上游 rawprep、recon-all、建模与解剖分别保留其实际时间。
- 同病例比较 JSON：官方自身全部组合、跨软件全部组合、FNIT 自身组合、完整范围与单侧门槛、节点语义和两条链各自 atlas SHA。

正式 FNIT CLI 当前保存四矩阵和影像，不保存 TCK/FOD。因此十例正式结果只做已有矩阵对照；人口分布诊断来自实际保留 TCK 的 pilot 五种子，分开报告。

## 3. 命令行及参数

```bash
# 所有路径来自本轮实际产物。old reference只用于二进制/helper身份校验。
raw_manifest=/path/to/frozen/input_manifest.json
raw_manifest_sha256=完整64位SHA256
anatomy_contract=/path/to/official_anatomy_raw10_v1/sub-CON03/consumer_contract.json
dwi_contract=/path/to/official_modeling_raw10_v2/sub-CON03/consumer_contract.json
verified_reference_manifest=/path/to/audited_fixed_reference/reference_manifest.json
verified_reference_manifest_sha256=完整64位SHA256
mrtrix_binary_directory=/path/to/pinned/MRtrix3/bin
new_output_directory=/path/to/new_official_tracking/sub-CON03

python tools/reference/benchmark_connectome_raw_official.py \
    --raw-manifest "$raw_manifest" --raw-manifest-sha256 "$raw_manifest_sha256" \
    --anatomy-contract "$anatomy_contract" --dwi-contract "$dwi_contract" \
    --verified-reference-manifest "$verified_reference_manifest" \
    --verified-reference-manifest-sha256 "$verified_reference_manifest_sha256" \
    --case-id sub-CON03 --mrtrix-bin "$mrtrix_binary_directory" \
    --n-seeds 100000 --seeds 0 1 2 3 4 --downstream-threads 8 \
    --output-dir "$new_output_directory"
```

- `--raw-manifest` / `--raw-manifest-sha256`：固定原始清单及其完整 SHA。
- `--anatomy-contract` / `--dwi-contract`：真实完成合同，解剖合同必须绑定这里传入的同一 DWI 合同。
- `--verified-reference-manifest` / 对应 SHA：已经成功审计的官方版本身份；所有七个二进制、helper SHA 和实际版本必须一致。
- `--case-id`：原始清单中的唯一病例 ID。
- `--mrtrix-bin`：已固定官方二进制目录。
- `--n-seeds`：每轮尝试种子数，不是接受轨迹数；默认 100000。
- `--seeds`：至少三种不同的非负种子；本轮预定 0–4。
- `--downstream-threads`：本轮固定 8。追踪固定 0，保持官方单线程 RNG 执行语义。
- `--output-dir`：全新目录。
- `--dry-run`：真实来源与文件检查加命令计划，不执行官方程序，不作为 benchmark。

`run_connectome_raw_official_cohort.py` 是 CPU 调度器：另传 `--official-dwi-root`、`--official-anatomy-root`、`--output-root`、`--python`，其余身份/种子参数相同；可用 `--case-ids` 选择原始清单病例。默认逐例、下游八线程；`--poll-seconds` 默认 15。它只等待明确传入的合同目录，源与清单发生变化则停止，不隐式重试或回退到旧结果。

```bash
python tools/benchmark_connectome_raw_cohort_envelope.py \
    --official-root /path/to/new_official_tracking/sub-CON03 \
    --fnit /path/to/fnit/sub-CON03/connectome \
    --fnit-gpu-reports /path/to/fnit/sub-CON03/gpu_report.json \
    --raw-manifest "$raw_manifest" --raw-manifest-sha256 "$raw_manifest_sha256" \
    --case-id sub-CON03 --fnit-seeds 0 --output /path/to/comparison/sub-CON03.json
```

这里要求 FNIT `gpu_report.json` 同病例且实际 exit0，原始输入在 before/after 逐项校验，读到的矩阵、atlas 和 nodes 文件摘要属于该真实输出报告。尝试数和种子标签核对实际 CLI command。

`preflight_connectome_raw_reference.py --config <冻结JSON> --config-sha256 <完整SHA> --output <新报告>` 只读核对真实工具/二进制、`-version`、原始文件与实际完成合同。配置列出源码 SHA、已审计 reference manifest 和 SHA、MRtrix 目录、raw manifest 和 SHA、明确 modeling/anatomy 根目录。加载器错误保留实际 returncode 与输出并使 `program_dependencies_ready=false`；合同未齐则 `required_dependency_ready=false`，不会因此启动计算或切换二进制。

## 4. 官方步骤与命令

| 阶段 | 本工具消费或执行的官方步骤 |
|---|---|
| 原始预处理 | 独立官方 TOPUP、EDDY 产物合同 |
| 解剖 | 真实官方 recon-all，随后官方配准、5TT、GMWMI、atlas 合同 |
| 建模 | 官方 Dhollander、MSMT-CSD、mtnormalise、张量 FA 合同 |
| 追踪 | `tckgen -algorithm iFOD2 -act 5tt -seed_gmwmi gmwmi -seeds 100000 -select 0 -maxlength 250 -angle 45 -cutoff 0.1 -power 0.5 -samples 3 -trials 1000 -max_attempts_per_seed 1000 -downsample 2 -nthreads 0` |
| SIFT2 | `tcksift2 tracks.tck wm_fod weights.txt -act 5tt -nthreads 8` |
| 标量 | `tckstats -dump lengths.txt`；`tcksample -precise -stat_tck mean` |
| 矩阵 | `tck2connectome -symmetric -assignment_radial_search 4`；FBC加 `-tck_weights_in`；长度/FA再加 `-scale_file -stat_edge mean` |

新官方自产链让已固定 C++ solver 从自己的真实 header 计算默认步长（0.5 voxel）和 ACT 最短长度（2 voxels），并记录实际 TCK header。固定 FNIT PT 的旧工具仍显式传入 PT 推导参数，保持原本算子定位设置。完整 argv、环境种子和各阶段参数以机器 manifest 为准。

四矩阵定义保持：轨迹数、`sum(w)`、`sum(w*length)/sum(w)`、`sum(w*preciseFA)/sum(w)`。自连接保留，未分配轨迹丢弃，不重新构造或改变已保存矩阵。

## 5. 精度、时间与真实图

新十例官方完整链的产物齐备后才进行真实矩阵评测。工具验收的是同原始病例和严格节点语义；两条独立处理链的中间图像及 atlas 内容分别记录。已有固定输入报告继续要求跨软件 atlas 来源 SHA 相同。

预先定义门槛：错误不超过官方自身最大观测错误，相似性不低于官方自身最小观测相似性。完整五次重复范围保留，不要求 FNIT 超过 MRtrix 自身重复；范围是有限样本观测，不是总体置信区间。共同 count 边上的 length/FA、全部严格上三角 count/FBC、support Dice 与 count Pearson 的数值定义复用原工具。

目前已完成的独立固定输入 CON03：官方 198 命令全部 exit0，一份 FNIT 对官方五份在八 atlas 主矩阵中 233/240 比较通过，7 个未通过。这个结果是算子定位，不是新十例完整官方链结果。已有 [真实点访问与人口分布图](../../validation/connectome/tenraw_20261002/task_04_repeat_reference/tractogram_population.png) 及 [矩阵图](../../validation/connectome/tenraw_20261002/task_04_repeat_reference/matrix_envelope.png)。

实际只读依赖预检：nodecw10 的七个固定二进制版本调用均 exit0，101 个原始唯一文件 SHA/header 已核对；完整 producer 合同尚未齐，`required_dependency_ready=false`。gpucw1 的旧系统 C++ 加载器失败独立保留，未改原程序或运行库。详见 [实际预检记录](../../validation/connectome/tenraw_20261002/task_04_raw_reference_preflight/README.md)。这不是追踪耗时或科学一致性结果。

## 6. 版本与 benchmark 记录

- 2026-10-03：增加独立官方原始链执行器、显式合同 CPU 控制器和同 raw/节点语义矩阵 envelope；最新 32 个 CPU 工具契约回归通过、1 个可选绘图库跳过，3.94 s。这些小数组检验工具协议，不是科学 benchmark。
- 2026-10-03：真实 fresh FS anatomy CON03 的官方 native atlas 已在另一阶段完成；官方 rawprep retry、modeling v2 与完整 DWI atlas 合同尚在推进。本工具不会提前写 completed。
- 既有固定输入和 FNIT 五种子 GPU 后处理分别保留 [独立说明](benchmark_connectome_fnit_repeats.md) 与实际结果命名空间。

## 7. 原代码与文献

- [UKB-connectomics](https://github.com/sina-mansour/UKB-connectomics)；[MRtrix3](https://github.com/MRtrix3/mrtrix3)，本轮已审计版本 `3.0.3-103-g026e850d`。
- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/)、[FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)、[OpenNeuro ds001226 固定 snapshot](https://github.com/OpenNeuroDatasets/ds001226/tree/fb4d0fda44f2ab7a732fb4ab6cd62add09dc1cd7)。原始清单逐文件 SHA 与原 snapshot/下载来源分别记录。
- Tournier et al. (2019), MRtrix3, *NeuroImage* 202:116137；Smith et al. (2012), ACT, *NeuroImage* 62:1924–1938；Smith et al. (2015), SIFT2, *NeuroImage* 119:338–351；Jeurissen et al. (2014), MSMT-CSD, *NeuroImage* 103:411–426。
