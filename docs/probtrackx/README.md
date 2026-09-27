# ProbtrackX 概率纤维束追踪

`TorchProbtrackX` 从 BEDPOSTX 的方向后验独立追踪。当前可在**同一扩散网格的体积掩膜**上输出 seed→voxel 路径密度、稀疏 voxel→voxel 矩阵，以及有向 region→region 矩阵；还可输出每个种子体素到各目标 ROI 的计数图。CPU 使用 PyTorch，GPU 步进使用 Triton，float32 张量且默认允许 TF32。追踪时不调用 FSL。官方 [ProbtrackX 输出定义](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/probtrackx.html)中的 `matrix3` 是目标体素两两共同经过次数，和 `matrix2` 的种子体素×目标体素含义不同。下方的[覆盖表](#与官方选项的差异)列出尚未等价的模式。

## 输入与 Python 用法

`run(samples_dir, output_dir, ...)` 每次处理一个被试。`samples_dir` 需有 `nodif_brain_mask.nii.gz` 和各条纤维的 `merged_th<i>samples.nii.gz`、`merged_ph<i>samples.nii.gz`、`merged_f<i>samples.nii.gz`。`seed=` 是一个非空 3D NIfTI 掩膜；`regions=` 是按矩阵行列顺序排列的至少两个非空、不重叠 3D ROI，二者须择一。所有 mask 均须与 BEDPOSTX mask 具有相同 shape 和 affine；`mask=` 可指定追踪 mask。`output_dir` 是绝对或相对结果目录；文件已存在时报错，除非设置 `overwrite=True`。

```python
from fnit import TorchProbtrackX

tracker = TorchProbtrackX(
    device="cuda:0",  # 计算设备；GPU 步进使用 Triton，可改为 "cpu"
    nsamples=5000,  # 每个种子体素发出的轨迹数
    nsteps=2000,  # 每条双向轨迹的总步数；必须是大于等于 2 的偶数
    steplength=0.5,  # 每一步的物理长度，单位 mm
    cthr=0.2,  # 相邻方向最小点积阈值，控制曲率
    fibthresh=0.01,  # 可选纤维方向的最小 volume fraction
    batch_size=2048,  # 同批并行推进的轨迹数
    seed=12345,  # 随机数种子，对应 FSL --rseed
    distthresh=0.0,  # 每个半轨迹的最短距离阈值，单位 mm
    sampvox=0.0,  # 种子体素内随机抖动半径，单位 mm
    fibst=None,  # 起始纤维编号；None 使用默认选择规则
    usef=False,  # 是否按局部 fiber fraction 选择方向
    randfib=0,  # 起始纤维随机选择模式，支持 0、1、2、3
    pathdist=False,  # 是否把密度值改为路径长度加权和
    mean_path_length=False,  # 是否另写平均首次抵达长度图
)

voxel = tracker.run(
    samples_dir="/absolute/path/subject.bedpostX",  # 输入：BEDPOSTX 后验目录
    output_dir="/absolute/path/voxel_result",  # 输出：本次 seed 模式结果目录
    seed="/absolute/path/seed.nii.gz",  # 输入：非空 3D seed mask
    regions=None,  # ROI network 模式输入；单 seed 模式必须为 None
    mask=None,  # 可选追踪 mask；None 使用 nodif_brain_mask.nii.gz
    avoid=None,  # 可选排除 mask
    stop=None,  # 可选进入后停止的 mask
    forcefirststep=False,  # 首步是否跳过 avoid/stop 条件
    waypoints=None,  # 可选 waypoint mask、路径列表或文本列表
    waycond="AND",  # waypoint 组合条件：AND 或 OR
    wayorder=False,  # 是否要求按列表顺序通过 waypoint
    onewaycondition=False,  # 是否要求每个半轨迹单独满足 waypoint
    wtstop=None,  # 可选进入后首次离开即停止的 mask
    matrix1=True,  # 输出 seed voxel×seed voxel 稀疏矩阵
    target2="/absolute/path/target_union.nii.gz",  # matrix2 的目标体素 mask
    target3="/absolute/path/target_union.nii.gz",  # matrix3 的行目标 mask
    lrtarget3=None,  # 可选 matrix3 列目标 mask；None 使用 target3
    distthresh1=0.0,  # matrix1 有效路径总长度下限，单位 mm
    distthresh3=0.0,  # matrix3 有效路径总长度下限，单位 mm
    targetmasks=[  # 每个种子体素要统计命中的目标 ROI，列表顺序定义输出列
        "/absolute/path/roi_01.nii.gz",
        "/absolute/path/roi_02.nii.gz",
    ],
    overwrite=False,  # False 时已有输出会停止运行
)

network = tracker.run(
    samples_dir="/absolute/path/subject.bedpostX",  # 输入：同一 BEDPOSTX 后验目录
    output_dir="/absolute/path/network_result",  # 输出：ROI network 结果目录
    seed=None,  # network 模式不使用单 seed mask
    regions=[  # 输入：至少两个非空、不重叠 ROI；列表顺序定义矩阵行列
        "/absolute/path/roi_01.nii.gz",
        "/absolute/path/roi_02.nii.gz",
    ],
    overwrite=False,  # 是否覆盖已有结果
)

constrained = tracker.run(
    samples_dir="/absolute/path/subject.bedpostX",  # 输入：BEDPOSTX 后验目录
    output_dir="/absolute/path/constrained_result",  # 输出：带约束的 seed 结果目录
    seed="/absolute/path/seed.nii.gz",  # 输入：非空 3D seed mask
    regions=None,  # 单 seed 模式必须为 None
    waypoints="/absolute/path/waypoint_list.txt",  # 输入：waypoint 路径列表
    waycond="AND",  # 必须通过全部 waypoint
    wayorder=True,  # 必须按列表顺序通过 waypoint
    onewaycondition=True,  # 每个半轨迹单独检查 waypoint
    wtstop="/absolute/path/wtstop.nii.gz",  # 进入后首次离开即停止的 mask
    overwrite=False,  # 是否覆盖已有结果
)

print(voxel.paths, voxel.matrix2, voxel.matrix3)
print(network.network_matrix, network.network_probability)
print(constrained.paths)
```

| 参数 | 输入和含义 | FSL 对应 |
| --- | --- | --- |
| `nsamples`, `nsteps`, `steplength` | 每个种子体素的轨迹数、双向总步数（偶数≥2）、物理步长 mm | `-P`, `-S`, `--steplength` |
| `cthr`, `fibthresh`, `seed` | 方向点积阈值、纤维分数阈值、随机种子 | `--cthr`, `--fibthresh`, `--rseed` |
| `distthresh`, `sampvox`, `fibst`, `randfib`, `usef` | 单半路径最短距离、seed 抖动半径、起始纤维与纤维选择 | 同名官方选项 |
| `avoid`, `stop`, `forcefirststep` | 排除 mask、进入后停止的 mask、首步条件；仅同网格体积 | 同名官方选项 |
| `waypoints`, `waycond`, `wayorder`, `onewaycondition` | 必经 mask、AND/OR、按列表顺序通过、对每个半轨迹分别判定 | `--waypoints`, `--waycond`, `--wayorder`, `--onewaycondition` |
| `wtstop` | 进入 mask 后在首次离开时停止该半轨迹 | `--wtstop` |
| `matrix1=True`, `distthresh1` | 种子体素之间的稀疏矩阵；有效路径总长度下限 | `--omatrix1`, `--distthresh1` |
| `target2=...` | 种子体素×目标体素；`target2` 必须同网格 | `--omatrix2 --target2=...` |
| `target3=...`, `lrtarget3=...`, `distthresh3` | 目标体素共访矩阵；可选不同的列目标；有效路径总长度下限 | `--omatrix3 --target3=... --lrtarget3=... --distthresh3` |
| `targetmasks=[...]` | 每个种子体素到多个目标 ROI 的命中次数 | `--targetmasks=... --os2t --s2tastext` |
| `pathdist`, `mean_path_length` | 路径长度加权与平均路径长度，仅已有密度及 ROI 网络模式支持；和新稀疏矩阵或 `targetmasks` 同时使用会报错 | `--pd`, `--ompl` |
| `device`, `batch_size` | `cpu` 或 `cuda:0`，每批并行轨迹数 | FNIT 选项 |

`distthresh1/3` 仅过滤对应稀疏矩阵的更新，不改变 `fdt_paths` 与 `waytotal`。`matrix1`、`matrix2`、`matrix3` 可以同次运行。每条有效轨迹对 matrix1/2 的同一列及 seed→target 的同一 ROI 最多计一次；matrix3 对该轨迹经过的每对目标体素各计一次。`targetmasks` 可为 Python 路径列表，或文本列表路径；文本列表中的相对路径按列表所在目录解析。`waypoints`、`wtstop` 也接受单个同网格 3D NIfTI、Python 路径列表或文本列表；默认 `waycond="AND"` 要求经过全部 waypoint，`"OR"` 要求经过至少一个。默认两个半轨迹可合并满足条件；`onewaycondition=True` 改为逐半轨迹判定。`wayorder=True` 仅与 `AND` 合用，并按 waypoint 列表顺序检查。`stop` 在进入 mask 时终止，`wtstop` 则允许进入并在离开后终止。在 9 组 9×5×5 合成直线场的 waypoint/`wtstop` 配对中，FNIT 与 FSL 6.0.7.22 的 `waytotal` 和 `fdt_paths` 逐项相同，AND 条件下的 matrix2 `.dot` 也逐项相同。本次默认及矩阵真实 DWI 配对未使用这些约束。当前公开真实 DWI benchmark 覆盖默认追踪、三类矩阵、target mask 与五区 network；waypoint/`wtstop` 的真实 DWI 数值和时间未列入当前发布证据。

## 输出及结构

```text
output_dir/
├── fdt_paths.nii.gz                         # seed→voxel 轨迹通过次数
├── fdt_paths_lengths.nii.gz                 # mean_path_length=True；平均首次抵达长度
├── waytotal                                 # 单 seed 一行；regions 每 ROI 一行
├── fdt_matrix1.dot                          # Nseed × Nseed；matrix1=True
├── coords_for_fdt_matrix1                   # 每行 x y z ROI编号 位置编号
├── fdt_matrix2.dot                          # Nseed × Ntarget2；target2=...
├── coords_for_fdt_matrix2                   # 行坐标，5列
├── tract_space_coords_for_fdt_matrix2       # 列坐标，3列
├── lookup_tractspace_fdt_matrix2.nii.gz     # 目标体素处为1起始列号，其余为0
├── fdt_matrix3.dot                          # Ntarget3 × Ntarget3或Nlrtarget3
├── coords_for_fdt_matrix3                   # 行坐标，5列
├── tract_space_coords_for_fdt_matrix3       # 仅lrtarget3时的列坐标，5列
├── fdt_network_matrix                       # regions模式；Nroi × Nroi，原始有向计数
├── fdt_network_matrix_lengths               # regions+mean_path_length；平均命中长度
├── fdt_network_matrix_probability           # FNIT归一化有向矩阵；计数模式
├── fdt_network_matrix_symmetric             # FNIT对称化矩阵；计数模式
├── seeds_to_<target>.nii.gz                 # targetmasks；每目标一张seed网格图
├── seeds_<roi>_to_<target>.nii.gz           # regions+targetmasks；roi从0编号
└── matrix_seeds_to_all_targets              # Nseed × Ntargetmasks文本矩阵
```

所有 NIfTI 输出保留输入 mask 的形状和 affine。`.dot` 文件按官方格式写入从 1 开始的 `row column count` 稀疏三元组，末行为 `Nrow Ncol 0` 维度标记；体素次序由配套坐标表定义，不能直接假定是 NIfTI 的线性索引。`matrix1` 排除自连接；`matrix3` 未指定 `lrtarget3` 时只存上三角且不含对角，指定后为行目标×列目标的有向组合。`fdt_paths` 是经过体素的采样轨迹次数，**不是解剖纤维条数**。

`run()` 返回 `ProbTrackXResult`。`output_dir`、`paths`、`waytotal` 为路径；启用相应输出时，`lengths`、`network_matrix`、`network_lengths`、`network_probability`、`network_symmetric`、`matrix1`、`matrix1_coords`、`matrix2`、`matrix2_coords`、`matrix2_target_coords`、`matrix2_lookup`、`matrix3`、`matrix3_coords`、`matrix3_target_coords`、`seed_to_targets_matrix` 为对应文件路径，否则为 `None`。`seed_to_targets` 是目标图路径元组；`seed_points`、`accepted_streamlines` 和 `elapsed_seconds` 分别是种子体素数、有效采样轨迹数与运行秒数。`regions` 与 `targetmasks` 同用时，种子行按 ROI 列表顺序拼接，目标图使用从 0 开始的 `<roi>` 编号。

`regions` 与稀疏矩阵或 `targetmasks` 同用时，追踪先应用网络模式的“命中其他 ROI”筛选，因此这些输出是条件连接结果；若需要不经网络筛选的 voxel×voxel 矩阵，应把 ROI 并集作为单个 `seed` 输入。

ROI 矩阵的 `C[i,j]` 是从 ROI *i* 发出的样本抵达 ROI *j* 的次数，行列顺序由 `regions` 决定。FNIT 派生有向矩阵定义为 `P[i,j] = C[i,j] / (Ni × nsamples)`，其中 `Ni` 是第 *i* 个 seed ROI 的体素数；对称矩阵是 `(P[i,j] + P[j,i]) / 2`。只把原始 `fdt_network_matrix` 与 FSL 同名文件直接对照。`waytotal` 是有效轨迹数，不能替代发出的 `Ni × nsamples`；regions 模式的密度图只计命中其他 ROI 的有效轨迹。`pathdist=True` 时 `fdt_paths`/原始网络矩阵改为长度和；`mean_path_length=True` 另写 `fdt_paths_lengths.nii.gz` 与网络的 `fdt_network_matrix_lengths`。

## CLI 与原版 FSL 命令

```bash
fnit probtrackx --samples-dir /absolute/path/subject.bedpostX \
  --seed /absolute/path/seed.nii.gz --output-dir /absolute/path/fnit_voxel \
  --device cuda:0 --nsamples 5000 --nsteps 2000 \
  --omatrix1 --omatrix2 --target2 /absolute/path/target_union.nii.gz \
  --omatrix3 --target3 /absolute/path/target_union.nii.gz
fnit probtrackx --samples-dir /absolute/path/subject.bedpostX \
  --roi-list /absolute/path/roi_list.txt --output-dir /absolute/path/fnit_network \
  --device cuda:0 --nsamples 5000 --nsteps 2000
fnit probtrackx --samples-dir /absolute/path/subject.bedpostX \
  --seed /absolute/path/seed.nii.gz --output-dir /absolute/path/fnit_constrained \
  --device cuda:0 --nsamples 5000 --nsteps 2000 \
  --waypoints /absolute/path/waypoint_list.txt --waycond AND \
  --wayorder --onewaycondition --wtstop /absolute/path/wtstop.nii.gz
```

`roi_list.txt` 每行一个 ROI NIfTI 路径。单 seed 的 voxel→ROI 分类另加 `--targetmasks /absolute/path/target_list.txt`。`fnit-probtrackx` 接受同样参数。下例将三种矩阵合并调用；真实 DWI benchmark 为每种矩阵分别运行。对应的官方 CPU 命令为：

```bash
"$FSLDIR/bin/probtrackx2" -s /absolute/path/subject.bedpostX/merged \
  -m /absolute/path/subject.bedpostX/nodif_brain_mask.nii.gz \
  -x /absolute/path/seed.nii.gz --dir=/absolute/path/fsl_voxel \
  --forcedir --opd -P 5000 -S 2000 --steplength=0.5 \
  --cthr=0.2 --fibthresh=0.01 --rseed=12345 \
  --omatrix1 --omatrix2 --target2=/absolute/path/target_union.nii.gz \
  --omatrix3 --target3=/absolute/path/target_union.nii.gz
"$FSLDIR/bin/probtrackx2" -s /absolute/path/subject.bedpostX/merged \
  -m /absolute/path/subject.bedpostX/nodif_brain_mask.nii.gz \
  -x /absolute/path/roi_list.txt --network --dir=/absolute/path/fsl_network \
  --forcedir --opd -P 5000 -S 2000 --steplength=0.5 \
  --cthr=0.2 --fibthresh=0.01 --rseed=12345
"$FSLDIR/bin/probtrackx2" -s /absolute/path/subject.bedpostX/merged \
  -m /absolute/path/subject.bedpostX/nodif_brain_mask.nii.gz \
  -x /absolute/path/seed.nii.gz --dir=/absolute/path/fsl_constrained \
  --forcedir --opd -P 5000 -S 2000 --steplength=0.5 \
  --cthr=0.2 --fibthresh=0.01 --rseed=12345 \
  --waypoints=/absolute/path/waypoint_list.txt --waycond=AND \
  --wayorder --onewaycondition --wtstop=/absolute/path/wtstop.nii.gz
```

官方 voxel→ROI 分类需 `--targetmasks=/absolute/path/target_list.txt --os2t --s2tastext`。官方 GPU 使用 `probtrackx2_gpu`，但其部分矩阵选项的组合受限；本页稀疏矩阵的正式参照为 `probtrackx2` CPU。

## 与官方选项的差异

依据 [FSL 官方说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/probtrackx.html)及[官方源码](https://git.fmrib.ox.ac.uk/fsl/ptx2)。下表是目前的可执行范围，不把同名文件视为完整数值等价。

| 类别 | 已实现 | 尚有差异 |
| --- | --- | --- |
| 输入和空间 | BEDPOSTX 后验、同网格体积 seed/ROI、独立追踪 mask、体积避让和停止 | `--simple` ASCII 点、表面 seed/target、`--seedref`、`--meshspace`、`--xfm`、`--invxfm`，以及非 network 多 ROI 输入；`target2` 低分辨率网格未支持 |
| 步进和过滤 | 双向 Euler、`-P/-S`、`--steplength`、`--cthr`、`--fibthresh`、`--randfib`、`--fibst`、`--usef`、`--sampvox`、`--distthresh`、`--rseed`、体积 `--avoid`、`--stop`、`--forcefirststep`、`--waypoints`/`--waycond`/`--wayorder`/`--onewaycondition`、`--wtstop` | `--modeuler`、`--loopcheck`、局部方向/曲率及表面约束选项；随机数流不逐轨迹一致；waypoint/停止组合已做合成场配对；单 waypoint 真实 DWI 配对已完成但指标未公开，其他约束组合仍待验证 |
| 输出 | 密度图、`--pd`/`--ompl` 的密度及网络结果、`--network`、同网格体积计数的 `--omatrix1/2/3` 与 `--targetmasks/--os2t` | 新矩阵和 seed→target 的 `--pd/--ompl`、`--omatrix4`、`--fopd`、`--opathdir`、`--otargetpaths`、`--savepaths`、`--closestvertex`；`--s2tastext` 当前自动写，不能独立控制 |
| 文件和运行控制 | FSL 风格矩阵 `.dot`、坐标表、target2 lookup，FNIT 自有归一化连接矩阵 | 官方 `-o/--out`、`--dir/--forcedir` 目录命名、`--verbose`；FNIT 固定写密度图 |

官方仍有以下当前未接入的具体选项：`--simple`、表面 seed/target、`--seedref`、`--meshspace`、`--xfm`、`--invxfm`、`--closestvertex`；`--modeuler`、`--loopcheck`；`--omatrix4`、`--target4`、`--colmask4`、`--fopd`、`--opathdir`、`--otargetpaths`、`--savepaths`。官方帮助还包含方向与纤维选择相关的 `--prefdir`、`--no_integrity`、`--onewayonly`、`--locfibchoice`、`--loccurvthresh`、`--noprobinterpol`。FNIT 当前固定写 `fdt_paths.nii.gz`，不支持官方 `-o/--out` 与目录自动命名语义。上述选项会影响部分三类 connectome 的数值或输出结构，因此当前结果只应按本页明确支持的同网格体积模式使用。

## 与 FSL 的真实 DWI benchmark

在 gpucw1 使用同一例真实 UK Biobank DWI 的 FSL BEDPOSTX 三纤维后验，分别以 FSL 6.0.7.22 和本次 FNIT 源码运行。默认密度与长度加权模式采用 400 总步、0.5 mm 步长、`cthr=0.2`、`fibthresh=0.01`、相同随机种子；单 seed 每体素 200 条，五区网络每体素 2000 条。FNIT 批大小 2048；CPU 为 8 线程。时间为进程启动、后验载入、追踪及写盘的总墙钟。双方随机数流不同，因此比较汇总图和矩阵，不要求逐轨迹相同。

### Seed→voxel 与 region→region

默认计数 `--opd`。Pearson r 在双方非零并集计算；支持 Dice 比较非零体素集合，top 10% Dice 比较强连接体素。网络矩阵 MAE 是原始 `fdt_network_matrix` 的平均绝对计数差。

| 任务 | FSL / FNIT (s) | 密度 r | 支持 Dice | top 10% Dice | ROI 矩阵 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单 seed CPU | 12.06 / 13.76 | 0.9937 | 0.6957 | 0.8929 | — |
| 单 seed GPU | 12.41 / 12.94 | 0.9937 | 0.6959 | 0.8856 | — |
| 五区网络 CPU | 37.93 / 53.80 | 0.9388 | 0.5620 | 0.8142 | 0.76 |
| 五区网络 GPU | 13.81 / 16.75 | 0.9537 | 0.5181 | 0.7949 | 0.60 |

长度加权 `--opd --pd --ompl`。平均路径长度的 r 和 MAE 仅在双方均为非零的体素上计算，MAE 单位为 mm。该模式的 ROI 矩阵为长度加权和，不能与计数矩阵直接相减。

| 任务 | FSL / FNIT (s) | 加权密度 r | 支持 Dice | 平均长度 r | 长度 MAE (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单 seed CPU | 13.42 / 15.68 | 0.9750 | 0.6956 | 0.8680 | 5.69 |
| 单 seed GPU | 14.18 / 13.14 | 0.9758 | 0.6959 | 0.8638 | 5.73 |
| 五区网络 CPU | 32.94 / 54.58 | 0.8298 | 0.5610 | 0.8440 | 6.81 |
| 五区网络 GPU | 13.29 / 18.38 | 0.8330 | 0.5166 | 0.8676 | 6.35 |

以上四类模式的配对数值与源码 SHA-256 分别见[默认计数报告](../../validation/probtrackx/report.default.latest.public.json)和[长度加权报告](../../validation/probtrackx/report.current.latest.public.json)。在这些配置下，FNIT CPU 网络慢于 FSL CPU；GPU 也未显示稳定的整体加速。

### Voxel→voxel 稀疏矩阵

同一组 ROI 的并集用于种子和目标，分别运行 matrix1（seed×seed）、matrix2（seed×target2）、matrix3（目标体素共访）；每体素 500 条。按官方坐标表对齐 `.dot` 边后，支持 Dice 比较非零边集合；r 和 MAE 在非零并集计算。以下均为 FSL CPU 与 FNIT CPU 的配对。

| 输出 | FSL / FNIT (s) | 非零边 FSL / FNIT | 支持 Dice | 边权 r | 边权 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| matrix1 | 17.21 / 31.25 | 155 / 144 | 0.7425 | 0.99983 | 1.86 |
| matrix2 | 18.55 / 30.03 | 190 / 179 | 0.7913 | 0.99990 | 1.57 |
| matrix3 | 18.51 / 30.89 | 134 / 122 | 0.7734 | 0.99991 | 2.94 |

高权重边使 r 接近 1，但支持 Dice 为 0.74–0.79，表示低计数边仍不一致；本次 FNIT CPU 三种稀疏矩阵均慢于 FSL CPU。逐模式数值和源码哈希见[matrix1](../../validation/probtrackx/report.matrix1.cpu.latest.public.json)、[matrix2](../../validation/probtrackx/report.matrix2.cpu.latest.public.json)、[matrix3](../../validation/probtrackx/report.matrix3.cpu.latest.public.json)。

### Voxel→ROI 与归一化 ROI 矩阵

单个真实 ROI seed 到 5 个目标的 `--targetmasks --os2t` 配对：FSL / FNIT 为 12.35 / 15.97 秒；每个 seed 体素×目标的平均绝对计数误差为 0.1714，最大误差为 2。报告见[seed→ROI](../../validation/probtrackx/report.targets.cpu.latest.public.json)。

另一组每体素 500 条的五区网络计数配对：FSL / FNIT 为 20.13 / 30.15 秒；原始有向矩阵 MAE 为 0.32。FNIT 的额外归一化有向矩阵和对称矩阵按上文公式生成，和 FSL 原始矩阵含义不同；该组与 FSL 原始矩阵比较后的 MAE 分别为 0.000091 与 0.000023（比较时先对 FSL 原始矩阵施加相同公式）。

![真实 UK Biobank dMRI 的五区长度加权连接矩阵](figures/probtrackx_real_ukb_network_pd_ompl.png)

上图来自同一例真实 DWI 的 CPU 五区网络 `--opd --pd --ompl` 配对。上排为累计长度加权连接矩阵，下排为命中轨迹的平均长度；从左到右依次为 FSL、当前 FNIT 和 FNIT−FSL。区域标签只保留通用的 JHU 解剖名称，不含病例编号或服务器路径。图中矩阵来自当前源码哈希绑定的同一组输出，与[长度加权报告](../../validation/probtrackx/report.current.latest.public.json)中的矩阵 MAE、密度图和时间统计对应。

原始 DWI、BEDPOSTX 后验、seed、逐体素密度图和完整文本矩阵仍保留在授权服务器；仓库公开六份汇总 JSON 和这一张去标识连接矩阵图。双方随机数流不同，这张图比较的是网络汇总结果，不能证明逐轨迹一致。当前证据只有一例数据、五个 ROI，且网络较稀疏；它不能替代多病例或全脑分区验证。[复现命令和指标定义](../../validation/probtrackx/README.md)列出比较方法。合成直线场的 FSL 逐项相同测试只验证计数规则，不替代这组真实数据误差。`tests/probtrackx/` 的 40 项 CPU/CUDA 测试已在 gpucw1 通过。
