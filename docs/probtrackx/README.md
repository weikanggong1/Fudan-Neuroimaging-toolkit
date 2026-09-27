# ProbtrackX 概率纤维束追踪

`TorchProbtrackX` 从 BEDPOSTX 的方向后验独立追踪。当前可在**同一扩散网格的体积掩膜**上输出 seed→voxel 路径密度、稀疏 voxel→voxel 矩阵，以及有向 region→region 矩阵；还可输出每个种子体素到各目标 ROI 的计数图。CPU 使用 PyTorch，GPU 步进使用 Triton，float32 张量且默认允许 TF32。追踪时不调用 FSL。官方 [ProbtrackX 输出定义](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/probtrackx.html)中的 `matrix3` 是目标体素两两共同经过次数，和 `matrix2` 的种子体素×目标体素含义不同。下方的[覆盖表](#与官方选项的差异)列出尚未等价的模式。

## 输入与 Python 用法

`run(samples_dir, output_dir, ...)` 每次处理一个被试。`samples_dir` 需有 `nodif_brain_mask.nii.gz` 和各条纤维的 `merged_th<i>samples.nii.gz`、`merged_ph<i>samples.nii.gz`、`merged_f<i>samples.nii.gz`。`seed=` 是一个非空 3D NIfTI 掩膜；`regions=` 是按矩阵行列顺序排列的至少两个非空、不重叠 3D ROI，二者须择一。所有 mask 均须与 BEDPOSTX mask 具有相同 shape 和 affine；`mask=` 可指定追踪 mask。`output_dir` 是绝对或相对结果目录；文件已存在时报错，除非设置 `overwrite=True`。

```python
from fnit import TorchProbtrackX

tracker = TorchProbtrackX(
    device="cuda:0", nsamples=5000, nsteps=2000, batch_size=2048,
    steplength=0.5, cthr=0.2, fibthresh=0.01, seed=12345,
)
# seed→voxel、seed voxel→target voxel、target voxel pair，均为计数模式
voxel = tracker.run(
    "/absolute/path/subject.bedpostX", "/absolute/path/voxel_result",
    seed="/absolute/path/seed.nii.gz", matrix1=True,
    target2="/absolute/path/target_union.nii.gz",
    target3="/absolute/path/target_union.nii.gz",
    targetmasks=["/absolute/path/roi_01.nii.gz", "/absolute/path/roi_02.nii.gz"],
)
# region→region：ROI 列表顺序即矩阵顺序
network = tracker.run(
    "/absolute/path/subject.bedpostX", "/absolute/path/network_result",
    regions=["/absolute/path/roi_01.nii.gz", "/absolute/path/roi_02.nii.gz"],
)
print(voxel.paths, voxel.matrix2, voxel.matrix3)
print(network.network_matrix, network.network_probability, network.network_symmetric)

# 同网格体积 waypoint 与退出后停止约束
constrained = tracker.run(
    "/absolute/path/subject.bedpostX", "/absolute/path/constrained_result",
    seed="/absolute/path/seed.nii.gz",
    waypoints="/absolute/path/waypoint_list.txt", waycond="AND",
    wayorder=True, onewaycondition=True,
    wtstop="/absolute/path/wtstop.nii.gz",
)
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

`distthresh1/3` 仅过滤对应稀疏矩阵的更新，不改变 `fdt_paths` 与 `waytotal`。`matrix1`、`matrix2`、`matrix3` 可以同次运行。每条有效轨迹对 matrix1/2 的同一列及 seed→target 的同一 ROI 最多计一次；matrix3 对该轨迹经过的每对目标体素各计一次。`targetmasks` 可为 Python 路径列表，或文本列表路径；文本列表中的相对路径按列表所在目录解析。`waypoints`、`wtstop` 也接受单个同网格 3D NIfTI、Python 路径列表或文本列表；默认 `waycond="AND"` 要求经过全部 waypoint，`"OR"` 要求经过至少一个。默认两个半轨迹可合并满足条件；`onewaycondition=True` 改为逐半轨迹判定。`wayorder=True` 仅与 `AND` 合用，并按 waypoint 列表顺序检查。`stop` 在进入 mask 时终止，`wtstop` 则允许进入并在离开后终止。在 9 组 9×5×5 合成直线场的 waypoint/`wtstop` 配对中，FNIT 与 FSL 6.0.7.22 的 `waytotal` 和 `fdt_paths` 逐项相同，AND 条件下的 matrix2 `.dot` 也逐项相同。本次默认及矩阵真实 DWI 配对未使用这些约束。另在 gpucw1 完成了单 waypoint 的 FSL CPU/FNIT CPU 真实 DWI 配对，已检查 FSL 完成日志和双方输出；精度、耗时指标仍仅保留在授权服务器，待授权后发布。其他 waypoint/`wtstop` 组合尚无真实 DWI 配对结果。

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

已在 gpucw1 使用同一真实 DWI 的 BEDPOSTX 后验，将 FSL 6.0.7.22 与本次 FNIT 源码配对运行。覆盖默认 seed→voxel、长度加权密度、稀疏 matrix1/2/3、seed→目标 ROI 和 region→region；逐体素与逐边结果及汇总指标保存在授权服务器。当前公开页不据此宣称数值等价或加速，指标和比较图待数据发布授权后补充。[复现方法](../../validation/probtrackx/README.md)列出输入、官方命令、运行参数及比较脚本。

同网格合成直线场用于计数规则回归：matrix1/2/3 的稀疏输出、坐标表、matrix2 lookup 和密度图与 FSL 逐项相同；9 组 waypoint/`wtstop` 配对的 `waytotal` 和密度图逐项相同。这些合成结果不作为真实数据精度或耗时结论。`tests/probtrackx/` 的 40 项 CPU/CUDA 测试已在 gpucw1 通过。
