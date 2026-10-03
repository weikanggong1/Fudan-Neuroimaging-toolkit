# aseg 与 wmparc 体积及强度统计

## 函数功能简介

`write_aseg_stats`、`write_wmparc_stats` 从已有重建的分割、归一化 T1 和全局体积记录生成 `aseg.stats`、`wmparc.stats`。两者共享 NumPy/Numba 的部分容积与强度计算，使用 nibabel 读 MGZ，在 CPU 上运行，不调用 FreeSurfer 程序，也不新增依赖。完整 recon-all 已调用这两个函数。

2026-10-03 十人真实 fMRI 流程的 CON04 发现一个成熟子函数问题：`_statistics` 对单体素标签先计算分母为零的样本方差，再选用既有标准差 0，产生无用警告。修复仅将方差表达式放入 `count > 1` 分支；单体素仍写 0，部分容积、标签、均值、其他方差及格式保持原值。正式十人 benchmark 仍用冻结 `1128bc52`，不热改或重标执行来源。

## Python 调用、输入与输出

```python
from pathlib import Path
from fnit.recon_all.segstats_aseg_python import write_aseg_stats
from fnit.recon_all.segstats_wmparc_python import write_wmparc_stats

subject_directory = Path("/data/fnit/subjects/sub-CON04")  # 已完成重建，只读输入
asset_directory = Path("/data/fnit/recon_assets")          # 已校验的外置 LUT
new_statistics_directory = Path("/data/fnit/statistics_recheck/sub-CON04")
brain_volume_statistics = subject_directory / "stats/brainvol.stats"
synthseg_intracranial_volume_file = subject_directory / "stats/synthseg.tiv.dat"

aseg_statistics_path = write_aseg_stats(
    subject=subject_directory,
    aseg_lut=asset_directory / "ASegStatsLUT.txt",
    output=new_statistics_directory / "aseg.stats",        # 新输出，保留原表
    brainvol_stats=brain_volume_statistics,
    stiv=synthseg_intracranial_volume_file,
)
wmparc_statistics_path = write_wmparc_stats(
    subject=subject_directory,
    wm_lut=asset_directory / "WMParcStatsLUT.txt",
    output=new_statistics_directory / "wmparc.stats",
    brainvol_stats=brain_volume_statistics,
    stiv=synthseg_intracranial_volume_file,
)
```

| 参数 | 输入格式、默认值与用途 |
|---|---|
| `subject` | 必填目录；分别读取 `mri/aseg.mgz`、`mri/wmparc.mgz`，共享 `mri/norm.mgz`、`mri/transforms/talairach.xfm`。aseg 另从双侧 `surf/{lh,rh}.orig.nofix` 计算原拓扑缺陷孔数。 |
| `aseg_lut` / `wm_lut` | 对应必填 UTF-8 标签表；数据行以整数标签和结构名开始。使用已核验的外置资产，函数不下载。 |
| `output` | 必填 UTF-8 `.stats` 路径，创建父目录后写文件，返回该 `Path`。直接调用会覆盖同名文件，复核时应指定新目录。 |
| `brainvol_stats` | 可选；默认 `subject/stats/brainvol.stats`。读取先前计算的脑、皮层、白质和 MaskVol 等记录，不在本步骤重算表面体积。 |
| `stiv` | 可选；默认 `subject/stats/synthseg.tiv.dat`，首个标量为 SynthSeg 总颅内容积 mm³。Talairach XFM 用于另算 eTIV。 |

分割是同一 conform 网格的三维非负整数标签 MGZ；`norm.mgz` 同形状，单位为归一化 MR。体素体积取分割头前三个 zooms 的乘积。原表面为 FreeSurfer 三角格式；这里只计数顶点、边和面，不改变几何。

输出含 `# Measure` 全局记录、`# TableCol` 列定义和十列数据：行编号、分割标签、硬分割体素数、部分容积 mm³、结构名、norm 均值/样本标准差/最小值/最大值/范围。两表只选 LUT 中不超过该分割最大标签的正整数 ID；aseg 再排除皮层灰白质标签 2/3/41/42，并为所选空结构保留零行，wmparc 仅写非空结构。单体素标准差为 0；多体素保留原 float32 均值及样本方差的运算顺序。全局体积为 mm³，孔数为计数，体积比无单位。

## 命令行调用

```bash
subject_directory=/data/fnit/subjects/sub-CON04
asset_directory=/data/fnit/recon_assets
new_statistics_directory=/data/fnit/statistics_recheck/sub-CON04
python -m fnit.recon_all.segstats_aseg_python \
  "$subject_directory" "$asset_directory/ASegStatsLUT.txt" \
  "$new_statistics_directory/aseg.stats"
python -m fnit.recon_all.segstats_wmparc_python \
  "$subject_directory" "$asset_directory/WMParcStatsLUT.txt" \
  "$new_statistics_directory/wmparc.stats"
```

三个位置参数对应 `subject`、LUT、`output`。CLI 使用默认 `brainvol_stats` 和 `stiv`；自定义这两个输入时使用 Python API。

独立完整回归的重现入口为 [verify_single_voxel_stats_real.py](../../validation/fmri/public_ten_20261003/verify_single_voxel_stats_real.py)。它绑定本轮 CON04 与上述固定新/旧模块 SHA；实际测量的 v1 源码保留在 [producer 归档](../../validation/fmri/public_ten_20261003/frozen_harnesses/verify_single_voxel_stats_real_v1.py)，当前 CLI 另加强了警告、失败退出码与受保护目录检查。

```bash
frozen_source_directory=/data/fnit/frozen_source_1128
completed_subject_directory=/data/fnit/formal_v4/CON04/reconstruction/subject
fixed_statistics_module=/data/fnit/stats_fix/segstats_wmparc_python.py
completed_pipeline_report=/data/fnit/formal_v4/CON04/report/report.public.json
frozen_source_manifest=/data/fnit/frozen_source_1128_manifest.private.json
raw_t1w_image=/data/public_ten/sub-CON04/ses-preop/anat/sub-CON04_ses-preop_T1w.nii.gz
raw_bold_image=/data/public_ten/sub-CON04/ses-preop/func/sub-CON04_ses-preop_task-rest_bold.nii.gz
new_regression_directory=/data/fnit/stats_regression_CON04_attempt01
python validation/fmri/public_ten_20261003/verify_single_voxel_stats_real.py \
  --subject "$completed_subject_directory" --source "$frozen_source_directory" \
  --new-module "$fixed_statistics_module" --candidate-report "$completed_pipeline_report" \
  --source-manifest "$frozen_source_manifest" \
  --raw-t1w "$raw_t1w_image" --raw-bold "$raw_bold_image" \
  --output-root "$new_regression_directory"
```

八个参数均必填：`subject` 是同例已完成重建；`source` 为原冻结源码；`new-module` 为修复后的完整统计模块；`candidate-report` 绑定原 complete 流程与 raw SHA；`source-manifest` 是冻结根相对文件名到 SHA 的 JSON；`raw-t1w/raw-bold` 是原完整配对输入；`output-root` 为父目录已存在的新独立目录，不得与源码、原始数据或被试目录互相包含。输出为公开回归报告、私有 old/new 完整统计文件、私有实际导入路径和独立 Numba 缓存。`--worker/--config` 仅由父脚本启动其内部两个进程。

## 原软件调用

以下沿用 CON04 实际 FreeSurfer 7.3.2 参考日志的全部 flags 和顺序，输入/输出改为占位路径；[原命令来源及日志 SHA](../../validation/fmri/public_ten_20261003/reconstruction_completed/CON04.original-segstats-commands.public.json)已保存。仅在独立原软件环境使用：

```bash
reference_subject_directory=/data/reference/subject
reference_subject_id=subject
reference_asset_directory=/opt/freesurfer
export SUBJECTS_DIR=/data/reference  # 原软件从此处读取该 subject 的表面
mri_segstats --seed 1234 \
  --seg "$reference_subject_directory/mri/wmparc.mgz" \
  --sum "$reference_subject_directory/stats/wmparc.stats" \
  --pv "$reference_subject_directory/mri/norm.mgz" --excludeid 0 \
  --brainmask "$reference_subject_directory/mri/brainmask.mgz" \
  --in "$reference_subject_directory/mri/norm.mgz" \
  --in-intensity-name norm --in-intensity-units MR \
  --subject "$reference_subject_id" --surf-wm-vol \
  --ctab "$reference_asset_directory/WMParcStatsLUT.txt" --etiv
mri_segstats --seed 1234 \
  --seg "$reference_subject_directory/mri/aseg.mgz" \
  --sum "$reference_subject_directory/stats/aseg.stats" \
  --pv "$reference_subject_directory/mri/norm.mgz" --empty \
  --brainmask "$reference_subject_directory/mri/brainmask.mgz" \
  --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray \
  --in "$reference_subject_directory/mri/norm.mgz" \
  --in-intensity-name norm --in-intensity-units MR --etiv \
  --surf-wm-vol --surf-ctx-vol --totalgray --euler \
  --ctab "$reference_asset_directory/ASegStatsLUT.txt" --subject "$reference_subject_id"
```

FNIT 固定实现对应 FreeSurfer 8.2，另读取 SynthSeg sTIV；[8.2 的固定原命令](../../validation/recon_all/python_gpu_port/experimental/SEGSTATS_WMPARC.md)还含 `--stiv stats/synthseg.tiv.dat`。不能将该选项加进本轮 7.3.2 的实际命令来源。

## 最新真实精度、耗时与脑图

正式 CON04 raw T1w/BOLD 整链已完成，原 `aseg_stats`、`wmparc_stats` stage 为 **12.743、10.823 秒**。aseg 标签 77 实际仅一个体素，保存 normMean=83、normStdDev=0、normMin=normMax=83，表内数值全部有限。

[真实同输入完整 writer 回归](../../validation/fmri/public_ten_20261003/reconstruction_completed/CON04.single-voxel-full-writer-regression.public.json)逐次重算完整部分容积，不使用 mock：旧/新 aseg 全文件 SHA 同为 `9ee0fe2a…`，wmparc 同为 `86e65e9e…`，也分别等于原生产保存文件。旧 aseg 捕获一次单体素 RuntimeWarning，新版两表均无警告。两个进程都确认实际导入模块及 aseg 绑定函数；460 份冻结源文件、277 份被试文件、raw 与 LUT 的前后 SHA 全部不变。

| 完整读取、PV 与保存，CPU 单线程 | 旧版（秒） | 修复版（秒） | 文本差异 |
|---|---:|---:|---|
| aseg | 15.025 | 14.668 | 全字节相同 |
| wmparc | 11.038 | 10.965 | 全字节相同 |

独立诊断总墙钟 **55.753 秒**，首次 writer 含 Numba 编译，各时间按自身边界报告，不形成提速结论、不拼入 MRI 整例。此次未新增原软件同输入重跑。历史原软件对照中，aseg 的 45 行/450 字段/21 个 Measure、wmparc 的 70 行/700 字段/7 个 Measure 完全匹配；Python/官方 CLI 分别为 10.34/22.66 秒、10.14/100.34 秒。它们只验证一个已有官方被试的步骤，不代表当前 raw 整链等价。原记录见 [aseg](../../validation/recon_all/python_gpu_port/ASEG_STATS.md)、[wmparc](../../validation/recon_all/python_gpu_port/experimental/SEGSTATS_WMPARC.md)。

当前上游分割和统计的真实差异及 CSF 脑图见[十人重建比较](../../validation/fmri/public_ten_20261003/reconstruction_comparison.md)。本修复不调整这些标签或体积。

## 最近版本与 benchmark 记录

| 版本 | 改动与证据 |
|---|---|
| 2026-10-03 警告修复，模块 `da987f95…` | count>1 才算方差；三个 focused 单元回归通过。真实 CON04 两表完整 PV 重算、全文本零差、旧警告消除。 |
| 正式十例 `1128bc52`，模块 `86422fbb…` | 原始配对数据的 mature 统计；单体素保存结果有限，仍有未使用方差警告。执行源码保持原样。 |
| 2026-09-28 原软件固定输入 | aseg、wmparc 的逐字段步骤验证；自产上游的 raw 整链另行比较。 |

## 参考文献与原软件代码库

- [FNIT aseg 实现](../../src/fnit/recon_all/segstats_aseg_python.py)、[共享 PV/强度实现](../../src/fnit/recon_all/segstats_wmparc_python.py)。
- [固定 FreeSurfer `mri_segstats` 源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/mri_segstats/mri_segstats.cpp)、[MRI/PV 原实现](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mri.cpp)。
- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [DOI](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
