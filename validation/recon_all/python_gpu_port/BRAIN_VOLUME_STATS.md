# 脑体积统计：真实数据验证

`compute_brain_volume_stats(subject, aseg_lut)` 对应 FreeSurfer 8.2 的 `ComputeBrainVolumeStats2`。`subject` 是包含 `mri/aseg.mgz`、`mri/brainmask.mgz`、`mri/transforms/talairach.xfm` 和双侧 `surf/{lh,rh}.{white,pial}` 的被试目录；`aseg_lut` 是 `ASegStatsLUT.txt` 路径。MGZ 在该被试的 conform 网格中，表面使用 surface RAS。函数返回 `{统计名: 体积}` 字典，单位为 mm³；主流程把结果写入 `stats/brainvol.stats`。输入缺失或格式错误时抛出异常。

```python
from fnit.recon_all.brain_volume_stats_python import compute_brain_volume_stats

brain_volumes = compute_brain_volume_stats(
    subject="/data/subjects/sub02",  # 已完成体积分割及双侧 white/pial 的被试目录
    aseg_lut="/data/fnit-assets/ASegStatsLUT.txt",  # 与当前图谱匹配的标签清单
)
print(brain_volumes["CerebralWhiteMatterVol"])  # 左右大脑白质总体积，单位 mm³
```

官方对应命令是 `SUBJECTS_DIR=/data/subjects mri_brainvol_stats --subject sub02`；[固定版本原实现](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/cma.cpp)的 `ComputeBrainVolumeStats2` 根据 `talairach.xfm` 把未分侧的标签 77 变换到 MNI305 RAS，以 X≤0 归左、X>0 归右。FNIT 使用 nibabel 的体素到 scanner RAS 仿射，再乘 XFM 第一行，按相同规则统计。2026-09-29 的独立真实 T1 `sub-02` 有 4 个标签 77 体素，左侧 2 个、右侧 2 个。同一候选输入由独立的官方 8.2 命令计算后，16 项中 12 项完全相同，左右白质体积均完全相同；其余 4 项仅是表面体积求和差异，最大 0.000826 mm³。官方程序只在隔离的验证目录运行，没有进入 FNIT 的重建调度器。

此前冻结官方 `sub-01` 输入的 16 项中，12 项与官方数值一致，另 4 项的最大体积误差为 0.000295 mm³，Python CPU 耗时 1.96 s；[原始逐项记录](brain_volume_stats_cpu_report.json)只证明同输入计算，不代表 FNIT 自产上游整例吻合。

参考文献：Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
