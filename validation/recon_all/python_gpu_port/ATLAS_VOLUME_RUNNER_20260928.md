# recon-all 脑区体积图接入：真实 T1 同输入核对

## 输入与输出

`fnit.recon_all.native_free` 现在按顺序调用仓库中的 `label_cortex_volume` 三次，再调用 `label_wm_volume`。输入是 `mri/aseg.mgz`、双侧 `surf/{lh,rh}.{white,pial}`、`label/{lh,rh}.cortex.label` 和三套 `label/{lh,rh}.*.annot`。前三次分别写入 `mri/aparc+aseg.mgz`、`mri/aparc.a2009s+aseg.mgz`、`mri/aparc.DKTatlas+aseg.mgz`；第四次读取刚生成的 `aparc+aseg.mgz`，写入 `mri/wmparc.mgz`。输出都是与输入 `aseg.mgz` 同网格的三维 MGZ 标签图。

皮层步骤在 white 和 pial 表面上寻找法线方向合格的近邻顶点；白质步骤按 5 mm 距离和半球侧别赋值。代码运行在 CPU，使用 NiBabel、NumPy、SciPy 和 PyTorch，不调用 FreeSurfer 程序。此前调度器只取最近 white 顶点，并直接从 `aseg.mgz` 生成 `wmparc.mgz`；这两处已经换成上述已验收函数。

Python 单独调用示例；每个实参分别给出输入、表面目录、标签目录和输出：

```python
from fnit.recon_all.surf2volseg_cortex_python import label_cortex_volume
from fnit.recon_all.surf2volseg_wm_python import label_wm_volume

label_cortex_volume(
    aseg_file="/path/to/sub01/mri/aseg.mgz",  # 三维 aseg 输入
    surface_dir="/path/to/sub01/surf",  # 双侧 white/pial 所在目录
    label_dir="/path/to/sub01/label",  # 双侧 cortex.label 与 aparc.annot 所在目录
    output_file="/path/to/sub01/mri/aparc+aseg.mgz",  # 皮层脑区体积图
    atlas="aparc",  # 可改为 aparc.a2009s 或 aparc.DKTatlas
)
label_wm_volume(
    aparc_aseg_file="/path/to/sub01/mri/aparc+aseg.mgz",  # 上一步输出
    surface_dir="/path/to/sub01/surf",  # 双侧 white 所在目录
    label_dir="/path/to/sub01/label",  # 双侧 cortex.label 与 aparc.annot 所在目录
    output_file="/path/to/sub01/mri/wmparc.mgz",  # 白质脑区体积图
)
```

对应 FreeSurfer 8.2 保存日志的命令是 `mri_surf2volseg --label-cortex --i aseg.mgz --o aparc+aseg.mgz --lh-annot lh.aparc.annot 1000 --rh-annot rh.aparc.annot 2000`，另传双侧 `--*-cortex-mask`、`--*-white`、`--*-pial` 与 `--threads 4`。a2009s 的两个偏移为 11100/12100，DKT 为 1000/2000。白质命令以 `--label-wm --i aparc+aseg.mgz --o wmparc.mgz` 替换皮层模式，两个注释偏移为 3000/4000，其他表面和标签输入相同。逐参数的官方完整命令记录在该被试的 `scripts/recon-all.log` 第 7580、7624、7668、7712 行。

## 冻结真实输入结果

输入为真实 T1 已完成的 FreeSurfer 8.2 被试 `a_official`。测试脚本只软链接其上述输入到新的隔离被试目录，调用当前 runner 的两个体积投影包装函数；官方输出仅用于结果比较，未传入推理。输入 SHA-256、路径和四张输出的文件 SHA-256 见[机器报告](atlas_projection_runner_20260928.json)，复现入口见[脚本](validate_project_atlas_chain.py)。

| 输出 | 与官方不同的体素 / 总体素 | 仿射最大差 |
| --- | ---: | ---: |
| `aparc+aseg.mgz` | 0 / 16,777,216 | 0 |
| `aparc.a2009s+aseg.mgz` | 0 / 16,777,216 | 0 |
| `aparc.DKTatlas+aseg.mgz` | 0 / 16,777,216 | 0 |
| `wmparc.mgz` | 0 / 16,777,216 | 0 |

本次 headcw 常驻 Python 进程中，三张皮层图合计 **11.74 秒**，白质图 **4.26 秒**，含 I/O。另一次独立同输入测试的官方 CPU 命令分别耗时 27.20、24.43、22.34、9.69 秒；执行方式和共享节点负载不同，不能用这两组数值计算受控提速。独立算法对照和边界见[皮层体积图](SURF2VOLSEG_CORTEX.md)及[白质体积图](experimental/SURF2VOLSEG_WM.md)。四张 `.mgz` 压缩文件的 SHA-256 与官方不同；本次可确认的是全部体素及仿射一致。

此结果使用冻结的官方 `aseg`、表面和注释。FNIT 自产的这些上游文件仍有差异，当前未重跑完整重建，因此四张体积图的整例一致性尚未成立。
