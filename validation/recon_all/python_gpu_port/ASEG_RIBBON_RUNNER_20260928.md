# recon-all ribbon 与最终 aseg：真实 T1 冻结输入核对

`fnit-recon-all` 已接入仓库已有的三个 Python 函数。`write_ribbon` 读取 `mri/aseg.presurf.mgz`、双侧 white/pial 和 `FreeSurferColorLUT.txt`，输出 `mri/{ribbon,lh.ribbon,rh.ribbon}.mgz`；`relabel_volume` 读取同一 `aseg.presurf.mgz` 和双侧 white，输出 `mri/aseg.presurf.hypos.mgz`；`fix_presurf_volume` 再读取 hypos、ribbon、双侧 white/pial 和 `label/{lh,rh}.cortex.label`，输出最终 `mri/aseg.mgz`。这些都是与原 `aseg.presurf.mgz` 同网格的三维 MGZ。当前调度器按官方日志先写 ribbon，再重标 hypointensity，最后修正 aseg；前两步只读同一冻结前序输入，互不依赖。

三个函数都在 CPU 上使用 NiBabel、NumPy、SciPy 或 PyTorch，不调用 FreeSurfer 可执行程序。Python 具名参数示例：

```python
from fnit.recon_all.volmask_python import write_ribbon
from fnit.recon_all.relabel_hypointensities_python import relabel_volume
from fnit.recon_all.surf2volseg_fix_python import fix_presurf_volume

write_ribbon(
    template_file="/path/to/sub01/mri/aseg.presurf.mgz",  # 输出体素网格
    surface_dir="/path/to/sub01/surf",  # 双侧 white/pial 表面目录
    output_dir="/path/to/sub01/mri",  # ribbon、lh.ribbon、rh.ribbon 的输出目录
    color_lut_file="/path/to/assets/FreeSurferColorLUT.txt",  # 外置颜色表
)
relabel_volume(
    aseg_file="/path/to/sub01/mri/aseg.presurf.mgz",  # 预表面分割
    surface_dir="/path/to/sub01/surf",  # 双侧 white 所在目录
    output_file="/path/to/sub01/mri/aseg.presurf.hypos.mgz",  # 重标后的分割
)
fix_presurf_volume(
    aseg_file="/path/to/sub01/mri/aseg.presurf.hypos.mgz",  # 上一步输出
    ribbon_file="/path/to/sub01/mri/ribbon.mgz",  # 皮层 ribbon
    surface_dir="/path/to/sub01/surf",  # 双侧 white/pial 所在目录
    label_dir="/path/to/sub01/label",  # 双侧 cortex.label 所在目录
    output_file="/path/to/sub01/mri/aseg.mgz",  # 最终体积分割
)
```

官方 FreeSurfer 8.2 保存日志中的对应命令是 `mris_volmask --aseg_name aseg.presurf --label_left_white 2 --label_left_ribbon 3 --label_right_white 41 --label_right_ribbon 42 --save_ribbon --parallel SUBJECT`、`mri_relabel_hypointensities aseg.presurf.mgz ../surf aseg.presurf.hypos.mgz`，以及 `mri_surf2volseg --i aseg.presurf.hypos.mgz --fix-presurf-with-ribbon ribbon.mgz --o aseg.mgz`，后一命令还传双侧 `--*-cortex-mask`、`--*-white`、`--*-pial`。完整命令见官方被试 `scripts/recon-all.log` 第 7018、7500、7515 行。

## 同输入结果与时间

一例真实 T1 完成的 FreeSurfer 8.2 被试提供 `aseg.presurf.mgz`、表面和皮层标签。测试在新隔离目录软链接这些输入，用仓库 Python 函数生成五张图，再读取官方结果核对。脚本及输入 SHA-256 见[验证入口](validate_aseg_ribbon_chain.py)和[机器报告](aseg_ribbon_runner_20260928.json)。验证脚本先运行重标、后写 ribbon；当前调度器为官方次序，两步没有共享输出或相互依赖，因此此单例的数值结果适用于这两种执行次序。调度器本身未在本次整例运行。

| 输出 | 与官方不同的体素 / 总体素 | 仿射最大差 |
| --- | ---: | ---: |
| `aseg.presurf.hypos.mgz` | 0 / 16,777,216 | 0 |
| `ribbon.mgz` | 0 / 16,777,216 | 0 |
| `lh.ribbon.mgz` | 0 / 16,777,216 | 0 |
| `rh.ribbon.mgz` | 0 / 16,777,216 | 0 |
| `aseg.mgz` | 0 / 16,777,216 | 0 |

本次 headcw 常驻 Python 测得 hypointensity 重标 **3.10 秒**、ribbon **2.95 秒**、最终 aseg 修正 **3.84 秒**，均含各自 I/O。此前独立同输入测试记录的官方原生命令分别为 11.14 秒（重标）、3.28 秒（aseg 修正）；ribbon 的本次配对官方时间未取得。测试环境和执行时间不同，不能由此计算整例提速。五张压缩 MGZ 与官方文件字节不同；本次验收范围是全体素及仿射一致。

这项验证沿用了官方正确的 `aseg.presurf` 与表面。FNIT 自产上游仍有差异；最终体积分割和随后的脑区统计尚无同一 T1 整例一致性证据。
