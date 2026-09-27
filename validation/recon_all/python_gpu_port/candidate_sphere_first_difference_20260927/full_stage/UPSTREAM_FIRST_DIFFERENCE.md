# 归档官方 `sphere` 差异的上游首差

修复后的候选 `lh.sphere` 与**同候选输入** FreeSurfer 8.2 输出逐坐标、逐面完全一致；两者相对 2026-09-24 归档官方 `lh.sphere` 均差 2.959344 mm。因此只读比较此前保存的候选与归档官方 LH 表面，按 recon-all 顺序查第一处有序几何差异。完整数据、每份文件 SHA-256 和顶点分布见[阶段审计](upstream_surface_audit.json)；此处没有重新运行 T1 重建或球面优化。

| 表面阶段 | 顶点 | 精确坐标分量 | 均值/P99/最大位移，mm | >0.1 mm 顶点 |
| --- | ---: | ---: | ---: | ---: |
| `orig.premesh` | 101,689 | 305,067 / 305,067 | 0 / 0 / 0 | 0 |
| `orig.nofix`、`smoothwm.nofix`、`inflated.nofix`、`qsphere.nofix` | 各 102,764 | 各 308,292 / 308,292 | 0 / 0 / 0 | 0 |
| `orig` | 106,622 | 319,866 / 319,866 | 0 / 0 / 0 | 0 |
| **`white.preaparc`** | 106,622 | **99,139 / 319,866** | **0.000421 / 0.006999 / 0.630004** | **50** |
| `smoothwm` | 106,622 | 150,225 / 319,866 | 0.000311 / 0.006172 / 0.188170 | 18 |
| `inflated` | 106,622 | 301 / 319,866 | 0.001061 / 0.002213 / 0.004201 | 0 |
| 修复后 `sphere` | 106,622 | 0 / 319,866 | 2.959344 / 5.437959 / 7.066581 | 106,579 |

各行的有序三角面均与对应官方阶段相同。前六张表面的全文件 SHA-256 有差异；逐坐标和逐面比较证实几何相同，差别限于几何之外的文件内容。`white.preaparc` 是**第一张已保存且几何不等**的表面，后续 `smoothwm`、`inflated` 和非线性球面优化保留并放大了这条输入链上的差异。本次只读比较不能单独分解 `inflated` 与 `smoothwm` 各自对球面位移的贡献。

该 `white.preaparc` 由 Conda 源码编译的 `mris_place_surface` 生成，执行的是官方的 `--white` 放置命令：

```bash
mris_place_surface --adgws-in surf/autodet.gw.stats.lh.dat --wm mri/wm.mgz \
  --threads 4 --invol mri/brain.finalsurfs.mgz --lh --i surf/lh.orig \
  --o surf/lh.white.preaparc --white --seg mri/aseg.presurf.mgz \
  --restore-255 --nsmooth 5 --rip-bg-no-annot --rip-bg --rip-bg-lof \
  --restore-255 --outvol mri/mrisps.wpa.mgz
```

[输入核对](white_place_input_audit.json)显示 `wm.mgz`、`aseg.presurf.mgz` 和 `brain.finalsurfs.mgz` 各 16,777,216 个体素逐个相同，空间仿射最大差 0 mm；`autodet.gw.stats.lh.dat` 字节相同；输入 `lh.orig` 有序几何也完全相同。三张 MGZ 的**文件**哈希不同，因此本轮不能把文件头因素和 Conda/官方二进制数值路径完全分开。此前[冻结官方输入的放置诊断](../../native_cpp_conda_20260927/WHITE_PREAPARC_FIRST_DIVERGENCE_20260927.md)在同一真实 T1、逐字节相同的输入文件上得到相同量级的 50 个 >0.1 mm 顶点，并将最早可见力差异约束在第 5 次更新后、全局 SSE 差异在第 7 次更新；它支持继续排查 C++ 浮点积分，但尚未证明唯一错误表达式。本轮没有改动该放置器。

`audit_upstream_surfaces.py` 的 CLI 输入为 `--candidate-surf-dir`（候选 `surf` 目录）、`--official-surf-dir`（归档官方 `surf` 目录）、`--repaired-sphere`（本轮修复后的 `lh.sphere`）及 `--report-json`（输出 JSON）；它返回每个保存阶段的文件哈希、顶点/面数、同索引精确数及位移分布。`audit_white_place_inputs.py` 的输入为 `--candidate-subject`、`--official-subject` 两个 subject 目录以及 `--report-json`；输出三张 MGZ 的体素差数、仿射差和阈值文件字节比较。两者只读，不调用预装神经影像程序。
