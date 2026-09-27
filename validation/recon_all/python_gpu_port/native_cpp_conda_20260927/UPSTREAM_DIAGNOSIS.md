# 上游 MRI 与左侧原始表面诊断

核对时间：2026-09-27 11:47 CST，gpucw1。候选为 `reconall_conda_cpp_e2e_20260927/subjects/sub01`，官方参考为 `reconall_benchmark_pair_ac_20260924/official_subjects/a_official`。两者使用同一 T1；下表是**完整流水线输出之间**的比较，不能据此判断替换的某个 C++ 命令与官方同输入算法是否一致。候选仍在运行，本报告只覆盖当时已有的文件。

## 体积文件

全部 MRI 均为 256×256×256，体素到 RAS 的仿射矩阵逐元素相同，因而下表可直接逐体素比较。非零 Dice 仅描述非零支持区域，不代表同值标签或强度一致。

| 文件 | 不同值体素 / 16,777,216 | 候选/官方非零体素 | 非零 Dice | 主要发现 |
| --- | ---: | ---: | ---: | --- |
| `orig.mgz` | 0 | 2,251,350 / 2,251,350 | 1.000000 | 全体素一致 |
| `nu.mgz` | 0 | 2,226,863 / 2,226,863 | 1.000000 | 全体素一致 |
| `T1.mgz` | 0 | 2,226,863 / 2,226,863 | 1.000000 | 全体素一致 |
| `brainmask.mgz` | 38 | 1,269,264 / 1,269,264 | 0.999985 | 仅 38 个体素值不同；非零数量相同 |
| `aseg.mgz` | 107,249 | 1,285,092 / 1,224,118 | 0.975369 | 候选缺少官方标签 251–255，双侧皮质标签差异明显 |
| `wm.mgz` | 435,774 | 352,799 / 416,149 | 0.863400 | 候选仅含 0/255；官方非零值分布为 73–255，包括大量 250 |
| `filled.mgz` | 90,420 | 352,788 / 401,211 | 0.880410 | 左侧 255 标签 Dice 0.877826，右侧 127 标签 Dice 0.881658 |

候选 `aseg.mgz` 是 float32 存储，但所有体素仍为整数标签；官方文件是 int32。关键标签 Dice：2 为 0.936103，3 为 0.870541，41 为 0.937317，42 为 0.848293。其余多种深部结构标签达到 1.0。`wm.mgz` 的不同值体素数同时包含灰度编码差异，因此判断白质空间范围时应看非零 Dice，不能把所有 435,774 个体素直接解释为白质边界错误。

## 差异出现在哪个阶段

同一网格上的 `synthseg.rca.mgz` 在两版间 **16,777,216/16,777,216 体素完全一致**，数据类型也同为 float32。`aseg.auto.mgz` 与 `aseg.presurf.mgz` 各有 2,263 个不同体素，主要是官方标签 251–255 在候选中仍为 2/41 等；例如官方 251→候选 41 有 486 个、官方 255→候选 41 有 365 个。候选的 `aseg.auto.mgz` 到最终 `aseg.mgz` 无任何变化，官方同两文件之间有 **104,986** 个体素变化。最终两版 `aseg.mgz` 的主要不同转移为官方 0→候选 42（35,154 个）、0→3（25,899 个）、2→3（14,628 个）、41→42（11,645 个）、42→41（9,305 个）、3→2（7,000 个）。因此该例的最终 `aseg` 偏差主要发生于 SynthSeg 之后的标签赋值与官方后处理，而非 SynthSeg 推理输出。

## `lh.orig` 几何

| 指标 | 候选 | 官方 | 差异 |
| --- | ---: | ---: | ---: |
| 顶点数 | 117,076 | 106,622 | +10,454（+9.80%） |
| 三角面数 | 234,148 | 213,240 | +20,908（+9.80%） |
| 三角面总面积 | 72,881.53 mm² | 64,683.41 mm² | +12.67% |
| 三角网格有向体积的绝对值 | 177,766.23 mm³ | 200,090.46 mm³ | −11.16% |

面积由原始三角面计算；体积为网格三重积的绝对值，只作为几何诊断，不能当作 FreeSurfer 报告的皮质或灰质体积。两版顶点数和面数不同，**当前 `lh.orig` 无法按顶点索引逐点比较厚度、曲率等值**；后续可做空间对应后的几何比较。候选 `rh.orig` 在本次取样时尚未生成，故未作右侧表面判断。`wm`、`filled` 和左侧网格均明显不同，足以解释为何下游即使使用同一 C++ 算法也无法自动得到与官方相同的表面指标；具体因果贡献还需对每个表面阶段做同输入配对测试。

方法：Conda 环境内 `nibabel` 解码 MGZ，使用 NumPy 比较体素值、标签和仿射矩阵；`lh.orig` 面积与体积直接由三角网格计算。读取期间未修改两个被试目录。


## 官方 `aseg.auto` 到最终 `aseg` 的命令链与可接入性

来源：官方参考被试 `scripts/recon-all.log` 的命令记录（约第 1891、1938、7018、7500、7515 行）；以下时间是该官方运行日志中的墙钟时间，不是新实现的 benchmark。

| 顺序 | 官方命令/产物 | 显式输入与本次候选状态 | 官方耗时 |
| --- | --- | --- | ---: |
| 1 | `seg2cc --s a_official` 调 `mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz -lta transforms/cc_up.lta a_official` | 官方 `aseg.auto_noCCseg.mgz` 是 `synthseg.rca.mgz` 的链接；候选已有全体素相同的 `synthseg.rca.mgz` 和 `norm.mgz`，但尚无此链接。`cc_up.lta` 是此命令的输出，不是额外输入。候选 `norm.mgz` 与官方有 494,932 个体素值不同。 | 130.45 s |
| 2 | `cp aseg.auto.mgz aseg.presurf.mgz` | 候选已有这两个文件，当前内容仍等于 SynthSeg 输出。 | — |
| 3 | `mris_volmask --aseg_name aseg.presurf --label_left_white 2 --label_left_ribbon 3 --label_right_white 41 --label_right_ribbon 42 --save_ribbon --parallel a_official`，生成 `mri/ribbon.mgz`、双侧 ribbon | 候选已有 `aseg.presurf.mgz` 和双侧 `white`/`pial`，但其表面与官方不同；当前尚无 `ribbon.mgz`。 | 日志报告 244.47 s |
| 4 | `mri_relabel_hypointensities aseg.presurf.mgz ../surf aseg.presurf.hypos.mgz` | 候选有 `aseg.presurf.mgz` 和双侧 `white`，尚无 `aseg.presurf.hypos.mgz`。官方此文件相对 `aseg.presurf.mgz` **实际仅 1 个体素值不同**。 | 27.13 s |
| 5 | `mri_surf2volseg --o aseg.mgz --i aseg.presurf.hypos.mgz --fix-presurf-with-ribbon mri/ribbon.mgz --threads 4 --lh-cortex-mask label/lh.cortex.label --lh-white surf/lh.white --lh-pial surf/lh.pial --rh-cortex-mask label/rh.cortex.label --rh-white surf/rh.white --rh-pial surf/rh.pial` | 候选已有双侧 `white`/`pial` 和左侧 cortex label；在 11:52 CST 快照中还缺 `ribbon.mgz`、`aseg.presurf.hypos.mgz`、右侧 cortex label。后者可能在运行中的流程稍后生成。 | 10.32 s |

官方 `mri_cc` 后的 2,263 个变化属于小范围标签补全。补编一个 `mri_cc` target 并使用现有 SynthSeg 与 norm 可以**尝试**复现这一步，不需新模型或模板；但候选 norm 不同，尚未实测同输入输出，不能承诺恰好恢复全部 2,263 个体素。更大范围的最终 `aseg` 差异来自表面与 ribbon 回填。若只增加两个 target，`mris_volmask` 加 `mri_surf2volseg` 是有意义的候选实验；必须先生成右侧 cortex label，并为缺失的 `aseg.presurf.hypos.mgz` 提供上游输出（该例仅 1 体素差，可在探索实验中明确记录近似）。这两个 target 依赖当前近似的 white/pial，因此不能保证最终标签更接近官方，更不能作为精度验收。要完整复现官方这条链，至少还须处理 `mri_cc` 和 `mri_relabel_hypointensities`，并首先改善输入表面。上述 C++ 替换均未在本轮编译或运行。
