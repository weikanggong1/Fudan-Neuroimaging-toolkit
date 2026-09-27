# WM 和 filled 上游链：2026-09-27 同 T1 受控验收

本次从当前候选的 `nu.mgz`、`aseg.presurf.mgz` 及修正后的 `norm.mgz`、`brainmask.mgz`、Talairach LTA 连续运行。输入 T1 与 2026-09-24 官方 `a_official` 相同。`nu`、`norm`、`brainmask`、`aseg.presurf` 和 Python EntoWM 均与官方 256³ 图像逐体素相同，详见 [输入 JSON](exact_prefix/exact_inputs_comparison.json)。使用旧 v2 `norm`/`brainmask` 的并行对照中，`brain`、`antsdn.brain`、`wm.seg` 分别仍差 332,394、334,658、108,393 个体素，详见 [旧输入](upstream_comparison.json)和[旧分割](denoise_segment_comparison.json)。这些对照定位了误差来自上游输入，而非这三步的同输入算法。

## 调度与结果

新默认 Conda runner 在 `mri_cc` 后依次调用以下步骤；官方命令均以被试的 `mri` 目录为当前目录。表中时间为 gpucw1 上单次墙钟，受共享负载和 NFS 缓存影响。C++ 两步来自固定 FreeSurfer 8.2 提交 `d932c45b7941662ea380a05efef580568b98d41a`，由项目脚本仅用 Conda 编译并链接；其余步骤调用已实现的 Python 函数，无需安装 FreeSurfer 运行包。

| 功能与当前用法 | 官方等价命令 | 对官方不同体素 / 16,777,216 | 当前耗时 |
|---|---|---:|---:|
| 第二次归一化 `normalize_t1_aseg(norm, aseg.presurf, brainmask, brain, device="cpu")` | `mri_normalize -seed 1234 -mprage -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz` | 0 | 134.16 s |
| EntoWM `mri_entowm_seg(nu, entowm, weights, device="cpu")` | `mri_entowm_seg --s SUBJECT` | 0 | 10.70 s |
| ANTs 去噪 `denoise_volume(brain, antsdn.brain)` | `AntsDenoiseImageFs -i brain.mgz -o antsdn.brain.mgz` | 0 | 52.33 s |
| Conda C++ 白质初分割 `_run_native_wm_segment(...)` | `mri_segment -wsizemm 13 -mprage antsdn.brain.mgz wm.seg.mgz` | 0 | 81.63 s |
| Conda C++ 白质编辑 `_run_native_wm_edit(...)` | `mri_edit_wm_with_aseg -keep-in -fix-ento-wm entowm.mgz 3 255 255 -fix-acj aseg.presurf.mgz 255 255 -fill-seg-wm -fix-scm-ha 1 wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz` | 0 | 52.88 s |
| `pretess_mgh(wm.asegedit, "wm", norm, wm)` | `mri_pretess wm.asegedit.mgz wm norm.mgz wm.mgz` | 后续编辑后 `wm.mgz` 为 0 | 7.42 s |
| `fix_ento_wm(wm, entowm, wm, level=3, ...)` | `mri_edit_wm_with_aseg -sa-fix-ento-wm entowm.mgz 3 255 255 wm.mgz wm.mgz` | 0；修改 1,453 个体素 | 2.24 s |
| `fix_ento_wm(wm, aseg.presurf, wm, acj=True, ...)` | `mri_edit_wm_with_aseg -sa-fix-acj aseg.presurf.mgz 255 255 wm.mgz wm.mgz` | 0；修改 598 个体素 | 4.28 s |
| `fill_mgz(wm, aseg.presurf, talairach.lta, SubCorticalMassLUT, filled, cut_log)` | `mri_fill -a ../scripts/ponscc.cut.log -xform transforms/talairach.lta -segmentation aseg.presurf.mgz -ctab SubCorticalMassLUT.txt wm.mgz filled.mgz` | 0 | 81.53 s |

[完整输出 JSON](exact_prefix/exact_chain_comparison.json)核对 `brain`、`antsdn.brain`、`wm.seg`、`wm.asegedit`、`wm`、`filled`：**六个文件各 0 个体素差**，uint8 类型、仿射和 284 字节 MGH 头部逐一相同；`ponscc.cut.log` 字节相同。压缩 MGZ 哈希因命令历史/压缩元数据可能不同。上述步骤在独立中间目录连续执行，尚不是更新后 runner 从空被试目录完成的新整例，不能推断 white/pial、球面配准或脑区统计已一致。

旧 v2 默认从 SynthSeg 标签直接生成 `wm/filled`，与官方分别有 435,674/92,188 个体素值差；其中 `wm` 包含强度编码差异。新默认 Conda 路径使用上述真实 WM 链。旧近似保留为显式 `experimental_approximate_wm=True` 的历史对照；重放旧 v2 输入所得 `wm/filled` 均与旧输出 0 体素差，见[对照 JSON](baseline_compatibility.json)。

## 构建、依赖与时间边界

`build_recon_all_fs_cpp_conda.sh` 在 gpucw1 对固定源码的八目标构建退出 0、185.01 秒；[来源](fs_cpp_eight/build-provenance.txt)和[八个二进制 SHA-256](fs_cpp_eight/bin.sha256)留档。新增 `mri_edit_wm_with_aseg` 对冻结官方输入也独立产生 0 体素差，单次 45.99 秒。官方原整例日志中相同命令为 28.42 秒，但两次负载/日期不同，不能视为稳定速度比较。Python `mri_fill` 的早期冻结同输入验收及 headcw 原生耗时见[既有报告](../../experimental/FILL_ASEG.md)；本次 gpucw1 时间较长，不推断加速。

目标节点 glibc 2.17 曾使 pip 版 `statsmodels 0.15.0`、`scikit-learn` 无法导入；将 `statsmodels 0.14.5`、`pandas 2.2.3`、`scikit-learn 1.5.2` 放进 Conda 依赖后，gpucw1 `import ants` 成功。更新的 [YAML 求解日志](yaml_eight_dryrun.log)在 `CONDA_OVERRIDE_GLIBC=2.17` 下退出 0；没有重复完整创建 YAML 环境。默认权重组新增 EntoWM 模型和 ctab（3,297,222 字节）；默认资产组新增原已校验但未列入默认下载的 `SubCorticalMassLUT.txt`（281 字节）。首次填充因该文件缺失而失败，加入默认资产并校验 SHA-256 后重跑成功。
