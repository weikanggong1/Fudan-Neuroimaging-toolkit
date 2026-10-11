# 当前条件调用清单

| 阶段 | 原版程序 | native缺省CUDA实际分支 | 独立Python实现 |
| --- | --- | --- | --- |
| N4 | fnit_n4_itk | 执行 | SimpleITK完整CPU/Torch完整拟合；`src/fnit/recon_all/input_n4_chain.py:36` |
| WM segmentation | mri_segment | 执行 | Numba有序核心+Torch histogram/planar；`src/fnit/recon_all/native_free.py:78` |
| WM aseg edit | mri_edit_wm_with_aseg | 执行 | 完整Numba/CPU编辑；hybrid另可显式选择；`src/fnit/recon_all/native_free.py:138` |
| Topology | mris_fix_topology_fnit | 执行 | digital或显式完整自有GA；两者算法不同；`src/fnit/recon_all/native_free.py:351` |
| nofix/inflated | mris_inflate | 执行 | 自有Torch inflation/CPU动态顺序；`src/fnit/recon_all/native_free.py:457` |
| white.preaparc | mris_place_surface | 执行 | 完整Python/Numba/Torch候选与清理；`src/fnit/recon_all/native_free.py:709` |
| white/pial | mris_place_surface | 执行 | 完整四轮自有放置；`src/fnit/recon_all/native_free.py:905` |
| avg_curv | mrisp_paint | 执行 | 自有采样及有序marked平均；`src/fnit/recon_all/native_free.py:581` |
| curv.stats | mris_curvature_stats | 执行 | 自有Torch离散曲率+Numba/system libm；`src/fnit/recon_all/native_free.py:609` |
| defects volume | mri_label2vol | 执行 | Torch raw原域投射，LH→RH共享卷；`src/fnit/recon_all/native_free.py:639` |
| GCA | mri_em_register | 默认CUDA auto已走自有Torch；CPUauto有能力探测 | 自有EM CPU+Torch候选评分/求逆；`src/fnit/recon_all/native_free.py:164` |
| vertex metrics | mris_place_surface | 默认CUDA自有指标；native仍发现该二进制供放置使用 | 自有area/thickness/curvature；`src/fnit/recon_all/native_free.py:509` |
| MNI postprocess | mri_warp_convert, mri_ca_register, mri_convert | 默认CUDA走自有postprocess | 自有GPU转换/逆场/采样check；`src/fnit/recon_all/mni_nonlinear_chain.py:20` |

上述原程序在本轮独立Python配方中均未选中；静态判断不代表实测执行次数为零。
